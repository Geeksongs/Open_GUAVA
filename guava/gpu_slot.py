"""Single-GPU-slot manager for serial heavy-model execution.

On a small GPU (e.g. an RTX 4060 with ~7-8 GB) the segmentation models
(OWL-ViT + SAM2, or SAM3) and the Contact-GraspNet model cannot all stay
resident at once -- their idle footprints plus GraspNet's inference transient
overflow VRAM.  But within a single ``grasp(object)`` call they are used in a
strict sequence: first the object is segmented, then a grasp is planned on that
mask.  So we only ever need **one model group on the GPU at a time**.

This manager runs each perception model as a subprocess (reusing CaP-X's
already-tested ``capx/serving/launch_*_server.py``) and keeps exactly one
*group* resident, tearing down the others to free their VRAM:

    group "seg"      -> OWL-ViT (8117) + SAM2 (8113)     [segmenter="sam2"]
                     -> SAM3 (8114)                       [segmenter="sam3"]
    group "graspnet" -> Contact-GraspNet (8115)

PyRoKi IK (8116) is light (JAX/CPU, ~0 VRAM) and is intentionally not managed
here -- launch it once and leave it resident.

The CaP-X client functions hit fixed localhost ports, so swapping the backing
server process is transparent to them.  The cost is a model (re)load whenever the
resident group changes -- the price of running everything on one small GPU.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

# name -> (server script, port, extra CLI args)
_SERVERS: dict[str, tuple[str, int, list[str]]] = {
    "owlvit": ("capx/serving/launch_owlvit_server.py", 8117,
               ["--model-name", "google/owlv2-base-patch16-ensemble"]),
    "sam2": ("capx/serving/launch_sam2_server.py", 8113, []),
    "sam3": ("capx/serving/launch_sam3_server.py", 8114, []),
    "graspnet": ("capx/serving/launch_contact_graspnet_server.py", 8115, []),
}

# Logical groups that are mutually exclusive on the GPU.
_GROUPS: dict[str, list[str]] = {
    "seg_sam2": ["owlvit", "sam2"],
    "seg_sam3": ["sam3"],
    "graspnet": ["graspnet"],
}


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


class GpuSlotManager:
    """Keeps exactly one mutually-exclusive model group resident on the GPU.

    Parameters
    ----------
    device:        CUDA device string for the servers.
    host:          Host the servers bind to (matches the client SERVICE_URLs).
    startup_timeout: Seconds to wait for a launched server's port to come up.
                   (CaP-X servers load their model *before* ``uvicorn.run``, so
                   an open port implies the model is ready.)
    python_exe:    Interpreter for the server subprocess (defaults to current).
    """

    def __init__(
        self,
        device: str = "cuda",
        host: str = "127.0.0.1",
        startup_timeout: float = 240.0,
        python_exe: str | None = None,
    ) -> None:
        self.device = device
        self.host = host
        self.startup_timeout = startup_timeout
        self.python_exe = python_exe or sys.executable
        self._running: dict[str, subprocess.Popen] = {}
        self._group: str | None = None

    # ------------------------------------------------------------------ #
    def ensure_group(self, group: str) -> None:
        """Make ``group`` the sole resident model group, evicting the others."""
        if group not in _GROUPS:
            raise ValueError(f"unknown group '{group}', expected {list(_GROUPS)}")
        if self._group == group and all(
            _port_open(self.host, _SERVERS[n][1]) for n in _GROUPS[group]
        ):
            return

        wanted = set(_GROUPS[group])
        # Evict any managed server not in the wanted group (frees VRAM).
        for name in list(self._running):
            if name not in wanted:
                self._teardown(name)

        # Start any wanted server not already serving.
        for name in _GROUPS[group]:
            self._start(name)

        self._group = group

    # ------------------------------------------------------------------ #
    def _start(self, name: str) -> None:
        script, port, extra = _SERVERS[name]
        if _port_open(self.host, port):
            return  # already serving (managed or externally launched)
        env = dict(os.environ)
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        proc = subprocess.Popen(
            [self.python_exe, script, "--device", self.device,
             "--port", str(port), "--host", self.host, *extra],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )
        self._running[name] = proc
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} server exited during startup (rc={proc.returncode})")
            if _port_open(self.host, port):
                return
            time.sleep(1.0)
        self._teardown(name)
        raise TimeoutError(f"{name} server did not come up within {self.startup_timeout}s")

    def _teardown(self, name: str) -> None:
        proc = self._running.pop(name, None)
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)
        time.sleep(1.0)  # let the driver reclaim VRAM before the next load

    def stop_all(self) -> None:
        for name in list(self._running):
            self._teardown(name)
        self._group = None

    def __enter__(self) -> "GpuSlotManager":
        return self

    def __exit__(self, *exc) -> None:
        self.stop_all()
