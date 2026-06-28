"""Single-GPU-slot manager for serial heavy-model execution.

On a small GPU (e.g. an RTX 4060 with ~7-8 GB) the SAM3 segmentation server and
the Contact-GraspNet server cannot comfortably stay resident at the same time.
But within a single ``grasp(object)`` call they are used *sequentially* -- first
SAM3 segments the object, then GraspNet plans a grasp on that mask.  So we only
ever need **one heavy model on the GPU at a time**.

This manager runs each heavy perception model as a subprocess (reusing CaP-X's
already-tested ``capx/serving/launch_*_server.py``) and guarantees that starting
one server first tears down the other, freeing its VRAM.  PyRoKi IK (8116) is
light and is intentionally *not* managed here -- keep it resident.

The CaP-X client functions (``init_sam3``/``init_contact_graspnet``) are thin
HTTP clients that hit fixed localhost ports, so swapping the backing server
process is transparent to them.

Usage
-----
    slot = GpuSlotManager(device="cuda")
    slot.ensure("sam3")       # SAM3 resident, GraspNet (if any) torn down
    ... call segment client (port 8114) ...
    slot.ensure("graspnet")   # SAM3 torn down, GraspNet resident
    ... call grasp client (port 8115) ...
    slot.stop_all()
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from dataclasses import dataclass

# name -> (server script, port).  Ports must match the client SERVICE_URLs in
# capx/integrations/vision/{sam3,graspnet}.py.
_SERVERS: dict[str, tuple[str, int]] = {
    "sam3": ("capx/serving/launch_sam3_server.py", 8114),
    "graspnet": ("capx/serving/launch_contact_graspnet_server.py", 8115),
}


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


@dataclass
class _Running:
    name: str
    proc: subprocess.Popen


class GpuSlotManager:
    """Ensures at most one heavy perception server holds the GPU at a time.

    Parameters
    ----------
    device:
        CUDA device string passed to the server (e.g. ``"cuda"`` or ``"cuda:0"``).
    host:
        Host the servers bind to (must match the client SERVICE_URLs).
    startup_timeout:
        Seconds to wait for a freshly launched server's port to come up.  The
        CaP-X servers load their model *before* calling ``uvicorn.run``, so an
        open port implies the model is ready.
    python_exe:
        Interpreter used to launch the server subprocess (defaults to the
        current interpreter, i.e. the capx conda env).
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
        self._current: _Running | None = None

    # ------------------------------------------------------------------ #
    def ensure(self, name: str) -> None:
        """Make ``name`` the resident heavy server, tearing down any other.

        Idempotent: if ``name`` is already resident (or an externally launched
        server is already serving its port), this is a no-op.
        """
        if name not in _SERVERS:
            raise ValueError(f"unknown heavy server '{name}', expected {list(_SERVERS)}")
        script, port = _SERVERS[name]

        # Already the current managed server?
        if self._current is not None and self._current.name == name:
            if self._current.proc.poll() is None and _port_open(self.host, port):
                return
            # died -> fall through to relaunch

        # An externally pre-launched server already owns this port: defer to it
        # (big-GPU setups where the user ran launch_servers.py themselves).
        if self._current is None and _port_open(self.host, port):
            return

        # Evict whatever is currently loaded to free VRAM.
        self._teardown_current()

        # Launch the requested server.
        proc = subprocess.Popen(
            [self.python_exe, script, "--device", self.device,
             "--port", str(port), "--host", self.host],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._current = _Running(name=name, proc=proc)
        self._wait_until_ready(name, port, proc)

    # ------------------------------------------------------------------ #
    def _wait_until_ready(self, name: str, port: int, proc: subprocess.Popen) -> None:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} server exited during startup (rc={proc.returncode})")
            if _port_open(self.host, port):
                return
            time.sleep(1.0)
        self._teardown_current()
        raise TimeoutError(f"{name} server did not come up within {self.startup_timeout}s")

    def _teardown_current(self) -> None:
        if self._current is None:
            return
        proc = self._current.proc
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)
        self._current = None
        # Give the driver a moment to reclaim VRAM before the next load.
        time.sleep(1.0)

    def stop_all(self) -> None:
        self._teardown_current()

    def __enter__(self) -> "GpuSlotManager":
        return self

    def __exit__(self, *exc) -> None:
        self.stop_all()
