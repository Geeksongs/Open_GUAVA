"""Auto start/stop of the background services GUAVA needs.

So a run is a single command: the runner starts the LLM proxy and PyRoKi (owning
them -- any stray instance on those ports is killed first), and on exit tears
EVERYTHING down (proxy, PyRoKi, and all perception servers). In serial-GPU mode
it also frees stray perception servers up front, since the single-GPU-slot
manager must own them exclusively -- a leftover GraspNet hogging VRAM is what
makes SAM3 fail to load.

Net effect: every run opens all services on start and closes all of them on
exit, leaving no background processes behind.
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import time

# Perception ports owned by the serial GPU-slot manager (must be free for it).
PERCEPTION_PORTS = (8114, 8113, 8115, 8117)  # SAM3, SAM2, GraspNet, OWL-ViT


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _pids_on_port(port: int) -> list[int]:
    """Return PIDs listening on ``port`` (via ``ss``)."""
    try:
        out = subprocess.run(
            ["ss", "-lntp"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    pids: list[int] = []
    for line in out.splitlines():
        if f":{port} " in line:
            pids += [int(m) for m in re.findall(r"pid=(\d+)", line)]
    return pids


def kill_port(port: int) -> None:
    for pid in _pids_on_port(port):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


class ServiceManager:
    """Brings up / tears down the LLM proxy and PyRoKi for a GUAVA run."""

    def __init__(
        self,
        key_file: str = ".openrouterkey",
        device: str = "cuda",
        host: str = "127.0.0.1",
        python_exe: str | None = None,
    ) -> None:
        self.key_file = key_file
        self.device = device
        self.host = host
        self.python_exe = python_exe or sys.executable
        self._started: list[tuple[str, subprocess.Popen]] = []

    # ------------------------------------------------------------------ #
    def _launch(self, name: str, cmd: list[str], port: int, timeout: float = 240.0) -> None:
        # Always own the service: if something is already on the port (a stray
        # from a previous run), kill it and start fresh so we can guarantee a
        # clean shutdown at the end.
        if port_open(port, self.host):
            print(f"[services] {name}: clearing stray server on :{port}")
            kill_port(port)
            time.sleep(2.0)
        print(f"[services] starting {name} on :{port} ...")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._started.append((name, proc))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited during startup (rc={proc.returncode})")
            if port_open(port, self.host):
                print(f"[services] {name} ready on :{port}")
                return
            time.sleep(1.0)
        raise TimeoutError(f"{name} did not come up within {timeout}s")

    def ensure_proxy(self, port: int = 8110) -> None:
        self._launch(
            "LLM proxy",
            [self.python_exe, "capx/serving/openrouter_server.py",
             "--key-file", self.key_file, "--port", str(port)],
            port,
        )

    def ensure_pyroki(self, port: int = 8116) -> None:
        self._launch(
            "PyRoKi",
            [self.python_exe, "capx/serving/launch_pyroki_server.py",
             "--device", self.device, "--port", str(port), "--host", self.host,
             "--robot", "panda_description", "--target-link", "panda_hand"],
            port,
        )

    def free_perception_ports(self) -> None:
        """Kill any stray SAM3/SAM2/GraspNet/OWL-ViT so the slot manager owns them."""
        for p in PERCEPTION_PORTS:
            if port_open(p, self.host):
                print(f"[services] freeing stray perception server on :{p}")
                kill_port(p)
        time.sleep(2.0)

    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Stop everything: the proxy/PyRoKi we started + any perception servers."""
        for name, proc in reversed(self._started):
            if proc.poll() is None:
                print(f"[services] stopping {name} ...")
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._started.clear()
        # Also clear any perception servers left by the GPU-slot manager.
        for p in PERCEPTION_PORTS:
            if port_open(p, self.host):
                print(f"[services] stopping perception server on :{p}")
                kill_port(p)

    def __enter__(self) -> "ServiceManager":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
