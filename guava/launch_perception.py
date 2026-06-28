"""Launch the perception/motion servers GUAVA needs, for a chosen segmenter.

Brings up the right set of CaP-X servers as background subprocesses and waits
until each port is serving (the CaP-X servers load their model *before*
``uvicorn.run``, so an open port implies the model is ready):

  segmenter="sam2"  -> OWL-ViT (8117) + SAM2 (8113) + GraspNet (8115) + PyRoKi (8116)
  segmenter="sam3"  -> SAM3 (8114) + GraspNet (8115) + PyRoKi (8116)

On a small GPU these can run resident together (OWL-ViT/SAM2/GraspNet are each a
few hundred MB to ~3 GB). If you OOM, use the serial-GPU slot manager instead
(``GuavaTools(serial_gpu=True)``).

Usage
-----
    python guava/launch_perception.py --segmenter sam2 --device cuda
    # leave running; Ctrl-C tears all servers down.
"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

import tyro

# name -> (server script, port)
_SERVERS = {
    "owlvit": ("capx/serving/launch_owlvit_server.py", 8117),
    "sam2": ("capx/serving/launch_sam2_server.py", 8113),
    "sam3": ("capx/serving/launch_sam3_server.py", 8114),
    "graspnet": ("capx/serving/launch_contact_graspnet_server.py", 8115),
    "pyroki": ("capx/serving/launch_pyroki_server.py", 8116),
}

_GROUPS = {
    "sam2": ["owlvit", "sam2", "graspnet", "pyroki"],
    "sam3": ["sam3", "graspnet", "pyroki"],
}

# PyRoKi takes extra CLI args (robot / target link) in the example configs.
_EXTRA_ARGS = {
    "pyroki": ["--robot", "panda_description", "--target-link", "panda_hand"],
}


@dataclass
class Args:
    segmenter: str = "sam2"
    device: str = "cuda"
    host: str = "127.0.0.1"
    startup_timeout: float = 300.0


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def main(args: Args) -> None:
    if args.segmenter not in _GROUPS:
        raise SystemExit(f"segmenter must be one of {list(_GROUPS)}")

    procs: list[tuple[str, subprocess.Popen]] = []

    def _shutdown(*_):
        print("\n[launch_perception] tearing down servers ...")
        for name, p in procs:
            if p.poll() is None:
                p.terminate()
        for name, p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for name in _GROUPS[args.segmenter]:
        script, port = _SERVERS[name]
        if _port_open(args.host, port):
            print(f"[launch_perception] {name} already serving on :{port}, reusing.")
            continue
        cmd = [sys.executable, script, "--device", args.device,
               "--port", str(port), "--host", args.host]
        cmd += _EXTRA_ARGS.get(name, [])
        print(f"[launch_perception] starting {name}: {' '.join(cmd)}")
        procs.append((name, subprocess.Popen(cmd)))

    # Wait for readiness.
    for name in _GROUPS[args.segmenter]:
        _, port = _SERVERS[name]
        deadline = time.time() + args.startup_timeout
        while time.time() < deadline:
            if _port_open(args.host, port):
                print(f"[launch_perception] {name} ready on :{port}")
                break
            time.sleep(1.0)
        else:
            print(f"[launch_perception] WARNING: {name} not ready within timeout")

    print("[launch_perception] all requested servers up. Ctrl-C to stop.")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main(tyro.cli(Args))
