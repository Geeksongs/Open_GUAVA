"""Offline end-to-end test of the GUAVA ReAct loop (no GPU / no robosuite).

Stubs out ``guava.tools`` heavy deps (viser, CaP-X perception) and the LLM, then
drives a full episode to prove the harness wiring -- observation building, tool
parsing, dispatch, recovery on bad calls, and success termination -- is sound.

Run: ``.venv/bin/python -m guava.tests.test_loop_offline`` from repo root.
"""

from __future__ import annotations

import sys
import types

import numpy as np


# --- Stub the perception/motion stack so guava.tools imports without a GPU ----
def _install_stubs() -> None:
    # viser.transforms shim
    viser = types.ModuleType("viser")
    vtf = types.ModuleType("viser.transforms")

    class _SE3:
        def __init__(self, *a, **k): ...
        @classmethod
        def from_matrix(cls, m): return cls()
        def inverse(self): return self
        def __matmul__(self, o): return self
        def translation(self): return np.zeros(3)
        def rotation(self): return _SO3()

    class _SO3:
        def __init__(self, *a, **k): ...
        @classmethod
        def from_matrix(cls, m): return cls()
        @classmethod
        def from_rpy_radians(cls, *a): return cls()
        def as_rpy_radians(self): return np.zeros(3)
        def __matmul__(self, o): return self
        @property
        def wxyz(self): return np.array([1.0, 0, 0, 0])

    vtf.SE3, vtf.SO3 = _SE3, _SO3
    viser.transforms = vtf
    sys.modules["viser"] = viser
    sys.modules["viser.transforms"] = vtf

    # capx.integrations.franka.control_reduced.FrankaControlApiReduced shim
    cr = types.ModuleType("capx.integrations.franka.control_reduced")

    class _Api:
        def __init__(self, env, use_sam3=True): self._env = env
        def get_observation(self): return self._env.get_observation()
        def segment_sam3_text_prompt(self, rgb, text_prompt):
            return [{"mask": np.ones((4, 4), bool), "score": 0.9}]
        def plan_grasp(self, depth, intrinsics, segmentation):
            return np.tile(np.eye(4), (1, 1, 1)), np.array([0.9])
        def solve_ik(self, pos, quat): return np.zeros(7)
        def move_to_joints(self, j): self._env.moved = True
        def open_gripper(self): self._env._gripper_fraction = 1.0
        def close_gripper(self): self._env._gripper_fraction = 0.5

    cr.FrankaControlApiReduced = _Api
    sys.modules["capx.integrations.franka.control_reduced"] = cr

    # depth_to_pointcloud shim
    du = types.ModuleType("capx.utils.depth_utils")
    du.depth_to_pointcloud = lambda d, K, **kw: np.ones((d.shape[0] * d.shape[1], 3))
    sys.modules["capx.utils.depth_utils"] = du


_install_stubs()

from guava.agent import GuavaAgent  # noqa: E402


class MockEnv:
    """Minimal env exposing the surface guava.tools/agent rely on."""

    def __init__(self):
        self._gripper_fraction = 1.0
        self.gripper_link_wxyz_xyz = np.array([1.0, 0, 0, 0, 0.4, 0.0, 0.3])
        self.base_link_wxyz_xyz = np.array([1.0, 0, 0, 0, 0.0, 0.0, 0.0])
        self.moved = False
        self._calls = 0

    def get_observation(self):
        return {"robot0_robotview": {
            "images": {"rgb": np.zeros((4, 4, 3), np.uint8),
                       "depth": np.ones((4, 4), np.float32)},
            "intrinsics": np.eye(3),
            "pose_mat": np.eye(4),
        }}

    def render(self): return np.zeros((4, 4, 3), np.uint8)

    def task_completed(self):
        # Succeed only after a grasp followed by a move/release sequence.
        return self._calls >= 3


class MockLLM:
    """Scripted VLM: one malformed turn (recovery), then a valid plan."""

    def __init__(self, env):
        self.env = env
        self.script = [
            "<think>where is it</think> oops no tool",                      # malformed
            '<think>align first</think><tool_call>{"name":"align","arguments":{"object":"cube","position":"top","clearance":"medium"}}</tool_call>',
            '<think>grasp</think><tool_call>{"name":"grasp","arguments":{"object":"cube"}}</tool_call>',
            '<think>lift</think><tool_call>{"name":"move","arguments":{"x":0.4,"y":0.0,"z":0.4}}</tool_call>',
            "<think>done, cube stacked</think> Task complete.",
        ]
        self.i = 0

    def __call__(self, messages):
        text = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        if "tool_call" in text:
            self.env._calls += 1
        return {"content": text, "usage_tokens": 120}


def main() -> None:
    env = MockEnv()
    agent = GuavaAgent(env, MockLLM(env), max_turns=10)
    res = agent.run_episode("stack the cube")

    print(f"success      = {res.success}")
    print(f"done_reason  = {res.done_reason}")
    print(f"num_turns    = {res.num_turns}")
    print(f"total_tokens = {res.total_tokens}")
    for s in res.steps:
        tag = f"{s.tool_name}({s.arguments})" if s.tool_name else "(no-tool)"
        print(f"  turn {s.turn}: {tag} -> {s.result!r}{' ERR:'+s.error if s.error else ''}")

    assert res.success, "episode should succeed"
    assert res.done_reason == "task_complete"
    # First turn was malformed -> recorded with an error, loop recovered.
    assert res.steps[0].error is not None
    assert any(s.tool_name == "grasp" for s in res.steps)
    print("\nOK: GUAVA ReAct loop wiring verified end-to-end.")


if __name__ == "__main__":
    main()
