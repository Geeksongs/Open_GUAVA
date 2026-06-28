# GUAVA harness (reproduced inside CaP-X)

A faithful re-implementation of the harness from **Liu et al. (2026), "Guava: An
Effective and Universal Harness for Embodied Manipulation"** (`../guava.pdf`),
built directly on top of the CaP-X low-level stack (RoboSuite env + SAM3 +
Contact-GraspNet + PyRoKi IK).

We reproduce the **harness only** — i.e. GUAVA's *teacher* structure (Section 3,
Appendices A.1 & B). We do **not** reproduce the 4B distillation/RL pipeline
(out of scope here).

## The three GUAVA principles → where they live

| Principle (paper §3.1)                       | File |
| -------------------------------------------- | ---- |
| Iterative perception-reasoning-action (ReAct) loop | `agent.py` |
| Semantic action abstractions (9 tools, Appendix B) | `tools.py` |
| Multimodal observation (image + symbolic gripper state) | `agent.py` (`_build_observation_content`) |
| Exact system prompt + tool spec              | `prompts.py` |

## The action space (Appendix B — verbatim, authoritative over Table 1)

`grasp(object)` · `align(object, position, clearance)` · `get_position(object)` ·
`get_position_and_size(object)` · `move(x,y,z)` · `rotate(angle_deg, axis)` ·
`close_gripper()` · `release()` · `home_pose()`

- `position ∈ {top, left, right, front, back}`, `clearance ∈ {small, medium, large}`, `axis ∈ {x, y, z}`.
- Each semantic tool is implemented by **composing CaP-X primitives** (SAM3 text
  segmentation → depth→point-cloud → Contact-GraspNet / PyRoKi IK → blocking
  joint move). The VLM never writes Python and never reasons about joint angles.

## How it differs from CaP-X (Code-as-Policy)

CaP-X generates a **whole Python program per turn** and `exec`s it. GUAVA emits
**exactly one semantic tool call per turn** (`<think>…</think><tool_call>{…}</tool_call>`),
then re-grounds on a fresh multimodal observation. That closed loop is what gives
GUAVA its failure recovery and token efficiency — so this package deliberately
**does not** use CaP-X's `exec`/multi-turn/video-differencing path.

## Run

```bash
# 1. start the CaP-X LLM proxy (repo README step 2) and perception servers
# 2. eval the harness over a RoboSuite task
uv run --no-sync --active guava/runner.py \
    --env franka_robosuite_cubes_low_level \
    --task "stack the red cube on top of the green cube" \
    --model openai/gpt-5.4 --trials 15
```

Reports **success rate** (paper Table 2) and **avg tokens/episode** (Figure 9).

## Offline test (no GPU / no robosuite)

```bash
PYTHONPATH=. .venv/bin/python guava/tests/test_loop_offline.py
```

Stubs the perception/motion stack and the LLM, then drives a full episode
(malformed-turn recovery → align → grasp → move → `Task complete`) to verify the
loop wiring end-to-end.
```
```
