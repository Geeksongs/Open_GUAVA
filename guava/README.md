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

## Setup

GUAVA runs inside the **main CaP-X environment** (RoboSuite + SAM3 +
Contact-GraspNet + PyRoKi). Follow the repo-root README to install it, then the
only extra step is an LLM key:

```bash
# OpenRouter key for the LLM proxy (git-ignored). Any OpenRouter-routable model works.
echo "sk-or-v1-your-key-here" > .openrouterkey
```

- **You do NOT need to start any servers by hand.** The runner owns the full
  service lifecycle: it launches the LLM proxy (`:8110`) and PyRoKi IK (`:8116`),
  reuses/clears stray instances on those ports, and tears everything down on exit.
  Pass `--no-manage-servers` only if you want to run the proxy/perception yourself.
- **SAM3** (default, paper-faithful) needs approved HuggingFace gated access. No
  access? Use `--segmenter sam2` (OWL-ViT + SAM2 fallback).
- **Small-VRAM GPU?** Add `--serial-gpu` to time-share SAM3 ↔ Contact-GraspNet on
  one card instead of loading both at once.

## Run

One command runs everything (services auto-managed):

```bash
# Default: lift the red cube, gpt-5-nano, 15 trials
uv run --no-sync --active guava/runner.py

# Pick a task + model explicitly
uv run --no-sync --active guava/runner.py \
    --env franka_robosuite_cubes_restack_low_level \
    --task "stack the red cube on top of the green cube" \
    --model openai/gpt-5.4 --trials 15

# Small-VRAM card + SAM2 fallback + live motion window
uv run --no-sync --active guava/runner.py --serial-gpu --segmenter sam2 --viz
```

By default it prints each turn's `<think>` reasoning and chosen tool **live**, then
reports **success rate** (paper Table 2) and **avg tokens/episode** (Figure 9).
Add `--no-trace` for just the summary.

## CLI flags (`tyro`, see `guava/runner.py:RunArgs`)

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--env` | `franka_robosuite_cube_lift_low_level` | CaP-X low-level env (see list below) |
| `--task` | `"Pick up the red cube and lift it."` | NL instruction given to the VLM |
| `--model` | `openai/gpt-5-nano` | **Any** OpenRouter-routable LLM id — open or closed, any provider. The default is just a cheap pick; pass e.g. `openai/gpt-5.4`, `anthropic/claude-opus-4-8`, `google/gemini-3-pro`, etc. |
| `--segmenter` | `sam3` | `sam3` (paper-faithful) or `sam2` (no-gated-access fallback) |
| `--trials` | `15` | Episodes to evaluate (success rate is over these) |
| `--max-turns` | `20` | ReAct turns per episode before timeout |
| `--temperature` | `0.0` | Greedy decoding (recommended — avoids malformed turns) |
| `--max-tokens` | `0` (auto) | `0` = query the model's own output limit; or set a cap |
| `--reasoning-effort` | `medium` | Reasoning-effort hint for models that support it |
| `--vlm-camera` | `agentview` | Fixed 3rd-person view the VLM sees (`''` = env default) |
| `--serial-gpu` / `--no-serial-gpu` | off | Time-share SAM3↔GraspNet on one GPU |
| `--viz` / `--no-viz` | off | Live OpenCV motion window (needs a display) |
| `--trace` / `--no-trace` | on | Print per-turn think+tool live |
| `--manage-servers` / `--no-manage-servers` | on | Auto start/stop proxy + PyRoKi |
| `--debug` | off | Verbose LLM-proxy I/O |

## Available environments

```
franka_robosuite_cube_lift_low_level      # lift the red cube (simplest, default)
franka_robosuite_cubes_low_level          # two cubes present
franka_robosuite_cubes_restack_low_level  # stacking
franka_robosuite_nut_assembly_low_level   # nut-on-peg
franka_robosuite_spill_wipe_low_level     # wipe a spill
franka_cubes_low_level                    # non-robosuite cube env
franka_libero_*_low_level                 # LIBERO low-level tasks (needs LIBERO venv)
franka_real_low_level                     # real Franka (see docs/real-franka.md)
```

## Offline test (no GPU / no robosuite)

```bash
PYTHONPATH=. .venv/bin/python guava/tests/test_loop_offline.py
```

Stubs the perception/motion stack and the LLM, then drives a full episode
(malformed-turn recovery → align → grasp → move → `Task complete`) to verify the
loop wiring end-to-end.
