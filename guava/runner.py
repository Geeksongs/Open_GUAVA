"""Run the GUAVA harness over CaP-X RoboSuite tasks.

Wires together:
  * a CaP-X low-level env (built via ``capx.envs.base.get_env``),
  * the CaP-X LLM proxy (``query_model`` -> local OpenRouter/vLLM server),
  * the GUAVA ReAct agent loop,

and evaluates a task over N trials, reporting success rate and tokens/episode
(the two headline metrics in the paper: Table 2 success and Figure 9 tokens).

Usage
-----
    # 1. start the CaP-X LLM proxy (see repo README, step 2)
    # 2. start perception servers or let them auto-load
    uv run --no-sync --active guava/runner.py \
        --env franka_robosuite_cubes_low_level \
        --task "stack the red cube on the green cube" \
        --model openai/gpt-5.4 --trials 15
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro

from capx.envs.base import get_env
from capx.llm.client import ModelQueryArgs, OPENROUTER_SERVER_URL, query_model
from guava.agent import GuavaAgent


@dataclass
class RunArgs:
    env: str = "franka_robosuite_cube_lift_low_level"
    """CaP-X low-level env name (see capx/envs/base.py registry).
    Default is the simplest task: lift the red cube."""
    task: str = "Pick up the red cube and lift it."
    """Natural-language task instruction passed to the GUAVA system prompt."""
    model: str = "minimax/minimax-m3"
    """LLM id (routed through the CaP-X proxy, OpenRouter-compatible)."""
    server_url: str = OPENROUTER_SERVER_URL
    segmenter: str = "sam3"
    """Segmentation backend: 'sam3' (paper-faithful, default) or 'sam2'
    (OWL-ViT+SAM2 fallback). SAM3 requires approved HuggingFace gated access."""
    serial_gpu: bool = False
    """Serialize SAM3<->GraspNet on one GPU (for small-VRAM cards)."""
    trials: int = 15
    max_turns: int = 20
    temperature: float = 0.2
    max_tokens: int = 512000
    """Output token cap. Set to MiniMax M3's maximum (512000) so generation is
    effectively unconstrained -- the model emits its full <think> + tool call."""
    reasoning_effort: str = "medium"
    debug: bool = False


def make_query_fn(args: RunArgs):
    """Return a ``messages -> {content, usage_tokens}`` callable over CaP-X proxy."""
    qargs = ModelQueryArgs(
        model=args.model,
        server_url=args.server_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    def _query(messages: list[dict]) -> dict:
        out = query_model(qargs, messages)
        # query_model returns {"content", "reasoning", ...}; usage may be absent.
        return {
            "content": out["content"],
            "usage_tokens": int(out.get("usage_tokens", out.get("total_tokens", 0)) or 0),
        }

    return _query


def main(args: RunArgs) -> None:
    query_fn = make_query_fn(args)

    successes = 0
    total_tokens = 0
    for trial in range(args.trials):
        env = get_env(args.env, enable_render=True)
        env.reset(seed=trial, options={"trial": trial})
        agent = GuavaAgent(
            env, query_fn, max_turns=args.max_turns,
            segmenter=args.segmenter, serial_gpu=args.serial_gpu,
        )
        res = agent.run_episode(args.task)
        successes += int(res.success)
        total_tokens += res.total_tokens
        print(
            f"[trial {trial:02d}] success={res.success} reason={res.done_reason} "
            f"turns={res.num_turns} tokens={res.total_tokens}"
        )

    n = args.trials
    print("=" * 60)
    print(f"GUAVA harness | model={args.model} | task={args.task!r}")
    print(f"Success rate : {successes}/{n} = {100.0 * successes / n:.1f}%")
    print(f"Avg tokens/ep: {total_tokens / max(n, 1):.0f}")


if __name__ == "__main__":
    main(tyro.cli(RunArgs))
