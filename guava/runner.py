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
    model: str = "openai/gpt-5-nano"
    """LLM id (routed through the CaP-X proxy, OpenRouter-compatible)."""
    server_url: str = OPENROUTER_SERVER_URL
    segmenter: str = "sam3"
    """Segmentation backend: 'sam3' (paper-faithful, default) or 'sam2'
    (OWL-ViT+SAM2 fallback). SAM3 requires approved HuggingFace gated access."""
    serial_gpu: bool = False
    """Serialize SAM3<->GraspNet on one GPU (for small-VRAM cards)."""
    viz: bool = False
    """Pop up a live OpenCV window animating the robot's motion each turn
    (requires a display). Records every sim step for smooth playback."""
    manage_servers: bool = True
    """Auto-start the LLM proxy + PyRoKi (and clear stray perception servers in
    serial mode) before running, and tear down what we started on exit. Set
    --no-manage-servers if you launch the servers yourself."""
    trace: bool = True
    """Print each turn's <think> reasoning and the chosen tool. --no-trace for
    just the success/tokens summary."""
    vlm_camera: str = "agentview"
    """Third-person camera the VLM (and viz window) sees. A fixed external view
    like 'agentview' / 'frontview' avoids the arm occluding objects (the env's
    default e.g. birdview is top-down and gets blocked). Perception is
    unaffected. Set '' to use the env's default camera."""
    trials: int = 15
    max_turns: int = 20
    temperature: float = 0.0
    """Deterministic decoding (greedy) -- avoids random malformed responses that
    can derail the ReAct loop."""
    max_tokens: int = 0
    """Output token cap. 0 = auto: query OpenRouter for this model's own output
    limit (max_completion_tokens) and use it, so each model gets its correct
    maximum (e.g. gpt-5-nano's ~400k context vs MiniMax M3's 512k output).
    Set a positive value to override."""
    reasoning_effort: str = "medium"
    debug: bool = False


def resolve_max_tokens(model: str, key_file: str = ".openrouterkey", fallback: int = 16384) -> int:
    """Look up this model's own maximum output length from OpenRouter.

    Returns the provider's ``max_completion_tokens`` if given, else the model's
    ``context_length``, else ``fallback`` -- so every model is driven at its own
    maximum without exceeding its context window.
    """
    try:
        import requests
        from pathlib import Path
        headers = {}
        kp = Path(key_file)
        if kp.exists():
            headers["Authorization"] = f"Bearer {kp.read_text().strip()}"
        data = requests.get(
            "https://openrouter.ai/api/v1/models", headers=headers, timeout=30
        ).json()["data"]
        mid = model[len("openrouter/"):] if model.startswith("openrouter/") else model
        for m in data:
            if m["id"] == mid:
                tp = m.get("top_provider") or {}
                mc = tp.get("max_completion_tokens")
                ctx = m.get("context_length") or tp.get("context_length")
                val = mc or ctx or fallback
                # Reserve input budget: output + input must fit in the context
                # window, so never request more than (context - 64k) as output
                # (64k comfortably covers GUAVA's prompt + accumulated images).
                if ctx:
                    val = min(val, max(int(ctx) - 64000, 4096))
                print(f"[runner] max_tokens auto for {mid}: {val} "
                      f"(reported max_completion={mc}, context={ctx})")
                return int(val)
    except Exception as exc:  # noqa: BLE001
        print(f"[runner] max_tokens auto lookup failed ({exc}); using {fallback}")
    return fallback


def make_query_fn(args: RunArgs):
    """Return a ``messages -> {content, usage_tokens}`` callable over CaP-X proxy."""
    max_tokens = args.max_tokens if args.max_tokens > 0 else resolve_max_tokens(args.model)
    qargs = ModelQueryArgs(
        model=args.model,
        server_url=args.server_url,
        temperature=args.temperature,
        max_tokens=max_tokens,
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
    from guava.services import ServiceManager

    services = ServiceManager()
    successes = 0
    total_tokens = 0
    last_agent = None
    try:
        if args.manage_servers:
            # Bring up everything this run needs; tear it down on exit.
            services.ensure_proxy()
            services.ensure_pyroki()
            if args.serial_gpu:
                # The slot manager must exclusively own SAM3/GraspNet, so clear
                # any stray perception servers that would hog VRAM.
                services.free_perception_ports()

        query_fn = make_query_fn(args)
        for trial in range(args.trials):
            env = get_env(args.env, enable_render=True)
            env.reset(seed=trial, options={"trial": trial})
            if args.trace:
                print(f"\n===== trial {trial:02d} =====", flush=True)
            agent = GuavaAgent(
                env, query_fn, max_turns=args.max_turns,
                segmenter=args.segmenter, serial_gpu=args.serial_gpu, viz=args.viz,
                trace=args.trace, vlm_camera=args.vlm_camera,
            )
            last_agent = agent
            res = agent.run_episode(args.task)  # prints think+tool live when trace=True
            agent.close_viz()
            agent._tools.close()  # tear down the serial GPU-slot servers
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
    finally:
        # Always clean up: GPU-slot servers + the proxy/PyRoKi we started.
        try:
            if last_agent is not None:
                last_agent.close_viz()
                last_agent._tools.close()
        except Exception:  # noqa: BLE001
            pass
        if args.manage_servers:
            services.shutdown()


if __name__ == "__main__":
    main(tyro.cli(RunArgs))
