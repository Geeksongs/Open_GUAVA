"""GUAVA ReAct agent loop: iterative perception-reasoning-action.

Implements the closed-loop interaction described in the paper (Section 3.1,
Appendix A.1, Figures 7/8/11/12/13):

    repeat:
        observation = (RGB image, symbolic gripper state, last tool result)
        response    = VLM(system_prompt, history + observation)
        <think> ...reasoning... </think>
        <tool_call>{"name": ..., "arguments": {...}}</tool_call>
        result      = tools.dispatch(name, arguments)
    until "Task complete" / "Task failed" / step budget exhausted

This is deliberately *not* CaP-X's one-shot code-generation executor: GUAVA
emits exactly one semantic tool call per turn and re-grounds on a fresh
multimodal observation, which is what makes it token-efficient and
failure-recoverable.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image

from guava.prompts import (
    GRIPPER_STATE_TEMPLATE,
    SYSTEM_PROMPT,
    TASK_TEMPLATE,
    TOOL_RESULT_TEMPLATE,
)
from guava.tools import GuavaToolError, GuavaTools

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_DONE_RE = re.compile(r"task\s+complete", re.IGNORECASE)
_FAIL_RE = re.compile(r"task\s+failed", re.IGNORECASE)


@dataclass
class StepRecord:
    turn: int
    response: str
    think: str
    tool_name: str | None
    arguments: dict[str, Any]
    result: str
    error: str | None = None


@dataclass
class EpisodeResult:
    task: str
    success: bool
    done_reason: str  # "task_complete" | "task_failed" | "step_limit" | "error"
    num_turns: int
    steps: list[StepRecord] = field(default_factory=list)
    total_tokens: int = 0


def _img_to_data_url(img: np.ndarray) -> str:
    pil = Image.fromarray(np.asarray(img).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _extract_tool_json(text: str) -> dict | None:
    """Pull the tool-call JSON out of a response, robustly.

    Handles models that omit the closing </tool_call> tag (e.g. gpt-5-nano emits
    `<tool_call>{...}` with no closer) and nested braces, by brace-matching the
    object that follows the `<tool_call>` marker.  Falls back to the first
    balanced JSON object containing a "name" key.
    """
    def _balanced_from(s: str, start: int) -> str | None:
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return None

    marker = text.find("<tool_call>")
    if marker != -1:
        brace = text.find("{", marker)
        if brace != -1:
            obj = _balanced_from(text, brace)
            if obj:
                try:
                    return json.loads(obj)
                except json.JSONDecodeError:
                    pass
    # Fallback: first balanced {...} that parses and has a "name".
    for i, c in enumerate(text):
        if c == "{":
            obj = _balanced_from(text, i)
            if obj:
                try:
                    d = json.loads(obj)
                    if isinstance(d, dict) and "name" in d:
                        return d
                except json.JSONDecodeError:
                    continue
    return None


def _parse_response(text: str) -> tuple[str, dict | None, bool, bool]:
    """Extract (think, tool_call_or_None, is_done, is_failed) from a response."""
    think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think = think_m.group(1).strip() if think_m else ""
    done, failed = bool(_DONE_RE.search(text)), bool(_FAIL_RE.search(text))
    tc = _extract_tool_json(text)
    return think, tc, done, failed


class GuavaAgent:
    """Drives one or more episodes of the GUAVA harness over a CaP-X env.

    Parameters
    ----------
    low_level_env:
        Constructed CaP-X low-level env.
    query_fn:
        Callable ``(messages: list[dict]) -> dict`` returning at least
        ``{"content": str, "usage_tokens": int}``.  Wraps whatever LLM backend
        you use (see ``guava.runner`` for the CaP-X OpenRouter proxy wiring).
    max_turns:
        Step budget per episode (the paper uses a finite execution horizon).
    """

    def __init__(
        self,
        low_level_env: Any,
        query_fn: Callable[[list[dict]], dict],
        max_turns: int = 20,
        segmenter: str = "sam3",
        serial_gpu: bool = False,
        viz: bool = False,
        trace: bool = False,
        vlm_camera: str = "agentview",
    ) -> None:
        self._env = low_level_env
        self._query = query_fn
        self._max_turns = max_turns
        self._trace = trace
        # Camera the VLM (and the viz window) sees. The env's default camera is
        # often robot-mounted / top-down (e.g. nut uses birdview), where the arm
        # occludes the objects it hovers over. A fixed third-person view like
        # agentview keeps the scene visible. Perception (SAM3/GraspNet) is
        # unaffected -- it still uses the env's own camera and grounds on object
        # names, so this only improves what the VLM sees to reason about.
        self._vlm_camera = vlm_camera
        self._tools = GuavaTools(low_level_env, segmenter=segmenter, serial_gpu=serial_gpu)

        # Live visualization: pop up an OpenCV window and play the frames
        # recorded during each tool's motion, so the robot is shown moving.
        self._viz = viz
        self._viz_im = None
        self._viz_txt = None
        if viz:
            # Record EVERY sim step (not the default every-5th) so the played-back
            # motion is a smooth animation rather than a coarse flip-book.
            if hasattr(self._env, "_subsample_rate"):
                self._env._subsample_rate = 1
            if hasattr(self._env, "enable_video_capture"):
                self._env.enable_video_capture(True, clear=True)
            # OpenCV here is the headless build (no GUI), so use matplotlib with
            # the Tk backend for a live, updating animation window.
            import matplotlib
            matplotlib.use("TkAgg", force=True)
            import matplotlib.pyplot as plt
            self._plt = plt
            plt.ion()
            self._viz_fig, self._viz_ax = plt.subplots(figsize=(6, 6))
            self._viz_ax.axis("off")
            self._viz_fig.canvas.manager.set_window_title("GUAVA")
            self._viz_im = self._viz_ax.imshow(np.zeros((512, 512, 3), np.uint8))
            self._viz_txt = self._viz_ax.text(
                8, 24, "", color="lime", fontsize=12, weight="bold")
            plt.show(block=False)

    # ------------------------------------------------------------------ #
    def _viz_show(self, frame: np.ndarray, label: str) -> None:
        self._viz_im.set_data(np.asarray(frame).astype(np.uint8))
        self._viz_txt.set_text(label)
        self._viz_fig.canvas.draw_idle()
        self._viz_fig.canvas.flush_events()
        self._plt.pause(0.001)

    def _viz_play_since(self, start_frame: int, label: str) -> None:
        """Update the live window with the current VLM camera view.

        Shows the same third-person ``vlm_camera`` the model sees, so the human
        watches exactly what the VLM is reasoning over (one frame per turn).
        """
        if not self._viz:
            return
        self._viz_show(self._render_image(), label)

    def _viz_frame_count(self) -> int:
        if self._viz and hasattr(self._env, "get_video_frame_count"):
            return self._env.get_video_frame_count()
        return 0

    def close_viz(self) -> None:
        if self._viz:
            self._plt.close(self._viz_fig)

    # ------------------------------------------------------------------ #
    def _render_image(self) -> np.ndarray:
        # Render the chosen third-person camera directly from the MuJoCo sim,
        # independent of the env's perception camera (works for any camera in
        # the model, even one not in the observation config).
        if self._vlm_camera and hasattr(self._env, "robosuite_env"):
            try:
                w = getattr(self._env, "_render_width", 512)
                h = getattr(self._env, "_render_height", 512)
                img = self._env.robosuite_env.sim.render(
                    width=w, height=h, camera_name=self._vlm_camera)
                return np.asarray(img)[::-1]  # MuJoCo returns bottom-up
            except Exception:  # noqa: BLE001 - fall back to env default camera
                pass
        if hasattr(self._env, "render"):
            return self._env.render()
        obs = self._env.get_observation()
        return obs[self._tools._camera]["images"]["rgb"]

    def _build_observation_content(self, last_result: str | None, last_tool: str | None) -> list[dict]:
        """Multimodal observation: image + symbolic gripper state + last result."""
        gs = self._tools.gripper_state()
        text = GRIPPER_STATE_TEMPLATE.format(
            px=gs["position"][0], py=gs["position"][1], pz=gs["position"][2],
            roll=gs["rpy_deg"][0], pitch=gs["rpy_deg"][1], yaw=gs["rpy_deg"][2],
            opening=gs["opening_pct"],
        )
        if last_result is not None:
            text = TOOL_RESULT_TEMPLATE.format(tool=last_tool, result=last_result) + "\n" + text
        return [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": _img_to_data_url(self._render_image())}},
        ]

    # ------------------------------------------------------------------ #
    def _trace_step(self, turn: int, tool: str, think: str, result: str, err: str | None) -> None:
        """Print one turn's reasoning + tool live, as it happens."""
        if not self._trace:
            return
        print(f"\n[turn {turn}] TOOL: {tool}", flush=True)
        print(f"  THINK : {think if think else '(none)'}", flush=True)
        print(f"  RESULT: {result!r}" + (f"  ERR: {err}" if err else ""), flush=True)

    # ------------------------------------------------------------------ #
    def run_episode(self, task: str) -> EpisodeResult:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # First user turn: task + initial multimodal observation.
        first = [{"type": "text", "text": TASK_TEMPLATE.format(task=task)}]
        first += self._build_observation_content(None, None)
        messages.append({"role": "user", "content": first})

        result = EpisodeResult(task=task, success=False, done_reason="step_limit", num_turns=0)
        last_result: str | None = None
        last_tool: str | None = None

        for turn in range(self._max_turns):
            out = self._query(messages)
            response = out["content"]
            result.total_tokens += int(out.get("usage_tokens", 0))
            messages.append({"role": "assistant", "content": response})

            think, tc, is_done, is_failed = _parse_response(response)

            if is_done or is_failed:
                result.num_turns = turn + 1
                result.done_reason = "task_complete" if is_done else "task_failed"
                result.success = bool(is_done) and self._check_success()
                result.steps.append(StepRecord(turn, response, think, None, {}, "", None))
                self._trace_step(turn, "(FINISH)", think,
                                 "task_complete" if is_done else "task_failed", None)
                break

            if tc is None or "name" not in tc:
                # Malformed turn: re-attach the current observation (image + state)
                # alongside a format reminder, rather than a bare text error.
                # Stripping the visual context here was causing a single bad
                # response to cascade into repeated malformed turns.
                err = ('Your last response had no valid <tool_call>. Respond with '
                       'a <think>...</think> block then exactly one '
                       '<tool_call>{"name": ..., "arguments": {...}}</tool_call>.')
                result.steps.append(StepRecord(turn, response, think, None, {}, "", err))
                self._trace_step(turn, "(malformed — no tool)", think, "", err)
                recovery = [{"type": "text", "text": err}]
                recovery += self._build_observation_content(last_result, last_tool)
                messages.append({"role": "user", "content": recovery})
                last_result, last_tool = None, None
                continue

            name, args = tc["name"], tc.get("arguments", {}) or {}
            viz_start = self._viz_frame_count()  # frame index before this tool moves
            try:
                tool_out = self._tools.dispatch(name, args)
                last_result = tool_out if isinstance(tool_out, str) else json.dumps(tool_out)
                err = None
            except GuavaToolError as exc:
                last_result = f"[err] {exc}"
                err = str(exc)
            except Exception as exc:  # noqa: BLE001 - surfaced to agent for recovery
                last_result = f"[err] motion failed: {exc}"
                err = str(exc)
            last_tool = name

            # Play the robot motion for this tool in the live window.
            self._viz_play_since(viz_start, f"turn {turn}: {name}")

            result.steps.append(StepRecord(turn, response, think, name, args, last_result, err))
            self._trace_step(turn, f"{name}({args})", think, last_result, err)

            # Early environment-side success check (sparse task reward).
            if self._check_success():
                result.num_turns = turn + 1
                result.done_reason = "task_complete"
                result.success = True
                break

            messages.append({"role": "user", "content": self._build_observation_content(last_result, last_tool)})
        else:
            result.num_turns = self._max_turns

        return result

    # ------------------------------------------------------------------ #
    def _check_success(self) -> bool:
        if hasattr(self._env, "task_completed"):
            try:
                return bool(self._env.task_completed())
            except Exception:  # noqa: BLE001
                return False
        if hasattr(self._env, "compute_reward"):
            try:
                return float(self._env.compute_reward()) >= 1.0
            except Exception:  # noqa: BLE001
                return False
        return False
