"""Exact GUAVA harness prompts and tool specification.

Transcribed verbatim from the GUAVA paper (Liu et al., 2026,
"Guava: An Effective and Universal Harness for Embodied Manipulation"):

  * System prompt  -> Appendix A.1 ("System Prompt for Data Generation").
  * Tool definitions -> Table 1 (main paper) + Appendix B ("Tools"), where the
    Appendix is authoritative (e.g. ``align(object, position, clearance)`` and
    ``get_position_and_size`` rather than the abbreviated Table-1 names).
  * Interaction format -> Appendix A.1 + Figures 7, 8, 11, 12, 13.

The whole point of GUAVA is the *harness*, so these strings are kept faithful to
the paper rather than paraphrased.  If you change them you are no longer
reproducing GUAVA.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tool definitions (Appendix B - the complete action space)
# ---------------------------------------------------------------------------
# Each entry: name -> human-readable definition shown to the model.  The
# parameter grammar (enums) is exactly as specified in Appendix B.

TOOL_DEFINITIONS: list[dict[str, str]] = [
    {
        "name": "grasp",
        "signature": 'grasp(object: str)',
        "doc": (
            "Grasps the specified object. The implementation first segments the "
            "target object from RGB-D observations using SAM3 and estimates a "
            "grasp pose. The robot approaches the grasp pose, closes the gripper, "
            "and returns either `grasped` when the gripper cannot fully close or "
            "`closed` if the gripper closes completely (i.e. nothing was grasped)."
        ),
    },
    {
        "name": "align",
        "signature": 'align(object: str, position: str, clearance: str)',
        "doc": (
            "Moves the gripper to a specified relative position around the target "
            "object. `position` in {top, left, right, front, back} defines the "
            "approach direction, while `clearance` in {small, medium, large} "
            "controls the standoff distance. Grounds execution in 3D geometry "
            "estimated from the segmented point cloud."
        ),
    },
    {
        "name": "get_position",
        "signature": 'get_position(object: str)',
        "doc": (
            "Returns the estimated 3D position [x, y, z] of `object` in the robot "
            "base frame, computed as the centroid of the segmented object point "
            "cloud after outlier removal."
        ),
    },
    {
        "name": "get_position_and_size",
        "signature": 'get_position_and_size(object: str)',
        "doc": (
            "Returns both the estimated object position [x, y, z] and the "
            "axis-aligned bounding-box dimensions [dx, dy, dz], enabling the agent "
            "to reason about object size and spatial constraints."
        ),
    },
    {
        "name": "move",
        "signature": 'move(x: float, y: float, z: float)',
        "doc": (
            "Moves the robot end-effector to the Cartesian position [x, y, z] "
            "(robot base frame, meters) via a position-based controller."
        ),
    },
    {
        "name": "rotate",
        "signature": 'rotate(angle_deg: float, axis: str)',
        "doc": (
            "Rotates the gripper in place by `angle_deg` degrees about the "
            "specified body-frame axis in {x, y, z}."
        ),
    },
    {
        "name": "close_gripper",
        "signature": 'close_gripper()',
        "doc": "Closes the gripper.",
    },
    {
        "name": "release",
        "signature": 'release()',
        "doc": (
            "Opens the gripper to release a grasped object. Optionally performs a "
            "short retraction motion to avoid post-release collisions."
        ),
    },
    {
        "name": "home_pose",
        "signature": 'home_pose()',
        "doc": (
            "Moves the robot to a predefined home configuration. Serves as a "
            "recovery action when the current pose is unsuitable for further "
            "manipulation."
        ),
    },
]


def _render_tool_definitions() -> str:
    lines: list[str] = []
    for t in TOOL_DEFINITIONS:
        lines.append(f"- {t['signature']}")
        lines.append(f"    {t['doc']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt (Appendix A.1)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an intelligent robot arm controller. You will be \
shown an image of the scene and given a task to complete using the available \
tools.

Interaction Loop
In every response, think step-by-step inside <think></think> tags first, then \
call exactly one tool. When the task is fully complete or irrecoverable, output \
your thinking and end with Task complete or Task failed.

Reasoning Guidelines
Inside <think>, analyze the current scene state - including object and gripper \
poses, progress of the task - what has been done and what remains, the result \
of the last action - was it a success, failure, or unexpected outcome, and \
propose the best next action based on your reasoning.

Tool Calling
Call exactly one tool by emitting a single line of the form:
<tool_call>{{"name": "tool_name", "arguments": {{"param": value}}}}</tool_call>

Gripper State
The gripper's current position [x, y, z], rotation [roll, pitch, yaw], and \
gripper opening %% are provided at every turn. Use these values directly in \
your reasoning when necessary.

Tool Definition
{tool_definitions}

Output Format (STRICT)
Every response MUST have exactly two parts, in this order:
  1. A <think>...</think> block containing your reasoning. This block is
     REQUIRED -- never omit it and never leave it empty.
  2. EITHER a single <tool_call>...</tool_call> line, OR the phrase
     "Task complete" / "Task failed" when finished.
Do not output anything else. Do not call more than one tool per response.

Examples (illustrative trajectories; follow this exact format)
{few_shot}
""".format(tool_definitions=_render_tool_definitions(), few_shot="{few_shot}")


# ---------------------------------------------------------------------------
# Few-shot examples (reasoning style transcribed from the GUAVA paper appendix
# figures 7/12/13).  IMPORTANT: each example is a SINGLE turn -- one observation
# -> one <think> + one tool call -- because that is exactly what the model emits
# per turn.  We deliberately do NOT show full multi-step trajectories: the
# running history already lives in the conversation, so replaying it here would
# only waste tokens and risk the model dumping several steps at once.
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """\
Example of one turn (this is a SINGLE response -- emit exactly one like this):
<think>The orange is on the table at front-left and the gripper is open above the back of the table. I should first move above the orange with some clearance before grasping.</think>
<tool_call>{"name": "align", "arguments": {"object": "orange", "position": "top", "clearance": "medium"}}</tool_call>
"""

# Use replace (not .format) here: the prompt already contains literal JSON
# braces like {"name": ...} from the Tool Calling section, which .format would
# misparse as fields.
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{few_shot}", FEW_SHOT_EXAMPLES)


# ---------------------------------------------------------------------------
# Per-turn observation templates
# ---------------------------------------------------------------------------

# Textual symbolic state that accompanies the image at every turn (Appendix A.1
# "Gripper State" + paper Section 3.1 "textual state representations").
GRIPPER_STATE_TEMPLATE = (
    "Gripper state:\n"
    "  position [x, y, z] = [{px:.3f}, {py:.3f}, {pz:.3f}] (m, robot base frame)\n"
    "  rotation [roll, pitch, yaw] = [{roll:.1f}, {pitch:.1f}, {yaw:.1f}] (deg)\n"
    "  gripper opening = {opening:.0f}%"
)

# Result of the previous tool call, fed back as part of the next observation.
TOOL_RESULT_TEMPLATE = "Result of last action ({tool}): {result}"

# First user turn: task instruction.
TASK_TEMPLATE = "Task: {task}"
