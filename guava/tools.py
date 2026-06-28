"""GUAVA semantic tool implementations on top of the CaP-X low-level stack.

This is the heart of the GUAVA reproduction: the 9 semantic action abstractions
from Appendix B, each implemented by *composing* CaP-X perception + motion
primitives (SAM3 segmentation, depth->point cloud, Contact-GraspNet, PyRoKi IK,
blocking joint moves).  The VLM never writes Python or reasons about joint
angles -- it only emits one semantic tool call per turn, exactly as in the paper.

Design notes
------------
* ``grasp(object)``  -> SAM3 text-prompt segment -> plan_grasp -> approach,
  descend, close_gripper.  Returns "grasped" / "closed" per Appendix B.
* ``align(object, position, clearance)`` -> segment -> object centroid/extent ->
  offset along {top,left,right,front,back} by {small,medium,large} standoff ->
  move + top-down orientation.
* ``get_position`` / ``get_position_and_size`` -> segment -> point-cloud centroid
  (+ AABB extent) in robot base frame.
* ``move`` / ``rotate`` / ``close_gripper`` / ``release`` / ``home_pose`` ->
  thin wrappers over IK + joint moves + gripper control.

The class exposes ``dispatch(name, arguments)`` used by the ReAct agent loop; no
``exec`` of model-authored code is involved (we are aligning with GUAVA, not
CaP-X's Code-as-Policy executor).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as vtf

from capx.integrations.franka.control_reduced import FrankaControlApiReduced
from capx.utils.depth_utils import depth_to_pointcloud


# Top-down grasp orientation (gripper pointing -Z), quaternion wxyz.
# Matches the "top-down" convention used throughout the CaP-X Franka stack.
_TOP_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

# Standoff distances (meters) for the `clearance` enum {small, medium, large}.
_CLEARANCE = {"small": 0.05, "medium": 0.10, "large": 0.18}

# Approach-direction unit offsets in the robot base frame for the `position`
# enum {top, left, right, front, back}.  +x = front, +y = left, +z = up.
_DIRECTION = {
    "top": np.array([0.0, 0.0, 1.0]),
    "left": np.array([0.0, 1.0, 0.0]),
    "right": np.array([0.0, -1.0, 0.0]),
    "front": np.array([1.0, 0.0, 0.0]),
    "back": np.array([-1.0, 0.0, 0.0]),
}

_HOME_JOINTS = np.array([0.0, -0.30, 0.0, -2.20, 0.0, 2.00, 0.79], dtype=np.float64)


class GuavaToolError(Exception):
    """Raised when a semantic tool fails in a way the agent should observe."""


class GuavaTools:
    """Semantic GUAVA action space backed by CaP-X primitives.

    Parameters
    ----------
    low_level_env:
        A constructed CaP-X low-level env (e.g. ``FrankaRobosuiteCubesLowLevel``).
    """

    def __init__(self, low_level_env: Any, segmenter: str = "sam3",
                 serial_gpu: bool = False, gpu_device: str = "cuda") -> None:
        self._env = low_level_env
        if segmenter not in ("sam3", "sam2"):
            raise ValueError("segmenter must be 'sam3' or 'sam2'")
        self._segmenter = segmenter
        # Reuse CaP-X's reduced control API for the perception / IK clients.
        # (These are lightweight HTTP clients; the heavy models live in the
        # perception server processes.)
        #   segmenter="sam3" -> SAM3 text-prompt segmentation (paper-faithful).
        #   segmenter="sam2" -> OWL-ViT detection + SAM2 (strongest hiera-large)
        #                       box-conditioned segmentation, an ungated
        #                       drop-in while SAM3 gated access is pending.
        self._api = FrankaControlApiReduced(low_level_env, use_sam3=(segmenter == "sam3"))
        self._camera = "robot0_robotview"

        # Single-GPU-slot serialization: when True, SAM3 and GraspNet are run as
        # mutually-exclusive server processes so only one occupies VRAM at a
        # time (needed on small GPUs).  When False, assume servers are already
        # running (e.g. pre-launched on a larger / separate GPU).
        self._slot = None
        if serial_gpu:
            from guava.gpu_slot import GpuSlotManager
            self._slot = GpuSlotManager(device=gpu_device)

    def _ensure_seg(self) -> None:
        """Make the segmentation model group resident (serial-GPU mode only)."""
        if self._slot is not None:
            self._slot.ensure_group("seg_sam2" if self._segmenter == "sam2" else "seg_sam3")

    def _ensure_grasp(self) -> None:
        """Make GraspNet resident, evicting the segmentation group (serial-GPU)."""
        if self._slot is not None:
            self._slot.ensure_group("graspnet")

    def close(self) -> None:
        """Tear down any managed perception server (frees VRAM)."""
        if self._slot is not None:
            self._slot.stop_all()

    # ------------------------------------------------------------------ #
    # Perception helpers (shared by grasp / align / get_position*)
    # ------------------------------------------------------------------ #
    def _segment_best_mask(self, rgb: np.ndarray, object_name: str) -> np.ndarray:
        """Return the best (H, W) bool mask for `object_name` via the chosen backend.

        sam3: native text-prompt segmentation.
        sam2: OWL-ViT open-vocabulary detection -> best box -> SAM2 box-prompt
              segmentation (paper-equivalent "text -> mask", ungated).
        """
        if self._segmenter == "sam3":
            results = self._api.segment_sam3_text_prompt(rgb, text_prompt=object_name)
            if not results:
                raise GuavaToolError(f"SAM3 found no mask for '{object_name}'.")
            best = max(results, key=lambda r: r.get("score", 0.0))
            return np.asarray(best["mask"]).astype(bool)

        # sam2 path: detect a box first, then segment within it.
        dets = self._api.detect_object_owlvit(rgb, text=object_name)
        if not dets:
            raise GuavaToolError(f"OWL-ViT found no '{object_name}'.")
        box = max(dets, key=lambda d: d.get("score", 0.0))["box"]
        masks = self._api.segment_sam2(rgb, box=box)
        if not masks:
            raise GuavaToolError(f"SAM2 returned no mask for '{object_name}'.")
        best = max(masks, key=lambda m: m.get("score", 0.0))
        return np.asarray(best["mask"]).astype(bool)

    def _segment_object_points(self, object_name: str) -> np.ndarray:
        """Segment `object_name` and return its 3D points (base frame)."""
        self._ensure_seg()  # serial-GPU: keep segmentation group resident
        obs = self._api.get_observation()
        cam = obs[self._camera]
        rgb = cam["images"]["rgb"]
        depth = cam["images"]["depth"]
        K = cam["intrinsics"]
        extrinsics = cam["pose_mat"]  # camera -> base (4,4)

        mask = self._segment_best_mask(rgb, object_name)

        # Per-pixel point cloud (camera frame), kept in 1:1 image correspondence
        # so we can index by the 2D mask, then transform to the base frame.
        pts_cam = depth_to_pointcloud(
            np.asarray(depth).reshape(mask.shape),
            np.asarray(K, dtype=np.float64),
            subsample_factor=1,
            filter_invalid=False,
        ).reshape(mask.shape[0], mask.shape[1], 3)

        obj_cam = pts_cam[mask]
        # Drop invalid / zero-depth points.
        valid = np.isfinite(obj_cam).all(axis=1) & (obj_cam[:, 2] > 1e-3)
        obj_cam = obj_cam[valid]
        if obj_cam.shape[0] < 10:
            raise GuavaToolError(f"Too few valid 3D points for '{object_name}'.")

        pts_h = np.concatenate([obj_cam, np.ones((obj_cam.shape[0], 1))], axis=1)
        pts_base = (np.asarray(extrinsics, dtype=np.float64) @ pts_h.T).T[:, :3]
        return pts_base

    @staticmethod
    def _centroid_outlier_removed(points: np.ndarray) -> np.ndarray:
        """Centroid after a simple per-axis 10-90 percentile inlier filter."""
        lo, hi = np.percentile(points, [10, 90], axis=0)
        inliers = points[np.all((points >= lo) & (points <= hi), axis=1)]
        if inliers.shape[0] == 0:
            inliers = points
        return inliers.mean(axis=0)

    def _gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position xyz, rpy degrees) of the gripper in the base frame."""
        wxyz_xyz = np.asarray(self._env.gripper_link_wxyz_xyz, dtype=np.float64)
        base = np.asarray(self._env.base_link_wxyz_xyz, dtype=np.float64)
        rel = vtf.SE3(wxyz_xyz=base).inverse() @ vtf.SE3(wxyz_xyz=wxyz_xyz)
        pos = rel.translation()
        rpy = np.degrees(rel.rotation().as_rpy_radians())
        return pos, np.asarray(rpy, dtype=np.float64)

    def gripper_state(self) -> dict[str, Any]:
        """Symbolic gripper state fed to the model every turn (Appendix A.1)."""
        pos, rpy = self._gripper_pose()
        opening = float(getattr(self._env, "_gripper_fraction", 1.0)) * 100.0
        return {
            "position": pos,
            "rpy_deg": rpy,
            "opening_pct": opening,
        }

    # ------------------------------------------------------------------ #
    # Internal motion helper
    # ------------------------------------------------------------------ #
    def _move_to(self, position: np.ndarray, quat_wxyz: np.ndarray | None = None) -> None:
        quat = _TOP_DOWN_QUAT_WXYZ if quat_wxyz is None else quat_wxyz
        joints = self._api.solve_ik(np.asarray(position, dtype=np.float64), quat)
        self._api.move_to_joints(joints)

    # ------------------------------------------------------------------ #
    # The 9 semantic tools (Appendix B)
    # ------------------------------------------------------------------ #
    def grasp(self, object: str) -> str:
        # --- Stage 1: segmentation (segmentation group resident on the GPU) ---
        self._ensure_seg()
        obs = self._api.get_observation()
        cam = obs[self._camera]
        rgb, depth, K = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"]
        extrinsics = cam["pose_mat"]

        seg = self._segment_best_mask(rgb, object).astype(np.int32)

        # --- Stage 2: grasp planning (evict segmentation group, load GraspNet) ---
        # Segmentation and GraspNet run strictly in sequence within a grasp, so
        # on a small GPU we free the segmentation models' VRAM before bringing up
        # GraspNet, then they reload for the next perception call.
        self._ensure_grasp()
        # Call the Contact-GraspNet client directly with forward_passes=1.
        # CaP-X's plan_grasp hardcodes forward_passes=3, whose inference
        # transient (~1.8 GB) OOMs a 7-8 GB GPU; a single pass fits comfortably
        # and still yields good grasps.  We then apply the same +0.12 m local-z
        # approach offset that plan_grasp uses.
        depth_np = np.asarray(depth)
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[:, :, 0]
        seg_np = seg[:, :, 0] if seg.ndim == 3 else seg
        grasp_cam, scores, _ = self._api.grasp_net_plan_fn(
            depth_np, np.asarray(K, dtype=np.float64), seg_np, 1,
            z_range=[0.2, 2.0], forward_passes=1,
        )
        if scores is None or len(scores) == 0:
            raise GuavaToolError(f"No grasp proposal for '{object}'.")
        grasp_cam = np.asarray(grasp_cam, dtype=np.float64)
        grasp_poses = (
            vtf.SE3.from_matrix(grasp_cam)
            @ vtf.SE3.from_translation(np.array([0.0, 0.0, 0.12]))
        ).as_matrix()
        best_T_cam = grasp_poses[int(np.argmax(scores))]
        T_base = np.asarray(extrinsics, dtype=np.float64) @ best_T_cam
        grasp_pos = T_base[:3, 3]
        grasp_quat = vtf.SO3.from_matrix(T_base[:3, :3]).wxyz

        self._api.open_gripper()
        # Pre-grasp standoff above target, then descend.
        self._move_to(grasp_pos + np.array([0.0, 0.0, 0.10]), grasp_quat)
        self._move_to(grasp_pos, grasp_quat)
        self._api.close_gripper()

        frac = float(getattr(self._env, "_gripper_fraction", 0.0))
        # Per Appendix B: gripper cannot fully close => something is grasped.
        return "grasped" if frac > 0.02 else "closed"

    def align(self, object: str, position: str, clearance: str) -> str:
        if position not in _DIRECTION:
            raise GuavaToolError(f"position must be one of {list(_DIRECTION)}")
        if clearance not in _CLEARANCE:
            raise GuavaToolError(f"clearance must be one of {list(_CLEARANCE)}")

        points = self._segment_object_points(object)
        center = self._centroid_outlier_removed(points)
        extent = points.max(axis=0) - points.min(axis=0)

        standoff = _CLEARANCE[clearance]
        direction = _DIRECTION[position]
        # Half-extent along the approach axis so clearance is measured from the
        # object surface, not its centroid.
        half = 0.5 * float(np.abs(extent @ np.abs(direction)))
        target = center + direction * (half + standoff)
        self._move_to(target, _TOP_DOWN_QUAT_WXYZ)
        return f"aligned {position} of {object} at {clearance} clearance"

    def get_position(self, object: str) -> list[float]:
        center = self._centroid_outlier_removed(self._segment_object_points(object))
        return [round(float(v), 4) for v in center]

    def get_position_and_size(self, object: str) -> dict[str, list[float]]:
        points = self._segment_object_points(object)
        center = self._centroid_outlier_removed(points)
        extent = points.max(axis=0) - points.min(axis=0)
        return {
            "position": [round(float(v), 4) for v in center],
            "size": [round(float(v), 4) for v in extent],
        }

    def move(self, x: float, y: float, z: float) -> str:
        self._move_to(np.array([x, y, z], dtype=np.float64), _TOP_DOWN_QUAT_WXYZ)
        return f"moved to [{x:.3f}, {y:.3f}, {z:.3f}]"

    def rotate(self, angle_deg: float, axis: str) -> str:
        if axis not in ("x", "y", "z"):
            raise GuavaToolError("axis must be one of {x, y, z}")
        pos, _ = self._gripper_pose()
        cur_quat = np.asarray(self._env.gripper_link_wxyz_xyz, dtype=np.float64)[:4]
        rpy = {"x": (np.radians(angle_deg), 0, 0),
               "y": (0, np.radians(angle_deg), 0),
               "z": (0, 0, np.radians(angle_deg))}[axis]
        new_quat = (vtf.SO3(wxyz=cur_quat) @ vtf.SO3.from_rpy_radians(*rpy)).wxyz
        self._move_to(pos, new_quat)
        return f"rotated {angle_deg} deg about {axis}"

    def close_gripper(self) -> str:
        self._api.close_gripper()
        return "gripper closed"

    def release(self) -> str:
        self._api.open_gripper()
        pos, _ = self._gripper_pose()
        # Short upward retraction to avoid post-release collisions (Appendix B).
        self._move_to(pos + np.array([0.0, 0.0, 0.05]), _TOP_DOWN_QUAT_WXYZ)
        return "released"

    def home_pose(self) -> str:
        self._api.move_to_joints(_HOME_JOINTS)
        return "returned to home pose"

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #
    _ARGFUL = {"grasp", "align", "get_position", "get_position_and_size", "move", "rotate"}

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute one semantic tool call and return its observation payload."""
        fn = getattr(self, name, None)
        if fn is None or name not in {t for t in dir(self)} or name.startswith("_"):
            raise GuavaToolError(f"unknown tool '{name}'")
        if name not in {
            "grasp", "align", "get_position", "get_position_and_size",
            "move", "rotate", "close_gripper", "release", "home_pose",
        }:
            raise GuavaToolError(f"'{name}' is not part of the GUAVA action space")
        return fn(**(arguments or {}))
