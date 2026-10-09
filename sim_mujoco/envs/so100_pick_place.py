"""Minimal SO-100 pick-and-place MuJoCo environment.

No robosuite/LIBERO dependency yet -- this is the raw MuJoCo scaffold
(arm + table + cube) that a robosuite RobotModel/task wrapper can be
built on top of later. See sim_mujoco/README.md for status and next steps.
"""

import os

import mujoco
import numpy as np

SCENE_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "so100", "scene_pick_place.xml")

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


class SO100PickPlaceEnv:
    """Simple joint-position-controlled pick-and-place task for the SO-100 arm."""

    def __init__(self, scene_path: str = SCENE_PATH):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES]
        )
        self.target_xy = np.array(self.model.geom("target_zone").pos[:2])

        self.reset()

    def reset(self) -> dict:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray, n_substeps: int = 5) -> tuple[dict, float, bool, dict]:
        """action: 6 target joint positions (rad), ordered per JOINT_NAMES."""
        self.data.ctrl[self.actuator_ids] = action
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        cube_xy = obs["cube_pos"][:2]
        cube_z = obs["cube_pos"][2]
        dist_to_target = float(np.linalg.norm(cube_xy - self.target_xy))
        success = dist_to_target < 0.05 and cube_z < 0.03
        reward = -dist_to_target
        return obs, reward, success, {"dist_to_target": dist_to_target}

    def _get_obs(self) -> dict:
        return {
            "qpos": self.data.qpos[: len(JOINT_NAMES)].copy(),
            "qvel": self.data.qvel[: len(JOINT_NAMES)].copy(),
            "cube_pos": self.data.xpos[self.cube_body_id].copy(),
            "cube_quat": self.data.xquat[self.cube_body_id].copy(),
        }
