"""Quick sanity check: load the SO-100 pick-place scene, reset, and step with a few
scripted actions. Run with: python sim_mujoco/smoke_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from envs.so100_pick_place import SO100PickPlaceEnv, JOINT_NAMES  # noqa: E402


def main():
    env = SO100PickPlaceEnv()
    obs = env.reset()
    print("[smoke_test] loaded scene OK")
    print(f"[smoke_test] joints: {JOINT_NAMES}")
    print(f"[smoke_test] initial qpos: {obs['qpos']}")
    print(f"[smoke_test] initial cube_pos: {obs['cube_pos']}")

    home_action = np.array([0, -1.57, 1.57, 1.57, -1.57, 0])
    for i in range(200):
        obs, reward, success, info = env.step(home_action)

    print(f"[smoke_test] after 200 steps holding home pose:")
    print(f"[smoke_test]   qpos: {obs['qpos']}")
    print(f"[smoke_test]   cube_pos: {obs['cube_pos']} (should still be resting on table)")
    print(f"[smoke_test]   dist_to_target: {info['dist_to_target']:.3f}, success: {success}")
    print("[smoke_test] PASS: physics stepped without errors")


if __name__ == "__main__":
    main()
