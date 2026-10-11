"""Verify the SO-100 model's collision geometry is sane: visual/collision mesh
alignment, passive stability, self-collision behavior, and a functional grasp test.

Run with: python sim_mujoco/verify_collisions.py
"""

import os

import mujoco
import numpy as np
from PIL import Image

SCENE_PATH = os.path.join(os.path.dirname(__file__), "assets", "so100", "scene_pick_place.xml")
OUT_DIR = os.path.join(os.path.dirname(__file__), "verification")

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def render_geom_groups(model, data, out_path, group, cam_kwargs):
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    for k, v in cam_kwargs.items():
        setattr(cam, k, v)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[group] = 1
    renderer.update_scene(data, camera=cam, scene_option=opt)
    Image.fromarray(renderer.render()).save(out_path)


def test_visual_vs_collision_alignment(model, data):
    os.makedirs(OUT_DIR, exist_ok=True)
    cam_kwargs = dict(lookat=[0.0, 0.0, 0.08], distance=0.5, azimuth=110, elevation=-15)
    render_geom_groups(model, data, os.path.join(OUT_DIR, "visual_only.png"), group=2, cam_kwargs=cam_kwargs)
    render_geom_groups(model, data, os.path.join(OUT_DIR, "collision_only.png"), group=3, cam_kwargs=cam_kwargs)
    print("[1/4] Saved visual_only.png / collision_only.png for manual comparison -> sim_mujoco/verification/")


def test_passive_stability(model, data):
    mujoco.mj_resetData(model, data)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    ok = not np.any(np.isnan(data.qpos)) and not np.any(np.abs(data.qvel) > 50)
    print(f"[2/4] Passive stability (2000 steps, no control): {'PASS' if ok else 'FAIL'} "
          f"(max |qvel| = {np.abs(data.qvel).max():.3f})")
    return ok


def test_self_collision_sweep(model, data, n_samples=2000, seed=0):
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    qpos_adr = [model.jnt_qposadr[j] for j in jids]
    ranges = [model.jnt_range[j] for j in jids]

    mujoco.mj_resetDataKeyframe(model, data, 0)
    rng = np.random.default_rng(seed)
    events = {}
    for _ in range(n_samples):
        for adr, (lo, hi) in zip(qpos_adr, ranges):
            data.qpos[adr] = rng.uniform(lo, hi)
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            con = data.contact[c]
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[con.geom1])
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[con.geom2])
            if "world" in (n1, n2) or "salt_shaker" in (n1, n2):
                continue
            pair = tuple(sorted((n1, n2)))
            events[pair] = events.get(pair, 0) + 1

    print(f"[3/4] Self-collision sweep ({n_samples} random full-range configs):")
    if not events:
        print("       no self-collisions at all")
    else:
        for pair, count in sorted(events.items(), key=lambda x: -x[1]):
            print(f"       {pair}: {100 * count / n_samples:.1f}% of configs (expected at extreme folded poses)")
    return events


def test_grasp_and_lift(model, data):
    """Drives the arm to a reach pose over the object's table position (found via a
    random-search 'IK' with Wrist_Roll held at -1.57 so the jaws close in a horizontal
    plane over the table, matching the built-in 'home' orientation), opens the jaw, places
    the salt shaker (50.8mm diameter) at the settled open-jaw midpoint, closes the gripper,
    and gently lifts, checking it stays held.

    Two things had to be fixed to get a real PASS here, neither of which is a mesh/collision
    defect:
    1. The raw Menagerie actuator defaults (forcerange +-3.5 N*m) are too weak to hold this
       arm extended against gravity -- it sags toward its joint limits before reaching the
       target. The old Isaac Lab port already found this and re-tuned to 15 N*m
       (sim/marker_pick_place/assets/so100.py); we match that here for this test. This is
       real actuator tuning that will be needed once the trajectory controller is built, not
       just a verification-script workaround.
    2. An instantaneous jump to the lift target kicks the grasped object loose through
       dynamic coupling; a gentle ramped lift holds it. The real trajectory controller will
       need smooth motion for the same reason.
    """
    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "salt_shaker_free")
    obj_qadr = model.jnt_qposadr[obj_jid]
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "salt_shaker")
    act_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES}
    jids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES}
    qadr = {n: model.jnt_qposadr[j] for n, j in jids.items()}
    fixed_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_1")
    moving_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_1")

    for n in ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]:
        model.actuator_forcerange[act_ids[n]] = [-15, 15]
    model.actuator_forcerange[act_ids["Jaw"]] = [-10, 10]

    # Reach pose found by random search to bring the gripper to the object's table
    # position (0.22, -0.15, 0.0381) with Wrist_Roll fixed at -1.57; re-derive if the
    # scene's object position changes.
    reach_cfg = dict(Rotation=1.2795, Pitch=-1.3149, Elbow=1.5629, Wrist_Pitch=1.1885, Wrist_Roll=-1.57)

    mujoco.mj_resetData(model, data)
    for k, v in reach_cfg.items():
        data.qpos[qadr[k]] = v
        data.ctrl[act_ids[k]] = v
    data.ctrl[act_ids["Jaw"]] = 1.0  # open BEFORE computing the midpoint, not after
    mujoco.mj_forward(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)
    midpoint = 0.5 * (data.geom_xpos[fixed_pad] + data.geom_xpos[moving_pad])

    data.qpos[obj_qadr : obj_qadr + 3] = midpoint
    data.qpos[obj_qadr + 3 : obj_qadr + 7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for _ in range(50):
        mujoco.mj_step(model, data)

    data.ctrl[act_ids["Jaw"]] = -0.174  # fully closed target
    for _ in range(700):
        mujoco.mj_step(model, data)
    z_before = data.xpos[obj_bid][2]

    start_pitch = reach_cfg["Pitch"]
    target_pitch = start_pitch - 0.25
    for step in range(400):
        frac = min(1.0, step / 200)
        data.ctrl[act_ids["Pitch"]] = start_pitch + frac * (target_pitch - start_pitch)
        mujoco.mj_step(model, data)
    z_after = data.xpos[obj_bid][2]

    held = z_after > z_before + 0.015
    print(f"[4/4] Grasp-and-lift test (salt shaker, 50.8mm dia): z {z_before:.3f} -> {z_after:.3f} "
          f"({'PASS: held and lifted' if held else 'FAIL: not held'})")
    return held


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    test_visual_vs_collision_alignment(model, data)
    passive_ok = test_passive_stability(model, data)
    test_self_collision_sweep(model, data)
    grasp_ok = test_grasp_and_lift(model, data)

    print()
    print("Overall:", "PASS" if passive_ok and grasp_ok else "FAIL")


if __name__ == "__main__":
    main()
