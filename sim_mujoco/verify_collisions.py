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
            if "world" in (n1, n2) or "cube" in (n1, n2):
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
    cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    cube_qadr = model.jnt_qposadr[cube_jid]
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    act_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES}

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[act_ids["Jaw"]] = 1.5  # open
    data.qpos[cube_qadr : cube_qadr + 3] = [-0.01, -0.235, 0.165]
    data.qpos[cube_qadr + 3 : cube_qadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    z_before = data.xpos[cube_bid][2]

    data.ctrl[act_ids["Jaw"]] = -0.1  # close
    for _ in range(500):
        mujoco.mj_step(model, data)

    data.ctrl[act_ids["Pitch"]] += -0.3  # lift
    for _ in range(800):
        mujoco.mj_step(model, data)
    z_after = data.xpos[cube_bid][2]

    held = z_after > z_before + 0.02
    print(f"[4/4] Grasp-and-lift test: cube z {z_before:.3f} -> {z_after:.3f} "
          f"({'PASS: cube held and lifted' if held else 'FAIL: cube not held'})")
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
