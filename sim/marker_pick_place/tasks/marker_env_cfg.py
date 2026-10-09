import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
import isaaclab.utils.math as math_utils
from isaacsim.core.utils.rotations import euler_angles_to_quat
from marker_pick_place.mdp import random_asset_pose, randomize_robot_color, reset_joints_by_offset
from isaaclab.sim import get_current_stage
from pxr import Gf, Sdf, UsdGeom

import torch


from marker_pick_place.assets.so100 import SO100_CFG
from marker_pick_place.assets.scene_objects import (
    BACK_BASEBOARD_CFG,
    BACK_WALL_CFG,
    BANANA_CFG,
    CUP_BACK_WALL_CFG,
    CUP_BOTTOM_CFG,
    CUP_FRONT_WALL_CFG,
    CUP_HANDLE_BOTTOM_CFG,
    CUP_HANDLE_OUTER_CFG,
    CUP_HANDLE_TOP_CFG,
    CUP_LEFT_WALL_CFG,
    CUP_RIGHT_WALL_CFG,
    FLOOR_CFG,
    LEFT_JOG_BASEBOARD_21_CFG,
    LEFT_JOG_BASEBOARD_42_5_CFG,
    LEFT_JOG_WALL_21_CFG,
    LEFT_JOG_WALL_42_5_CFG,
    LEFT_LONG_BASEBOARD_100_CFG,
    LEFT_LONG_WALL_100_CFG,
    RIGHT_BASEBOARD_CFG,
    RIGHT_FACE_BASEBOARD_CFG,
    RIGHT_UP_BASEBOARD_CFG,
    RIGHT_WALL_CFG,
    RIGHT_WALL_FACE_CFG,
    RIGHT_WALL_UP_CFG,
    TABLE_LEG_BACK_LEFT_CFG,
    TABLE_LEG_BACK_RIGHT_CFG,
    TABLE_LEG_FRONT_LEFT_CFG,
    TABLE_LEG_FRONT_RIGHT_CFG,
    TABLE_TOP_CFG,
)

from marker_pick_place.mdp import (
    JointPositionActionCfg,
    ee_frame_state,
    image,
    joint_pos,
    joint_pos_rel,
)


camera_cfg = CameraCfg(
    prim_path="",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        focal_length=13.5,
        f_stop=100.0,
        focus_distance=0.05,
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=euler_angles_to_quat(np.array([0, 0, 0]), degrees=True),
        convention="opengl",
    ),
)

# Gripper-local camera pose.
# Position comes from: inv(T_world_gripper) * T_world_camera.
# Rotation is a direct wxyz quaternion. Do not convert this through Euler degrees.
GRIPPER_CAMERA_POS = (0.02660, -0.00600, -0.12000)

# This rotation points the camera optical axis back toward the gripper pose.
# Quaternion convention: w, x, y, z.
GRIPPER_CAMERA_ROT = (-0.15123, 0.40528, -0.90154, -0.01034)

ENV_CAMERA_POS = (0.75, -0.75, 1.25)
ENV_CAMERA_ROT_DEG = (52.9746765, 0.0, 45.0)
ENV_CAMERA_ROT = tuple(euler_angles_to_quat(np.array(ENV_CAMERA_ROT_DEG), degrees=True))

VISUAL_RANDOMIZATION_GROUPS = {
    "table": [
        "table_top",
        "table_leg_front_left",
        "table_leg_front_right",
        "table_leg_back_left",
        "table_leg_back_right",
    ],
    "floor": ["floor"],
    "walls": [
        "back_wall",
        "right_wall",
        "right_wall_face",
        "right_wall_up",
        "left_jog_wall_42_5",
        "left_jog_wall_21",
        "left_long_wall_100",
        "back_baseboard",
        "right_baseboard",
        "right_face_baseboard",
        "right_up_baseboard",
        "left_jog_baseboard_42_5",
        "left_jog_baseboard_21",
        "left_long_baseboard_100",
    ],
    "cup": [
        "cup_bottom",
        "cup_front_wall",
        "cup_back_wall",
        "cup_left_wall",
        "cup_right_wall",
        "cup_handle_top",
        "cup_handle_bottom",
        "cup_handle_outer",
    ],
}

VISUAL_RANDOMIZATION_COLOR_PALETTES = {
    "table": [
        (0.04, 0.04, 0.04),  # black laminate
        (0.42, 0.25, 0.13),  # brown wood
        (0.58, 0.34, 0.18),  # warm tabletop
        (0.52, 0.30, 0.20),  # reddish wood
        (0.70, 0.55, 0.36),  # light plywood
    ],
    "floor": [
        (0.38, 0.38, 0.36),
        (0.58, 0.55, 0.50),
        (0.72, 0.68, 0.60),
        (0.50, 0.46, 0.40),
    ],
    "walls": [
        (0.92, 0.90, 0.84),
        (0.72, 0.69, 0.62),
        (0.48, 0.35, 0.72),  # purple sim backdrop
        (0.58, 0.68, 0.72),
        (0.64, 0.72, 0.58),
    ],
    "cup": [
        (0.05, 0.22, 0.85),
        (0.85, 0.10, 0.06),
        (0.05, 0.45, 0.18),
        (0.90, 0.78, 0.10),
        (0.92, 0.92, 0.84),
    ],
}

VISUAL_RANDOMIZATION_COLOR_JITTER = {
    "table": 0.05,
    "floor": 0.06,
    "walls": 0.08,
    "cup": 0.04,
}

VISUAL_RANDOMIZATION_MATERIAL_RANGES = {
    "table": {"roughness": (0.35, 0.95), "metallic": (0.0, 0.02)},
    "floor": {"roughness": (0.45, 1.0), "metallic": (0.0, 0.0)},
    "walls": {"roughness": (0.55, 1.0), "metallic": (0.0, 0.0)},
    "cup": {"roughness": (0.25, 0.85), "metallic": (0.0, 0.04)},
}


def _xform_attr(prim, attr_name, add_op):
    attr = prim.GetAttribute(attr_name)
    if attr.IsValid():
        return attr
    return add_op(UsdGeom.Xformable(prim)).GetAttr()


def _force_camera_usd_pose_writes(camera):
    """Camera render products consume USD-authored transforms more reliably than Fabric writes."""
    if hasattr(camera, "_view"):
        camera._view._use_fabric = False


def refresh_config_cameras(env, env_ids, camera_names=None):
    """Re-apply configured camera xforms so render products initialize cleanly."""
    if camera_names is None:
        camera_names = ["env_cam", "gripper_cam"]

    stage = get_current_stage()
    camera_updates = []
    for name in camera_names:
        camera = env.scene[name]
        for env_idx in range(env.num_envs):
            prim_path = camera.cfg.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_idx}")
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                camera_updates.append((prim, camera.cfg.offset.pos, camera.cfg.offset.rot))

    with Sdf.ChangeBlock():
        for prim, pos, _ in camera_updates:
            translate_attr = _xform_attr(prim, "xformOp:translate", lambda xform: xform.AddTranslateOp())
            translate_attr.Set(Gf.Vec3d(pos[0] + 1.0e-6, pos[1], pos[2]))

    with Sdf.ChangeBlock():
        for prim, pos, quat in camera_updates:
            translate_attr = _xform_attr(prim, "xformOp:translate", lambda xform: xform.AddTranslateOp())
            orient_attr = _xform_attr(prim, "xformOp:orient", lambda xform: xform.AddOrientOp())
            translate_attr.Set(Gf.Vec3d(*pos))
            orient_attr.Set(Gf.Quatd(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))


def sync_gripper_camera_to_frame(
    env,
    env_ids,
    camera_cfg=SceneEntityCfg("gripper_cam"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
):
    """Keep the external gripper camera riding on the gripper frame."""
    camera = env.scene[camera_cfg.name]
    _force_camera_usd_pose_writes(camera)
    ee_frame = env.scene[ee_frame_cfg.name]
    gripper_pos = ee_frame.data.target_pos_w[:, 0, :]
    gripper_quat = ee_frame.data.target_quat_w[:, 0, :]
    offset_pos = torch.tensor(camera.cfg.offset.pos, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    offset_quat = torch.tensor(camera.cfg.offset.rot, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    camera_pos, camera_quat = math_utils.combine_frame_transforms(
        gripper_pos,
        gripper_quat,
        offset_pos,
        offset_quat,
    )
    camera.set_world_poses(camera_pos.to(dtype=torch.float32), camera_quat.to(dtype=torch.float32), convention="opengl")


def sync_env_camera_to_world(
    env,
    env_ids,
    camera_cfg=SceneEntityCfg("env_cam"),
):
    """Keep the env camera authored at its configured world pose."""
    camera = env.scene[camera_cfg.name]
    _force_camera_usd_pose_writes(camera)
    env_camera_pos = torch.tensor(ENV_CAMERA_POS, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    env_camera_pos = env_camera_pos + env.scene.env_origins
    env_camera_quat = torch.tensor(
        euler_angles_to_quat(np.array(ENV_CAMERA_ROT_DEG), degrees=True),
        device=env.device,
        dtype=torch.float32,
    ).repeat(env.num_envs, 1)
    camera.set_world_poses(
        env_camera_pos,
        env_camera_quat,
        convention="opengl",
    )


def randomize_banana_on_table(
    env,
    env_ids,
    asset_cfg=SceneEntityCfg("banana"),
):
    """Place the banana on the left side of the tabletop for clean data collection."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    banana = env.scene[asset_cfg.name]
    root_states = banana.data.default_root_state[env_ids].clone()
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]

    positions[:, 0] = 0.13
    positions[:, 1] = -0.20
    positions[:, 2] = 0.485

    yaw = torch.zeros(len(env_ids), device=banana.device)
    roll = torch.zeros_like(yaw)
    pitch = torch.zeros_like(yaw)
    orientations_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    banana.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    zero_velocity = torch.zeros((len(env_ids), 6), device=banana.device)
    banana.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def randomize_banana_on_table_dr(
    env,
    env_ids,
    asset_cfg=SceneEntityCfg("banana"),
):
    """Randomize banana pose on the left side of the tabletop while avoiding the cup."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    banana = env.scene[asset_cfg.name]
    root_states = banana.data.default_root_state[env_ids].clone()
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]

    positions[:, 0] = torch.empty(len(env_ids), device=banana.device).uniform_(0.05, 0.18)
    positions[:, 1] = torch.empty(len(env_ids), device=banana.device).uniform_(-0.24, -0.10)
    positions[:, 2] = 0.485

    yaw = torch.empty(len(env_ids), device=banana.device).uniform_(-0.8, 0.8)
    roll = torch.zeros_like(yaw)
    pitch = torch.zeros_like(yaw)
    orientations_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    banana.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    zero_velocity = torch.zeros((len(env_ids), 6), device=banana.device)
    banana.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def _sample_palette_color(group_name):
    palette = VISUAL_RANDOMIZATION_COLOR_PALETTES[group_name]
    base_color = palette[torch.randint(0, len(palette), (1,), device="cpu").item()]
    jitter = VISUAL_RANDOMIZATION_COLOR_JITTER[group_name]
    channels = []
    for channel in base_color:
        value = channel + torch.empty(1, device="cpu").uniform_(-jitter, jitter).item()
        channels.append(max(0.0, min(1.0, value)))
    return tuple(channels)


def _sample_range(value_range):
    return torch.empty(1, device="cpu").uniform_(*value_range).item()


def _set_asset_material(env, asset_name, color, material_ranges):
    asset = env.scene[asset_name]
    prim_path = asset.prim_paths[0]
    material_path = f"{prim_path}/Looks/visualMaterial/Shader"
    prims = sim_utils.find_matching_prims(material_path)
    if not prims:
        return
    shader = prims[0]
    diffuse_attr = shader.GetAttribute("inputs:diffuse_color_constant")
    roughness_attr = shader.GetAttribute("inputs:roughness")
    metallic_attr = shader.GetAttribute("inputs:metallic")
    if diffuse_attr.IsValid():
        diffuse_attr.Set(color)
    if roughness_attr.IsValid() and "roughness" in material_ranges:
        roughness_attr.Set(float(_sample_range(material_ranges["roughness"])))
    if metallic_attr.IsValid() and "metallic" in material_ranges:
        metallic_attr.Set(float(_sample_range(material_ranges["metallic"])))


def randomize_scene_visuals(env, env_ids):
    """Randomize USD materials and dome light to create visible sim-to-real variation."""
    with Sdf.ChangeBlock():
        for group_name, asset_names in VISUAL_RANDOMIZATION_GROUPS.items():
            color = _sample_palette_color(group_name)
            material_ranges = VISUAL_RANDOMIZATION_MATERIAL_RANGES[group_name]
            for asset_name in asset_names:
                _set_asset_material(env, asset_name, color, material_ranges)

        light_prim = get_current_stage().GetPrimAtPath("/World/Light")
        if light_prim.IsValid():
            intensity = _sample_range((1200.0, 5200.0))
            color = (
                _sample_range((0.70, 1.0)),
                _sample_range((0.70, 1.0)),
                _sample_range((0.65, 1.0)),
            )
            intensity_attr = light_prim.GetAttribute("inputs:intensity")
            color_attr = light_prim.GetAttribute("inputs:color")
            if intensity_attr.IsValid():
                intensity_attr.Set(float(intensity))
            if color_attr.IsValid():
                color_attr.Set(Gf.Vec3f(*color))


@configclass
class MarkerSceneCfg(InteractiveSceneCfg):
    """Scene for banana pick-place demo collection."""

    num_envs = 1
    env_spacing = 2.0

    robot = SO100_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper",
                name="gripper",
            ),
        ],
    )

    env_cam_mount = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/EnvCamMount",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.02),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=ENV_CAMERA_POS,
            rot=ENV_CAMERA_ROT,
        ),
    )

    env_cam = camera_cfg.replace()
    env_cam.prim_path = "{ENV_REGEX_NS}/EnvCamMount/EnvCam"
    env_cam.offset.pos = (0.0, 0.0, 0.0)
    env_cam.offset.rot = (1.0, 0.0, 0.0, 0.0)

    gripper_cam = camera_cfg.replace()
    gripper_cam.prim_path = "{ENV_REGEX_NS}/GripperCamera"
    gripper_cam.width = 640
    gripper_cam.height = 360
    gripper_cam.spawn.focus_distance = 0.5
    gripper_cam.offset.pos = GRIPPER_CAMERA_POS
    gripper_cam.offset.rot = GRIPPER_CAMERA_ROT

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2800.0,
            color=(0.85, 0.85, 0.85),
        ),
    )

    floor = FLOOR_CFG

    back_wall = BACK_WALL_CFG
    right_wall = RIGHT_WALL_CFG
    right_wall_face = RIGHT_WALL_FACE_CFG
    right_wall_up = RIGHT_WALL_UP_CFG
    left_jog_wall_42_5 = LEFT_JOG_WALL_42_5_CFG
    left_jog_wall_21 = LEFT_JOG_WALL_21_CFG
    left_long_wall_100 = LEFT_LONG_WALL_100_CFG

    back_baseboard = BACK_BASEBOARD_CFG
    right_baseboard = RIGHT_BASEBOARD_CFG
    right_face_baseboard = RIGHT_FACE_BASEBOARD_CFG
    right_up_baseboard = RIGHT_UP_BASEBOARD_CFG
    left_jog_baseboard_42_5 = LEFT_JOG_BASEBOARD_42_5_CFG
    left_jog_baseboard_21 = LEFT_JOG_BASEBOARD_21_CFG
    left_long_baseboard_100 = LEFT_LONG_BASEBOARD_100_CFG

    table_top = TABLE_TOP_CFG
    table_leg_front_left = TABLE_LEG_FRONT_LEFT_CFG
    table_leg_front_right = TABLE_LEG_FRONT_RIGHT_CFG
    table_leg_back_left = TABLE_LEG_BACK_LEFT_CFG
    table_leg_back_right = TABLE_LEG_BACK_RIGHT_CFG

    cup_bottom = CUP_BOTTOM_CFG
    cup_front_wall = CUP_FRONT_WALL_CFG
    cup_back_wall = CUP_BACK_WALL_CFG
    cup_left_wall = CUP_LEFT_WALL_CFG
    cup_right_wall = CUP_RIGHT_WALL_CFG
    cup_handle_top = CUP_HANDLE_TOP_CFG
    cup_handle_bottom = CUP_HANDLE_BOTTOM_CFG
    cup_handle_outer = CUP_HANDLE_OUTER_CFG

    banana = BANANA_CFG


@configclass
class ActionsCfg:
    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        scale=1,
        use_default_offset=False,
    )

@configclass
class EventCfg:
    startup_refresh_camera_xforms = EventTerm(
        func=refresh_config_cameras,
        mode="startup",
        params={
            "camera_names": ["env_cam"],
        },
    )

    startup_sync_env_camera = EventTerm(
        func=sync_env_camera_to_world,
        mode="startup",
    )

    refresh_camera_xforms = EventTerm(
        func=refresh_config_cameras,
        mode="reset",
        params={
            "camera_names": ["env_cam"],
        },
    )

    sync_env_camera = None

    sync_gripper_camera = EventTerm(
        func=sync_gripper_camera_to_frame,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
    )

    reset_robot_position = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ],
            ),
            "position_range": (-0.04, 0.04),
            "velocity_range": (0, 0),
        },
    )

    randomize_banana_pose = EventTerm(
        func=randomize_banana_on_table,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("banana"),
        },
    )

    randomize_banana_pose_dr = None

    reset_sync_gripper_camera = EventTerm(
        func=sync_gripper_camera_to_frame,
        mode="reset",
    )

    reset_sync_env_camera = EventTerm(
        func=sync_env_camera_to_world,
        mode="reset",
    )

    randomize_robot_appearance = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={
            "color_names": ["white"],
        },
    )

    randomize_robot_appearance_dr = None

    randomize_scene_visuals_dr = None

def banana_pose(env):
    banana = env.scene["banana"]
    return torch.cat(
        [
            banana.data.root_pos_w,
            banana.data.root_quat_w,
        ],
        dim=-1,
    )

def cup_pose(env):
    cup_pos = torch.tensor(
        [-0.275 + 0.23, -0.275 + 0.12, 0.45 + 0.004],
        device=env.device,
    ).repeat(env.num_envs, 1)
    cup_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        device=env.device,
    ).repeat(env.num_envs, 1)
    return torch.cat([cup_pos, cup_quat], dim=-1)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_obs = ObsTerm(func=joint_pos)
        joint_pos_rel = ObsTerm(func=joint_pos_rel)
        banana_pose = ObsTerm(func=banana_pose)
        cup_pose = ObsTerm(func=cup_pose)
        ee_frame_state = ObsTerm(
            func=ee_frame_state,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class VisualCfg(ObsGroup):
        rgb_env = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("env_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        rgb_gripper = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("gripper_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    visual: VisualCfg = VisualCfg()


@configclass
class MarkerTeleopEnvCfg(ManagerBasedRLEnvCfg):
    """Teleop/IL environment config for banana pick-place demo collection."""

    scene: MarkerSceneCfg = MarkerSceneCfg()

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    events: EventCfg = EventCfg()
    rewards = None
    terminations = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0

        self.scene.num_envs = 1

        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 3
        self.image_obs_list = ["env_cam", "gripper_cam"]
        self.sim.physx.enable_ccd = True
        self.sim.physx.enable_stabilization = True

        self.viewer.eye = (0.75, -0.75, 1.25)
        self.viewer.lookat = (0.0, 0.0, 0.45)
