import os
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaacsim.core.utils.rotations import euler_angles_to_quat


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "../.."))


@clone
def spawn_so100_with_gripper_physics(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    return spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)

SO100_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_so100_with_gripper_physics,
        usd_path=os.path.join(
            PROJECT_ROOT,
            "assets",
            "so100.usd",
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=48,
            solver_velocity_iteration_count=8,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.26, 0.0, 0.45),
        rot=euler_angles_to_quat(np.array([0, 0, 270]), degrees=True),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 1.57,
            "elbow_flex": -1.57,
            "wrist_flex": 1.0,
            "wrist_roll": -1.57,
            "gripper": 1.2,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=15.0,
            velocity_limit_sim=8.0,
            stiffness={
                "shoulder_pan": 600.0,
                "shoulder_lift": 500.0,
                "elbow_flex": 360.0,
                "wrist_flex": 240.0,
                "wrist_roll": 160.0,
            },
            damping={
                "shoulder_pan": 120.0,
                "shoulder_lift": 100.0,
                "elbow_flex": 70.0,
                "wrist_flex": 45.0,
                "wrist_roll": 30.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10.0,
            velocity_limit_sim=5.0,
            stiffness=180.0,
            damping=50.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
