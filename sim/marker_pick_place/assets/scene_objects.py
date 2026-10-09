import os

import isaaclab.sim as sim_utils
from pxr import Usd
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import bind_physics_material, clone, get_current_stage


WALL_WHITE = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.90, 0.86))
WALL_GREY = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.86, 0.82))
BASEBOARD_WHITE = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.94, 0.93, 0.90))
TABLE_BLACK = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.005, 0.005, 0.004))
CUP_BLUE = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.22, 0.85))

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "../.."))
BANANA_USD_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "banana",
    "usd",
    "banana_convex_decomposition.usd",
)
BANANA_MASS = 0.12
BANANA_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=2.0,
    dynamic_friction=1.4,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="min",
)


@clone
def spawn_banana_with_physics(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    from isaaclab.sim import schemas
    from isaaclab.sim.schemas import schemas_cfg

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = get_current_stage()
    schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    material_path = f"{prim_path}/physicsMaterial"
    BANANA_PHYSICS_MATERIAL.func(material_path, BANANA_PHYSICS_MATERIAL)

    collision_cfg = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.002,
        rest_offset=0.0,
    )
    mesh_collision_cfg = schemas_cfg.ConvexDecompositionPropertiesCfg(
        max_convex_hulls=16,
        voxel_resolution=250000,
        error_percentage=3.0,
    )
    for descendant in Usd.PrimRange(prim):
        if descendant.GetTypeName() == "Mesh":
            mesh_path = str(descendant.GetPath())
            schemas.define_collision_properties(mesh_path, collision_cfg, stage=stage)
            schemas.define_mesh_collision_properties(mesh_path, mesh_collision_cfg, stage=stage)
            bind_physics_material(mesh_path, material_path, stage=stage)
    return prim


def fixed_box_cfg(prim_path, size, pos, material):
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=material,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


FLOOR_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/Floor",
    (1.80, 2.60, 0.01),
    (0.25, 0.0, -0.005),
    sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.42)),
)

BACK_WALL_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/BackWall",
    (0.04, 1.55, 1.40),
    (-0.295, -0.23, 0.70),
    WALL_WHITE,
)

RIGHT_WALL_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/RightWall",
    (0.35, 0.04, 1.40),
    (-0.295 + 0.35 / 2.0, 0.545, 0.70),
    WALL_GREY,
)

RIGHT_WALL_FACE_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/RightWallFace",
    (0.04, 0.35, 1.40),
    (0.055, 0.545 + 0.35 / 2.0, 0.70),
    WALL_GREY,
)

RIGHT_WALL_UP_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/RightWallUp",
    (0.35, 0.04, 1.40),
    (0.055 - 0.35 / 2.0, 0.545 + 0.35, 0.70),
    WALL_WHITE,
)

LEFT_JOG_WALL_42_5_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/LeftJogWall42_5",
    (0.425, 0.04, 1.40),
    (-0.295 + 0.425 / 2.0, -1.005, 0.70),
    WALL_WHITE,
)

LEFT_JOG_WALL_21_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/LeftJogWall21",
    (0.04, 0.21, 1.40),
    (-0.295 + 0.425, -1.005 - 0.21 / 2.0, 0.70),
    WALL_GREY,
)

LEFT_LONG_WALL_100_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/LeftLongWall100",
    (1.00, 0.04, 1.40),
    (-0.295 + 0.425 + 1.00 / 2.0, -1.005 - 0.21, 0.70),
    WALL_WHITE,
)

BACK_BASEBOARD_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/BackBaseboard",
    (0.035, 1.55, 0.07),
    (-0.270, -0.23, 0.035),
    BASEBOARD_WHITE,
)

RIGHT_BASEBOARD_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/RightBaseboard",
    (0.35, 0.035, 0.07),
    (-0.295 + 0.35 / 2.0, 0.520, 0.035),
    BASEBOARD_WHITE,
)

RIGHT_FACE_BASEBOARD_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/RightFaceBaseboard",
    (0.035, 0.35, 0.07),
    (0.030, 0.545 + 0.35 / 2.0, 0.035),
    BASEBOARD_WHITE,
)

RIGHT_UP_BASEBOARD_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/RightUpBaseboard",
    (0.35, 0.035, 0.07),
    (0.055 - 0.35 / 2.0, 0.545 + 0.35 - 0.025, 0.035),
    BASEBOARD_WHITE,
)

LEFT_JOG_BASEBOARD_42_5_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/LeftJogBaseboard42_5",
    (0.425, 0.035, 0.07),
    (-0.295 + 0.425 / 2.0, -0.980, 0.035),
    BASEBOARD_WHITE,
)

LEFT_JOG_BASEBOARD_21_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/LeftJogBaseboard21",
    (0.035, 0.21, 0.07),
    (-0.295 + 0.400, -1.005 - 0.21 / 2.0, 0.035),
    BASEBOARD_WHITE,
)

LEFT_LONG_BASEBOARD_100_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/LeftLongBaseboard100",
    (1.00, 0.035, 0.07),
    (-0.295 + 0.425 + 1.00 / 2.0, -1.005 - 0.185, 0.035),
    BASEBOARD_WHITE,
)

TABLE_TOP_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/TableTop",
    (0.55, 0.55, 0.04),
    (0.0, 0.0, 0.45 - 0.04 / 2.0),
    TABLE_BLACK,
)

TABLE_LEG_FRONT_LEFT_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/TableLegFrontLeft",
    (0.055, 0.055, 0.43),
    (0.23, 0.23, 0.43 / 2.0),
    TABLE_BLACK,
)

TABLE_LEG_FRONT_RIGHT_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/TableLegFrontRight",
    (0.055, 0.055, 0.43),
    (0.23, -0.23, 0.43 / 2.0),
    TABLE_BLACK,
)

TABLE_LEG_BACK_LEFT_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/TableLegBackLeft",
    (0.055, 0.055, 0.43),
    (-0.23, 0.23, 0.43 / 2.0),
    TABLE_BLACK,
)

TABLE_LEG_BACK_RIGHT_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/TableLegBackRight",
    (0.055, 0.055, 0.43),
    (-0.23, -0.23, 0.43 / 2.0),
    TABLE_BLACK,
)

CUP_BOTTOM_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/CupBottom",
    (0.08, 0.08, 0.008),
    (-0.275 + 0.23, -0.275 + 0.12, 0.45 + 0.004),
    CUP_BLUE,
)

CUP_FRONT_WALL_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/CupFrontWall",
    (0.006, 0.08, 0.095),
    (-0.275 + 0.23 + 0.04, -0.275 + 0.12, 0.45 + 0.095 / 2.0),
    CUP_BLUE,
)

CUP_BACK_WALL_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/CupBackWall",
    (0.006, 0.08, 0.095),
    (-0.275 + 0.23 - 0.04, -0.275 + 0.12, 0.45 + 0.095 / 2.0),
    CUP_BLUE,
)

CUP_LEFT_WALL_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/CupLeftWall",
    (0.08, 0.006, 0.095),
    (-0.275 + 0.23, -0.275 + 0.12 + 0.04, 0.45 + 0.095 / 2.0),
    CUP_BLUE,
)

CUP_RIGHT_WALL_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/CupRightWall",
    (0.08, 0.006, 0.095),
    (-0.275 + 0.23, -0.275 + 0.12 - 0.04, 0.45 + 0.095 / 2.0),
    CUP_BLUE,
)

CUP_HANDLE_TOP_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/CupHandleTop",
    (0.012, 0.045, 0.010),
    (-0.275 + 0.23, -0.275 + 0.12 - 0.065, 0.45 + 0.072),
    CUP_BLUE,
)

CUP_HANDLE_BOTTOM_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/CupHandleBottom",
    (0.012, 0.045, 0.010),
    (-0.275 + 0.23, -0.275 + 0.12 - 0.065, 0.45 + 0.032),
    CUP_BLUE,
)

CUP_HANDLE_OUTER_CFG = fixed_box_cfg(
    "{ENV_REGEX_NS}/CupHandleOuter",
    (0.012, 0.010, 0.050),
    (-0.275 + 0.23, -0.275 + 0.12 - 0.085, 0.45 + 0.052),
    CUP_BLUE,
)

BANANA_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Banana",
    spawn=sim_utils.UsdFileCfg(
        func=spawn_banana_with_physics,
        usd_path=BANANA_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            max_linear_velocity=1.0,
            max_angular_velocity=180.0,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
            linear_damping=0.03,
            angular_damping=0.03,
            sleep_threshold=0.001,
            stabilization_threshold=0.0001,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=BANANA_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.002,
            rest_offset=0.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.08, -0.10, 0.485),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)
