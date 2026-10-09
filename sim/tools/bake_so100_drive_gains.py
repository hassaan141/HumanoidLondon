import argparse
from pathlib import Path

from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade


JOINT_GAINS = {
    "shoulder_pan": {"stiffness": 600.0, "damping": 120.0, "max_force": 15.0},
    "shoulder_lift": {"stiffness": 500.0, "damping": 100.0, "max_force": 15.0},
    "elbow_flex": {"stiffness": 360.0, "damping": 70.0, "max_force": 15.0},
    "wrist_flex": {"stiffness": 240.0, "damping": 45.0, "max_force": 15.0},
    "wrist_roll": {"stiffness": 160.0, "damping": 30.0, "max_force": 15.0},
    "gripper": {"stiffness": 180.0, "damping": 50.0, "max_force": 10.0},
}

DISABLED_XFORM_COLLIDERS = [
    "/so_arm100/gripper/collisions",
    "/so_arm100/jaw/collisions",
]

CONVEX_DECOMPOSITION_COLLIDERS = [
    "/colliders/gripper/Fixed_Jaw/mesh",
    "/colliders/jaw/Moving_Jaw/mesh",
]

GRIPPER_PHYSICS_MATERIAL_PATH = "/PhysicsMaterials/gripper_high_friction"


def apply_drive(joint_prim, gains):
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive.GetStiffnessAttr().Set(gains["stiffness"])
    drive.GetDampingAttr().Set(gains["damping"])
    drive.GetMaxForceAttr().Set(gains["max_force"])


def apply_convex_decomposition(collider_prim):
    collision = UsdPhysics.CollisionAPI(collider_prim)
    if not collision:
        collision = UsdPhysics.CollisionAPI.Apply(collider_prim)
    collision.GetCollisionEnabledAttr().Set(True)

    mesh_collision = UsdPhysics.MeshCollisionAPI(collider_prim)
    if not mesh_collision:
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collider_prim)
    mesh_collision.GetApproximationAttr().Set("convexDecomposition")

    convex_decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI(collider_prim)
    if not convex_decomposition:
        convex_decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(collider_prim)
    convex_decomposition.GetMaxConvexHullsAttr().Set(24)
    convex_decomposition.GetVoxelResolutionAttr().Set(200000)
    convex_decomposition.GetErrorPercentageAttr().Set(3.0)


def ensure_gripper_physics_material(stage):
    material_prim = stage.DefinePrim(GRIPPER_PHYSICS_MATERIAL_PATH, "Material")
    material = UsdPhysics.MaterialAPI(material_prim)
    if not material:
        material = UsdPhysics.MaterialAPI.Apply(material_prim)
    material.GetStaticFrictionAttr().Set(2.0)
    material.GetDynamicFrictionAttr().Set(1.4)
    material.GetRestitutionAttr().Set(0.0)

    physx_material = PhysxSchema.PhysxMaterialAPI(material_prim)
    if not physx_material:
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    physx_material.GetFrictionCombineModeAttr().Set("max")
    physx_material.GetRestitutionCombineModeAttr().Set("min")
    return UsdShade.Material(material_prim)


def bind_physics_material(collider_prim, material):
    binding = UsdShade.MaterialBindingAPI(collider_prim)
    if not binding:
        binding = UsdShade.MaterialBindingAPI.Apply(collider_prim)
    binding.Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def disable_collision(collider_prim):
    collision = UsdPhysics.CollisionAPI(collider_prim)
    if collision:
        collision.GetCollisionEnabledAttr().Set(False)


def main():
    parser = argparse.ArgumentParser(description="Bake SO100 drive gains into USD.")
    parser.add_argument(
        "--usd",
        required=True,
        help="Path to the SO100 USD file",
    )
    args = parser.parse_args()

    usd_path = Path(args.usd)
    if not usd_path.exists():
        raise FileNotFoundError(f"USD not found: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    updated = []
    for prim in stage.Traverse():
        name = prim.GetName()
        if name in JOINT_GAINS and "Joint" in prim.GetTypeName():
            apply_drive(prim, JOINT_GAINS[name])
            updated.append(name)

    gripper_material = ensure_gripper_physics_material(stage)
    updated_colliders = []
    for prim_path in DISABLED_XFORM_COLLIDERS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing collider prim: {prim_path}")
        disable_collision(prim)

    for prim_path in CONVEX_DECOMPOSITION_COLLIDERS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing collider prim: {prim_path}")
        apply_convex_decomposition(prim)
        bind_physics_material(prim, gripper_material)
        updated_colliders.append(prim_path)

    stage.Save()
    print("[INFO] Updated joints:")
    for name in sorted(updated):
        print(f" - {name}")
    print("[INFO] Updated convex decomposition colliders:")
    for prim_path in updated_colliders:
        print(f" - {prim_path}")


if __name__ == "__main__":
    main()
