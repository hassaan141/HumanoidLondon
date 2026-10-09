import argparse

from isaaclab.app import AppLauncher

from marker_pick_place.teleop.dataset_recorder import RecorderConfig, TeleopDatasetRecorder
from marker_pick_place.teleop.lerobot_interface import SO100LeaderTeleop, SO100TeleopConfig


TASK_NAME = "Marker-PickPlace-Teleop"
HARD_CODED_PORT = "/dev/ttyACM0"
HARD_CODED_ID = "leader_arm_1"


parser = argparse.ArgumentParser(description="SO100 marker pick-place.")
parser.add_argument("--teleop", action="store_true", default=False)
parser.add_argument("--record", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--debug-control", action="store_true", default=False)
parser.add_argument("--debug-control-interval", type=int, default=30)
parser.add_argument("--repo-id", type=str, default="local/banana_pick")
parser.add_argument("--repo-root", type=str, default="datasets/banana_pick_lerobot")
parser.add_argument("--task-name", type=str, default="pick up the banana and place it in the cup")
parser.add_argument("--domain-randomization", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import get_current_stage
from pxr import Usd, UsdPhysics, UsdShade
import carb
import omni.appwindow

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import marker_pick_place.tasks  # noqa: F401
from marker_pick_place.tasks.marker_env_cfg import (
    randomize_banana_on_table_dr,
    randomize_scene_visuals,
    refresh_config_cameras,
    sync_env_camera_to_world,
    sync_gripper_camera_to_frame,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from marker_pick_place.mdp import randomize_robot_color


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
CAMERA_SENSOR_NAMES = ["env_cam", "gripper_cam"]


class TeleopHotkeys:
    """Collect key presses from the Isaac Sim viewport."""

    def __init__(self) -> None:
        self.start_recording = False
        self.stop_recording = False
        self.reset_env = False
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )

    def _on_keyboard_event(self, event) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if event.input == carb.input.KeyboardInput.R:
            self.start_recording = True
        elif event.input == carb.input.KeyboardInput.Y:
            self.stop_recording = True
        elif event.input == carb.input.KeyboardInput.T:
            self.reset_env = True
        return True


def _print_robot_actuator_config(env) -> None:
    robot = env.unwrapped.scene["robot"]
    print("[INFO]: Loaded robot actuator config:")
    for actuator_name, actuator_cfg in robot.cfg.actuators.items():
        print(
            "[INFO]: "
            f"{actuator_name}: joints={actuator_cfg.joint_names_expr}, "
            f"effort={actuator_cfg.effort_limit_sim}, "
            f"velocity={actuator_cfg.velocity_limit_sim}, "
            f"stiffness={actuator_cfg.stiffness}, "
            f"damping={actuator_cfg.damping}"
        )
    print(f"[INFO]: Robot joint order: {robot.data.joint_names}")
    print(f"[INFO]: Action space: {env.action_space}")


def _format_joint_values(values: torch.Tensor, precision: int = 3) -> str:
    values_cpu = values.detach().cpu().tolist()
    return ", ".join(
        f"{name}={value:.{precision}f}" for name, value in zip(JOINT_NAMES, values_cpu)
    )


def _resolve_joint_indices(env) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    indices = []
    for joint_name in JOINT_NAMES:
        if joint_name not in robot.data.joint_names:
            raise RuntimeError(
                f"Joint '{joint_name}' not found in robot joints: {robot.data.joint_names}"
            )
        indices.append(robot.data.joint_names.index(joint_name))
    return torch.tensor(indices, dtype=torch.long, device=env.unwrapped.device)


def _print_runtime_joint_metadata(env, joint_indices: torch.Tensor) -> None:
    robot = env.unwrapped.scene["robot"]
    limits = robot.data.soft_joint_pos_limits[0, joint_indices].detach().cpu()
    print("[INFO]: Runtime joint position limits (rad):")
    for name, limit in zip(JOINT_NAMES, limits):
        print(f"[INFO]:   {name}: lower={limit[0]:.4f}, upper={limit[1]:.4f}")


def _print_physics_debug(env) -> None:
    stage = get_current_stage()
    robot = env.unwrapped.scene["robot"]
    print("[INFO]: Runtime physics debug:")
    print(
        "[INFO]:   robot self-collisions="
        f"{robot.cfg.spawn.articulation_props.enabled_self_collisions}, "
        f"contact_sensors={robot.cfg.spawn.activate_contact_sensors}"
    )

    paths = [
        "/World/envs/env_0/Robot/gripper/collisions/Fixed_Jaw/mesh",
        "/World/envs/env_0/Robot/jaw/collisions/Moving_Jaw/mesh",
    ]
    banana_prim = stage.GetPrimAtPath("/World/envs/env_0/Banana")
    if banana_prim.IsValid():
        for descendant in Usd.PrimRange(banana_prim):
            if descendant.GetTypeName() == "Mesh":
                paths.append(str(descendant.GetPath()))

    for prim_path in paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[WARN]:   missing physics prim: {prim_path}")
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
        binding = UsdShade.MaterialBindingAPI(prim)
        material_rel = binding.GetDirectBindingRel("physics") if binding else None
        print(
            "[INFO]:   "
            f"{prim_path}: collision="
            f"{collision.GetCollisionEnabledAttr().Get() if collision else None}, "
            f"approx={mesh_collision.GetApproximationAttr().Get() if mesh_collision else None}, "
            f"physics_material={material_rel.GetTargets() if material_rel else []}"
        )


def _print_control_debug(
    step_count: int,
    sample,
    action: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> None:
    error = action - joint_pos
    print(f"[CONTROL DEBUG] step={step_count}")
    print(f"[CONTROL DEBUG] raw leader deg: {_format_joint_values(sample.raw_deg)}")
    print(f"[CONTROL DEBUG] normalized: {_format_joint_values(sample.normalized)}")
    print(f"[CONTROL DEBUG] target rad: {_format_joint_values(action)}")
    print(f"[CONTROL DEBUG] actual rad: {_format_joint_values(joint_pos)}")
    print(f"[CONTROL DEBUG] error rad: {_format_joint_values(error)}")
    print(f"[CONTROL DEBUG] velocity rad/s: {_format_joint_values(joint_vel)}")


def _disable_camera_pipeline(env_cfg) -> None:
    """Keep control-only runs from paying the camera sensor/render cost."""
    env_cfg.scene.env_cam = None
    env_cfg.scene.gripper_cam = None
    env_cfg.events.startup_refresh_camera_xforms = None
    env_cfg.events.startup_sync_env_camera = None
    env_cfg.events.refresh_camera_xforms = None
    env_cfg.events.sync_env_camera = None
    env_cfg.events.reset_sync_env_camera = None
    env_cfg.events.reset_sync_gripper_camera = None
    env_cfg.events.sync_gripper_camera = None

    env_cfg.observations.visual.rgb_env = None
    env_cfg.observations.visual.rgb_gripper = None


def _enable_domain_randomization(env_cfg) -> None:
    env_cfg.events.reset_robot_position.params["position_range"] = (-0.04, 0.04)
    env_cfg.events.randomize_banana_pose = None
    env_cfg.events.randomize_banana_pose_dr = EventTerm(
        func=randomize_banana_on_table_dr,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("banana"),
        },
    )
    env_cfg.events.randomize_robot_appearance = None
    env_cfg.events.randomize_robot_appearance_dr = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={
            "color_names": ["orange", "teal", "black"],
        },
    )
    env_cfg.events.randomize_scene_visuals_dr = EventTerm(
        func=randomize_scene_visuals,
        mode="reset",
    )


def _disable_domain_randomization(env_cfg) -> None:
    env_cfg.events.reset_robot_position.params["position_range"] = (0.0, 0.0)
    env_cfg.events.randomize_banana_pose_dr = None
    env_cfg.events.randomize_robot_appearance_dr = None
    env_cfg.events.randomize_scene_visuals_dr = None


def _warmup_render(steps: int = 12) -> None:
    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        return
    for _ in range(steps):
        sim.render()


def _force_camera_refresh(env, cycles: int = 3) -> None:
    sim = sim_utils.SimulationContext.instance()
    for _ in range(cycles):
        refresh_config_cameras(env.unwrapped, None, ["env_cam"])
        sync_env_camera_to_world(env.unwrapped, None)
        sync_gripper_camera_to_frame(env.unwrapped, None)

        if sim is not None:
            sim.render()

        for name in CAMERA_SENSOR_NAMES:
            if name in env.unwrapped.scene.keys():
                env.unwrapped.scene[name].update(0.0, force_recompute=True)


def _camera_prim_path(env, camera_name: str) -> str:
    camera_cfg = getattr(env.unwrapped.scene.cfg, camera_name)
    return camera_cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")


def _rebind_viewport_cameras(env) -> None:
    try:
        from omni.kit.viewport.utility import get_active_viewport
    except Exception as exc:
        print(f"[WARN]: Could not import viewport utility for camera rebind: {exc}")
        return

    viewport = get_active_viewport()
    if viewport is None:
        return

    sim = sim_utils.SimulationContext.instance()
    previous_camera = viewport.camera_path
    for name in CAMERA_SENSOR_NAMES:
        viewport.set_active_camera(_camera_prim_path(env, name))
        if sim is not None:
            sim.render()
    if previous_camera:
        viewport.set_active_camera(previous_camera.pathString)


def _collect_cameras(env) -> dict:
    cameras = {}
    for obj in CAMERA_SENSOR_NAMES:
        if obj in env.unwrapped.scene.keys():
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.removesuffix("_cam")] = {
                "height": camera_cfg.height,
                "width": camera_cfg.width,
            }
    return cameras


def _make_env():
    env_cfg = parse_env_cfg(
        TASK_NAME,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    if not args_cli.enable_cameras:
        _disable_camera_pipeline(env_cfg)
    if args_cli.domain_randomization:
        _enable_domain_randomization(env_cfg)
    else:
        _disable_domain_randomization(env_cfg)

    env = gym.make(TASK_NAME, cfg=env_cfg)
    _print_robot_actuator_config(env)
    env.reset()
    joint_indices = _resolve_joint_indices(env)
    _print_runtime_joint_metadata(env, joint_indices)
    _print_physics_debug(env)
    if args_cli.enable_cameras:
        _force_camera_refresh(env)
        _rebind_viewport_cameras(env)
        _warmup_render()

    return env, joint_indices


def _run_viewer(env) -> None:
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    print("[INFO]: Marker pick-place environment running. Pass --teleop for leader-arm control.")
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(actions)


def _run_teleop(env, joint_indices: torch.Tensor) -> None:
    cameras = _collect_cameras(env) if args_cli.enable_cameras else {}
    if len(cameras) == 0:
        print("[INFO]: No cameras found - visual recording disabled")

    leader = SO100LeaderTeleop(
        device=env.unwrapped.device,
        config=SO100TeleopConfig(port=HARD_CODED_PORT, robot_id=HARD_CODED_ID),
    )
    leader.connect()

    recorder = TeleopDatasetRecorder(
        RecorderConfig(
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
            task_name=args_cli.task_name,
            fps=30,
        ),
        cameras=cameras,
        device=env.unwrapped.device,
    )
    hotkeys = TeleopHotkeys()

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    print("[INFO]: Teleop running with cameras/dataset recording.")
    print("[INFO]: Commands:")
    print("[INFO]:   R - start recording")
    print("[INFO]:   Y - stop and save recording")
    print("[INFO]:   T - reset env and randomize domain with Isaac Lab events")
    print("[INFO]:   Close Isaac Sim - exit teleop")
    print(f"[INFO]: Domain randomization: {'enabled' if args_cli.domain_randomization else 'disabled'}")
    if args_cli.record:
        print("[INFO]: Recording armed. Press R to start.")
    step_count = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if hotkeys.start_recording:
                    hotkeys.start_recording = False
                    if recorder.enabled:
                        print("[INFO]: Recording is already running")
                    else:
                        recorder.start()

                if hotkeys.stop_recording:
                    hotkeys.stop_recording = False
                    if recorder.enabled:
                        recorder.stop()
                    else:
                        print("[INFO]: Recording is not running")

                if hotkeys.reset_env:
                    hotkeys.reset_env = False
                    env.reset()
                    if args_cli.enable_cameras:
                        _force_camera_refresh(env)
                    actions.zero_()
                    print(
                        "[INFO]: Environment reset"
                        f" ({'domain randomization enabled' if args_cli.domain_randomization else 'clean mode'})"
                    )

                sample = leader.read_sample()
                actions[0] = sample.target_rad
                obs, _, _, _, _ = env.step(actions)
                step_count += 1

                if (
                    args_cli.debug_control
                    and step_count % max(args_cli.debug_control_interval, 1) == 0
                ):
                    robot = env.unwrapped.scene["robot"]
                    joint_pos = robot.data.joint_pos[0, joint_indices]
                    joint_vel = robot.data.joint_vel[0, joint_indices]
                    _print_control_debug(
                        step_count,
                        sample,
                        actions[0],
                        joint_pos,
                        joint_vel,
                    )

                if recorder.enabled:
                    joint_pos = obs["policy"]["joint_pos_obs"][0]
                    recorder.push(
                        sample.raw_deg,
                        leader.sim_radians_to_raw_deg(joint_pos),
                        obs.get("visual", {}),
                    )
    finally:
        if recorder.enabled:
            recorder.stop()


def main() -> None:
    env, joint_indices = _make_env()
    try:
        if args_cli.teleop:
            _run_teleop(env, joint_indices)
        else:
            _run_viewer(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
