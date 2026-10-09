import os
from dataclasses import dataclass

import torch

from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from lerobot.robots import make_robot_from_config


@dataclass(frozen=True)
class SO100TeleopConfig:
    port: str = "/dev/ttyACM0"
    robot_id: str = "leader_arm_1"


@dataclass(frozen=True)
class SO100LeaderSample:
    raw_deg: torch.Tensor
    normalized: torch.Tensor
    target_rad: torch.Tensor


class SO100LeaderTeleop:
    """Minimal leader teleop interface for SO100.

    Reads joint targets from the real leader and maps them to sim joint radians.
    """

    # Joint order matches the USD articulation joint order.
    JOINT_ORDER = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    # Joint limits in degrees from the local SO100 USD/URDF.
    JOINT_LIMITS_DEG = {
        "shoulder_pan": (-114.5916, 114.5916),
        "shoulder_lift": (0.0, 200.5352),
        "elbow_flex": (-179.9993, 0.0),
        "wrist_flex": (-143.2394, 68.7549),
        "wrist_roll": (-179.9993, 179.9993),
        "gripper": (-11.4592, 114.5916),
    }

    # The local USD shoulder-pan joint axis is opposite the LeRobot leader convention.
    JOINT_TARGET_SIGNS = {
        "shoulder_pan": -1.0,
        "shoulder_lift": 1.0,
        "elbow_flex": 1.0,
        "wrist_flex": 1.0,
        "wrist_roll": 1.0,
        "gripper": 1.0,
    }

    def __init__(self, device: str, config: SO100TeleopConfig | None = None) -> None:
        self.device = device
        self.config = config or SO100TeleopConfig(
            port=os.getenv("TELEOP_PORT", SO100TeleopConfig.port),
            robot_id=os.getenv("TELEOP_ID", SO100TeleopConfig.robot_id),
        )
        self._robot = None

        self._joint_names = [name.split(".")[0] for name in self.JOINT_ORDER]
        self._joint_mins = torch.tensor(
            [self.JOINT_LIMITS_DEG[name][0] for name in self._joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self._joint_maxs = torch.tensor(
            [self.JOINT_LIMITS_DEG[name][1] for name in self._joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self._joint_target_signs = torch.tensor(
            [self.JOINT_TARGET_SIGNS[name] for name in self._joint_names],
            dtype=torch.float32,
            device=self.device,
        )

    def connect(self) -> None:
        # LeRobot exposes this hardware interface under the SO101 leader name.
        cfg = SO101LeaderConfig(port=self.config.port, id=self.config.robot_id)
        self._robot = make_robot_from_config(cfg)
        self._robot.connect()
        print(
            f"[INFO]: Connected to leader on {self.config.port} (id={self.config.robot_id})"
        )

    def _normalize_leader(self, raw_values: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(raw_values)
        normalized[:-1] = (raw_values[:-1] + 100.0) / 200.0
        normalized[-1] = raw_values[-1] / 100.0
        return normalized

    def _map_to_limits(self, normalized: torch.Tensor) -> torch.Tensor:
        mapped_deg = self._joint_mins + normalized * (self._joint_maxs - self._joint_mins)
        mapped_rad = mapped_deg * torch.pi / 180.0
        return mapped_rad * self._joint_target_signs

    def sim_radians_to_raw_deg(self, joint_pos_rad: torch.Tensor) -> torch.Tensor:
        mapped_deg = joint_pos_rad / self._joint_target_signs * 180.0 / torch.pi
        normalized = (mapped_deg - self._joint_mins) / (self._joint_maxs - self._joint_mins)
        raw_deg = torch.zeros_like(normalized)
        raw_deg[:-1] = normalized[:-1] * 200.0 - 100.0
        raw_deg[-1] = normalized[-1] * 100.0
        return raw_deg

    def read_sample(self) -> SO100LeaderSample:
        if self._robot is None:
            raise RuntimeError("Leader robot not connected. Call connect() first.")
        raw_action = self._robot.get_action()
        raw_tensor = torch.tensor(
            [raw_action[joint] for joint in self.JOINT_ORDER],
            dtype=torch.float32,
            device=self.device,
        )
        normalized = self._normalize_leader(raw_tensor)
        target = self._map_to_limits(normalized)
        return SO100LeaderSample(
            raw_deg=raw_tensor,
            normalized=normalized,
            target_rad=target,
        )

    def read_action(self) -> torch.Tensor:
        return self.read_sample().target_rad
