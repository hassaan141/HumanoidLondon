from dataclasses import dataclass
from pathlib import Path

import torch


JOINT_FEATURE_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


@dataclass
class RecorderConfig:
    repo_id: str = "local/banana_pick"
    dataset_root: str = "datasets/banana_pick_lerobot"
    task_name: str = "pick up the banana and place it in the cup"
    fps: int = 30
    robot_type: str = "so101_follower"


class TeleopDatasetRecorder:
    """LeRobot-backed recorder for teleop episodes."""

    def __init__(self, config: RecorderConfig, cameras: dict, device: str) -> None:
        self.config = config
        self.cameras = cameras
        self.device = device
        self._enabled = False
        self._dataset = None

    def _features(self) -> dict:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": JOINT_FEATURE_NAMES,
                "fps": self.config.fps,
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": JOINT_FEATURE_NAMES,
                "fps": self.config.fps,
            },
        }
        for camera_name, camera in self.cameras.items():
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": (camera["height"], camera["width"], 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _init_dataset(self) -> None:
        if self._dataset is not None:
            return

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset_root = Path(self.config.dataset_root)
        metadata_paths = [
            dataset_root / "meta" / "tasks.parquet",
            dataset_root / "meta" / "info.json",
        ]
        if dataset_root.exists() and any(path.exists() for path in metadata_paths):
            self._dataset = LeRobotDataset(self.config.repo_id, root=self.config.dataset_root)
            print(f"[INFO]: Existing LeRobot dataset initialized: {self.config.dataset_root}")
            return
        if dataset_root.exists() and any(dataset_root.iterdir()):
            raise RuntimeError(
                f"Dataset root exists but is not an initialized LeRobot dataset: {self.config.dataset_root}. "
                "Remove it or choose a fresh --repo-root."
            )

        self._dataset = LeRobotDataset.create(
            self.config.repo_id,
            fps=self.config.fps,
            features=self._features(),
            root=self.config.dataset_root,
            robot_type=self.config.robot_type,
        )
        print(f"[INFO]: New LeRobot dataset initialized: {self.config.dataset_root}")

    def start(self) -> None:
        self._init_dataset()
        self._enabled = True
        print(f"[INFO]: Started LeRobot episode recording to {self.config.dataset_root}")

    def stop(self) -> None:
        if not self._enabled:
            return
        self._dataset.save_episode()
        self._dataset.finalize()
        self._dataset = None
        self._enabled = False
        print(f"[INFO]: LeRobot episode saved to {self.config.dataset_root}")

    def push(self, action: torch.Tensor, joint_pos: torch.Tensor, visual_obs: dict) -> None:
        if not self._enabled:
            return

        frame = {
            "action": action.detach().cpu().numpy().astype("float32"),
            "observation.state": joint_pos.detach().cpu().numpy().astype("float32"),
            "task": self.config.task_name,
        }
        for camera_name in self.cameras.keys():
            rgb_key = f"rgb_{camera_name}"
            if rgb_key in visual_obs:
                frame[f"observation.images.{camera_name}"] = (
                    visual_obs[rgb_key][0].detach().cpu().to(torch.uint8).numpy()
                )
        self._dataset.add_frame(frame)

    @property
    def enabled(self) -> bool:
        return self._enabled
