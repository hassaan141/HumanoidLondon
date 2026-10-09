from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from marker_pick_place.tasks.marker_env_cfg import MarkerSceneCfg, MarkerTeleopEnvCfg


@configclass
class MarkerRLSceneCfg(MarkerSceneCfg):
    """Scene extras for RL/evaluation-style task logic."""

    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Banana",
        ],
    )


@configclass
class MarkerRLEnvCfg(MarkerTeleopEnvCfg):
    """Banana environment variant for future RL/evaluation logic."""

    scene: MarkerRLSceneCfg = MarkerRLSceneCfg()
