import gymnasium as gym


gym.register(
    id="Marker-PickPlace-Teleop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marker_pick_place.tasks.marker_env_cfg:MarkerTeleopEnvCfg",
    },
)

gym.register(
    id="Marker-PickPlace-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "marker_pick_place.tasks.marker_rl_env_cfg:MarkerRLEnvCfg",
    },
)
