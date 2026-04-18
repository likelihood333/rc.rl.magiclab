import gymnasium as gym

gym.register(
    id="Magiclab-Z1-23dof-Standup-Ground",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standup_env_cfg:Z1StandupGroundEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standup_env_cfg:Z1StandupGroundPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "magiclab_rl_lab.tasks.standingup.agents.rsl_rl_standup_ppo_cfg:StandupPPORunnerCfg",
    },
)

gym.register(
    id="Magiclab-Z1-23dof-Standup-Prone",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standup_env_cfg:Z1StandupProneEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standup_env_cfg:Z1StandupPronePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "magiclab_rl_lab.tasks.standingup.agents.rsl_rl_standup_ppo_cfg:StandupPPORunnerCfg",
    },
)

gym.register(
    id="Magiclab-Z1-23dof-Standup-Platform",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standup_env_cfg:Z1StandupPlatformEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standup_env_cfg:Z1StandupPlatformPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "magiclab_rl_lab.tasks.standingup.agents.rsl_rl_standup_ppo_cfg:StandupPPORunnerCfg",
    },
)

gym.register(
    id="Magiclab-Z1-23dof-Standup-Slope",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standup_env_cfg:Z1StandupSlopeEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standup_env_cfg:Z1StandupSlopePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "magiclab_rl_lab.tasks.standingup.agents.rsl_rl_standup_ppo_cfg:StandupPPORunnerCfg",
    },
)

gym.register(
    id="Magiclab-Z1-23dof-Standup-Wall",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standup_env_cfg:Z1StandupWallEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standup_env_cfg:Z1StandupWallPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "magiclab_rl_lab.tasks.standingup.agents.rsl_rl_standup_ppo_cfg:StandupPPORunnerCfg",
    },
)

