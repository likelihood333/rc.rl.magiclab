# HoST 迁移改动说明（23DoF 版本）

本文档记录本次基于 `Z1 23dof` 的迁移落地改动，以及每个文件承担的功能。

## 一、改动原则

- 以“最小侵入”为主：优先新增文件，不改已有 locomotion 任务逻辑。
- 若必须改原文件，已在代码中用注释标识：`[HoST migration]`。

## 二、原文件改动（仅 1 个）

### 1) `source/magiclab_rl_lab/magiclab_rl_lab/assets/robots/magiclab.py`

- 改动类型：**修改原文件**（已加注释标识）。
- 新增内容：`MAGICLAB_Z1_23DOF_CFG`。
- 作用：
  - 提供 23DoF 机器人资产配置入口；
  - 使用 `MagicBotZ1_23dof.urdf`；
  - 补齐腰/手臂执行器分组，供 stand-up 任务调用。

## 三、新增任务包（standingup）

### 2) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/__init__.py`

- 作用：standingup 任务包入口，导入 robots 子包完成任务注册链路。

### 3) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/agents/rsl_rl_standup_ppo_cfg.py`

- 作用：独立 PPO 配置 `StandupPPORunnerCfg`（不污染原 velocity PPO 配置）。

### 4) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/mdp/__init__.py`

- 作用：聚合导出 standingup 自定义 MDP 项（reward/event/obs/curriculum）。

### 5) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/mdp/rewards_standup.py`

- 作用：实现 stand-up 关键奖励/终止辅助函数：
  - `upright_reward`
  - `head_height_reward`
  - `joint_power_penalty`
  - `action_smoothness_penalty`
  - `head_below_minimum`

### 6) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/mdp/events_standup.py`

- 作用：实现跌倒姿态初始化事件 `reset_root_pose_fallen`，支持按 roll/pitch/yaw 范围随机重置。

### 7) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/mdp/observations_standup.py`

- 作用：新增 episode 进度观测 `normalized_episode_progress`，辅助策略阶段感知。

### 8) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/mdp/curriculums_standup.py`

- 作用：提供 stand-up 成功信号 `standup_success_rate`，用于课程调参和监控。

## 四、新增 23DoF 任务注册与环境配置

### 9) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/robots/__init__.py`

- 作用：robots 聚合入口。

### 10) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/robots/z1/__init__.py`

- 作用：显式导入 `23dof` 子包并触发注册。

### 11) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/robots/z1/23dof/__init__.py`

- 作用：注册 5 个 stand-up 任务 ID：
  - `Magiclab-Z1-23dof-Standup-Ground`
  - `Magiclab-Z1-23dof-Standup-Prone`
  - `Magiclab-Z1-23dof-Standup-Platform`
  - `Magiclab-Z1-23dof-Standup-Slope`
  - `Magiclab-Z1-23dof-Standup-Wall`

### 12) `source/magiclab_rl_lab/magiclab_rl_lab/tasks/standingup/robots/z1/23dof/standup_env_cfg.py`

- 作用：standingup 环境主配置文件（核心）：
  - scene / actions / observations / rewards / events / terminations / curriculum；
  - 使用 `MAGICLAB_Z1_23DOF_CFG`；
  - 提供 ground/prone/platform/slope/wall 的训练/play cfg 类。
- 说明：
  - 当前 `platform/slope/wall` 为第一版占位配置（先保证迁移链路和 task 结构完整）；
  - 后续可逐步替换为更接近 HoST 的专用地形与障碍建模。

## 五、新增评估脚本

### 13) `scripts/rsl_rl/eval_standup.py`

- 作用：新增 stand-up 指标评估脚本，支持 checkpoint 推理并输出：
  - `success_rate`
  - `avg_energy`
  - `avg_smoothness`
  - `avg_feet_movement`
- 设计目标：对齐 HoST 评估思路，同时保持与当前 `rsl_rl` 推理流程兼容。

## 六、建议启动方式

- 训练（ground）：
  - `python scripts/rsl_rl/train.py --headless --task Magiclab-Z1-23dof-Standup-Ground`
- 回放（ground）：
  - `python scripts/rsl_rl/play.py --task Magiclab-Z1-23dof-Standup-Ground --checkpoint <path>`
- 评估（ground）：
  - `python scripts/rsl_rl/eval_standup.py --task Magiclab-Z1-23dof-Standup-Ground --checkpoint <path> --num_episodes 10`

