# GitHub RL 实现调研

目标是给 Z1 的 Isaac Lab 训练增加一个可独立维护的 PPO 后端，保留现有 RSL-RL 入口。

| 实现 | 适合借鉴的部分 | 本仓库接入成本 |
| --- | --- | --- |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | GPU 并行 rollout、非对称 actor/critic、PPO、超时 bootstrap | 最低；现有训练已经采用相同任务组织 |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | [连续动作 PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) 的更新流程、指标与 seed 管理 | 中；单文件入口要拆出环境接口，Isaac Gym 脚本不能直接当 Isaac Lab 入口 |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | [PPO](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/ppo)、callback、保存恢复 | 较高；需要连接 SB3 VecEnv 与当前 GPU tensor 环境 |

选择 RSL-RL 的一个固定版本作为首个移植对象。这里的“借鉴”包括保留上游源码、注明来源并在本地写适配层；不将第三方实现作为原创算法。

CleanRL 的 [PPO 文档](https://docs.cleanrl.dev/rl-algorithms/ppo/) 可用于核对 GAE、mini-batch 更新和 KL 指标；SB3 的 [PPO 文档](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) 可用于核对接口约定。

许可证应按实际下载版本核对。当前查阅的 CleanRL 与 SB3 LICENSE 为 MIT；本次选定的 RSL-RL v2.2.4 为 BSD-3-Clause。

本轮仅查阅和静态适配，没有训练曲线或性能结论。
