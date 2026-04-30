# RSL-RL 移植范围

上游：[leggedrobotics/rsl_rl v2.2.4](https://github.com/leggedrobotics/rsl_rl/tree/v2.2.4)。
固定提交：`f80d4750fbdfb62cfdb0c362b7063450f427cf35`。

首批一次性引入整个 `rsl_rl/` 包：18 个 Python 文件，加 LICENSE 和来源说明，共 20 个文件。包含 PPO、rollout storage、MLP / recurrent actor-critic、normalizer、runner 与日志工具。没有复制上游训练环境、权重或依赖二进制。

本地将包命名空间隔离，避免覆盖现有安装的 `rsl_rl`。现有 `scripts/rsl_rl/train.py` 继续使用环境里安装的后端；新入口显式选择引入的实现。

适配层负责：

- 将 Isaac Lab 的五元组 step 输出转为 runner 的四元组；保留 `critic` 观测与 timeout。
- 从实际观测 tensor 获取输入尺寸，不把 `76 * 6` 写死到算法里。
- 维护 12DOF 速度跟踪与 23DOF 起身配置，保留已有任务注册。
- 保存环境、训练参数和上游版本，再支持恢复训练与导出 actor。

限制：v2.2.4 后端先支持单进程、MLP PPO、TensorBoard。本地适配不启用 recurrent、RND、symmetry 或多 GPU。上游源码保留这些模块，但入口不承诺它们已适配。

检查重点：[runner](https://github.com/leggedrobotics/rsl_rl/blob/v2.2.4/rsl_rl/runners/on_policy_runner.py)、[PPO](https://github.com/leggedrobotics/rsl_rl/blob/v2.2.4/rsl_rl/algorithms/ppo.py)、[ActorCritic](https://github.com/leggedrobotics/rsl_rl/blob/v2.2.4/rsl_rl/modules/actor_critic.py)。上游 actor 最后一层是线性层，部署不能自动增加 tanh。
