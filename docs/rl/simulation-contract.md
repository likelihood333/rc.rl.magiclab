# 仿真与策略接口

| 项目 | 约定 |
| --- | --- |
| 12DOF 任务 | `Magiclab-Z1-12dof-Velocity` |
| 23DOF 任务 | `Magiclab-Z1-23dof-Standup-Ground`，另有 Prone / Platform / Slope / Wall |
| actor 输入 | 环境返回的 `policy` tensor，形状 `[num_envs, actor_dim]` |
| critic 输入 | 优先 `critic`，缺省使用 `policy`；维度可与 actor 不同 |
| step | Isaac Lab `(observations, rewards, terminated, truncated, extras)` |
| episode 结束 | `terminated | truncated`；只在无限时域任务中为 truncation 提供 bootstrap mask |
| 自动 reset | 交给 Isaac Lab；适配层不在 step 后重复 reset |
| 历史观测 | 交给 observation manager；适配层不再叠一份历史 |

23DOF 当前训练观测末项是 `normalized_episode_progress`；HoST 部署末项则为常量 `action_rescale_obs`。此外，训练 observation manager 的历史排列不能凭 `76 * 6` 推断与部署按帧拼接一致。这两个语义问题需要单独对齐，新训练权重不能直接替换现有 HoST 真机权重。

12DOF 也需要核对观测项、关节顺序、动作 offset / scale 和控制周期。输入输出尺寸一致只是接口检查，不代表 sim2sim 或真机验证完成。

新后端的初始范围是仿真训练与 actor 导出，沿用现有部署入口前先完成上述映射。
