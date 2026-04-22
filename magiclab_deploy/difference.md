# `deploy_real_z1_12dof.py` 与 `deploy_real_z1_23dof.py` 差异说明

对应文件（与 `deploy_z1_base.py` 共用工具）：

| 项目 | 12DOF | 23DOF |
|------|--------|--------|
| 入口脚本 | `deploy_real_z1_12dof.py` | `deploy_real_z1_23dof.py` |
| 配置 | `configs/z1_12dof.yaml` | `configs/z1_23dof.yaml` |

---

## 1. 任务与自由度

| | 12DOF | 23DOF |
|---|--------|--------|
| 策略关节 | **仅 12 个腿关节**在观测/动作里 | **23 维**：腿 12 + 腰 1 + 臂 10（策略序；与 SDK 序有重映射） |
| 上肢 / 腰 / 头 | 腿控时 **固定为 YAML 默认角**（`send_position_commands(..., default_arm_angles, ...)`） | 腰与 10 个受控臂关节 **在策略内**，非控制臂关节用 **hold 模式**（`arm_hold_operation_mode`） |
| 步态相位 | 观测里含 **gait 历史**（正弦相位 → 左右脚支撑 `stance_mask`） | **无** gait 项；与 HoST 起身等任务一致 |

---

## 2. 观测（obs）

| | 12DOF | 23DOF |
|---|--------|--------|
| 结构 | 分量历史：`ang_vel`、`gravity`、`cmd`、`qj`、`dqj`、`actions`、`gait`，用 `update_hist` 拼成 `history_obs` | 单帧 obs 为 **拼好的向量**（角速、重力、qj、dqj、last_action、`action_rescale_obs` 标量），再对 `history_obs` 做 **滑动窗口**（`cat` 当前帧与历史） |
| 重力项 | `rpy_scale`（`config.rpy_scale`） | `gravity_scale`（`gravity_scale`） |
| 指令 | 手柄 `cmd` 进历史 | **无** 速度指令项（与训练 HoST 设定一致） |
| 腿序 | 与 SDK 一致使用 | **显式重映射**：`leg_sdk_to_policy` / `leg_policy_to_sdk` |
| 状态来源 | 各回调写到成员，再 `get_joint_*` / IMU 组装 | `RobotStateBuffer` 异步写、主线程 `snapshot()` |

---

## 3. 动作语义与下发

| | 12DOF | 23DOF |
|---|--------|--------|
| 网络输出含义 | 经 `action_scale` 后作为 **相对默认腿姿的位移**，**目标腿角** = `action * action_scale + default_leg_angles` | **有界动作**（通常 tanh）经 `clip(-1,1) * action_scale` 为 **相对当前关节角的增量**：`target = q + delta`，再可做 `max_joint_delta` 限幅 |
| 关节命令 | 聚合为一次 `send_position_commands`（腿 + 固定臂腰头） | **分腿 / 腰 / 臂** `publish_*`，`make_joint_command`，`kps`/`kds` 与 `operation_mode` 可配 |

---

## 4. 策略加载（`load_policy`）

| | 12DOF | 23DOF |
|---|--------|--------|
| `policy_format` | 默认 **`auto`**：先试 JIT，再 checkpoint | **仅 `checkpoint`**（`load_23dof_settings` 内强制），需与训练 `torch.save` 的 `model_state_dict` 一致 |
| `actor_activation` | 仅在 **checkpoint 重建** MLP 时用 | 同上 |
| 设备 | `cuda` 若可用，否则 `cpu` | **YAML `device`**：`cpu` / `cuda` 显式指定 |

---

## 5. 频率与主循环

| | 12DOF | 23DOF |
|---|--------|--------|
| 主循环节拍 | 外层 `interval = 0.002` s（约 **500Hz** 调 `run_active_control`） | 单循环内双时钟：`next_infer_time` / `next_cmd_time` |
| 推理与发令 | `control_decimation`：每 N 次循环做一次观测+推理+发送（`simulation_dt * N` → 如 **50Hz**） | **`infer_hz`** 推理、**`cmd_hz`** 发令，可 **不同**；中间帧复用 `cached_action` |
| 二者关系 | 推理与发令 **同一步** 完成 | 推理与发令 **解耦** |

---

## 6. 上电 / 进低层流程

| | 12DOF | 23DOF |
|---|--------|--------|
| 流程 | `connect` → `switch_to_low_level` → 订阅读取回调 → 等传感器 | 连接后 **故障检查** → **可反复** `HighLevel` 重置 + **gait**（`RECOVERY_STAND` 或回退 `PASSIVE`）+ `switch_to_low_level` 重试 → 订阅 → 可选 **暖机零力矩 publish** → `_wait_state_ready` |
| 交互 | **手柄状态机**：零力矩 → 回默认位 → 默认位确认 → 主动控（`gamepad` 线程） | **无** 该套状态机；`main` 为 `initialize_robot()` → `run()` |
| 安全退出 | `shutdown` 阻尼/断开等 | `cleanup`：`release_on_exit` 时零力矩释放序列，否则 **safe-stop**（保持当前位弱增益发一段时间）等 |

---

## 7. 配置方式

| | 12DOF | 23DOF |
|---|--------|--------|
| 加载 | `Config(config_path)`，属性在 `config_z1.Config` 上 | `load_23dof_settings` → 冻结的 **`Z1Deploy23dofSettings`**，字段校验更严 |
| `--config` | 默认 `g1.yaml`，相对 `configs/` | 默认用 **`Z1_23DOF_CONFIG` 或脚本旁 `z1_23dof.yaml`**，支持绝对路径 |
| 其它 | 见 `z1_12dof.yaml`（`leg_mode`、`upper_*` 等） | 见 `z1_23dof.yaml`（`infer_hz`、`cmd_hz`、`dry_run`、全链 `kps`/`kds` 等） |

---

## 8. 小结

- **12DOF**：偏 **双足步态 + 手柄引导**，**只控腿 12 维**，**观测里含 gait 与速度指令**；**绝对目标角** 控制腿，腰臂头用默认；**单频**（由 `simulation_dt` 与 `control_decimation` 决定）。  
- **23DOF**：**全身 23 维**（HoST/起身类），**增量式** 目标、**分频** 推理/发令、**强约束** 切低层前 gait 与**故障**、**无** 游戏手柄状态机；策略加载固定 **checkpoint**。

共用逻辑见 `deploy_z1_base.py`（如 `get_gravity_orientation`、`load_policy`、部分连接与低层切换辅助等）。
