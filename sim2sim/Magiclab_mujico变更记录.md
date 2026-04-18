# MuJoCo sim2sim 23dof 变更记录

## 说明
- 原有 `12dof` 文件保持原样：
  - `deploy_mujoco/z1_12dof_prop_mujoco.py`
  - `deploy_mujoco/configs/z1_12dof.yaml`
  - `README.md`
- 新能力全部通过**新增文件**提供，不覆盖旧流程。

## 新增文件
- `deploy_mujoco/z1_23dof_host_sim2sim.py`
- `deploy_mujoco/configs/z1_23dof_host.yaml`

## FallRecovery-23dof sim2sim 功能
1. 支持加载两类模型文件：
   - TorchScript (`torch.jit.load`)
   - rsl_rl checkpoint (`torch.load` + 自动重建 actor)
2. 观测结构对齐 `host_z1 host_ground`：
   - one-step obs: `ang_vel(3) + gravity(3) + q(23) + dq(23) + action(23) + action_rescale(1) = 76`
   - history: `his_obs=6`
3. 新增执行器映射配置：
   - `controlled_actuator_indices`
   - 将策略输出 `23d policy` 映射到 `z1_full_dof.xml`（24dof），跳过不需要的执行器（13号 head actuator）
4. 控制模式：
   - `relative_current_pos`（target = 当前关节角 q + action * action_scale，和 host 训练侧一致）
   - `default_angles`（target = 默认站立角 default_angles + action * action_scale）
5. 无图形界面运行：
   - 新增 `--headless` 参数，服务器无 `DISPLAY` 时可直接运行。
   - 支持 `--max_steps` 控制无头模式步数（`0` 表示一直跑）

## 默认配置假设（请确认）
- `z1_full_dof.xml` 通常有 24 个执行器，默认跳过索引 `13`（head）。
- 采用 23 维 action，对应索引：
  `[0..12, 14..23]`

## 其他
1. **模型文件位置**
   - `magiclab_mujoco/model/z1_dof/host_gym_model_12000.pt`
2. **动作维度与顺序**
   - 必须确认训练时 action 顺序与 `controlled_actuator_indices` 一致。
   - 若顺序不同，需要重排该索引。
3. **观测缩放和历史长度**
   - 请确认训练配置是否为：
   - `num_obs=76`, `his_obs=6`, `ang_vel_scale=0.25`, `dof_vel_scale=0.05`, `action_scale=0.5`
4. **PD 参数与默认姿态**
   - 已按 `z1_config_ground.py` 对齐 `kps/kds/default_angles`（按 MuJoCo actuator 顺序映射）。
5. **初始化与 warmup**
   - 已对齐 `init_state.pos/rot/target_joint_angles/default_joint_angles`。
   - 新增 `unactuated_policy_steps=30`，前 30 个 policy step 动作置零并观测置零，行为贴近训练环境。
6. **可切换初始化模式（本次实现 prone）**
   - `z1_23dof_host.yaml` 新增 `init_mode + init_presets`。
   - 已新增 `supine`（正面仰躺）preset，并将 `init_mode` 切换为 `supine`。
   - 如需切回趴地，改回 `init_mode: prone` 即可。
7. **周期推扰动 + 周期重置（便于可视化验证）**
   - 推扰动新增时间基准：`push_time_basis: wall_time | sim_time`（默认 `wall_time`，和肉眼观察时间一致）。
   - 推扰动配置：`enable_periodic_push / push_interval_s / push_start_after_s / push_velocity_magnitude / push_direction / push_random_yaw`。
   - 新增周期重置到初始化 preset（如 `supine`）：`enable_periodic_reset / reset_interval_s / reset_start_after_s / reset_time_basis`。
   - 当前默认：每 10 秒（墙钟）推一次，每 20 秒（墙钟）重置到 `init_mode`，便于观察“推倒-重置-再推倒”循环。
8. **关键稳定性修复（对齐 host_ground 力矩裁剪）**
   - 发现 `z1_full_dof.xml` 的 actuator 未设置 `forcerange`，MuJoCo 侧若不手动裁剪会出现过大力矩、四肢发散、QACC 不稳定。
   - `z1_23dof_host_sim2sim.py` 新增 `torque_limits` 裁剪：`tau = clip(tau, -torque_limits, torque_limits)`，行为对齐 `host_ground.py`。
   - 配置 `z1_23dof_host.yaml` 新增 23 维 `torque_limits`（按 URDF effort 顺序映射）。
   - 为做“和 gym play 一致”的基线对比，当前默认关闭 `enable_periodic_push` 与 `enable_periodic_reset`（需要时再打开）。
9. **初始“悬空”定位与修复**
   - 用 MuJoCo 前向检查确认：`supine` 旧参数 `init_base_pos.z=0.24` 时初始接触数 `ncon=0`，机器人并未贴地。
   - 已将 `supine.init_base_pos` 调整为 `0.12`，确保初始就有地面接触（避免“飘在空中”观感）。
10. **“越飞越高/远”二次修复**
   - 现象定位：MuJoCo 在复杂 mesh 接触下对步长敏感，且初始化过低会产生接触反弹能量，导致弹飞。
   - 配置调整：
     - `simulation_dt: 0.002`，`control_decimation: 10`（保持控制周期约 50Hz，同时提升稳定性）。
     - `action_scale: 0.4`（降低策略输出激进度）。
     - `supine.init_base_pos.z: 0.128`（避免 `0.12` 的深接触反弹）。
   - 运行保护：
     - 新增根速度裁剪 `root_lin_vel_clip / root_ang_vel_clip`。
     - 新增越界安全重置 `enable_safety_reset`（高度/水平距离超限自动回到 `init_mode`）。

## 运行方式
```bash
python deploy_mujoco/z1_23dof_host_sim2sim.py --config_file deploy_mujoco/configs/z1_23dof_host.yaml
```

无图形界面（推荐服务器上使用）：
```bash
python deploy_mujoco/z1_23dof_host_sim2sim.py --config_file deploy_mujoco/configs/z1_23dof_host.yaml --headless --max_steps 5000
```

## 备注
- 若启动后报 `checkpoint input/output mismatch`，说明 `num_obs*his_obs` 或 `num_actions` 与训练模型不一致，请优先核对训练配置。
