# MagicBot Z1 真机部署（Ubuntu 22.04）

> **Author: Mingzhu Li**  
> **Last Modified: 2026-04-21 01:42:13 CST**

本目录提供一版基于 `magicbot-z1_sdk` 的真机部署脚本，用于运行模型：

- `model/host_gym_model_12000.pt`

脚本文件：

- `deploy_host_policy_real.py`

## 1. 环境要求

- Ubuntu 22.04（推荐）
- Python 3.10
- 已编译并可导入 `magicbot_z1_python`
  - `python3 -c "import magicbot_z1_python as m; print('ok', m.get_robot_model())"`
- 网络与系统参数按 SDK README 配置：
  - `rtprio`（`/etc/security/limits.conf`）
  - `net.core.rmem/wmem`（`/etc/sysctl.conf`）
  - 与机器人同网段（推荐 `192.168.54.X`，开发机可用 `192.168.54.111`）
  - 组播路由配置：
    - `sudo ifconfig <nic> multicast`
    - `sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev <nic>`

## 2. 运行方式

先确保 SDK Python binding 可用（示例，按你本机安装路径改）：

```bash
export PYTHONPATH=/opt/magic_robotics/magicbot_z1_sdk/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/magic_robotics/magicbot_z1_sdk/lib:$LD_LIBRARY_PATH
```

再运行部署脚本：

```bash
cd /home/ubuntu/rc.rl.magiclab/sim2real
python3 deploy_host_policy_real.py \
  --model_path /home/ubuntu/rc.rl.magiclab/sim2real/model/host_gym_model_12000.pt \
  --local_ip 192.168.54.111 \
  --device cpu \
  --control_hz 100
```

## 3. 脚本做了什么

- 连接机器人并切换到低层控制。
- 订阅 `leg/waist/arm/imu` 状态。
- 按训练侧 `host_ground_v1` 构造观测（76 维，历史 6 帧）。
- 加载 `.pt` 策略（自动兼容 TorchScript / checkpoint）。
- 推理输出 23 维动作并下发到：
  - 腿：12 关节
  - 腰：1 关节
  - 手臂：10 关节（通过 `--arm_control_indices` 映射到 SDK 的 14 关节序）

## 4. 关键可调参数

- `--arm_control_indices`：策略 10 维手臂动作映射到 SDK `arm_state` 的索引（默认 `0,1,2,3,4,7,8,9,10,11`）。
- `--leg_operation_mode/--waist_operation_mode/--arm_operation_mode`：默认 `4`（位置控制常用模式）。
- `--arm_hold_operation_mode`：非策略控制关节默认 `200`（准备态）。
- `--max_joint_delta`：单周期最大关节变化（默认 `0.25 rad`）。
- `--dry_run`：只推理不下发命令。

## 5. 安全建议

- 首次上机请先 `--dry_run` 验证观测和动作维度是否正确。
- 真实机器人请配合吊具/急停，先低增益、低频率、小动作范围测试。
- 若关节顺序与本脚本默认不一致，先修正 `--arm_control_indices` 再执行实控。

## 6. sim2real 可能的 gap（当前版本）

以下是当前脚本与训练环境之间，仍可能存在的差异点：

- **控制接口差异**：训练中是 `action -> PD torque -> torque clip`，真机侧当前是下发位置目标 + `kp/kd`（`operation_mode=4`），底层执行链路并不完全等价。
- **执行器/通信延迟差异**：训练里有 domain randomization（含 delay），真机实际网络抖动与控制周期抖动更复杂，可能带来动作滞后或相位误差。
- **关节顺序映射风险**：策略动作顺序基于训练侧 DOF 顺序；真机脚本目前仍依赖 SDK 回调数组顺序与 `--arm_control_indices`，若机型/固件变更可能错位。
- **传感器噪声与偏置**：训练中噪声模型是仿真噪声，真机 IMU/关节编码器偏置、温漂与饱和行为不一定一致。
- **接触与碰撞模型差异**：仿真地面摩擦、碰撞几何、脚踝接触细节与真实机器人有偏差，易造成起身阶段轨迹偏离。
- **动作尺度差异**：训练中的 `action_rescale` 存在随机扰动；真机当前使用固定值，可能导致动作保守或过激。

建议优先做以下对齐以降低 gap：

- 显式 joint-name 映射（不要依赖回调天然顺序）。
- 增加 IMU/关节低通滤波与异常值保护。
- 加入启动分阶段策略（`dry_run -> 小增益小幅度 -> 全量`）。
- 对 `kp/kd/action_scale/control_hz` 做真机网格调参并记录稳定区间。
