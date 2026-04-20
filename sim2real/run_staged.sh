#!/usr/bin/env bash
# ===================== Metadata =====================
# Author: Mingzhu Li
# Last Modified: 2026-04-21 01:42:13 CST
# ====================================================

# 分阶段真机部署脚本（dry_run -> 小增益小幅度 -> 全量）
# 使用前请确认：
# 1) 已正确配置 magicbot_z1_python 的 PYTHONPATH / LD_LIBRARY_PATH
# 2) 机器人已上电、网络可达、急停可用、具备吊具保护

set -euo pipefail

# ===== 按需修改的基础参数 =====
MODEL_PATH="/home/ubuntu/rc.rl.magiclab/sim2real/model/host_gym_model_12000.pt"
SCRIPT_PATH="/home/ubuntu/rc.rl.magiclab/sim2real/deploy_real.py"
LOCAL_IP="192.168.54.119" # 使用端侧小电脑

# 阶段2（保守模式）使用的小增益（约为默认值的 50%）
KP_SOFT=(
  75 75 75 100 20 20
  75 75 75 100 20 20
  50
  50 50 50 50 50
  50 50 50 50 50
)
KD_SOFT=(
  2 2 2 3 1 1
  2 2 2 3 1 1
  2
  2 2 2 2 2
  2 2 2 2 2
)

# ============================================================
# 命令1：dry_run（仅推理，不下发任何关节命令）
# 目的：确认观测维度、模型推理、频率、状态回调都正常
python3 "${SCRIPT_PATH}" \
  --model_path "${MODEL_PATH}" \
  --local_ip "${LOCAL_IP}" \
  --device cpu \
  --control_hz 100 \
  --dry_run

# # 命令2：小增益小幅度实控（保守模式）
# # 目的：低风险验证关节映射和动作方向是否正确
# python3 "${SCRIPT_PATH}" \
#   --model_path "${MODEL_PATH}" \
#   --local_ip "${LOCAL_IP}" \
#   --device cpu \
#   --control_hz 50 \
#   --action_scale 0.10 \
#   --max_joint_delta 0.05 \
#   --kps "${KP_SOFT[@]}" \
#   --kds "${KD_SOFT[@]}"

# # 命令3：全量参数实控（接近训练设置）
# # 目的：在前两阶段稳定后恢复目标性能
# python3 "${SCRIPT_PATH}" \
#   --model_path "${MODEL_PATH}" \
#   --local_ip "${LOCAL_IP}" \
#   --device cpu \
#   --control_hz 100 \
#   --action_scale 0.40 \
#   --max_joint_delta 0.25

