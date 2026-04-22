#!/usr/bin/env bash
# 23DOF 真机分阶段部署（dry_run -> 保守模式 -> 正常模式）

set -euo pipefail

SCRIPT_PATH="/home/eame/FallRecovery/magiclab_deploy/deploy_real_z1_23dof.py"
CONFIG_PATH="/home/eame/FallRecovery/magiclab_deploy/configs/z1_23dof.yaml"

# ============================================================
# 阶段1: dry_run — 单独 YAML 设 dry_run: true，例如：
# python3 "${SCRIPT_PATH}" --config /path/to/z1_23dof_dryrun.yaml
# dryrun yaml 建议含：dry_run, infer_hz/cmd_hz, prepare_gait, gait_settle_s,
# lowlevel_retry, state_timeout_s 等。

# ============================================================
# 阶段2: 保守实控 — 使用更小 action_scale、max_joint_delta、infer_hz/cmd_hz 的 YAML。
# python3 "${SCRIPT_PATH}" --config /path/to/conservative.yaml

# ============================================================
# 阶段3: 正常实控（默认 CONFIG_PATH 已与 z1_23dof.yaml 对齐）
python3 "${SCRIPT_PATH}" --config "${CONFIG_PATH}"
