#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CFG_FILE="${MLAB_TRAIN_CFG:-scripts/rsl_rl/standup_train_cfg.json}"
ARGS=("$@")
if [[ ! " ${ARGS[*]} " =~ " --task " ]]; then
    ARGS+=("--task" "Magiclab-Z1-23dof-Standup-Ground")
fi
if [[ ! " ${ARGS[*]} " =~ " --headless " ]]; then
    ARGS+=("--headless")
fi

python scripts/rsl_rl/train.py --cfg "$CFG_FILE" "${ARGS[@]}"
