---
name: magiclab-real-deploy
description: Prepare Magicbot SDK configuration and staged Z1 real-robot deployment in rc.rl.magiclab, including dry-run diagnosis.
---

Work from the checkout's `magiclab_deploy/`. Read the chosen entry point, YAML, and `README.md`. This is a Linux SDK workflow; the macOS checkout is suitable for edits, not evidence of hardware connectivity. Determine the actual SDK installation directory on the deployment host before exporting `PYTHONPATH` and `LD_LIBRARY_PATH`; use `LD_PRELOAD` only when that host needs the documented libstdc++ workaround.

Entrypoints:

```bash
python deploy_real_z1_12dof.py --config z1_12dof.yaml
python deploy_real_z1_23dof.py --config /absolute/path/to/z1_23dof_dryrun.yaml
```

`Config` resolves `policy_path` relative to the YAML directory and `policy_root`. Copying a YAML elsewhere changes that base: use an absolute policy root in host-specific configs. Check policy format, action/observation dimensions, joint maps, gains, action scale, and infer/publish rates.

Prepare a separate 23DOF YAML with `dry_run: true`, then a conservative profile and the final profile. Read the initialization flow: dry-run suppresses policy command publishing but still connects, changes controller level, and may send preparation/exit commands. Treat dry-run as hardware interaction. `run_staged_23dof.sh` currently runs its final stage directly and contains host-specific paths; do not use it as an automatic staged validator.

Code/config preparation does not authorize connecting to or commanding a robot. For an explicitly authorized live deployment, confirm the selected host, robot and profile against the user's instruction, verify emergency stop and physical support arrangements, inspect active faults, then perform the requested stage. On faults, stale state, non-finite actions, or interrupted control, stop and follow the entry point's safe-stop/release behavior rather than retrying motion automatically.

Read `docs/rl/simulation-contract.md` before replacing HoST weights with new standup weights. Dimension equality does not resolve progress-term or history-order differences. Report exactly which stages ran and what the robot/logs showed.
