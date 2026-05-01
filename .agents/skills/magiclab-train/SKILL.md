---
name: magiclab-train
description: Configure, launch, resume, or diagnose Isaac Lab PPO training in rc.rl.magiclab for Z1 locomotion and standup tasks.
---

Locate the rc.rl.magiclab checkout from the active workspace; this machine's checkout is `/Users/manju/Documents/GitHub/likelihood333/rc.rl.magiclab`. Work from `magiclab_rl_lab/` for training commands.

Read `scripts/rsl_rl/train.py`, `scripts/rsl_rl/cli_args.py`, and the selected task's configuration before choosing overrides. The existing entry bootstraps Isaac Sim using `MLAB_ISAACLAB_PATH` or the JSON runtime section. `MLAB_RUNTIME_READY=1` is valid only when the Isaac Sim runtime has already been sourced. Use the Python environment with Isaac Lab and this extension installed; do not infer a remote interpreter from the macOS checkout.

For existing 12DOF training:

```bash
python scripts/rsl_rl/train.py --task Magiclab-Z1-12dof-Velocity --headless --device cuda:0 --num_envs 2048 --max_iterations 10000 --run_name z1_velocity
```

For existing 23DOF training, read `scripts/rsl_rl/standup_train_cfg.json`, replace its runtime path for the target host, and use:

```bash
python scripts/rsl_rl/train.py --cfg /absolute/path/to/train_cfg.json --task Magiclab-Z1-23dof-Standup-Ground --headless --device cuda:0 --num_envs 2048
```

Use `--resume --load_run <run> --checkpoint <file>` for existing-backend restoration. Logs are under `logs/rsl_rl/<experiment>/<run>/`; inspect actual log growth and checkpoint files when reporting progress. Do not report performance from config settings or filenames.

The vendored PPO entry, when requested, is documented in `docs/rl/vendor-training.md`; it has its own config and resume flags. Read that document before launching. Honor instructions to write or inspect only: a training request's scope determines whether execution is authorized.
