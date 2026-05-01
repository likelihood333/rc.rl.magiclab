---
name: magiclab-sim2sim
description: Prepare and diagnose MuJoCo sim2sim policy playback for Magiclab Z1 12DOF and HoST 23DOF in rc.rl.magiclab.
---

Find rc.rl.magiclab in the active workspace; use its `sim2sim/` as working directory. Read `deploy_mujoco/configs/` and the chosen script's argument parser. The environment specification is `magiclab_mujoco.yaml`.

12DOF playback:

```bash
python deploy_mujoco/z1_12dof_prop_mujoco.py --config deploy_mujoco/configs/z1_12dof.yaml
```

23DOF HoST playback:

```bash
python deploy_mujoco/z1_23dof_host_sim2sim.py --config deploy_mujoco/configs/z1_23dof_host.yaml --headless --max_steps 1000
```

Before playback, resolve the actual policy and XML paths, check actuator selection excludes the head for the 23DOF policy, and compare joint order, scales, control period, and history layout with training. `policy_format: auto` tries TorchScript then a raw checkpoint; architecture metadata matters for the latter.

Read `docs/rl/simulation-contract.md` when evaluating newly trained standup policies. A new actor export is not automatically compatible with the HoST playback observation. Preserve supplied checkpoint files and existing config profiles; put experiment changes in a named config. When execution is authorized, evaluate both prone and supine starts and report observed resets and clipping. For a code-only request, prepare commands and configuration without launching MuJoCo.
