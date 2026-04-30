### Install
```bash
conda env create -f magiclab_deploy.yaml -n magiclab_deploy
conda activate magiclab_deploy
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

### Run

prepare environment variables:
```
export PYTHONPATH=/opt/magic_robotics/magic_z1_sdk/lib
export LD_LIBRARY_PATH=/opt/magic_robotics/magic_z1_sdk/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

Run deployment:
```bash
python deploy_real_z1_12dof.py --config=z1_12dof.yaml
```

策略加载在 `policy_loader.py`，23DOF 配置、观测和动作映射分别在 `settings_23dof.py`、`observations_23dof.py`、`action_mapping.py`；相对权重路径从 YAML 所在目录的 `policy_root` 解析。
两个入口里的主类均名为 `MagicBotController`，差异主要在：受控关节维度、观测构造与控制循环（手柄状态机 vs HoST 分频推理）。

Packages to install for keyboard control:
`sudo python -m pip install absl-py keyboard `

Test command input program (in another terminal):
```bash
python gamepad_reader_btp.py
```

# FallRecovery
test:

1. 清故障信息
2. 跑监控，查看电机是否异常
```bash
python /home/eame/magicbot-z1_sdk/example/python/monitor_example.py
```
3. 真机部署（dry_run 模式需要在 YAML 中设 `dry_run: true`）
```bash
python /home/eame/FallRecovery/magiclab_deploy/deploy_real_z1_23dof.py \
  --config z1_23dof.yaml
```

