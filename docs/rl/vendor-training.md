# Vendored PPO 运行与维护

入口：`magiclab_rl_lab/scripts/vendor_ppo/train.py`。算法源码来自 [RSL-RL v2.2.4](https://github.com/leggedrobotics/rsl_rl/tree/v2.2.4)，在 `third_party/rsl_rl/magiclab_rsl/`。来源和许可证见该目录的 `UPSTREAM.md`、`LICENSE`。

## 环境

使用 Linux 上已安装 Isaac Lab 的 Python 环境。本地扩展需可导入；在该环境中可从仓库根目录执行安装命令：

```bash
python -m pip install -e magiclab_rl_lab/source/magiclab_rl_lab
python -m pip install GitPython tensorboard
```

这些是后续运行指令，本次没有安装依赖或执行项目。Isaac Lab / Isaac Sim、GPU 驱动和资产路径沿用现有部署。新入口不使用旧训练脚本的 runtime JSON bootstrap：用目标安装的 `isaaclab.sh -p` 提供运行环境，或先正确 source Isaac Sim 的 runtime 环境再用 Python。

## 训练

先设 `ISAACLAB_PATH` 为目标机器的实际安装目录，再进入仓库的 `magiclab_rl_lab/`：

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/vendor_ppo/train.py \
  --task Magiclab-Z1-12dof-Velocity \
  --config scripts/vendor_ppo/configs/velocity.json \
  --headless --device cuda:0 --num_envs 2048 \
  --max_iterations 10000 --run_name velocity_ppo

"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/vendor_ppo/train.py \
  --task Magiclab-Z1-23dof-Standup-Ground \
  --config scripts/vendor_ppo/configs/standup.json \
  --headless --device cuda:0 --num_envs 2048 \
  --max_iterations 20000 --run_name standup_ppo
```

起身的 Prone / Platform / Slope / Wall 注册任务也可使用同一后端。配置从任务默认值开始，再应用 JSON，最后由 CLI 覆盖 seed 和迭代数。JSON 只支持 `runner`、`policy`、`algorithm` 三个 section，不读取现有 `standup_train_cfg.json` 的 runtime/loss 字段。

初始范围：单进程、MLP PPO、TensorBoard；保持现有任务定义、动作语义和 observation manager 历史。现有 `scripts/rsl_rl/train.py` 不受影响。

## 保存与恢复

日志默认在 `logs/vendor_ppo/<experiment>/<timestamp>_<run_name>/`。每个 run 保存 `params/env.yaml`、`params/runner.json`、`params/run.json` 和 `model_<iteration>.pt`。起身 profile 每 100 次更新保存；这只是频率设置，没有产生新权重。

```bash
"${ISAACLAB_PATH}/isaaclab.sh" -p scripts/vendor_ppo/train.py \
  --task Magiclab-Z1-23dof-Standup-Ground \
  --config scripts/vendor_ppo/configs/standup.json \
  --headless --device cuda:0 --num_envs 2048 \
  --resume /absolute/path/to/run/model_1000.pt --max_iterations 2000
```

`--max_iterations` 是本次继续执行的更新次数。增加 `--weights_only` 会重新创建 optimizer 并从迭代 0 开始；不会复制原 run 的随机数状态或仿真状态，所以恢复不保证逐步复现。恢复需要原 checkpoint 相邻的 `params/run.json`，并检查上游版本、任务、观测配置、动作配置、网络与 normalization 设置；不自动猜测其它后端的 checkpoint 格式。

## 导出

导出不需要启动 Isaac Sim，但需要可导入本地扩展、PyTorch 与 GitPython 的环境：

```bash
python scripts/vendor_ppo/export.py \
  --checkpoint /absolute/path/to/run/model_1000.pt \
  --output_dir /absolute/path/to/exports/standup_model_1000
```

输出目录必须尚不存在，得到 `policy.pt` 和 `policy.json`。TorchScript 包含 actor 和可选观测 normalization，保留线性输出；checkpoint SHA256、源版本与观测契约一并记录。导出代码包含一次零输入形状/有限值检查，该检查只有后续实际执行导出时才运行。

送入 MuJoCo 或真机之前先读 [simulation-contract.md](simulation-contract.md)。当前起身任务的 progress 项和 observation manager 历史排列尚未与 HoST 部署对齐；不能直接替换现有 HoST 权重。记录参数有助于完成映射，不等于部署验证。

## Agent skills

仓库源文件在 `.agents/skills/`，分别为 `magiclab-train`、`magiclab-sim2sim`、`magiclab-real-deploy`。本机安装副本在 `~/.codex/skills/`；维护时同步对应的 `SKILL.md`。运行、连接或控制真机仍由当前用户请求的范围决定。

本轮验证范围：源码静态阅读、上游目录比对与 Git diff 检查；训练、仿真、导出和真机步骤都未执行。
