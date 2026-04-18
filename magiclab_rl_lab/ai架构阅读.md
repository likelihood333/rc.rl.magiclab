# magiclab_rl_lab 仓库文件作用说明

## 根目录

- `.dockerignore`：Docker 构建忽略规则。
- `.flake8`：Python 代码规范检查配置。
- `.gitattributes`：Git 属性设置（换行、文件类型等）。
- `.gitignore`：Git 忽略规则。
- `.pre-commit-config.yaml`：提交前自动检查配置。
- `LICENCE`：仓库许可证。
- `README.md`：项目介绍、安装与训练/推理方法。
- `magiclab_rl_lab.sh`：项目管理脚本（安装、列任务等）。
- `train_bash.sh`：训练快捷启动脚本。
- `play_bash.sh`：策略回放快捷脚本。
- `play_keyboard_bash.sh`：键盘控制回放脚本。
- `pyproject.toml`：项目构建和工具配置。

## doc/licenses

- `doc/licenses/isaaclab-license.txt`：IsaacLab 许可证文本。
- `doc/licenses/onnxruntime-license.txt`：ONNX Runtime 许可证文本。

## scripts

- `scripts/list_envs.py`：列出并打印已注册的 Magiclab 环境任务。
- `scripts/rsl_rl/cli_args.py`：RSL-RL 的命令行参数定义与配置覆盖逻辑。
- `scripts/rsl_rl/train.py`：训练入口（创建环境、Runner、训练循环、日志导出）。
- `scripts/rsl_rl/play.py`：加载 checkpoint 进行推理回放，并导出 `policy.pt` / `policy.onnx`。
- `scripts/rsl_rl/play_keyboard.py`：支持键盘控制的推理回放版本。
- `scripts/rsl_rl/rl_utils.py`：辅助功能（当前是相机跟随函数）。

## source/magiclab_rl_lab

- `source/magiclab_rl_lab/config/extension.toml`：Isaac 扩展配置入口。
- `source/magiclab_rl_lab/docs/CHANGELOG.rst`：版本变更记录。
- `source/magiclab_rl_lab/pyproject.toml`：子包构建配置。
- `source/magiclab_rl_lab/setup.py`：安装脚本。

## source/magiclab_rl_lab/magiclab_rl_lab

- `__init__.py`：包入口（当前为空）。
- `ui_extension_example.py`：UI 扩展示例。
- `utils/export_deploy_cfg.py`：导出部署配置。
- `utils/parser_cfg.py`：解析环境/任务配置。

### assets/robots

- `assets/robots/magiclab.py`：Magiclab 机器人资产主配置（URDF、执行器、初始姿态等）。

### data/robots/magicbot-Z1/urdf

- `MagicBotZ1_12dof.urdf`：12 自由度模型。
- `MagicBotZ1_12dof_arm_ready_pos.urdf`：12 自由度 + 手臂 ready 位姿版本。
- `MagicBotZ1_23dof.urdf`：23 自由度模型。

### data/robots/magicbot-Z1/meshes

- `*.STL`：机器人各连杆网格文件（头、躯干、骨盆、左右髋/膝/踝、肩/肘/腕等），供仿真与可视化加载。

## tasks

- `tasks/__init__.py`：任务包自动导入与注册。
- `tasks/locomotion/__init__.py`：locomotion 子模块入口。
- `tasks/locomotion/agents/__init__.py`：agent 配置入口。
- `tasks/locomotion/agents/rsl_rl_ppo_cfg.py`：PPO 默认超参数配置。
- `tasks/locomotion/mdp/__init__.py`：整合导出命令、课程、观测、奖励。
- `tasks/locomotion/mdp/curriculums.py`：课程学习逻辑（命令范围渐进提升）。
- `tasks/locomotion/mdp/observations.py`：自定义观测（如 gait phase、contact mask）。
- `tasks/locomotion/mdp/rewards.py`：自定义奖励函数集合。
- `tasks/locomotion/mdp/commands/__init__.py`：命令模块导出。
- `tasks/locomotion/mdp/commands/velocity_command.py`：速度命令配置（含 `limit_ranges`）。
- `tasks/locomotion/robots/__init__.py`：机器人任务包入口（空文件）。
- `tasks/locomotion/robots/z1/__init__.py`：Z1 子包入口（空文件）。
- `tasks/locomotion/robots/z1/12dof/__init__.py`：12dof 任务注册入口。
- `tasks/locomotion/robots/z1/12dof/velocity_env_cfg.py`：核心环境配置（场景、动作、观测、奖励、终止、事件、课程、训练/回放配置）。