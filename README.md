<div align="center">

# `rc.rl.magiclab`

**Magiclab Z1 强化学习全 Pipeline 实现：**
**从 Isaac Lab 训练 → MuJoCo 对仿 → 真机部署**

[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.0-76B900?style=flat)](https://isaac-sim.github.io/IsaacLab)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-FF6F00?style=flat)](https://mujoco.org/)


</div>

本仓库是围绕 **Magiclab Z1**（12DOF 行走 / 23DOF 起身等）的 **RL 工具链**：训练环境、仿真器侧部署、以及实机低层上的策略运行脚本，分模块维护，各子目录有独立说明。

---

## 目录结构

| 路径 | 说明 |
|------|------|
| [**`magiclab_rl_lab/`**](magiclab_rl_lab/) | 基于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的训练环境与任务：Z1-12 速度跟踪、Z1-23 起身等。 |
| [**`sim2sim/`**](sim2sim/) | **Sim-to-sim**：MuJoCo 中的策略回放 / 手操验证（含 Xbox / 键盘等输入示例）。 |
| [**`magiclab_deploy/`**](magiclab_deploy/) | **真机部署**：连接 Magicbot SDK，加载 PyTorch 策略，12DOF / 23DOF 双入口。 |

更细的安装、依赖与运行命令，请点进上表各目录下的 **`README.md`**。

---

## 工作流关系（概念）

```text
Isaac Lab 中训练 (magiclab_rl_lab)
         │
         ▼  导出/拷贝 checkpoint / 导出的 policy
         │
  MuJoCo 中回灌验证 (sim2sim)  ──►  调通观测与动作语义
         │
         ▼
  实机低层 (magiclab_deploy)  ──►  真机安全流程、频率与 YAML 标定
```

- **12DOF** 与 **23DOF** 的观测、动作维度、控制频率在各自 `configs/*.yaml` 与脚本的注释中说明。  
- 真机与仿真配置 **不共用同一份 YAML**，以各子项目为准。

---

## 快速链接

- 训练与任务列表 → [`magiclab_rl_lab/README.md`](magiclab_rl_lab/README.md)  
- MuJoCo 跑法 → [`sim2sim/README.md`](sim2sim/README.md)  
- 真机环境变量、12/23 入口、dry-run → [`magiclab_deploy/README.md`](magiclab_deploy/README.md)  
- Mujoco 相关变更记录 → [`sim2sim/Magiclab_mujico变更记录.md`](sim2sim/Magiclab_mujico变更记录.md)

---

## 许可与引用

子项目可能各自带有许可证或第三方依赖（Isaac、MuJoCo、厂商 SDK 等），使用前请同时遵守对应文档与**实机安全规范**（急停、支撑、场地区域等）。

---

<div align="center">

**⸻  RoboCup / Magiclab 赛事实验用代码整理  ⸻**

</div>
