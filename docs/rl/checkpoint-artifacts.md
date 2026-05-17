# Checkpoint 文件

权重由本地文件或外部 artifact 保存，Git 维护引用配置。此次取消跟踪保留本地文件，不删除历史提交中的权重。

| 文件 | SHA256 |
| --- | --- |
| `magiclab_deploy/policy/z1_12dof/policy_gait1_v13_8_8600.pt` | `8d16f60f697ff129bfbf666bc27fe2af1d943635d0cc2626f4b9977bd26ff295` |
| `sim2sim/model/z1_dof/policy_gait1_v13_8_8600.pt` | `8d16f60f697ff129bfbf666bc27fe2af1d943635d0cc2626f4b9977bd26ff295` |
| `sim2sim/model/z1_dof/host_gym_model_12000.pt` | `4db8a2ae2a41f39d56577ddc204c8f654c436b6e63b332f91f4685358579dafb` |

新 clone 需另行放置上述文件。尚未配置外部下载源时，可从取消跟踪前的提交 `bd9e662` 恢复指定文件，例如在仓库根目录：

```bash
mkdir -p sim2sim/model/z1_dof
git show bd9e662:sim2sim/model/z1_dof/host_gym_model_12000.pt > sim2sim/model/z1_dof/host_gym_model_12000.pt
```

后续训练产物保留 `params/` 元数据；切换策略时提交配置与来源记录。哈希用于核对文件身份，没有验证策略表现。
