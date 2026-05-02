# RSL-RL source snapshot

- Repository: https://github.com/leggedrobotics/rsl_rl
- Tag: v2.2.4
- Commit: f80d4750fbdfb62cfdb0c362b7063450f427cf35
- License: BSD-3-Clause, retained in LICENSE and source headers.
- Imported subtree: rsl_rl/ (18 Python files).
- Import date: 2026-10-15.

The initial import copies the package unchanged. Local follow-up commits isolate
its namespace and add the Magiclab integration outside this snapshot. Dependency
implementations and binaries are not redistributed. Upstream dependency license
notices can be found at the pinned repository's licenses/dependencies/ directory.

Runtime imports use torch, numpy and GitPython. TensorBoard is needed for the
selected local logging mode. Upstream optional logging modules require their own
wandb/neptune installations; the local adapter does not enable those modes.
