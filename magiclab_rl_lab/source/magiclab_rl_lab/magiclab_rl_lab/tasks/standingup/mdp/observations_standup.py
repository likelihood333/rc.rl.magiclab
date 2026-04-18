from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def normalized_episode_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose normalized episode progress for stand-up phase awareness."""
    return (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1).clamp(0.0, 1.0)

