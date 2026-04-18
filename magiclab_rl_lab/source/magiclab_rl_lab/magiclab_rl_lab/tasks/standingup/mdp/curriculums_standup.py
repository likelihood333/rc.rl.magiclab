from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def standup_success_rate(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    reward_term_name: str = "head_height",
) -> torch.Tensor:
    """Expose stand-up curriculum signal.

    This does not mutate cfg yet; it provides a stable scalar for logging/tuning.
    """
    if len(env_ids) == 0:
        return torch.tensor(0.0, device=env.device)
    reward_sum = env.reward_manager._episode_sums[reward_term_name][env_ids]
    return torch.mean(reward_sum) / env.max_episode_length_s

