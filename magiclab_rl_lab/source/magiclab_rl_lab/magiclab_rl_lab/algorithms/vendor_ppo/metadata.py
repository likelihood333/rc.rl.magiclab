"""Record the training policy contract and reject incompatible resumes."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


UPSTREAM_COMMIT = "f80d4750fbdfb62cfdb0c362b7063450f427cf35"


def policy_contract(env, task: str, cfg) -> dict:
    policy, extras = env.get_observations()
    critic = extras["observations"].get("critic", policy)
    obs_cfg = env.cfg.observations.policy
    return {
        "task": task,
        "actor_dim": policy.shape[1],
        "critic_dim": critic.shape[1],
        "num_actions": env.num_actions,
        "step_dt": env.unwrapped.step_dt,
        "policy_terms": list(env.unwrapped.observation_manager.active_terms["policy"]),
        "history_length": obs_cfg.history_length,
        "flatten_history_dim": obs_cfg.flatten_history_dim,
        "joint_names": list(env.unwrapped.scene["robot"].data.joint_names),
        "policy": cfg.to_runner_dict()["policy"],
        "empirical_normalization": cfg.empirical_normalization,
        "clip_actions": cfg.clip_actions,
        "actor_output": "linear",
        "history_owner": "isaaclab_observation_manager",
    }


def save_run_metadata(path: Path, *, contract: dict, repo_root: Path, seed: int, resume, weights_only: bool):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = {
        "schema_version": 1,
        "backend": "magiclab_rsl",
        "upstream_commit": UPSTREAM_COMMIT,
        "repo_commit": commit,
        "seed": seed,
        "resume_checkpoint": str(resume) if resume is not None else None,
        "weights_only": weights_only,
        "contract": contract,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def validate_resume(checkpoint: Path, contract: dict) -> None:
    metadata_file = checkpoint.parent / "params" / "run.json"
    if not metadata_file.is_file():
        raise ValueError(f"Resume requires the source run metadata: {metadata_file}")
    with metadata_file.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("backend") != "magiclab_rsl" or metadata.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Checkpoint backend differs from the pinned PPO implementation.")
    if metadata.get("contract") != contract:
        raise ValueError("Checkpoint task, observations, action mapping or network config differs from this run.")
