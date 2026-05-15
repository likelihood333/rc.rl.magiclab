"""Export a pinned-backend actor and its observation contract without launching simulation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from backend import activate_backend


def export_actor(checkpoint: Path, output_dir: Path) -> None:
    activate_backend()
    from magiclab_rsl.modules import ActorCritic, EmpiricalNormalization
    from magiclab_rl_lab.algorithms.vendor_ppo.metadata import UPSTREAM_COMMIT

    metadata_path = checkpoint.parent / "params" / "run.json"
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("backend") != "magiclab_rsl" or metadata.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Expected a checkpoint from the vendored PPO backend.")
    contract = metadata["contract"]
    policy_cfg = copy.deepcopy(contract["policy"])
    if policy_cfg.pop("class_name") != "ActorCritic":
        raise ValueError("Only the feed-forward ActorCritic export is supported.")
    actor_critic = ActorCritic(
        contract["actor_dim"], contract["critic_dim"], contract["num_actions"], **policy_cfg,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor_critic.load_state_dict(saved["model_state_dict"], strict=True)
    normalizer = torch.nn.Identity()
    if contract["empirical_normalization"]:
        normalizer = EmpiricalNormalization(shape=[contract["actor_dim"]])
        normalizer.load_state_dict(saved["obs_norm_state_dict"], strict=True)
    # Keep the original linear output; clipping belongs to the environment adapter.
    actor = torch.nn.Sequential(normalizer, actor_critic.actor).cpu().eval()
    with torch.inference_mode():
        probe = actor(torch.zeros(1, contract["actor_dim"]))
    if probe.shape != (1, contract["num_actions"]) or not torch.isfinite(probe).all():
        raise ValueError("Exported actor produced an invalid output.")
    scripted = torch.jit.script(actor)
    # Each export gets its own directory so policy and metadata remain paired.
    output_dir.mkdir(parents=True, exist_ok=False)
    scripted.save(str(output_dir / "policy.pt"))
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata["source_checkpoint"] = str(checkpoint)
    metadata["source_sha256"] = digest.hexdigest()
    metadata["policy_format"] = "jit"
    metadata["deployment_status"] = "requires_observation_and_action_mapping_validation"
    with (output_dir / "policy.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"[INFO] Exported policy and contract to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    export_actor(checkpoint, args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
