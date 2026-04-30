"""Policy loading without a robot SDK dependency."""

from collections import OrderedDict

import torch


def get_activation_module(act_name: str) -> torch.nn.Module:
    act_name = act_name.lower()
    if act_name == "elu":
        return torch.nn.ELU()
    if act_name == "selu":
        return torch.nn.SELU()
    if act_name == "relu":
        return torch.nn.ReLU()
    if act_name == "lrelu":
        return torch.nn.LeakyReLU()
    if act_name == "tanh":
        return torch.nn.Tanh()
    if act_name == "sigmoid":
        return torch.nn.Sigmoid()
    raise ValueError(f"Unsupported actor activation: {act_name}")


def build_actor_from_state_dict(
    actor_state_dict: dict[str, torch.Tensor], activation: str = "elu"
) -> torch.nn.Module:
    linear_indices = sorted(
        int(k.split(".")[0]) for k in actor_state_dict.keys() if k.endswith(".weight")
    )
    if not linear_indices:
        raise ValueError("No linear layers found in actor_state_dict.")

    modules: OrderedDict[str, torch.nn.Module] = OrderedDict()
    for i, idx in enumerate(linear_indices):
        out_dim, in_dim = actor_state_dict[f"{idx}.weight"].shape
        modules[str(idx)] = torch.nn.Linear(in_dim, out_dim)
        modules[str(idx + 1)] = (
            get_activation_module(activation)
            if i < len(linear_indices) - 1
            else torch.nn.Tanh()
        )
    actor = torch.nn.Sequential(modules)
    actor.load_state_dict(actor_state_dict, strict=False)
    return actor


def load_policy(
    policy_path: str,
    device: torch.device,
    obs_dim: int,
    action_dim: int,
    policy_format: str = "auto",
    actor_activation: str = "elu",
) -> torch.nn.Module:
    errors: list[str] = []

    if policy_format in ("auto", "jit"):
        try:
            policy = torch.jit.load(policy_path, map_location=device)
            policy.eval()
            out = policy(torch.zeros(1, obs_dim, dtype=torch.float32, device=device))
            if out.shape[-1] != action_dim:
                raise ValueError(f"JIT output={out.shape[-1]} expected={action_dim}")
            return policy
        except Exception as exc:  # noqa: BLE001
            errors.append(f"jit={exc}")
            if policy_format == "jit":
                raise

    if policy_format in ("auto", "checkpoint"):
        try:
            checkpoint = torch.load(policy_path, map_location=device)
            state_dict = (
                checkpoint["model_state_dict"]
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                else checkpoint
            )
            actor_state_dict = {
                k[len("actor.") :]: v
                for k, v in state_dict.items()
                if k.startswith("actor.")
            }
            if not actor_state_dict:
                raise ValueError("No actor.* keys found in checkpoint.")

            in_dim = actor_state_dict["0.weight"].shape[1]
            out_idx = max(
                int(k.split(".")[0])
                for k in actor_state_dict.keys()
                if k.endswith(".weight")
            )
            out_dim = actor_state_dict[f"{out_idx}.weight"].shape[0]

            if in_dim != obs_dim:
                raise ValueError(f"checkpoint input={in_dim} expected={obs_dim}")
            if out_dim != action_dim:
                raise ValueError(f"checkpoint output={out_dim} expected={action_dim}")

            actor = build_actor_from_state_dict(
                actor_state_dict, activation=actor_activation
            )
            actor.to(device).eval()
            return actor
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ckpt={exc}")
            if policy_format == "checkpoint":
                raise

    raise RuntimeError(
        f"Failed to load policy format={policy_format}, details={' | '.join(errors)}"
    )
