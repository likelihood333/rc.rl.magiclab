#!/usr/bin/env python3
# ===================== Metadata =====================
# Author: Mingzhu Li
# Last Modified: 2026-04-21 01:42:13 CST
# ====================================================
"""
Benchmark max policy inference frequency on edge computer.

Usage example:
python3 benchmark_inference_fps.py \
  --model_path /home/ubuntu/rc.rl.magiclab/sim2real/model/host_gym_model_12000.pt \
  --device cpu \
  --test_seconds 20
"""

from __future__ import annotations

import argparse
import time
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
        raise ValueError("No linear layers found in actor state_dict.")

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


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    idx = int((len(values) - 1) * p)
    return sorted(values)[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark policy inference FPS.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ubuntu/rc.rl.magiclab/sim2real/model/host_gym_model_12000.pt",
    )
    parser.add_argument("--obs_dim", type=int, default=456)
    parser.add_argument("--action_dim", type=int, default=23)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--policy_format", type=str, default="auto", choices=["auto", "jit", "checkpoint"])
    parser.add_argument("--actor_activation", type=str, default="elu")
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--test_seconds", type=float, default=20.0)
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="Set torch CPU threads (>0 to enable).",
    )
    args = parser.parse_args()

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    device = torch.device(args.device)
    print(f"[INFO] device={device}, torch_threads={torch.get_num_threads()}")
    print(f"[INFO] loading model: {args.model_path}")

    model = load_policy(
        policy_path=args.model_path,
        device=device,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        policy_format=args.policy_format,
        actor_activation=args.actor_activation,
    )
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()

    x = torch.randn(args.batch_size, args.obs_dim, device=device, dtype=torch.float32)

    # Warm-up
    with torch.inference_mode():
        for _ in range(args.warmup_steps):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed benchmark
    latencies_ms: list[float] = []
    n_iter = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        while (time.perf_counter() - t0) < args.test_seconds:
            t_start = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)
            n_iter += 1
    elapsed = time.perf_counter() - t0

    fps = n_iter / elapsed if elapsed > 0 else float("nan")
    p50 = percentile(latencies_ms, 0.50)
    p95 = percentile(latencies_ms, 0.95)
    avg = sum(latencies_ms) / len(latencies_ms) if latencies_ms else float("nan")

    print("========== Inference Benchmark ==========")
    print(f"iterations: {n_iter}")
    print(f"elapsed_s: {elapsed:.3f}")
    print(f"max_fps_estimate: {fps:.2f}")
    print(f"latency_avg_ms: {avg:.3f}")
    print(f"latency_p50_ms: {p50:.3f}")
    print(f"latency_p95_ms: {p95:.3f}")
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

