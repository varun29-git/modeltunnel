"""Optimistic dense H100 training ceilings, never sparsity-assisted peaks."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict

from .integrity import write_json_atomic
from .preflight import count_model
from .train import load_config


# Per GPU, not the two-GPU NVL pair. Decimal TFLOP/s and TB/s.
# NVIDIA H100 Tensor Core GPU datasheet / H100 NVL product brief.
H100 = {
    "sxm": {"bf16_tflops": 989.5, "fp8_tflops": 1979., "hbm_tb_s": 3.35},
    "nvl": {"bf16_tflops": 835.5, "fp8_tflops": 1671., "hbm_tb_s": 3.9},
    "pcie": {"bf16_tflops": 756.5, "fp8_tflops": 1513., "hbm_tb_s": 2.0},
}


def causal_pairs(length, window):
    window = min(length, window)
    return window * (window + 1) // 2 + (length - window) * window


def training_roofline(config, training, *, variant="nvl", batch_size=1, target_tokens_s=67000., measured_tokens_s=None):
    if variant not in H100 or batch_size < 1 or not math.isfinite(target_tokens_s) or target_tokens_s <= 0:
        raise ValueError("invalid hardware, batch, or throughput target")
    if measured_tokens_s is not None and (not math.isfinite(measured_tokens_s) or measured_tokens_s <= 0):
        raise ValueError("measured throughput must be finite and positive")
    if training.dtype != "bf16":
        raise ValueError("this tensor-core ceiling models BF16/FP8 training, not FP32")
    counts = count_model(config)
    projection = counts["ternary"]
    embedding_dim = config.embedding_dim or config.dim
    head = config.vocab_size * embedding_dim
    embedding_projections = 2 * config.dim * embedding_dim if config.use_embedding_projection or embedding_dim != config.dim else 0
    pairs = sum(causal_pairs(config.max_seq_len, config.max_seq_len if (layer + 1) % config.global_attention_every == 0
                             else config.local_attention_window) for layer in range(config.n_layers))
    attention_forward = 4 * config.dim * pairs / config.max_seq_len
    # Fwd + dgrad + wgrad for projections; QK/AV fwd and four backward GEMMs.
    useful_fp8 = (2 * projection * (1 + (config.fp8_backward != "bf16") + (config.fp8_backward == "full"))
                  if config.linear_backend == "deepgemm" else 0)
    useful_bf16 = 6 * (projection + head + embedding_projections) - useful_fp8 + 3 * attention_forward
    extra_fp8 = 2 * projection if config.gradient_checkpointing and config.linear_backend == "deepgemm" else 0
    extra_bf16 = (2 * projection if config.gradient_checkpointing and config.linear_backend != "deepgemm" else 0)
    extra_bf16 += attention_forward if config.gradient_checkpointing else 0
    extra_bf16 += 2 * head if config.loss_backend == "checkpoint" else 0
    # Flash-style attention recomputes QK during backward instead of retaining
    # the quadratic probability matrix, even without block checkpointing.
    extra_bf16 += attention_forward / 2 if config.attention_backend in {"fa3", "flex"} else 0
    hardware = H100[variant]
    useful_seconds = useful_fp8 / (hardware["fp8_tflops"] * 1e12) + useful_bf16 / (hardware["bf16_tflops"] * 1e12)
    compute_seconds = useful_seconds + extra_fp8 / (hardware["fp8_tflops"] * 1e12) + extra_bf16 / (hardware["bf16_tflops"] * 1e12)
    update_tokens = batch_size * config.max_seq_len * training.gradient_accumulation_steps
    # Fused AdamW: read grad + read/write latent/m/v = 28 bytes/parameter.
    # Clipping: norm read + read/write scale = 12 more. Does not include
    # activations, GEMM operands, gradient accumulation or communication.
    state_traffic = 40 * counts["total"]
    cache_traffic = (4 + (2 if config.fp8_backward != "bf16" else 1)) * projection if config.fp8_cache_weights else 0
    hbm_seconds = (state_traffic + cache_traffic) / update_tokens / (hardware["hbm_tb_s"] * 1e12)
    ceiling = 1 / max(compute_seconds, hbm_seconds)
    projection_flops = 2 * projection * update_tokens
    fp8_peak, bf16_peak = hardware["fp8_tflops"] * 1e12, hardware["bf16_tflops"] * 1e12
    component_ms = {
        "projection_forward": projection_flops / (fp8_peak if config.linear_backend == "deepgemm" else bf16_peak) * 1e3,
        "projection_dgrad": projection_flops / (fp8_peak if config.linear_backend == "deepgemm" and config.fp8_backward != "bf16" else bf16_peak) * 1e3,
        "projection_wgrad": projection_flops / (fp8_peak if config.linear_backend == "deepgemm" and config.fp8_backward == "full" else bf16_peak) * 1e3,
        "head": (6 + (2 if config.loss_backend == "checkpoint" else 0)) * head * update_tokens / bf16_peak * 1e3,
        "attention_including_replay": attention_forward * (3 + int(config.gradient_checkpointing)
            + (.5 if config.attention_backend in {"fa3", "flex"} else 0)) * update_tokens / bf16_peak * 1e3,
        "embedding_projections": 6 * embedding_projections * update_tokens / bf16_peak * 1e3,
        "projection_checkpoint_replay": (extra_fp8 / fp8_peak
            + (2 * projection / bf16_peak if config.gradient_checkpointing and config.linear_backend != "deepgemm" else 0)) * update_tokens * 1e3,
    }
    return {
        "hardware": {"variant": variant, **hardware, "sparse_tensor_cores_assumed": False},
        "model": asdict(config), "parameters": counts["total"], "tokens_per_update": update_tokens,
        "useful_flops_per_token": {"fp8": useful_fp8, "bf16": useful_bf16},
        "recomputation_flops_per_token": {"fp8": extra_fp8, "bf16": extra_bf16},
        "ideal_compute_us_per_token": compute_seconds * 1e6,
        "ideal_component_compute_ms_per_update": component_ms,
        "target_update_ms": update_tokens / target_tokens_s * 1e3,
        "measured_update_ms": update_tokens / measured_tokens_s * 1e3 if measured_tokens_s else None,
        "required_time_reduction_fraction": max(0., 1 - measured_tokens_s / target_tokens_s) if measured_tokens_s else None,
        "optimizer_and_cache_hbm_floor_us_per_token": hbm_seconds * 1e6,
        "optimistic_tokens_per_second_ceiling": ceiling,
        "target_tokens_per_second": target_tokens_s,
        "target_fraction_of_compute_peak": target_tokens_s * compute_seconds,
        "measured_tokens_per_second": measured_tokens_s,
        "measured_fraction_of_compute_peak_including_recomputation": measured_tokens_s * compute_seconds if measured_tokens_s else None,
        "measured_useful_model_compute_fraction": measured_tokens_s * useful_seconds if measured_tokens_s else None,
        "scope": "Analytical upper bound at advertised dense clocks, not a forecast. Counts valid causal attention pairs; tile padding adds work. Omits non-GEMM math, launch latency, most HBM traffic, communication, IO, evaluation and checkpoints. Ternary zeros do not meet NVIDIA 2:4 structured sparsity requirements."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--h100", choices=H100, default="nvl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-tokens-s", type=float, default=67000.)
    parser.add_argument("--measured-tokens-s", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()
    config, training, _ = load_config(args.config)
    report = training_roofline(config, training, variant=args.h100, batch_size=args.batch_size,
                               target_tokens_s=args.target_tokens_s, measured_tokens_s=args.measured_tokens_s)
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
