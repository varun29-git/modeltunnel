"""Audit the verified 10B dense and 2B QAT checkpoints and event logs.

Raw checkpoints/logs are inputs only. Outputs are compact metrics and figures.
Run from the repository root with the SHA-verified downloaded artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


INDIC = ("hi", "mr", "kn", "te", "ta", "gu", "bn")
DENSE_SHA = "3d116e88048c93a0403ab2920b1680b066a9f29cbc07fa8ae1c2c67f2144684b"
QAT_SHA = "ceee823262d11e914eb63d6a396c3df90fb47e5dd9301f7db0f8add0a468c29f"
DENSE_MIX = {"en": .5, "code": .1, "math": .1, "bn": .043, "gu": .043,
             "hi": .043, "kn": .042, "mr": .043, "ta": .043, "te": .043}
QAT_MIX = {"en": .37, "code": .13, "math": .13, "bn": .053, "gu": .053,
           "hi": .053, "kn": .053, "mr": .053, "ta": .053, "te": .052}
PROJECTIONS = (
    ("q", "attn.qkv_proj.weight", slice(0, 640)),
    ("k", "attn.qkv_proj.weight", slice(640, 768)),
    ("v", "attn.qkv_proj.weight", slice(768, 896)),
    ("o", "attn.o_proj.weight", slice(None)),
    ("gate", "ffn.gate_up_proj.weight", slice(0, 1664)),
    ("up", "ffn.gate_up_proj.weight", slice(1664, 3328)),
    ("down", "ffn.down_proj.weight", slice(None)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_events(path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    events = [json.loads(line) for line in path.open() if line.strip()]
    return (events, [event for event in events if event.get("event") == "train"],
            [event for event in events if event.get("event") == "validation"])


def weighted_loss(event: dict, mix: dict[str, float]) -> float:
    return sum(mix[source] * event["languages"][source]["loss"] for source in mix)


def stage_metrics(events_path: Path, mix: dict[str, float], bounds: list[tuple[str, int, int]]) -> dict:
    events, train, validation = read_events(events_path)
    phases = {}
    for label, lo, hi in bounds:
        updates = [item for item in train if lo < item["tokens_seen"] <= hi]
        checkpoints = [item for item in validation if lo < item["tokens_seen"] <= hi]
        phases[label] = {
            "updates": len(updates),
            "clipped_updates": sum(item["grad_clip_coefficient"] < 1 for item in updates),
            "max_gradient_norm": max(item["grad_norm_before_clip"] for item in updates),
            "median_update_tokens_per_second": statistics.median(item["tokens_per_second"] for item in updates),
            "first_validation_tokens": checkpoints[0]["tokens_seen"],
            "last_validation_tokens": checkpoints[-1]["tokens_seen"],
            "first_weighted_validation": weighted_loss(checkpoints[0], mix),
            "last_weighted_validation": weighted_loss(checkpoints[-1], mix),
        }
    spikes = [train[i]["updates"] for i in range(20, len(train))
              if train[i]["loss"] > 2 * statistics.median(x["loss"] for x in train[i-20:i])]
    return {
        "train_events": len(train), "validation_events": len(validation),
        "first_train_loss": train[0]["loss"], "last_train_loss": train[-1]["loss"],
        "loss_spikes_over_2x_20_update_median": spikes,
        "last_clipped_update": max((item["updates"] for item in train if item["grad_clip_coefficient"] < 1), default=None),
        "final_validation": {key: val["loss"] for key, val in validation[-1]["languages"].items()},
        "final_validation_dense_mix": weighted_loss(validation[-1], DENSE_MIX),
        "final_validation_qat_mix": weighted_loss(validation[-1], QAT_MIX),
        "phases": phases,
        "start_time_unix": events[0]["time_unix"], "end_time_unix": events[-1]["time_unix"],
        "peak_allocated_gib": max(item["peak_memory_bytes"] for item in train) / (1024 ** 3),
        "peak_reserved_gib": max(item["reserved_memory_bytes"] for item in train) / (1024 ** 3),
    }


def rms(values: torch.Tensor) -> float:
    return math.sqrt(values.double().square().mean().item())


def quantize_ternary(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = weight.shape
    assert rows % 128 == 0 and columns % 128 == 0
    blocks = weight.reshape(rows // 128, 128, columns // 128, 128)
    scale = blocks.abs().mean(dim=(1, 3)).clamp_min(1e-6)
    expanded = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    codes = (weight / expanded).round().clamp(-1, 1)
    return codes, codes * expanded


def matrix_sums(dense: torch.Tensor, qat: torch.Tensor) -> dict[str, float]:
    dense = dense.float().contiguous()
    qat = qat.float().contiguous()
    dense_codes, dense_ternary = quantize_ternary(dense)
    qat_codes, qat_ternary = quantize_ternary(qat)
    return {
        "count": dense.numel(),
        "dense_sq": dense.square().sum().item(),
        "qat_sq": qat.square().sum().item(),
        "dense_q_sq": dense_ternary.square().sum().item(),
        "qat_q_sq": qat_ternary.square().sum().item(),
        "dense_q_dot": (dense * dense_ternary).sum().item(),
        "qat_q_dot": (qat * qat_ternary).sum().item(),
        "dense_q_err_sq": (dense - dense_ternary).square().sum().item(),
        "qat_q_err_sq": (qat - qat_ternary).square().sum().item(),
        "master_dot": (dense * qat).sum().item(),
        "master_err_sq": (dense - qat).square().sum().item(),
        "effective_err_sq": (dense - qat_ternary).square().sum().item(),
        "effective_dot": (dense * qat_ternary).sum().item(),
        "dense_zero": (dense_codes == 0).sum().item(),
        "qat_zero": (qat_codes == 0).sum().item(),
    }


def summarize_sums(sums: dict[str, float]) -> dict:
    count, d2, q2 = sums["count"], sums["dense_sq"], sums["qat_sq"]
    dq2, qq2 = sums["dense_q_sq"], sums["qat_q_sq"]
    return {
        "parameters": int(count),
        "dense_master_rms": math.sqrt(d2 / count), "qat_master_rms": math.sqrt(q2 / count),
        "master_rms_ratio": math.sqrt(q2 / d2),
        "master_cosine": sums["master_dot"] / math.sqrt(d2 * q2),
        "master_relative_rms_change": math.sqrt(sums["master_err_sq"] / d2),
        "dense_ternary_zero_fraction": sums["dense_zero"] / count,
        "qat_ternary_zero_fraction": sums["qat_zero"] / count,
        "dense_ternary_rms_ratio": math.sqrt(dq2 / d2),
        "qat_ternary_rms_ratio": math.sqrt(qq2 / q2),
        "dense_ternary_relative_rms_error": math.sqrt(sums["dense_q_err_sq"] / d2),
        "qat_ternary_relative_rms_error": math.sqrt(sums["qat_q_err_sq"] / q2),
        "dense_ternary_cosine": sums["dense_q_dot"] / math.sqrt(d2 * dq2),
        "qat_ternary_cosine": sums["qat_q_dot"] / math.sqrt(q2 * qq2),
        "qat_effective_vs_dense_relative_rms_change": math.sqrt(sums["effective_err_sq"] / d2),
        "qat_effective_vs_dense_cosine": sums["effective_dot"] / math.sqrt(d2 * qq2),
    }


def add_sums(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def vector_stats(dense: torch.Tensor, qat: torch.Tensor) -> dict:
    dense, qat = dense.float(), qat.float()
    result = {
        "parameters": dense.numel(), "dense_rms": rms(dense), "qat_rms": rms(qat),
        "dense_abs_max": dense.abs().max().item(), "qat_abs_max": qat.abs().max().item(),
        "dense_negative_fraction": (dense < 0).float().mean().item(),
        "qat_negative_fraction": (qat < 0).float().mean().item(),
        "qat_vs_dense_relative_rms_change": rms(qat - dense) / rms(dense),
    }
    for label, values in (("dense", dense), ("qat", qat)):
        result[f"{label}_abs_below_0_01_fraction"] = (values.abs() < .01).float().mean().item()
        result[f"{label}_largest_coordinate_energy_fraction"] = (
            values.square().max().item() / values.square().sum().item())
    return result


def percentile(values: torch.Tensor, percent: float) -> float:
    return torch.quantile(values, percent).item()


def checkpoint_metrics(dense_path: Path, qat_path: Path) -> dict:
    dense = torch.load(dense_path, map_location="cpu", weights_only=True, mmap=True)
    qat = torch.load(qat_path, map_location="cpu", weights_only=True, mmap=True)
    dmodel, qmodel = dense["model"], qat["model"]
    assert dmodel.keys() == qmodel.keys()
    assert dense["state"]["tokens_seen"] == 9_999_998_976
    assert qat["state"]["tokens_seen"] == 1_999_998_976
    aggregated = defaultdict(dict)
    by_layer = []
    norm_by_layer = []
    for layer in range(25):
        matrix_row = {"layer": layer}
        norm_row = {"layer": layer}
        for label, suffix, selected_rows in PROJECTIONS:
            key = f"layers.{layer}.{suffix}"
            sums = matrix_sums(dmodel[key][selected_rows], qmodel[key][selected_rows])
            add_sums(aggregated[label], sums)
            matrix_row[label] = summarize_sums(sums)
        for label in ("attn_norm", "ffn_norm"):
            key = f"layers.{layer}.{label}.weight"
            norm_row[label] = vector_stats(dmodel[key], qmodel[key])
        by_layer.append(matrix_row)
        norm_by_layer.append(norm_row)
    d_embed, q_embed = dmodel["tok_embeddings.weight"], qmodel["tok_embeddings.weight"]
    embedding = vector_stats(d_embed, q_embed)
    for label, weights in (("dense", d_embed), ("qat", q_embed)):
        rows = weights.double().square().mean(dim=1).sqrt().float()
        embedding[f"{label}_row_rms_p01"] = percentile(rows, .01)
        embedding[f"{label}_row_rms_median"] = percentile(rows, .5)
        embedding[f"{label}_row_rms_p99"] = percentile(rows, .99)
        embedding[f"{label}_row_rms_max"] = rows.max().item()
    embedding["master_cosine"] = (d_embed.double() * q_embed.double()).sum().item() / (
        math.sqrt(d_embed.double().square().sum().item() * q_embed.double().square().sum().item()))
    optimizer = {}
    for label, checkpoint in (("dense", dense), ("qat", qat)):
        groups = checkpoint["optimizer"]["param_groups"]
        optimizer[label] = {
            "group_count": len(groups), "state_entries": len(checkpoint["optimizer"]["state"]),
            "projection_groups": sum(bool(g.get("use_muon")) for g in groups),
            "matrix_peak_lrs": sorted(set(round(g["base_lr"], 10) for g in groups if g.get("use_muon"))),
            "embedding_peak_lr": groups[-2]["base_lr"], "norm_peak_lr": groups[-1]["base_lr"],
            "embedding_weight_decay": groups[-2]["weight_decay"],
            "norm_weight_decay": groups[-1]["weight_decay"],
        }
    return {
        "trainable_parameters": sum(v.numel() for k, v in dmodel.items() if k.endswith(".weight")),
        "projection_groups": {name: summarize_sums(sums) for name, sums in aggregated.items()},
        "projection_by_layer": by_layer,
        "norm_by_layer": norm_by_layer,
        "final_norm": vector_stats(dmodel["norm.weight"], qmodel["norm.weight"]),
        "embedding": embedding,
        "optimizer": optimizer,
    }


def probe_metrics(path: Path) -> dict:
    errors, zeros, entropy, logits, ratios = [], [], [], [], []
    nonfinite = 0
    final = None
    events = 0
    attention_low_by_layer = Counter()
    last_residual_max = (0.0, 0)
    first_gain_below = {}
    for line in path.open():
        if not line.strip():
            continue
        record = json.loads(line)
        events += 1
        final = record
        last_residual = record["probes"]["layer_24.residual"]["rms"]
        if last_residual > last_residual_max[0]:
            last_residual_max = (last_residual, record["updates"])
        first_gain = record["probes"]["layers.0.attn_norm.weight.parameter_sample"]["weight"]["rms"]
        for threshold in (.5, .1, .05, .03, .02):
            if first_gain < threshold and str(threshold) not in first_gain_below:
                first_gain_below[str(threshold)] = {
                    "updates": record["updates"], "tokens_seen": record["tokens_seen"]}
        for name, probe in record["probes"].items():
            if name.endswith(".fp8_input"):
                errors.append(probe["relative_rms_error"])
                zeros.append(probe["zero_fraction"])
            elif name.endswith(".attention_sample"):
                entropy.append(probe["min_entropy"])
                logits.append(probe["logits"]["max_abs"])
                if probe["min_entropy"] < .1:
                    attention_low_by_layer[name.split(".")[0]] += 1
            elif name.endswith(".parameter_sample"):
                ratios.append(probe["update_to_weight_rms"])
            def scan(obj):
                nonlocal nonfinite
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == "nonfinite_fraction" and value > 0:
                            nonfinite += 1
                        elif isinstance(value, dict):
                            scan(value)
            scan(probe)
    assert final is not None
    return {
        "events": events, "nonfinite_probe_values": nonfinite,
        "fp8_relative_rms_error_median": statistics.median(errors),
        "fp8_relative_rms_error_p99": sorted(errors)[int(.99 * len(errors))],
        "fp8_relative_rms_error_max": max(errors),
        "fp8_zero_fraction_max": max(zeros),
        "attention_samples": len(entropy),
        "attention_min_entropy_lt_0_1": sum(value < .1 for value in entropy),
        "attention_max_logit_gt_30": sum(value > 30 for value in logits),
        "attention_max_logit": max(logits),
        "attention_low_entropy_by_layer": dict(attention_low_by_layer),
        "parameter_update_to_weight_rms_gt_0_1": sum(value > .1 for value in ratios),
        "parameter_update_to_weight_rms_max": max(ratios),
        "first_attention_norm_sample_rms_final": final["probes"]["layers.0.attn_norm.weight.parameter_sample"]["weight"]["rms"],
        "last_layer_residual_rms_final": final["probes"]["layer_24.residual"]["rms"],
        "last_layer_residual_rms_max": last_residual_max[0],
        "last_layer_residual_rms_max_update": last_residual_max[1],
        "first_attention_norm_gain_first_below": first_gain_below,
        "final_residual_rms_by_layer": [final["probes"][f"layer_{layer:02d}.residual"]["rms"] for layer in range(25)],
        "final_attention_output_rms_by_layer": [final["probes"][f"layer_{layer:02d}.attention_output"]["rms"] for layer in range(25)],
        "final_ffn_output_rms_by_layer": [final["probes"][f"layer_{layer:02d}.ffn_output"]["rms"] for layer in range(25)],
    }


def plots(metrics: dict, dense_events: Path, qat_events: Path, output: Path) -> None:
    _, _, dv = read_events(dense_events)
    _, _, qv = read_events(qat_events)
    x_dense = [item["tokens_seen"] / 1e9 for item in dv]
    x_qat = [10 + item["tokens_seen"] / 1e9 for item in qv]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for index, (ax, mix, label) in enumerate(((axes[0], DENSE_MIX, "Dense-mixture weights"),
                                               (axes[1], QAT_MIX, "QAT-mixture weights"))):
        ax.plot(x_dense, [weighted_loss(item, mix) for item in dv], label="Dense FP8", color="#1769aa")
        ax.plot(x_qat, [weighted_loss(item, mix) for item in qv], label="Ternary QAT", color="#b04e32")
        if index == 0:
            ax.scatter([10], [sum(mix[k] * metrics["qat_initial_validation"][k] for k in mix)],
                       color="#b04e32", marker="x", label="One-shot ternary conversion")
        for boundary in (1, 8, 10, 10.2, 11.6):
            ax.axvline(boundary, color="#aaaaaa", linewidth=.6, linestyle="--")
        ax.set_ylim((2.2, 12.5) if index == 0 else (2.2, 3.7))
        ax.set_ylabel("Held-out NLL/token")
        ax.set_title(label)
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Cumulative training tokens (billions)")
    fig.tight_layout()
    fig.savefig(output / "combined_loss.png", dpi=170)
    plt.close(fig)

    names = [name for name, *_ in PROJECTIONS]
    groups = metrics["checkpoints"]["projection_groups"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    positions = list(range(len(names)))
    width = .36
    axes[0].bar([p-width/2 for p in positions], [groups[n]["dense_ternary_relative_rms_error"] for n in names],
                width, label="Dense checkpoint converted to ternary", color="#1769aa")
    axes[0].bar([p+width/2 for p in positions], [groups[n]["qat_ternary_relative_rms_error"] for n in names],
                width, label="After QAT", color="#b04e32")
    axes[0].set_ylabel("Weight quantization error / latent RMS")
    axes[0].legend(frameon=False)
    axes[1].bar([p-width/2 for p in positions], [groups[n]["dense_ternary_zero_fraction"] for n in names],
                width, color="#1769aa")
    axes[1].bar([p+width/2 for p in positions], [groups[n]["qat_ternary_zero_fraction"] for n in names],
                width, color="#b04e32")
    axes[1].set_ylabel("Ternary zero fraction")
    axes[1].set_xticks(positions, names)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "projection_groups.png", dpi=170)
    plt.close(fig)

    norms = metrics["checkpoints"]["norm_by_layer"]
    layers = list(range(25))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for group, color in (("attn_norm", "#1769aa"), ("ffn_norm", "#34855d")):
        ax.plot(layers, [row[group]["dense_rms"] for row in norms], label=f"Dense {group}", color=color)
        ax.plot(layers, [row[group]["qat_rms"] for row in norms], label=f"QAT {group}",
                color=color, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Gain RMS (log scale)")
    ax.grid(alpha=.2)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "norm_gains.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for stage, color in (("dense", "#1769aa"), ("qat", "#b04e32")):
        ax.plot(layers, metrics[f"{stage}_probes"]["final_residual_rms_by_layer"],
                label=f"{stage.upper()} final", color=color)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Sampled residual RMS")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "residual_scales.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dense-checkpoint", "qat-checkpoint", "dense-events", "qat-events",
                 "dense-stability", "qat-stability", "qat-initial-eval", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.dense_checkpoint) != DENSE_SHA or sha256(args.qat_checkpoint) != QAT_SHA:
        raise ValueError("checkpoint SHA-256 mismatch")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema": 1,
        "checkpoint_sha256": {"dense": DENSE_SHA, "qat": QAT_SHA},
        "dense": stage_metrics(args.dense_events, DENSE_MIX, [
            ("warmup", 0, 1_000_000_000), ("stable", 1_000_000_000, 8_000_000_000),
            ("decay", 8_000_000_000, 10_000_000_000)]),
        "qat": stage_metrics(args.qat_events, QAT_MIX, [
            ("warmup", 0, 200_000_000), ("stable", 200_000_000, 1_600_000_000),
            ("decay", 1_600_000_000, 2_000_000_000)]),
        "qat_initial_validation": {key: val["loss"] for key, val in
                                   json.loads(args.qat_initial_eval.read_text())["languages"].items()},
        "dense_probes": probe_metrics(args.dense_stability),
        "qat_probes": probe_metrics(args.qat_stability),
        "checkpoints": checkpoint_metrics(args.dense_checkpoint, args.qat_checkpoint),
    }
    (output / "combined_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    plots(metrics, args.dense_events, args.qat_events, output)
    print(f"Analyzed {metrics['checkpoints']['trainable_parameters']:,} parameters in both SHA-verified checkpoints")
    print(f"Saved metrics and four figures in {output}")


if __name__ == "__main__":
    main()
