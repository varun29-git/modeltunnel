"""Reproduce the final 10B run's loss plots and compact log statistics.

Usage:
  python analyze_final_run.py --run-root /path/to/downloaded/run \
      --output-dir /path/to/report/figures

The inputs are the verified Azure Blob copies of events.jsonl, stability.jsonl,
audit_final.json and views/manifest.dense.json. Raw logs stay outside git.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHASES = (("warmup", 0, 1), ("stable", 1, 8), ("decay", 8, 10))
INDIC = ("hi", "mr", "kn", "te", "ta", "gu", "bn")


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def quantiles(values):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "median": float(np.median(array)),
            "p95": float(np.quantile(array, .95)), "p99": float(np.quantile(array, .99)),
            "max": float(array.max())}


def phase_name(tokens: int) -> str:
    billions = tokens / 1e9
    if billions <= 1:
        return "warmup"
    if billions <= 8:
        return "stable"
    return "decay"


def style_axis(ax, *, ylabel: str, xlabel: bool = True):
    ax.axvline(1, color="#999999", linewidth=.9, linestyle="--")
    ax.axvline(8, color="#999999", linewidth=.9, linestyle="--")
    ax.grid(alpha=.18)
    ax.set_xlim(0, 10)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Training tokens (billions)")


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_losses(train, validation, output_dir):
    train_x = np.asarray([item["tokens_seen"] / 1e9 for item in train])
    train_loss = np.asarray([item["loss"] for item in train])
    window = 201
    smoothed = np.convolve(train_loss, np.ones(window) / window, mode="same")
    smoothed[:window // 2] = np.nan
    smoothed[-window // 2:] = np.nan
    val_x = [item["tokens_seen"] / 1e9 for item in validation]
    val_y = [item["weighted_loss"] for item in validation]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=False)
    axes[0].plot(train_x, smoothed, color="#c4742d", linewidth=1.5,
                 label="Training loss (201-update mean)")
    axes[0].plot(val_x, val_y, color="#1769aa", linewidth=1.6,
                 label="Mixture-weighted held-out loss")
    axes[0].set_ylim(2.2, 6.2)
    axes[0].legend(frameon=False)
    axes[0].set_title("Terminus Lean 150M: complete 10B-token loss trajectory")
    style_axis(axes[0], ylabel="Negative log-likelihood / token", xlabel=False)
    axes[1].plot(train_x, smoothed, color="#c4742d", linewidth=1.2)
    axes[1].plot(val_x, val_y, color="#1769aa", linewidth=1.8)
    axes[1].set_xlim(7.5, 10)
    axes[1].set_ylim(2.30, 2.86)
    axes[1].axvline(8, color="#999999", linewidth=.9, linestyle="--")
    axes[1].grid(alpha=.18)
    axes[1].set_xlabel("Training tokens (billions)")
    axes[1].set_ylabel("Loss, zoomed decay")
    fig.tight_layout()
    save_fig(fig, output_dir / "loss_overall.png")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
    for key, label, color, ax in [
        ("en", "English", "#1769aa", axes[0]),
        ("indic_weighted", "Indic weighted average", "#ab3670", axes[0]),
        ("code", "Code", "#287a42", axes[1]),
        ("math", "Math", "#b46b24", axes[1]),
    ]:
        ax.plot(val_x, [item[key] for item in validation], color=color,
                linewidth=1.6, label=label)
    for ax in axes:
        style_axis(ax, ylabel="Held-out loss")
        ax.legend(frameon=False)
    axes[0].set_title("Held-out loss by domain")
    fig.tight_layout()
    save_fig(fig, output_dir / "loss_domains.png")

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for key in INDIC:
        ax.plot(val_x, [item[key] for item in validation], linewidth=1.3,
                label=key.upper())
    style_axis(ax, ylabel="Held-out loss")
    ax.set_ylim(2.55, 4.5)
    ax.set_title("Seven Indic validation streams")
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    save_fig(fig, output_dir / "loss_indic.png")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    axes[0].plot(train_x, [item["lr_scale"] for item in train],
                 linewidth=1.4, color="#7552a3")
    style_axis(axes[0], ylabel="Peak LR multiplier", xlabel=False)
    axes[0].set_title("WSD schedule and per-update gradient norm")
    axes[1].plot(train_x, [item["grad_norm_before_clip"] for item in train],
                 linewidth=.5, alpha=.6, color="#47836d")
    axes[1].set_yscale("log")
    style_axis(axes[1], ylabel="Pre-clip gradient norm")
    fig.tight_layout()
    save_fig(fig, output_dir / "lr_gradient.png")


def analyze_stability(path: Path):
    fp8_errors = []
    fp8_zero_fraction = []
    attention_max_logits = []
    attention_min_entropy = []
    norm_update_ratios = []
    norm_weight_rms = []
    layer00_rms = []
    layer24_rms = []
    nonfinite_probes = 0
    sampled_updates = []
    series = collections.defaultdict(list)
    final_probes = None
    for item in read_jsonl(path):
        sampled_updates.append(item["updates"])
        series["tokens_b"].append(item["tokens_seen"] / 1e9)
        probes = item["probes"]
        final_probes = probes
        fp8 = [v for k, v in probes.items() if k.endswith(".fp8_input")]
        attention = [v for k, v in probes.items() if k.endswith(".attention_sample")]
        fp8_errors.extend(v["relative_rms_error"] for v in fp8
                          if v.get("relative_rms_error") is not None)
        fp8_zero_fraction.extend(v["zero_fraction"] for v in fp8
                                 if v.get("zero_fraction") is not None)
        attention_max_logits.append(max(v["logits"]["max_abs"] for v in attention))
        attention_min_entropy.append(min(v["min_entropy"] for v in attention))
        series["max_attention_logit"].append(attention_max_logits[-1])
        series["min_attention_entropy"].append(attention_min_entropy[-1])
        series["max_fp8_error"].append(max(v["relative_rms_error"] for v in fp8))
        norm = probes.get("layers.0.attn_norm.weight.parameter_sample")
        if norm:
            norm_update_ratios.append(norm["update_to_weight_rms"])
            norm_weight_rms.append(norm["weight"]["rms"])
            series["layer0_norm_gain_rms"].append(norm["weight"]["rms"])
            series["layer0_norm_update_ratio"].append(norm["update_to_weight_rms"])
        else:
            series["layer0_norm_gain_rms"].append(np.nan)
            series["layer0_norm_update_ratio"].append(np.nan)
        layer00_rms.append(probes["layer_00.residual"]["rms"])
        layer24_rms.append(probes["layer_24.residual"]["rms"])
        series["layer0_residual_rms"].append(layer00_rms[-1])
        series["layer24_residual_rms"].append(layer24_rms[-1])
        for value in probes.values():
            stack = [value]
            while stack:
                child = stack.pop()
                if isinstance(child, dict):
                    if child.get("nonfinite_fraction", 0) and child["nonfinite_fraction"] > 0:
                        nonfinite_probes += 1
                    stack.extend(v for v in child.values() if isinstance(v, dict))
    final_layers = [{
        "layer": i,
        "attention_norm_gain_rms": final_probes[
            f"layers.{i}.attn_norm.weight.parameter_sample"]["weight"]["rms"],
        "ffn_norm_gain_rms": final_probes[
            f"layers.{i}.ffn_norm.weight.parameter_sample"]["weight"]["rms"],
        "residual_rms": final_probes[f"layer_{i:02d}.residual"]["rms"],
    } for i in range(25)]
    return {
        "sampled_updates": len(sampled_updates),
        "fp8_relative_rms_error": quantiles(fp8_errors),
        "fp8_zero_fraction": quantiles(fp8_zero_fraction),
        "attention_sample_max_logit": quantiles(attention_max_logits),
        "attention_sample_min_entropy": quantiles(attention_min_entropy),
        "layer0_attn_norm_update_ratio": quantiles(norm_update_ratios),
        "layer0_attn_norm_weight_rms": quantiles(norm_weight_rms),
        "layer0_residual_rms": quantiles(layer00_rms),
        "layer24_residual_rms": quantiles(layer24_rms),
        "nonfinite_probe_values": nonfinite_probes,
        "attention_gt30_sampled_updates": sum(v > 30 for v in attention_max_logits),
        "entropy_lt0_1_sampled_updates": sum(v < .1 for v in attention_min_entropy),
        "norm_update_gt0_1_sampled_updates": sum(v > .1 for v in norm_update_ratios),
        "final_layer_scales": final_layers,
    }, series


def plot_stability(series, final_layers, output_dir):
    x = series["tokens_b"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)
    axes[0].plot(x, series["layer0_norm_gain_rms"], color="#a2405c",
                 linewidth=1.1, label="Layer 0 attention RMSNorm gain")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    style_axis(axes[0], ylabel="Sampled weight RMS", xlabel=False)
    axes[0].set_title("Scale drift in the dense FP8 run")
    axes[1].plot(x, series["layer0_residual_rms"], color="#1769aa",
                 linewidth=1, label="Layer 0 residual")
    axes[1].plot(x, series["layer24_residual_rms"], color="#b46b24",
                 linewidth=1, label="Layer 24 residual")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)
    style_axis(axes[1], ylabel="Sampled activation RMS")
    fig.tight_layout()
    save_fig(fig, output_dir / "norm_residual_scales.png")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
    axes[0].plot(x, series["max_attention_logit"], color="#70529b", linewidth=1)
    axes[0].axhline(30, color="#777777", linestyle=":", linewidth=.9)
    style_axis(axes[0], ylabel="Max |attention logit|", xlabel=False)
    axes[0].set_title("Sampled attention, FP8, and norm-update probes")
    axes[1].plot(x, series["max_fp8_error"], color="#287a42", linewidth=1)
    axes[1].axhline(.1, color="#777777", linestyle=":", linewidth=.9)
    style_axis(axes[1], ylabel="Max FP8 relative RMS error", xlabel=False)
    axes[2].plot(x, series["layer0_norm_update_ratio"], color="#a2405c", linewidth=1)
    axes[2].axhline(.1, color="#777777", linestyle=":", linewidth=.9)
    style_axis(axes[2], ylabel="Layer 0 norm update / weight")
    fig.tight_layout()
    save_fig(fig, output_dir / "stability_signals.png")

    layers = [item["layer"] + 1 for item in final_layers]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)
    axes[0].plot(layers, [item["attention_norm_gain_rms"] for item in final_layers],
                 marker="o", markersize=3, label="Attention RMSNorm gain")
    axes[0].plot(layers, [item["ffn_norm_gain_rms"] for item in final_layers],
                 marker="o", markersize=3, label="FFN RMSNorm gain")
    axes[0].set_ylabel("Sampled gain RMS")
    axes[0].set_title("Scale by layer at the final checkpoint")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.18)
    axes[1].plot(layers, [item["residual_rms"] for item in final_layers],
                 marker="o", markersize=3, color="#b46b24")
    axes[1].set_ylabel("Sampled residual RMS")
    axes[1].set_xlabel("Layer number")
    axes[1].grid(alpha=.18)
    axes[1].set_xlim(1, 25)
    fig.tight_layout()
    save_fig(fig, output_dir / "final_layer_scales.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output_dir = args.run_root, args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(root / "dense/events.jsonl"))
    train = [x for x in rows if x["event"] == "train"]
    val = [x for x in rows if x["event"] == "validation"]
    assert [x["updates"] for x in train] == list(range(1, 25433))
    assert train[-1]["tokens_seen"] == 9999998976
    mix = json.loads((root / "views/manifest.dense.json").read_text())["config"]["source_mix"]
    assert abs(sum(mix.values()) - 1) < 1e-9
    validation = []
    for item in val:
        source = {key: item["languages"][key]["loss"] for key in mix}
        source.update({"updates": item["updates"], "tokens_seen": item["tokens_seen"],
                       "weighted_loss": sum(mix[k] * source[k] for k in mix),
                       "indic_weighted": sum(mix[k] * source[k] for k in INDIC) / .3})
        validation.append(source)
    audit = json.loads((root / "audit_final.json").read_text())
    alerts = collections.Counter(item["reason"] for item in audit["probe_alerts"])
    alerts_by_phase = {phase: dict(collections.Counter(
        item["reason"] for item in audit["probe_alerts"]
        if phase_name(train[item["updates"] - 1]["tokens_seen"]) == phase))
        for phase, _, _ in PHASES}
    starts = [x for x in rows if x["event"] == "startup"]
    ends = [x for x in rows if x["event"] in {"stopped", "complete"}]
    assert len(starts) == len(ends) == 3
    invocations = [{"start_unix": a["time_unix"], "end_unix": b["time_unix"],
                    "elapsed_seconds": b["time_unix"] - a["time_unix"],
                    "start_update": a["state"]["updates"], "end_update": b["updates"]}
                   for a, b in zip(starts, ends)]
    active_seconds = sum(item["elapsed_seconds"] for item in invocations)
    stability, stability_series = analyze_stability(root / "dense/stability.jsonl")
    summary = {
        "training_updates": len(train), "validation_events": len(val),
        "tokens_seen": train[-1]["tokens_seen"], "source_mix": mix,
        "first_train_loss": train[0]["loss"], "final_train_batch_loss": train[-1]["loss"],
        "final_weighted_validation_loss": validation[-1]["weighted_loss"],
        "best_weighted_validation": min((x["weighted_loss"], x["updates"]) for x in validation),
        "validation_milestones": {
            str(billions): min(validation, key=lambda x: abs(x["tokens_seen"] / 1e9 - billions))
            for billions in (1, 5, 8, 9, 10)},
        "invocations": invocations,
        "active_seconds": active_seconds,
        "elapsed_wall_seconds": ends[-1]["time_unix"] - starts[0]["time_unix"],
        "active_wall_tokens_per_second": train[-1]["tokens_seen"] / active_seconds,
        "step_tokens_per_second": quantiles([x["tokens_per_second"] for x in train[100:-1]]),
        "peak_memory_gib": max(x["peak_memory_bytes"] for x in train) / 2**30,
        "peak_reserved_gib": max(x["reserved_memory_bytes"] for x in train) / 2**30,
        "gradient_norm": quantiles([x["grad_norm_before_clip"] for x in train]),
        "clipped_updates": sum(x["grad_clip_coefficient"] < 1 for x in train),
        "clip_below_0_1": len(audit["clip_coefficient_lt_0.1_updates"]),
        "clip_below_0_1_updates": audit["clip_coefficient_lt_0.1_updates"],
        "incidents": audit["incident_count"],
        "loss_spikes_over_2x": len(audit["loss_spikes_gt_2x_previous_20_median"]),
        "probe_alert_counts": dict(alerts), "probe_alert_counts_by_phase": alerts_by_phase,
        "stability": stability,
    }
    plot_losses(train, validation, output_dir)
    plot_stability(stability_series, stability["final_layer_scales"], output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "training_updates", "validation_events", "tokens_seen", "final_weighted_validation_loss",
        "active_wall_tokens_per_second", "clipped_updates", "probe_alert_counts")}, indent=2))


if __name__ == "__main__":
    main()
