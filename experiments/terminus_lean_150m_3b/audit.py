"""Audit the *single* dense FP8 run's records and checkpoint receipts.

Thresholds are investigation cues, not a claim of universal stability limits.
No dataset or GPU access is required.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terminus.integrity import file_sha256, write_json_atomic  # noqa: E402
from plan import MIDPOINT_TOKENS, MIDPOINT_UPDATES, TOTAL_UPDATES, USABLE_TOTAL_TOKENS  # noqa: E402


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def has_nonfinite_fraction(value):
    if isinstance(value, dict):
        return (value.get("nonfinite_fraction") is not None and value["nonfinite_fraction"] > 0
                or any(has_nonfinite_fraction(item) for item in value.values()))
    return False


def audit(run_dir, *, phase):
    run_dir = Path(run_dir)
    if phase not in {"midpoint", "final"}:
        raise ValueError("phase must be midpoint or final")
    contract = json.loads((run_dir / "run.json").read_text())["contract"]
    if contract["model"]["weight_mode"] != "dense":
        raise ValueError("this audit is only for the dense run")
    expected = [(MIDPOINT_UPDATES, MIDPOINT_TOKENS)]
    if phase == "final":
        expected.append((TOTAL_UPDATES, USABLE_TOTAL_TOKENS))
    checkpoints = []
    for update, tokens in expected:
        stem = f"step_{update:08d}"
        receipt = json.loads((run_dir / "checkpoints" / f"{stem}.json").read_text())
        checkpoint = run_dir / "checkpoints" / receipt["file"]
        if (receipt["state"]["updates"] != update or receipt["state"]["tokens_seen"] != tokens
                or receipt["size_bytes"] != checkpoint.stat().st_size
                or receipt["sha256"] != file_sha256(checkpoint)):
            raise ValueError(f"checkpoint receipt or SHA mismatch: {stem}")
        checkpoints.append({"updates": update, "tokens_seen": tokens, "file": str(checkpoint),
                            "sha256": receipt["sha256"]})
    events = read_jsonl(run_dir / "events.jsonl")
    train = [event for event in events if event.get("event") == "train"]
    validation = [event for event in events if event.get("event") == "validation"]
    stability = read_jsonl(run_dir / "stability.jsonl")
    incidents = read_jsonl(run_dir / "incidents.jsonl")
    end = expected[-1][0]
    incidents = [item for item in incidents if (item.get("state") or {}).get("updates", 0) <= end]
    updates = [event["updates"] for event in train if event["updates"] <= end]
    if updates != list(range(1, end + 1)):
        raise ValueError("per-update training records have a gap, duplicate, or changed order")
    if end not in {event["updates"] for event in validation}:
        raise ValueError("checkpoint boundary has no held-out validation record")
    sampled_updates = {item["updates"] for item in stability if item["updates"] <= end}
    if not {1, MIDPOINT_UPDATES}.issubset(sampled_updates) or (phase == "final" and TOTAL_UPDATES not in sampled_updates):
        raise ValueError("required first/midpoint/final numerical probes are missing")
    losses = [event["loss"] for event in train[:end]]
    spikes = []
    for index in range(20, len(losses)):
        baseline = statistics.median(losses[index - 20:index])
        if baseline > 0 and losses[index] > 2 * baseline:
            spikes.append({"updates": index + 1, "loss": losses[index], "previous_20_median": baseline})
    heavy_clip = [event["updates"] for event in train[:end] if event["grad_clip_coefficient"] < 0.1]
    probe_alerts = []
    for record in stability:
        if record["updates"] > end:
            continue
        for name, metrics in record["probes"].items():
            reason = None
            if has_nonfinite_fraction(metrics):
                reason = "nonfinite_values"
            elif "fp8_input" in name and metrics.get("relative_rms_error") is not None and metrics["relative_rms_error"] > 0.1:
                reason = "fp8_relative_rms_error_gt_0.1"
            elif "attention_sample" in name and metrics.get("logits", {}).get("max_abs", 0) > 30:
                reason = "sampled_attention_logit_gt_30"
            elif "attention_sample" in name and metrics.get("min_entropy") is not None and metrics["min_entropy"] < 0.1:
                reason = "sampled_attention_entropy_lt_0.1"
            elif "parameter_sample" in name and metrics.get("update_to_weight_rms") is not None and metrics["update_to_weight_rms"] > 0.1:
                reason = "sampled_update_weight_ratio_gt_0.1"
            if reason:
                probe_alerts.append({"updates": record["updates"], "probe": name, "reason": reason})
    return {
        "schema": 1, "phase": phase, "scope": "Terminus Lean dense FP8 only",
        "checkpoints": checkpoints, "training_updates": len(updates),
        "training_loss_first": losses[0], "training_loss_last": losses[-1],
        "validation_events": len([event for event in validation if event["updates"] <= end]),
        "stability_probe_updates": len([item for item in stability if item["updates"] <= end]),
        "incident_count": len(incidents), "incidents": incidents,
        "loss_spikes_gt_2x_previous_20_median": spikes,
        "clip_coefficient_lt_0.1_updates": heavy_clip,
        "probe_alerts": probe_alerts,
        "thresholds_are_investigation_cues": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", choices=("midpoint", "final"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(args.run_dir, phase=args.phase)
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
