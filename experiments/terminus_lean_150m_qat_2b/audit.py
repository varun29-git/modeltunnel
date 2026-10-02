"""Verify the completed ternary QAT continuation and summarize its signals."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terminus.checkpoint import load_checkpoint  # noqa: E402
from terminus.integrity import write_json_atomic  # noqa: E402

from experiments.terminus_lean_150m_qat_2b.plan import (  # noqa: E402
    PARENT_CHECKPOINT_SHA256, TOTAL_UPDATES, USABLE_TOKENS,
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(root: Path) -> dict:
    run = root / "qat"
    for name in ("plan.json", "config.json", "views/manifest.qat.json",
                 "views/view_receipt.json", "qat/transition_receipt.json",
                 "qat/run.json", "qat/events.jsonl", "qat/stability.jsonl",
                 "qat_initial_eval.json"):
        if not (root / name).is_file():
            raise ValueError(f"missing QAT artifact: {name}")
    parent = json.loads((run / "transition_receipt.json").read_text())
    if (parent["parent"]["checkpoint_sha256"] != PARENT_CHECKPOINT_SHA256
            or parent["data_policy"] != "restart"
            or set(parent["intentional_resets"]) != {"data_cursor", "update_counter"}):
        raise ValueError("QAT transition is not the verified fresh-offset restart")
    contract = json.loads((run / "run.json").read_text())["contract"]
    if (contract["model"]["weight_mode"] != "ternary"
            or not contract["model"]["quantize_activations"]
            or contract["model"]["linear_backend"] != "deepgemm"):
        raise ValueError("run contract is not native ternary-weight/FP8-activation training")
    for name in ("muon_lr", "adamw_lr", "norm_lr"):
        if contract["optimizer"][name] != .005:
            raise ValueError(f"QAT {name} differs from the half-peak plan")
    if contract["data"]["manifest_sha256"] != json.loads((root / "views/view_receipt.json").read_text())["qat_manifest_sha256"]:
        raise ValueError("trained QAT manifest differs from the checked fresh-offset view")
    checkpoint = load_checkpoint(run / f"checkpoints/step_{TOTAL_UPDATES:08d}.pt", contract)
    state = checkpoint["state"]
    if state["updates"] != TOTAL_UPDATES or state["tokens_seen"] != USABLE_TOKENS:
        raise ValueError("final checkpoint does not reach the 2B QAT budget")
    initial = load_checkpoint(run / "checkpoints/step_00000000.pt", contract)
    if initial["state"] != {"next_micro_step": 0, "tokens_seen": 0, "updates": 0}:
        raise ValueError("QAT initial checkpoint did not reset the phase counters")
    old_rates = [group["base_lr"] for group in checkpoint["optimizer"]["param_groups"]]
    initial_rates = [group["base_lr"] for group in initial["optimizer"]["param_groups"]]
    if old_rates != initial_rates or not old_rates:
        raise ValueError("QAT optimizer base learning rates changed after transition")
    events = read_jsonl(run / "events.jsonl")
    train = [event for event in events if event.get("event") == "train"]
    validation = [event for event in events if event.get("event") == "validation"]
    if [event["updates"] for event in train] != list(range(1, TOTAL_UPDATES + 1)):
        raise ValueError("QAT training updates are missing, repeated, or out of order")
    if not validation or validation[-1]["updates"] != TOTAL_UPDATES:
        raise ValueError("final QAT held-out validation is missing")
    initial_eval = json.loads((root / "qat_initial_eval.json").read_text())
    if initial_eval.get("updates") != 0 or set(initial_eval.get("languages", {})) != set(validation[-1]["languages"]):
        raise ValueError("QAT initial validation is missing a source")
    if not events or events[-1].get("event") != "complete":
        raise ValueError("QAT completion event is missing")
    stability = read_jsonl(run / "stability.jsonl")
    if not stability or stability[-1]["updates"] != TOTAL_UPDATES:
        raise ValueError("final QAT stability probe is missing")
    incidents = read_jsonl(run / "incidents.jsonl")
    if incidents:
        raise ValueError(f"QAT incidents require review: {len(incidents)}")
    if any(not math.isfinite(event["loss"]) for event in train):
        raise ValueError("QAT loss contains non-finite values")
    return {
        "phase": "final", "weight_mode": "ternary", "fp8_activations": True,
        "training_updates": len(train), "tokens_seen": state["tokens_seen"],
        "validation_events": len(validation), "stability_events": len(stability),
        "clipped_updates": sum(event["grad_clip_coefficient"] < 1 for event in train),
        "training_loss_first": train[0]["loss"], "training_loss_last": train[-1]["loss"],
        "final_validation": validation[-1]["languages"],
        "initial_ternary_validation": initial_eval["languages"],
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "initial_checkpoint_sha256": json.loads((run / "checkpoints/step_00000000.json").read_text())["sha256"],
        "final_checkpoint_sha256": json.loads((run / f"checkpoints/step_{TOTAL_UPDATES:08d}.json").read_text())["sha256"],
        "incidents": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run_root)
    write_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
