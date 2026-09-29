"""Deliberate dense -> ternary-QAT phase transition.

`convert_checkpoint` turns an actually trained dense-stage checkpoint into the
initial checkpoint of a QAT run, with an explicit receipt. It:

- verifies the parent checkpoint against the parent run's own contract
  (integrity + counters) before converting;
- builds the child (QAT) contract exactly as `train.py` will compute it at
  resume time, so exact-resume protection within the QAT phase is unchanged —
  run this tool on the training host, with the same world size, because the
  contract binds runtime and native-source identity;
- carries model weights, compatible optimizer state, per-rank RNG, and the
  data cursor (unless a reset is explicitly requested and recorded);
- supports checkpoint branches: any parent checkpoint of the dense run may be
  converted, and the receipt records which one; different pre-switch histories
  are never presented as shared.

Data policies:
- "continue": the child manifest covers the same stream with a larger budget
  (parent tokens_seen + QAT budget); the carried cursor guarantees no data
  repeats or disappears across the switch.
- "phase_mix": the child manifest uses phase_source_mixes whose phase boundary
  must equal the parent's tokens_seen, so the router switches mixture exactly
  at the transition.
- "restart": the data cursor (and optionally RNG) is reset; the receipt
  records this as an intentional reset.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import TokenManifest
from .integrity import fingerprint, source_hashes, validate_manifest, write_json_atomic
from .model import BitNetLlama
from .optim import build_optimizer
from .train import load_config, make_resume_contract

POLICIES = ("continue", "phase_mix", "restart")
RESETTABLE = {"optimizer", "data_cursor", "rng"}


def _cumulative_phase_boundaries(config):
    boundaries = []
    total = 0
    for phase in config.phase_source_mixes:
        total += phase["tokens"]
        boundaries.append(total)
    return boundaries


def convert_checkpoint(parent_run_dir, qat_config_path, qat_manifest_path, qat_data_root,
                       output_dir, *, world_size=1, device_name="cpu", parent_checkpoint=None,
                       data_policy="continue", intentional_resets=()):
    parent_run_dir = Path(parent_run_dir)
    output_dir = Path(output_dir)
    if data_policy not in POLICIES:
        raise ValueError(f"data_policy must be one of {POLICIES}")
    resets = set(intentional_resets)
    if unknown := resets - RESETTABLE:
        raise ValueError(f"unknown intentional resets: {sorted(unknown)}")
    if data_policy == "restart":
        resets.add("data_cursor")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("the QAT run directory must be fresh")
    parent_run = json.loads((parent_run_dir / "run.json").read_text())
    parent_contract = parent_run["contract"]
    parent_source = parent_checkpoint or parent_run_dir / "checkpoints"
    parent_payload = load_checkpoint(parent_source, parent_contract)
    parent_state = parent_payload["state"]

    model_config, training, optimizer_config = load_config(qat_config_path)
    if model_config.weight_mode != "ternary":
        raise ValueError("the QAT phase must run the ternary weight mode")
    manifest = TokenManifest.load(qat_manifest_path)
    manifest_report = validate_manifest(manifest, qat_data_root, model_config,
                                        development_mode=training.development_mode)
    parent_tokenizer = parent_contract["data"].get("tokenizer") or {}
    child_tokenizer = manifest_report.get("tokenizer") or {}
    if parent_tokenizer and child_tokenizer and \
            parent_tokenizer.get("sha256") != child_tokenizer.get("sha256"):
        raise ValueError("parent and child manifests use different tokenizers")
    device = torch.device(device_name)
    contract = make_resume_contract(model_config, training, optimizer_config, manifest_report,
                                    world_size, device)

    model = BitNetLlama(model_config)
    optimizer = build_optimizer(model, optimizer_config)
    model.load_state_dict(parent_payload["model"])  # strict: shapes must match
    resets_done = []
    if "optimizer" in resets:
        resets_done.append("optimizer")
    else:
        optimizer.load_state_dict(parent_payload["optimizer"])
    total_tokens = manifest.config.usable_train_tokens
    micro_tokens = manifest.config.tokens_per_step * world_size
    total_micro_steps = -(-manifest.config.total_examples //
                          (manifest.config.batch_size * world_size))
    if "data_cursor" in resets:
        state = {"next_micro_step": 0, "tokens_seen": 0, "updates": parent_state["updates"]}
        resets_done.append("data_cursor")
    else:
        state = dict(parent_state)
        if state["tokens_seen"] % micro_tokens:
            raise ValueError("carried token cursor is not divisible by the child microbatch tokens; "
                             "keep the batch decomposition or request data_policy='restart'")
        if state["next_micro_step"] != state["tokens_seen"] // micro_tokens:
            raise ValueError("carried data cursor does not align with the child batch decomposition")
        if state["next_micro_step"] != min(state["updates"] * training.gradient_accumulation_steps,
                                           total_micro_steps):
            raise ValueError("carried update counter does not align with child accumulation")
        if state["tokens_seen"] >= total_tokens:
            raise ValueError("child budget must extend beyond the parent's consumed tokens")
        if data_policy == "phase_mix":
            boundaries = _cumulative_phase_boundaries(manifest.config)
            if state["tokens_seen"] not in boundaries:
                raise ValueError("phase_mix requires a phase boundary exactly at the parent's "
                                 f"consumed tokens ({state['tokens_seen']}); boundaries: {boundaries}")
    if len(parent_payload["rank_rng"]) != world_size:
        if "rng" not in resets:
            raise ValueError("world size changed; request intentional_resets=('rng',) explicitly")
        resets_done.append("rng")
        import random as random_module

        import numpy as np

        from .checkpoint import rng_state
        rank_rng = []
        for rank in range(world_size):  # same fresh-run seeding train.py applies per rank
            random_module.seed(training.seed + rank)
            np.random.seed((training.seed + rank) % (2**32))
            torch.manual_seed(training.seed + rank)
            rank_rng.append(rng_state())
    else:
        rank_rng = parent_payload["rank_rng"]

    checkpoints = output_dir / "checkpoints"
    path = save_checkpoint(checkpoints, model, optimizer, state, contract, rank_rng)
    write_json_atomic(output_dir / "run.json",
                      {"contract_sha256": fingerprint(contract), "contract": contract})
    parent_receipt_path = parent_payload_receipt(parent_source)
    parent_receipt = json.loads(parent_receipt_path.read_text()) if parent_receipt_path else {}
    child_training = asdict(training)
    for name in ("checkpoint_interval", "validation_interval", "validation_batches_per_language",
                 "log_interval", "prefetch_batches", "time_limit_seconds"):
        child_training.pop(name)
    changed = {"model": _diff(parent_contract["model"], asdict(model_config)),
               "training": _diff(parent_contract["training"], child_training),
               "optimizer": _diff(parent_contract["optimizer"],
                                  json.loads(json.dumps(asdict(optimizer_config))))}
    receipt = {"event": "dense_to_qat_transition", "time_unix": time.time(),
               "parent": {"run_dir": str(parent_run_dir),
                          "checkpoint_sha256": parent_receipt.get("sha256"),
                          "checkpoint_file": parent_receipt.get("file"),
                          "contract_sha256": parent_run["contract_sha256"],
                          "state": parent_state},
               "child": {"config_sha256": fingerprint(json.loads(Path(qat_config_path).read_text())),
                         "contract_sha256": fingerprint(contract),
                         "manifest_sha256": manifest_report["manifest_sha256"],
                         "initial_checkpoint": path.name, "state": state},
               "data_policy": data_policy, "intentional_resets": sorted(resets_done),
               "changed_fields": changed,
               "tokenizer_sha256": child_tokenizer.get("sha256"),
               "source": source_hashes()}
    write_json_atomic(output_dir / "transition_receipt.json", receipt)
    return receipt


def parent_payload_receipt(source):
    source = Path(source)
    if source.is_dir():
        candidate = source / "latest.json"
        return candidate if candidate.exists() else None
    candidate = source.with_suffix(".json")
    return candidate if candidate.exists() else None


def _diff(parent, child):
    changed = {}
    for key in sorted(set(parent) | set(child)):
        if parent.get(key) != child.get(key):
            changed[key] = {"parent": parent.get(key), "child": child.get(key)}
    return changed


def main():
    parser = argparse.ArgumentParser(description="Convert a dense-stage checkpoint into a QAT run")
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--config", required=True, help="QAT phase config JSON")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-policy", choices=POLICIES, default="continue")
    parser.add_argument("--intentional-resets", default="", help="comma-separated: optimizer,data_cursor,rng")
    args = parser.parse_args()
    resets = tuple(name for name in args.intentional_resets.split(",") if name)
    receipt = convert_checkpoint(args.parent_run, args.config, args.manifest, args.data_root,
                                 args.output_dir, world_size=args.world_size, device_name=args.device,
                                 parent_checkpoint=args.parent_checkpoint,
                                 data_policy=args.data_policy, intentional_resets=resets)
    print(json.dumps({"transition": receipt["child"]["contract_sha256"],
                      "parent_state": receipt["parent"]["state"],
                      "child_state": receipt["child"]["state"]}, indent=1))


if __name__ == "__main__":
    main()
