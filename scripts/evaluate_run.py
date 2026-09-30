"""Score a trained checkpoint on versioned tasks and held-out streams.

Loads a run checkpoint (verified against its receipt), the tokenizer named by
the run's data manifest, and the evaluation manifest; writes per-item records
and an aggregate receipt with bootstrap CIs. Development tasks only unless
--include-final is passed (the untouched final set stays protected).

Example:
  .venv/bin/python scripts/evaluate_run.py --run-dir runs/x --config runs/x/config.json \
      --manifest data/manifest.json --data-root data --tokenizer data/tokenizer.json \
      --eval-manifest eval/eval_manifest.json --eval-root eval --max-items 64 \
      --output runs/x/eval_receipt.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from terminus.checkpoint import load_checkpoint
from terminus.integrity import file_sha256, fingerprint, write_json_atomic
from terminus.model import BitNetLlama
from terminus.task_eval import (aggregate, load_scalingbench_items, score_items)
from terminus.tokenizer import load_tokenizer
from terminus.train import load_config


def load_verified_model(config_path, checkpoint_dir):
    model_config, training, _ = load_config(config_path)
    checkpoint_path = Path(checkpoint_dir)
    run_file = (checkpoint_path.parent if checkpoint_path.is_dir() else checkpoint_path.parent.parent) / "run.json"
    if not run_file.exists():
        raise ValueError("checkpoint location lacks a run.json contract")
    contract = json.loads(run_file.read_text())["contract"]
    payload = load_checkpoint(checkpoint_dir, contract)
    model = BitNetLlama(model_config)
    model.load_state_dict(payload["model"])
    model.eval()
    receipt_path = (checkpoint_path / "latest.json" if checkpoint_path.is_dir()
                    else checkpoint_path.with_suffix(".json"))
    receipt = json.loads(receipt_path.read_text())
    return model, model_config, payload["state"], receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="run checkpoints directory or exact step_*.pt file")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--tasks", default="bbh,human_eval,math,mbpp,mmlu")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--device", default="auto", help="auto (cuda if available), cpu, or cuda:N")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.time()
    if args.device == "auto":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_default_dtype(torch.float32)
    model, model_config, state, checkpoint_receipt = load_verified_model(args.config,
                                                                         args.checkpoint)
    model = model.to(device)
    tokenizer = load_tokenizer(args.tokenizer)
    eval_manifest = json.loads(Path(args.eval_manifest).read_text())
    items = []
    for task in args.tasks.split(","):
        entry = eval_manifest["tasks"].get(f"scalingbench/{task}")
        if not entry or entry.get("status") != "fetched":
            print(f"skipping {task}: not fetched", flush=True)
            continue
        if entry.get("role") == "final" and not args.include_final:
            continue
        items.extend(load_scalingbench_items(Path(args.eval_root) / entry["file"],
                                             task=task, role=entry["role"],
                                             max_items=args.max_items))
    records = score_items(model, tokenizer, items, device=device,
                          max_seq_len=model_config.max_seq_len)
    summary = aggregate(records)
    receipt = {
        "evaluated_at_unix": time.time(), "wall_seconds": time.time() - started,
        "checkpoint": {"file": checkpoint_receipt["file"], "sha256": checkpoint_receipt["sha256"],
                       "state": state},
        "device": str(device),
        "config_sha256": fingerprint(json.loads(Path(args.config).read_text())),
        "tokenizer_sha256": file_sha256(args.tokenizer),
        "eval_manifest_sha256": file_sha256(args.eval_manifest),
        "tasks": sorted({record["task"] for record in records}),
        "summary": summary,
        "records_path": None,
    }
    output = Path(args.output)
    records_path = output.with_suffix(".items.jsonl")
    with records_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    receipt["records_path"] = str(records_path)
    write_json_atomic(output, receipt)
    print(json.dumps({"macro_task_loss": summary["macro_task_loss"],
                      "items": summary["items"]}, indent=1))


if __name__ == "__main__":
    main()
