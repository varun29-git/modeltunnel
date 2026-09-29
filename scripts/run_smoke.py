"""Bounded end-to-end smoke test of the ModelTunnelv2 pipeline.

Route: S3 raw sample (pre-built by scripts/prepare_campaign_sample.py)
-> frozen 65k tokenizer -> versioned train/dev manifest -> tiny dense-FP8
CPU training (emulated FP8 via the torch quantizer) -> checkpoint ->
dense->ternary QAT transition with LR re-warmup -> interrupted/exact-resumed
-> task (ScalingBench, bounded) + held-out evaluation -> result ledger ->
receipt.

This proves the system end-to-end on CPU in development mode. It is NOT a
native-GPU receipt: DeepGEMM/FA3/preflight gates are explicitly marked not
run, and no quality claim is made from smoke losses.

Usage: .venv/bin/python scripts/run_smoke.py --corpus sample/corpus --work runs/smoke
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from terminus.data import TokenManifest
from terminus.integrity import (CAMPAIGN_DOMAINS, CAMPAIGN_LANGUAGES, CAMPAIGN_SCOPE_V1,
                                file_sha256, validate_manifest, write_json_atomic)
from terminus.prepare_campaign_data import prepare_campaign_data
from terminus.registry import Ledger
from terminus.task_eval import aggregate, load_scalingbench_items, score_items
from terminus.train import run_training
from terminus.transition import convert_checkpoint

SEQ_LEN = 1024
TOTAL_TOKENS = 32768
DENSE_UPDATES = 6
MIX = {"en": 0.38, "hi": 0.15, "mr": 0.08, "kn": 0.08, "te": 0.08, "ta": 0.08,
       "gu": 0.08, "bn": 0.07, "math": 0.04, "code": 0.04}

SMOKE_MODEL = {"vocab_size": 65536, "dim": 256, "n_layers": 4, "n_heads": 4, "n_kv_heads": 2,
               "ffn_dim": 512, "max_seq_len": SEQ_LEN, "local_attention_window": 256,
               "global_attention_every": 2, "attention_backend": "sdpa_reference",
               "weight_init": "normal", "pad_token_id": 3, "loss_chunk_size": 256,
               "quantize_activations": True, "weight_scale_layout": "blockwise",
               "linear_backend": "torch", "loss_backend": "streaming"}


def write_config(path, *, weight_mode, warmup, decay, muon_lr, phase_start=None):
    training = {"device": "cpu", "dtype": "fp32", "development_mode": True, "seed": 1337,
                "gradient_accumulation_steps": 2, "warmup_tokens": warmup,
                "final_lr_fraction": 0.1, "schedule": "wsd", "decay_tokens": decay,
                "decay_style": "linear", "grad_clip": 1.0, "checkpoint_interval": 1000,
                "validation_interval": 1000, "validation_batches_per_language": 1,
                "log_interval": 1, "compile": False}
    if phase_start is not None:
        training["phase_start_tokens"] = phase_start
    payload = {"model": {**SMOKE_MODEL, "weight_mode": weight_mode}, "training": training,
               "optimizer": {"kind": "muon", "muon_lr": muon_lr, "adamw_lr": muon_lr,
                             "norm_lr": 0.0002, "weight_decay": 0.1}}
    Path(path).write_text(json.dumps(payload, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="sample corpus dir with streams.json")
    parser.add_argument("--work", required=True)
    parser.add_argument("--eval-root", default="eval")
    parser.add_argument("--eval-items-per-task", type=int, default=8)
    args = parser.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)
    streams = json.loads((corpus / "streams.json").read_text())
    receipt = {"route": "s3-sample -> tokenizer -> manifest -> dense-FP8 -> checkpoint -> "
                        "ternary-QAT -> evaluation -> ledger -> exact-resume",
               "started_unix": time.time(), "steps": {}}
    ledger = Ledger(work / "ledger.jsonl")

    # Step 1-2: tokenize into a versioned manifest, then production-style gate.
    scope = {"name": CAMPAIGN_SCOPE_V1, "languages": list(CAMPAIGN_LANGUAGES),
             "domains": list(CAMPAIGN_DOMAINS)}
    data_dir = work / "data"
    manifest = prepare_campaign_data(streams, "tokenizer/tokenizer-65k.json", data_dir,
                                     source_mix=MIX, total_train_tokens=TOTAL_TOKENS,
                                     seq_len=SEQ_LEN, scope=scope)
    gate = validate_manifest(TokenManifest.load(data_dir / "manifest.json"), data_dir,
                             __import__("terminus.model", fromlist=["BitNetLlamaConfig"])
                             .BitNetLlamaConfig(**SMOKE_MODEL), development_mode=False)
    receipt["steps"]["manifest"] = {"manifest_sha256": gate["manifest_sha256"],
                                    "tokens_by_key": gate["tokens_by_language"],
                                    "tokenizer_sha256": gate["tokenizer"]["sha256"]}
    print("manifest validated (production gate)", flush=True)

    # Step 3: dense-FP8 stage with an interrupt + exact resume.
    dense_config = work / "dense_config.json"
    write_config(dense_config, weight_mode="dense", warmup=4096, decay=8192, muon_lr=0.003)
    dense_attempt = ledger.begin_attempt({"experiment_id": "smoke/dense", "stage": "smoke",
                                          "kind": "dense_fp8", "config": json.loads(dense_config.read_text()),
                                          "data_manifest_sha256": gate["manifest_sha256"],
                                          "tokenizer_sha256": gate["tokenizer"]["sha256"]})
    with contextlib.redirect_stdout(io.StringIO()):
        run_training(dense_config, data_dir / "manifest.json", data_dir, work / "dense",
                     stop_after_updates=DENSE_UPDATES // 2)
        interrupted = run_training(dense_config, data_dir / "manifest.json", data_dir,
                                   work / "dense", resume=work / "dense/checkpoints",
                                   stop_after_updates=DENSE_UPDATES)
    receipt["steps"]["dense"] = {"updates": interrupted["updates"],
                                 "tokens_seen": interrupted["tokens_seen"]}
    ledger.finish_attempt(dense_attempt, outcome="completed",
                          realized={"tokens_seen": interrupted["tokens_seen"]},
                          cost={"gpu_hours": 0.0}, metrics=None)
    print(f"dense stage: {interrupted['tokens_seen']} tokens, {interrupted['updates']} updates",
          flush=True)

    # Step 4: transition to ternary QAT with LR re-warmup; interrupt + resume inside QAT.
    qat_config = work / "qat_config_input.json"
    write_config(qat_config, weight_mode="ternary", warmup=2048, decay=4096, muon_lr=0.001)
    transition_receipt = convert_checkpoint(work / "dense", qat_config, data_dir / "manifest.json",
                                            data_dir, work / "qat", data_policy="continue")
    with contextlib.redirect_stdout(io.StringIO()):
        run_training(work / "qat/qat_config.json", data_dir / "manifest.json", data_dir,
                     work / "qat", resume=work / "qat/checkpoints",
                     stop_after_updates=DENSE_UPDATES + 2)
        final = run_training(work / "qat/qat_config.json", data_dir / "manifest.json", data_dir,
                             work / "qat", resume=work / "qat/checkpoints")
    qat_attempt = ledger.begin_attempt({"experiment_id": "smoke/qat", "stage": "smoke",
                                        "kind": "qat_branch",
                                        "config": json.loads((work / "qat/qat_config.json").read_text()),
                                        "data_manifest_sha256": gate["manifest_sha256"],
                                        "tokenizer_sha256": gate["tokenizer"]["sha256"]})
    ledger.finish_attempt(qat_attempt, outcome="completed",
                          realized={"tokens_seen": final["tokens_seen"]}, cost={"gpu_hours": 0.0})
    receipt["steps"]["qat"] = {"updates": final["updates"], "tokens_seen": final["tokens_seen"],
                               "transition": {"parent_state": transition_receipt["parent"]["state"],
                                              "policy": transition_receipt["data_policy"]}}
    print(f"qat stage complete at {final['tokens_seen']} tokens", flush=True)

    # Step 5: task + held-out evaluation of the final ternary checkpoint.
    from terminus.model import BitNetLlama, BitNetLlamaConfig
    from terminus.checkpoint import load_checkpoint
    contract = json.loads((work / "qat/run.json").read_text())["contract"]
    payload = load_checkpoint(work / "qat/checkpoints", contract)
    model = BitNetLlama(BitNetLlamaConfig(**{**SMOKE_MODEL, "weight_mode": "ternary"}))
    model.load_state_dict(payload["model"])
    model.eval()
    from terminus.tokenizer import load_tokenizer
    tokenizer = load_tokenizer(data_dir / "tokenizer.json")
    items = []
    eval_manifest = json.loads((Path(args.eval_root) / "eval_manifest.json").read_text())
    for task in ("bbh", "math", "mbpp", "mmlu", "human_eval"):
        entry = eval_manifest["tasks"][f"scalingbench/{task}"]
        if entry.get("status") == "fetched":
            items.extend(load_scalingbench_items(Path(args.eval_root) / entry["file"], task=task,
                                                 max_items=args.eval_items_per_task))
    records = score_items(model, tokenizer, items, device=torch.device("cpu"),
                          max_seq_len=SEQ_LEN)
    summary = aggregate(records)
    eval_attempt = ledger.begin_attempt({"experiment_id": "smoke/eval", "stage": "smoke",
                                         "kind": "task_eval",
                                         "config": json.loads((work / "qat/qat_config.json").read_text()),
                                         "data_manifest_sha256": gate["manifest_sha256"],
                                         "tokenizer_sha256": gate["tokenizer"]["sha256"],
                                         "eval_manifest_sha256": file_sha256(Path(args.eval_root) / "eval_manifest.json")})
    ledger.finish_attempt(eval_attempt, outcome="completed",
                          metrics={"macro_task_loss": summary["macro_task_loss"]},
                          cost={"gpu_hours": 0.0})
    receipt["steps"]["evaluation"] = {"items": summary["items"],
                                      "per_task": {k: v["mean"] for k, v in summary["per_task"].items()},
                                      "macro_task_loss": summary["macro_task_loss"]}
    print("evaluation:", json.dumps(receipt["steps"]["evaluation"]), flush=True)

    receipt["gpu_native_checks"] = "NOT RUN (CPU host): DeepGEMM dense/ternary GEMMs, FA3, Triton kernels, preflight"
    receipt["quality_note"] = ("smoke losses prove plumbing only; no quality or cost claim")
    receipt["finished_unix"] = time.time()
    receipt["ledger"] = str(work / "ledger.jsonl")
    write_json_atomic(work / "smoke_receipt.json", receipt)
    print(f"receipt: {work / 'smoke_receipt.json'}")


if __name__ == "__main__":
    main()
