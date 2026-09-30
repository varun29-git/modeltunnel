"""Generate and verify the data-free Terminus Lean H100 experiment contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terminus.integrity import file_sha256, write_json_atomic  # noqa: E402
from terminus.model import BitNetLlamaConfig  # noqa: E402
from terminus.preflight import count_model  # noqa: E402
from terminus.registry import make_run_config  # noqa: E402
from terminus.train import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
TOKENIZER = ROOT / "tokenizer/tokenizer-65k.json"
TOKENIZER_SHA256 = "c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29"
SEQ_LEN = 4096
GLOBAL_BATCH_TOKENS = 262_144
NOMINAL_TOTAL_TOKENS = 3_000_000_000
USABLE_TOTAL_TOKENS = (NOMINAL_TOTAL_TOKENS // SEQ_LEN) * SEQ_LEN
TOTAL_UPDATES = (USABLE_TOTAL_TOKENS + GLOBAL_BATCH_TOKENS - 1) // GLOBAL_BATCH_TOKENS
MIDPOINT_UPDATES = TOTAL_UPDATES // 2
MIDPOINT_TOKENS = MIDPOINT_UPDATES * GLOBAL_BATCH_TOKENS
SHAPE = {"dim": 640, "n_layers": 25, "n_heads": 5, "n_kv_heads": 1,
         "ffn_dim": 1664, "global_attention_every": 5, "max_seq_len": SEQ_LEN}
EXPECTED_COUNT = 146_423_680


def render(microbatch: int):
    if microbatch not in (8, 16, 32, 64):
        raise ValueError("microbatch must be one of 8, 16, 32, 64")
    accumulation = GLOBAL_BATCH_TOKENS // (microbatch * SEQ_LEN)
    config = make_run_config("200m", weight_mode="dense", schedule="wsd",
        warmup_tokens=300_000_000, decay_tokens=600_000_000,
        muon_lr=.01, adamw_lr=.01, norm_lr=.01,
        gradient_accumulation_steps=accumulation, seed=1337,
        extra_model={**SHAPE, "use_subln": False, "qk_norm_mode": "none"})
    config["training"].update(checkpoint_interval=20_000, validation_interval=100,
                              log_interval=1, stability_log_interval=10)
    configs = {"lean_dense": config}
    sweep = json.loads(json.dumps(config))
    sweep["training"]["gradient_accumulation_steps"] = 64
    configs["lean_sweep"] = sweep
    plan = {
        "status": "prepared_not_launched", "device": "single Azure H100 NVL (SM90)",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "model_parameters": EXPECTED_COUNT,
        "weight_mode": "dense", "nominal_total_tokens": NOMINAL_TOTAL_TOKENS,
        "usable_total_tokens": USABLE_TOTAL_TOKENS,
        "total_updates": TOTAL_UPDATES,
        "midpoint_stop_after_updates": MIDPOINT_UPDATES,
        "midpoint_tokens": MIDPOINT_TOKENS,
        "normal_checkpoint_updates": [MIDPOINT_UPDATES, TOTAL_UPDATES],
        "peak_learning_rate": .01,
        "learning_rate_source": "BitCPM4 5M/2B dense FP8 pilot; unvalidated at 146M with Muon",
        "stability_file": "stability.jsonl", "incident_file": "incidents.jsonl",
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "microbatch_per_gpu": microbatch,
        "gradient_accumulation_steps": accumulation,
        "data_parallel_world_size": 1,
        "sequence_length": SEQ_LEN,
        "evaluation": "dense midpoint and final held-out loss/tasks; no QAT in this run",
        "hyperparameters": "BitCPM4 pilot LR is a direct trial setting, not a validated 146M optimum",
    }
    return configs, plan


def verify(configs, plan):
    if file_sha256(TOKENIZER) != TOKENIZER_SHA256:
        raise ValueError("frozen tokenizer SHA-256 mismatch")
    config = configs["lean_dense"]
    model = BitNetLlamaConfig(**config["model"])
    counts = count_model(model)
    if counts["total"] != EXPECTED_COUNT:
        raise ValueError(f"lean parameter count changed: {counts['total']}")
    if (model.linear_backend, model.attention_backend, model.fp8_backward,
        model.fp8_quantizer, model.loss_backend, model.residual_dtype) != (
        "deepgemm", "fa3", "full", "triton_split", "triton", "bf16"):
        raise ValueError("production H100 stack changed")
    if not (model.quantize_activations and model.fused_mlp and model.fused_qkv
            and model.fused_pointwise and model.fuse_projections and model.fp8_cache_weights):
        raise ValueError("production fusion or FP8 activations disabled")
    if model.vocab_size != 65536 or model.pad_token_id != 3 or model.weight_mode != "dense":
        raise ValueError("frozen tokenizer or dense geometry changed")
    if model.use_subln or model.qk_norm_mode != "none" or model.n_layers != 25 or model.global_attention_every != 5:
        raise ValueError("lean cuts or layer schedule changed")
    if (config["optimizer"]["muon_lr"], config["optimizer"]["adamw_lr"],
        config["optimizer"]["norm_lr"]) != (.01, .01, .01):
        raise ValueError("BitCPM4 pilot learning rate changed")
    if (config["training"]["checkpoint_interval"] <= TOTAL_UPDATES
            or config["training"]["log_interval"] != 1
            or config["training"]["stability_log_interval"] != 10
            or config["training"]["validation_interval"] != 100):
        raise ValueError("checkpoint or per-update telemetry contract changed")
    if plan["microbatch_per_gpu"] * plan["gradient_accumulation_steps"] * SEQ_LEN != GLOBAL_BATCH_TOKENS:
        raise ValueError("global batch decomposition changed")
    if MIDPOINT_TOKENS >= USABLE_TOTAL_TOKENS or MIDPOINT_TOKENS % GLOBAL_BATCH_TOKENS:
        raise ValueError("midpoint does not align to an optimizer update")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "check"))
    parser.add_argument("--microbatch", type=int, default=32)
    args = parser.parse_args()
    configs, plan = render(args.microbatch)
    verify(configs, plan)
    if args.command == "generate":
        CONFIGS.mkdir(exist_ok=True)
        for name, config in configs.items():
            write_json_atomic(CONFIGS / f"{name}.json", config)
        write_json_atomic(HERE / "plan.json", plan)
    else:
        if json.loads((HERE / "plan.json").read_text()) != plan:
            raise ValueError("plan.json differs from generated contract")
        unexpected = {path.stem for path in CONFIGS.glob("*.json")} - set(configs)
        if unexpected:
            raise ValueError(f"inactive configs remain in the dense-only folder: {sorted(unexpected)}")
        for name, config in configs.items():
            path = CONFIGS / f"{name}.json"
            if json.loads(path.read_text()) != config:
                raise ValueError(f"{path} differs from generated contract")
            load_config(path)
    print(json.dumps({"status": "generated" if args.command == "generate" else "verified",
                      "microbatch": args.microbatch, "global_batch_tokens": GLOBAL_BATCH_TOKENS,
                      "usable_tokens": USABLE_TOTAL_TOKENS, "midpoint_tokens": MIDPOINT_TOKENS,
                      "parameters": EXPECTED_COUNT}))


if __name__ == "__main__":
    main()
