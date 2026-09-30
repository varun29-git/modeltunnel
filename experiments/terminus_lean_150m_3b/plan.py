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
DENSE_UPDATES = 6867
DENSE_TOKENS = DENSE_UPDATES * GLOBAL_BATCH_TOKENS
NOMINAL_TOTAL_TOKENS = 3_000_000_000
USABLE_TOTAL_TOKENS = (NOMINAL_TOTAL_TOKENS // SEQ_LEN) * SEQ_LEN
SHAPE = {"dim": 640, "n_layers": 25, "n_heads": 5, "n_kv_heads": 1,
         "ffn_dim": 1664, "global_attention_every": 5, "max_seq_len": SEQ_LEN}
EXPECTED_COUNTS = {"control": 146_423_730, "lean": 146_423_680}


def render(microbatch: int):
    if microbatch not in (8, 16, 32, 64):
        raise ValueError("microbatch must be one of 8, 16, 32, 64")
    accumulation = GLOBAL_BATCH_TOKENS // (microbatch * SEQ_LEN)
    configs = {}
    for variant in ("control", "lean"):
        cuts = {"use_subln": variant != "lean",
                "qk_norm_mode": "unit_rms" if variant == "control" else "none"}
        for phase, peak_lr, warmup, decay in (("dense", .003, 90_000_000, 300_000_000),
                                              ("qat", .001, 60_000_000, 300_000_000)):
            config = make_run_config("200m", weight_mode="dense" if phase == "dense" else "ternary",
                schedule="wsd", warmup_tokens=warmup, decay_tokens=decay,
                muon_lr=peak_lr, adamw_lr=peak_lr, norm_lr=.0002,
                gradient_accumulation_steps=accumulation, seed=1337,
                extra_model={**SHAPE, **cuts})
            configs[f"{variant}_{phase}"] = config
    sweep = json.loads(json.dumps(configs["lean_dense"]))
    sweep["training"]["gradient_accumulation_steps"] = 64
    configs["lean_sweep"] = sweep
    plan = {
        "status": "prepared_not_launched", "device": "single Azure H100 NVL (SM90)",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "model_parameters": EXPECTED_COUNTS,
        "nominal_total_tokens_per_variant": NOMINAL_TOTAL_TOKENS,
        "usable_total_tokens_per_variant": USABLE_TOTAL_TOKENS,
        "dense_tokens_per_variant": DENSE_TOKENS,
        "qat_tokens_per_variant": USABLE_TOTAL_TOKENS - DENSE_TOKENS,
        "dense_stop_after_updates": DENSE_UPDATES,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "microbatch_per_gpu": microbatch,
        "gradient_accumulation_steps": accumulation,
        "data_parallel_world_size": 1,
        "sequence_length": SEQ_LEN,
        "evaluation": "paired dense and post-QAT held-out loss/tasks at equal tokens and GPU-hours",
        "hyperparameters": "provisional; do not infer an optimum from this plan",
    }
    return configs, plan


def verify(configs, plan):
    if file_sha256(TOKENIZER) != TOKENIZER_SHA256:
        raise ValueError("frozen tokenizer SHA-256 mismatch")
    for variant in ("control", "lean"):
        dense, qat = (configs[f"{variant}_{phase}"] for phase in ("dense", "qat"))
        assert {k: v for k, v in dense["model"].items() if k != "weight_mode"} == {
            k: v for k, v in qat["model"].items() if k != "weight_mode"}
        model = BitNetLlamaConfig(**dense["model"])
        counts = count_model(model)
        if counts["total"] != EXPECTED_COUNTS[variant]:
            raise ValueError(f"{variant} parameter count changed: {counts['total']}")
        if (model.linear_backend, model.attention_backend, model.fp8_backward,
            model.fp8_quantizer, model.loss_backend, model.residual_dtype) != (
            "deepgemm", "fa3", "full", "triton_split", "triton", "bf16"):
            raise ValueError("production H100 stack changed")
        if not (model.quantize_activations and model.fused_mlp and model.fused_qkv
                and model.fused_pointwise and model.fuse_projections and model.fp8_cache_weights):
            raise ValueError("production fusion or FP8 activations disabled")
        if model.vocab_size != 65536 or model.pad_token_id != 3:
            raise ValueError("frozen tokenizer geometry changed")
    if plan["microbatch_per_gpu"] * plan["gradient_accumulation_steps"] * SEQ_LEN != GLOBAL_BATCH_TOKENS:
        raise ValueError("global batch decomposition changed")
    if DENSE_TOKENS >= USABLE_TOTAL_TOKENS or DENSE_TOKENS % GLOBAL_BATCH_TOKENS:
        raise ValueError("dense/QAT boundary does not align to optimizer updates")


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
        for name, config in configs.items():
            path = CONFIGS / f"{name}.json"
            if json.loads(path.read_text()) != config:
                raise ValueError(f"{path} differs from generated contract")
            load_config(path)
    print(json.dumps({"status": "generated" if args.command == "generate" else "verified",
                      "microbatch": args.microbatch, "global_batch_tokens": GLOBAL_BATCH_TOKENS,
                      "dense_tokens": DENSE_TOKENS, "qat_tokens": USABLE_TOTAL_TOKENS - DENSE_TOKENS,
                      "parameters": EXPECTED_COUNTS}))


if __name__ == "__main__":
    main()
