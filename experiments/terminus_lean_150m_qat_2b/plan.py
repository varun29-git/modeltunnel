"""Frozen 2B-token ternary continuation contract for the completed dense run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terminus.integrity import file_sha256, write_json_atomic  # noqa: E402
from terminus.train import load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
DENSE_CONFIG = ROOT / "experiments/terminus_lean_150m_10b/configs/lean_dense.json"
TOKENIZER = ROOT / "tokenizer/tokenizer-65k.json"
TOKENIZER_SHA256 = "c53e0e95cd19e2954a8c7114c0a580eeb16698b153fdf84c5d80a7187d778d29"
PARENT_CHECKPOINT_SHA256 = "3d116e88048c93a0403ab2920b1680b066a9f29cbc07fa8ae1c2c67f2144684b"
PARENT_TOKENS = 9_999_998_976
PARENT_UPDATES = 25_432
TOTAL_TOKENS = 2_000_000_000
SEQ_LEN = 4096
MICROBATCH = 48
ACCUMULATION = 2
GLOBAL_BATCH_TOKENS = SEQ_LEN * MICROBATCH * ACCUMULATION
USABLE_TOKENS = TOTAL_TOKENS // SEQ_LEN * SEQ_LEN
TOTAL_UPDATES = math.ceil(USABLE_TOKENS / GLOBAL_BATCH_TOKENS)
QAT_MIX = {"en": .370, "code": .130, "math": .130,
           "bn": .053, "gu": .053, "hi": .053, "kn": .053,
           "mr": .053, "ta": .053, "te": .052}


def render() -> tuple[dict, dict]:
    dense = json.loads(DENSE_CONFIG.read_text())
    config = deepcopy(dense)
    config["model"]["weight_mode"] = "ternary"
    config["model"]["quantization_strength"] = 1.0
    for key in ("muon_lr", "adamw_lr", "norm_lr"):
        config["optimizer"][key] = dense["optimizer"][key] / 2
    config["training"].update(
        warmup_tokens=200_000_000,
        decay_tokens=400_000_000,
        checkpoint_interval=500,
        phase_start_tokens=None,
        quantization_warmup_tokens=0,
        quantization_start_strength=1.0,
    )
    plan = {
        "status": "authorized_qat_continuation",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_tokens_seen": PARENT_TOKENS,
        "parent_updates": PARENT_UPDATES,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "weight_mode": "ternary",
        "activation_quantization": "fp8_e4m3",
        "data_policy": "restart_at_fresh_source_offsets",
        "source_mix": QAT_MIX,
        "source_mix_reason": "The packed corpus has 2.890B unused tokens, but only 748M unused English; a fresh 2B QAT stage cannot retain the dense 50% English mix.",
        "nominal_qat_tokens": TOTAL_TOKENS,
        "usable_qat_tokens": USABLE_TOKENS,
        "total_qat_updates": TOTAL_UPDATES,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "microbatch": MICROBATCH,
        "gradient_accumulation_steps": ACCUMULATION,
        "data_parallel_world_size": 1,
        "schedule": {"warmup_tokens": 200_000_000, "stable_tokens": 1_400_000_000,
                     "decay_tokens": 400_000_000, "final_lr_fraction": .1},
        "peak_base_learning_rates": {key: config["optimizer"][key] for key in
                                      ("muon_lr", "adamw_lr", "norm_lr")},
        "note": "2B QAT is the requested duration. The completed dense run also had a 2B decay; twice that decay would be 4B.",
    }
    return config, plan


def verify(config: dict, plan: dict) -> None:
    if file_sha256(TOKENIZER) != TOKENIZER_SHA256:
        raise ValueError("frozen tokenizer changed")
    dense = json.loads(DENSE_CONFIG.read_text())
    if config["model"] != {**dense["model"], "weight_mode": "ternary", "quantization_strength": 1.0}:
        raise ValueError("QAT model differs from the trained dense architecture beyond weight mode")
    if config["model"]["linear_backend"] != "deepgemm" or not config["model"]["quantize_activations"]:
        raise ValueError("QAT must keep native DeepGEMM and FP8 activations")
    if any(config["optimizer"][key] != dense["optimizer"][key] / 2
           for key in ("muon_lr", "adamw_lr", "norm_lr")):
        raise ValueError("QAT peak base learning rates must be half of dense")
    if not math.isclose(sum(QAT_MIX.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("QAT source mixture must sum to one")
    if plan["total_qat_updates"] != TOTAL_UPDATES:
        raise ValueError("QAT update count changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "check"))
    args = parser.parse_args()
    config, plan = render()
    verify(config, plan)
    config_path = HERE / "configs/lean_ternary_qat.json"
    plan_path = HERE / "plan.json"
    if args.command == "generate":
        config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(config_path, config)
        write_json_atomic(plan_path, plan)
    elif json.loads(config_path.read_text()) != config or json.loads(plan_path.read_text()) != plan:
        raise ValueError("committed QAT config/plan differs from generated contract")
    load_config(config_path)
    print(json.dumps({"status": args.command, "qat_updates": TOTAL_UPDATES,
                      "qat_tokens": USABLE_TOKENS, "peak_lr": .005}))


if __name__ == "__main__":
    main()
