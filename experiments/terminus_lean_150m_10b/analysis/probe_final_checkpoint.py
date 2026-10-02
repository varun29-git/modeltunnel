"""Run a fixed, reproducible completion probe on the dense-FP8 checkpoint.

This is qualitative probing, not a downstream benchmark. The CPU reference
prequantizes each dense projection to the run's blockwise FP8 E4M3 weight
representation and quantizes its activations in BitLinear. It uses BF16
autocast and SDPA reference attention. It does not execute DeepGEMM/FA3.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from terminus.checkpoint import load_checkpoint  # noqa: E402
from terminus.generate import generate  # noqa: E402
from terminus.model import BitNetLlama  # noqa: E402
from terminus.quantization import (  # noqa: E402
    dequantize_weight_fp8_e4m3_2d,
    quantize_weight_dense_fp8_e4m3_2d,
)
from terminus.tokenizer65k import Frozen65kTokenizer  # noqa: E402
from terminus.train import load_config  # noqa: E402


PROMPTS = [
    ("english_fact", "France capital cloze", "The capital of France is"),
    ("english_fact", "India capital cloze", "The capital of India is"),
    ("english_fact", "Earth orbit cloze", "The Earth revolves around the"),
    ("english_fact", "Freezing point cloze", "Water freezes at"),
    ("english_qa", "France capital QA", "Q: What is the capital of France?\nA:"),
    ("english_qa", "Week days QA", "Q: How many days are there in a week?\nA:"),
    ("math", "Two plus two", "2 + 2 ="),
    ("math", "Twelve plus nineteen", "12 + 19 ="),
    ("math", "Square root", "The square root of 81 is"),
    ("math", "Few-shot arithmetic", "Question: What is 2 + 3?\nAnswer: 5\n\nQuestion: What is 7 + 4?\nAnswer: 11\n\nQuestion: What is 12 + 19?\nAnswer:"),
    ("code", "Add function", "def add(a, b):\n    "),
    ("code", "Parity function", "def is_even(n):\n    "),
    ("code", "Factorial continuation", "def factorial(n):\n    if n <= 1:\n        return 1\n    "),
    ("english_prose", "Story continuation", "Once upon a time, a small fox entered a forest and"),
    ("english_prose", "Internet history", "The history of the Internet began with"),
    ("hindi", "India capital Hindi", "भारत की राजधानी है"),
    ("hindi", "Maharashtra capital Hindi", "भारत की राजधानी नई दिल्ली है। महाराष्ट्र की राजधानी"),
    ("marathi", "Maharashtra capital Marathi", "महाराष्ट्राची राजधानी"),
    ("bengali", "Bangladesh capital Bengali", "বাংলাদেশের রাজধানী"),
    ("kannada", "India capital Kannada", "ಭಾರತದ ರಾಜಧಾನಿ"),
    ("gujarati", "India capital Gujarati", "ભારતની રાજધાની"),
    ("tamil", "India capital Tamil", "இந்தியாவின் தலைநகரம்"),
    ("telugu", "India capital Telugu", "భారతదేశ రాజధాని"),
]


def decode_output(tokenizer, token_ids):
    try:
        return tokenizer.decode(token_ids, render_specials=True), None
    except UnicodeDecodeError as exc:
        # Preserve the failure as evidence; an incomplete UTF-8 tail should
        # not crash the whole fixed suite.
        return None, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    config, _, _ = load_config(args.run_root / "config.json")
    contract = json.loads((args.run_root / "dense/run.json").read_text())["contract"]
    checkpoint = args.run_root / "dense/checkpoints/step_00025432.pt"
    payload = load_checkpoint(checkpoint, contract)
    reference = replace(
        config, attention_backend="sdpa_reference", linear_backend="torch",
        fp8_backward="bf16", fp8_quantizer="torch", fp8_cache_weights=False,
        loss_backend="checkpoint", fused_pointwise=False, fused_mlp=False,
        fused_qkv=False,
    )
    model = BitNetLlama(reference)
    model.load_state_dict(payload["model"])
    model.eval()
    with torch.no_grad():
        for _, layer in model.bitlinear_named_modules():
            codes, scales = quantize_weight_dense_fp8_e4m3_2d(layer.weight)
            layer.weight.copy_(dequantize_weight_fp8_e4m3_2d(codes, scales))
    tokenizer = Frozen65kTokenizer(args.tokenizer)

    results = []
    for index, (category, name, prompt) in enumerate(PROMPTS):
        input_ids = [tokenizer.eod_id()] + tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        started_prompt = time.monotonic()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = generate(model, input_tensor, max_new_tokens=args.max_new_tokens,
                              eos_token_id=tokenizer.eod_id(), temperature=0)
        output_ids = result[0, len(input_ids):].tolist()
        completion, decode_error = decode_output(tokenizer, output_ids)
        entry = {
            "index": index + 1, "category": category, "name": name, "prompt": prompt,
            "prompt_tokens": len(input_ids), "generated_tokens": len(output_ids),
            "ended_with_eod": bool(output_ids and output_ids[-1] == tokenizer.eod_id()),
            "completion": completion, "decode_error": decode_error,
            "seconds": round(time.monotonic() - started_prompt, 3),
        }
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    receipt = {
        "checkpoint_file": checkpoint.name, "checkpoint_sha256":
            json.loads(checkpoint.with_suffix(".json").read_text())["sha256"],
        "checkpoint_updates": payload["state"]["updates"],
        "checkpoint_tokens": payload["state"]["tokens_seen"],
        "tokenizer_sha256": tokenizer.sha256,
        "device": "cpu", "arithmetic": "FP8 E4M3 projection reference; BF16 autocast",
        "attention": "SDPA reference", "decoding": "greedy, temperature=0",
        "max_new_tokens": args.max_new_tokens,
        "wall_seconds": round(time.time() - started, 3),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
