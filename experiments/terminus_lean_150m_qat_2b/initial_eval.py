"""Measure held-out loss immediately after dense-to-ternary conversion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terminus.checkpoint import load_checkpoint  # noqa: E402
from terminus.data import TokenManifest  # noqa: E402
from terminus.integrity import write_json_atomic  # noqa: E402
from terminus.model import BitNetLlama, BitNetLlamaConfig  # noqa: E402
from terminus.train import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("native QAT initial evaluation requires CUDA")
    contract = json.loads((args.run_dir / "run.json").read_text())["contract"]
    checkpoint = load_checkpoint(args.run_dir / "checkpoints/step_00000000.pt", contract)
    model = BitNetLlama(BitNetLlamaConfig(**contract["model"])).to("cuda")
    model.load_state_dict(checkpoint["model"])
    model.clear_fp8_weight_cache()
    del checkpoint
    result = evaluate(model, TokenManifest.load(args.manifest), args.data_root,
                      torch.device("cuda"), torch.bfloat16, 8, 0, 1)
    payload = {"event": "pretraining_ternary_conversion_validation", "updates": 0,
               "tokens_seen": 0, "languages": result}
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
