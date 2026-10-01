"""After data is delivered, derive one full dense view without retokenizing.

This script is intentionally not part of plan.py: it reads prepared token shards
and must only be run after the owner declares the data ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terminus.data import TokenManifest  # noqa: E402
from terminus.integrity import (CAMPAIGN_DOMAINS, CAMPAIGN_LANGUAGES, CAMPAIGN_SCOPE_V1,
                                manifest_scope, validate_manifest, write_json_atomic)  # noqa: E402
from terminus.model import BitNetLlamaConfig  # noqa: E402

from plan import (HERE, NOMINAL_TOTAL_TOKENS, SEQ_LEN,
                  TOKENIZER_SHA256)  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    base = TokenManifest.load(args.base_manifest)
    plan = json.loads((HERE / "plan.json").read_text())
    if base.tokenizer.get("sha256") != TOKENIZER_SHA256:
        raise ValueError("prepared data does not use the frozen Terminus tokenizer")
    if base.config.seq_len != SEQ_LEN or not base.config.source_mix or base.config.phase_source_mixes:
        raise ValueError("this dense run requires 4096-token sequences and one fixed source mix")
    if base.config.cosmopedia_stories_prefix_tokens:
        raise ValueError("this dense run requires no separate prefix curriculum")
    scope = manifest_scope(base)
    if (scope["name"] != CAMPAIGN_SCOPE_V1 or set(scope["languages"]) != set(CAMPAIGN_LANGUAGES)
            or set(scope["domains"]) != set(CAMPAIGN_DOMAINS)):
        raise ValueError("prepared data scope differs from the eight-language campaign")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty output directory for immutable manifest views")
    model = BitNetLlamaConfig(**json.loads((HERE / "configs/lean_dense.json").read_text())["model"])
    config = replace(base.config, batch_size=plan["microbatch_per_gpu"], total_train_tokens=NOMINAL_TOTAL_TOKENS)
    view = TokenManifest(config, base.shards, base.tokenizer, base.provenance)
    report = validate_manifest(view, args.data_root, model, development_mode=False)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "manifest.dense.json", asdict(view))
    print(json.dumps({"view": "dense", "manifest_sha256": report["manifest_sha256"],
                      "usable_train_tokens": view.config.usable_train_tokens,
                      "batch_size": view.config.batch_size}))


if __name__ == "__main__":
    main()
