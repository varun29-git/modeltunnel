"""Build an audited QAT view of unused packed tokens; do not re-tokenize."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terminus.data import TokenManifest, TrainingTokenRouter  # noqa: E402
from terminus.integrity import fingerprint, validate_manifest, write_json_atomic  # noqa: E402
from terminus.model import BitNetLlamaConfig  # noqa: E402

from experiments.terminus_lean_150m_qat_2b.plan import (  # noqa: E402
    HERE, MICROBATCH, PARENT_CHECKPOINT_SHA256, PARENT_TOKENS,
    PARENT_UPDATES, QAT_MIX, SEQ_LEN, TOTAL_TOKENS,
)


def build_view(base: TokenManifest, dense_view: TokenManifest, parent_receipt: dict) -> tuple[TokenManifest, dict]:
    state = parent_receipt["state"]
    if (parent_receipt["sha256"] != PARENT_CHECKPOINT_SHA256
            or state["tokens_seen"] != PARENT_TOKENS or state["updates"] != PARENT_UPDATES):
        raise ValueError("parent checkpoint is not the verified 10B dense finish")
    if dense_view.shards != base.shards or dense_view.tokenizer != base.tokenizer:
        raise ValueError("the dense training view differs from the source packed corpus")
    if (dense_view.config.seq_len != SEQ_LEN or dense_view.config.batch_size != MICROBATCH
            or dense_view.config.usable_train_tokens != PARENT_TOKENS
            or dense_view.config.phase_source_mixes or dense_view.config.source_start_examples):
        raise ValueError("unexpected dense data geometry or curriculum")
    if base.config.seq_len != SEQ_LEN or set(base.config.source_mix) != set(QAT_MIX):
        raise ValueError("packed corpus lacks a QAT source")
    if parent_receipt["state"]["tokens_seen"] % SEQ_LEN:
        raise ValueError("parent training ended inside a sequence")

    dense_schedule = TrainingTokenRouter._weighted_schedule(dense_view.config.source_mix)
    dense_examples = PARENT_TOKENS // SEQ_LEN
    source_start_examples = {
        source: TrainingTokenRouter._source_count_before(source, dense_examples, dense_schedule)
        for source in sorted(QAT_MIX)
    }
    config = replace(base.config, batch_size=MICROBATCH, total_train_tokens=TOTAL_TOKENS,
                     source_mix=QAT_MIX, phase_source_mixes=[],
                     source_start_examples=source_start_examples)
    provenance = {**base.provenance, "qat_transition": {
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_tokens_seen": PARENT_TOKENS,
        "dense_manifest_sha256": fingerprint(asdict(dense_view)),
        "source_start_examples": source_start_examples,
        "fresh_token_policy": "continue_each_source_at_its_first_unused_4096_token_example",
    }}
    view = TokenManifest(config, base.shards, base.tokenizer, provenance)
    qat_schedule = TrainingTokenRouter._weighted_schedule(QAT_MIX)
    exposure = {
        source: TrainingTokenRouter._source_count_before(source, config.total_examples, qat_schedule) * SEQ_LEN
        for source in sorted(QAT_MIX)
    }
    receipt = {
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_dense_manifest_sha256": fingerprint(asdict(dense_view)),
        "source_start_examples": source_start_examples,
        "source_start_tokens": {source: examples * SEQ_LEN for source, examples in source_start_examples.items()},
        "qat_source_tokens": exposure,
        "qat_usable_tokens": config.usable_train_tokens,
        "data_retokenized": False,
    }
    return view, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--dense-manifest", required=True)
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("QAT view directory must be fresh")
    base = TokenManifest.load(args.base_manifest)
    dense_view = TokenManifest.load(args.dense_manifest)
    parent_run = Path(args.parent_run)
    parent_contract = json.loads((parent_run / "run.json").read_text())["contract"]
    if parent_contract["data"]["manifest_sha256"] != fingerprint(asdict(dense_view)):
        raise ValueError("dense view does not match the parent checkpoint contract")
    parent_receipt = json.loads((parent_run / "checkpoints/step_00025432.json").read_text())
    view, receipt = build_view(base, dense_view, parent_receipt)
    model = BitNetLlamaConfig(**json.loads((HERE / "configs/lean_ternary_qat.json").read_text())["model"])
    report = validate_manifest(view, args.data_root, model, development_mode=False)
    receipt["qat_manifest_sha256"] = report["manifest_sha256"]
    receipt["validated_shards"] = report["validated_shards"]
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "manifest.qat.json", asdict(view))
    write_json_atomic(output / "view_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
