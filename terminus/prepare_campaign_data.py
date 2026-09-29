"""Tokenize campaign corpus JSONL into packed, scope-aware token manifests.

Reads the record streams written by `corpus.CorpusBuilder` (or any JSONL with
the same record shape), tokenizes with the frozen 65k tokenizer, and packs
per-(source, split) token streams. Mapping of corpus splits: `train` ->
manifest train split, `dev` -> manifest val split; `test` records are never
tokenized into the training manifest (the final test set stays untouched).

Each shard keeps both `source` and `language`, so source-level mixtures
(including math/code domains) remain auditable; `integrity.validate_manifest`
enforces scope coverage, the tokenizer contract, and per-phase exposure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import TokenDataConfig, TokenManifest, TokenShardSpec, TrainingTokenRouter
from .integrity import file_sha256, manifest_scope, write_json_atomic
from .tokenizer import encode_text, load_tokenizer, tokenizer_contract


def read_records(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("uid", "doc_id", "source", "language", "license", "split", "text",
                        "text_sha256"):
                if key not in record:
                    raise ValueError(f"corpus record at line {line_number} lacks {key!r}")
            yield record


def prepare_campaign_data(record_files, tokenizer_path, output_dir, *, source_mix=None,
                          phase_source_mixes=None, total_train_tokens, seq_len=4096,
                          batch_size=1, shard_tokens=10_000_000, scope):
    """Build a token manifest from corpus record JSONL files.

    `record_files` maps source name -> JSONL path. The scope declares the
    versioned language/domain coverage enforced by validation.
    """
    if shard_tokens <= 0:
        raise ValueError("shard token budget must be positive")
    if not (source_mix or phase_source_mixes):
        raise ValueError("an explicit source_mix or phase_source_mixes is required")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("data preparation requires an empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_path)
    suffix = Path(tokenizer_path).suffix
    tokenizer_name = {".txt": "tokenizer.txt", ".json": "tokenizer.json"}.get(suffix, "tokenizer.model")
    import shutil
    shutil.copyfile(tokenizer_path, output_dir / tokenizer_name)
    with (output_dir / tokenizer_name).open("rb") as handle:
        os.fsync(handle.fileno())
    dtype = np.uint16 if tokenizer.get_piece_size() <= 65536 else np.uint32
    token_dtype = np.dtype(dtype).name

    streams, counts, indices = {}, defaultdict(int), defaultdict(int)
    totals, documents = defaultdict(int), defaultdict(int)
    shards, licenses = [], {}
    split_map = {"train": "train", "dev": "val"}
    inventory = defaultdict(lambda: {"documents": 0, "utf8_bytes": 0, "tokens": 0,
                                     "byte_fallback_tokens": 0})
    corpus_digests = {}

    def finish_shard(key):
        handle = streams.pop(key, None)
        if handle is None:
            return
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        source, language, split = key
        path = output_dir / f"{source}.{split}.{indices[key]:05d}.bin"
        shards.append(TokenShardSpec(path.name, source, split, counts[key], dtype=token_dtype,
                                     language=language, sha256=file_sha256(path)))
        indices[key] += 1
        counts[key] = 0

    try:
        for source, record_file in sorted(record_files.items()):
            digest = file_sha256(record_file)
            corpus_digests[source] = {"path": str(record_file), "sha256": digest}
            for record in read_records(record_file):
                if record["source"] != source:
                    raise ValueError(f"record source {record['source']!r} not in its file {source!r}")
                if record["split"] not in split_map:
                    if record["split"] != "test":
                        raise ValueError(f"unknown split {record['split']!r}")
                    continue  # the final test split is never tokenized into training data
                text = record["text"]
                if hashlib.sha256(text.encode()).hexdigest() != record["text_sha256"]:
                    raise ValueError("corpus record text hash mismatch")
                if record["license"] != licenses.setdefault(source, record["license"]):
                    raise ValueError("inconsistent license metadata for a source")
                language, split = record["language"], split_map[record["split"]]
                tokens = encode_text(tokenizer, text) + [tokenizer.eos_id()]
                key = (source, language, split)
                documents[key] += 1
                slot = inventory[f"{source}/{split}"]
                slot["documents"] += 1
                slot["utf8_bytes"] += len(text.encode("utf-8"))
                slot["tokens"] += len(tokens)
                slot["byte_fallback_tokens"] += sum(tokenizer.is_byte(t) for t in tokens) \
                    if hasattr(tokenizer, "is_byte") else 0
                array = np.asarray(tokens, dtype=dtype)
                start = 0
                while start < len(array):
                    if key not in streams:
                        path = output_dir / f"{source}.{split}.{indices[key]:05d}.bin"
                        streams[key] = path.open("xb")
                    take = min(len(array) - start, shard_tokens - counts[key])
                    streams[key].write(array[start:start + take].tobytes())
                    counts[key] += take
                    totals[key] += take
                    start += take
                    if counts[key] == shard_tokens:
                        finish_shard(key)
        for key in list(streams):
            finish_shard(key)
        config = TokenDataConfig(
            seq_len=seq_len, batch_size=batch_size, total_train_tokens=total_train_tokens,
            validation_tokens=sum(max(0, count - 1) for (source, language, split), count
                                  in totals.items() if split == "val"),
            token_dtype=token_dtype, source_mix=source_mix or None,
            phase_source_mixes=phase_source_mixes or [])
        contract = tokenizer_contract(tokenizer, output_dir / tokenizer_name)
        contract["path"] = tokenizer_name
        provenance = {"normalization": "NFC", "packing": "continuous_eos",
                      "document_dedup": "id_exact_minhash_v1",
                      "document_separator_id": tokenizer.eos_id(),
                      "licenses": licenses, "corpus_inputs": corpus_digests,
                      "scope": scope,
                      "token_inventory": {key: value for key, value in sorted(inventory.items())},
                      "documents": {f"{source}/{language}/{split}": count
                                    for (source, language, split), count in documents.items()}}
        manifest = TokenManifest(config, sorted(shards, key=lambda spec: spec.path),
                                 contract, provenance)
        manifest_scope(manifest)  # structural validation of the declared scope
        TrainingTokenRouter(manifest, root=output_dir)
        write_json_atomic(output_dir / "manifest.json", asdict(manifest))
        return manifest
    finally:
        for handle in streams.values():
            handle.close()


def main():
    parser = argparse.ArgumentParser(description="Tokenize campaign corpus records into packed token streams")
    parser.add_argument("--records", required=True, help="JSON map source -> records JSONL path")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-mix", help="JSON file mapping source names to mixture weights")
    parser.add_argument("--phase-mixes", help="JSON file with a phase_source_mixes list")
    parser.add_argument("--scope", required=True, help="JSON file with the versioned scope declaration")
    parser.add_argument("--total-train-tokens", required=True, type=int)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-tokens", type=int, default=10_000_000)
    args = parser.parse_args()
    records = json.loads(Path(args.records).read_text())
    scope = json.loads(Path(args.scope).read_text())
    source_mix = json.loads(Path(args.source_mix).read_text()) if args.source_mix else None
    phase_mixes = json.loads(Path(args.phase_mixes).read_text()) if args.phase_mixes else None
    prepare_campaign_data(records, args.tokenizer, args.output_dir, source_mix=source_mix,
                          phase_source_mixes=phase_mixes, total_train_tokens=args.total_train_tokens,
                          seq_len=args.seq_len, batch_size=args.batch_size,
                          shard_tokens=args.shard_tokens, scope=scope)


if __name__ == "__main__":
    main()
