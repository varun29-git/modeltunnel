"""Build a token manifest from pre-tokenized uint16 shard files.

The parallel tokenizer (scripts/tokenize_parallel.py) writes per-(stream,
split) shard .bin files plus sidecar JSON inventories; this module assembles
the versioned TokenManifest (scope, mixture, budget, provenance) without
re-tokenizing, and validates it. Shards are consumed in manifest order; the
router's deterministic schedule makes a smaller-budget manifest over the same
shards an exact prefix of a larger one (nested token sets for free).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import TokenDataConfig, TokenManifest, TokenShardSpec, TrainingTokenRouter
from .integrity import file_sha256, manifest_scope, write_json_atomic
from .tokenizer import load_tokenizer, tokenizer_contract


def pack_token_streams(output_dir, tokenizer_path, *, source_mix, total_train_tokens,
                       seq_len, batch_size, scope, languages_by_stream, split_names=("train", "val")):
    """Assemble manifest.json from {stream}.{split}.{index:05d}.bin + .json sidecars."""
    output_dir = Path(output_dir)
    tokenizer = load_tokenizer(tokenizer_path)
    suffix = Path(tokenizer_path).suffix
    tokenizer_name = {".txt": "tokenizer.txt", ".json": "tokenizer.json"}.get(suffix, "tokenizer.model")
    if not (output_dir / tokenizer_name).exists():
        import shutil
        shutil.copyfile(tokenizer_path, output_dir / tokenizer_name)
    contract = tokenizer_contract(tokenizer, output_dir / tokenizer_name)
    contract["path"] = tokenizer_name

    shards = []
    inventory = {}
    for path in sorted(output_dir.glob("*.bin")):
        stream, split, _ = path.name.rsplit(".", 2)[0], None, None
        parts = path.name.split(".")
        stream, split = parts[0], parts[1]
        num_tokens = path.stat().st_size // 2
        sidecar = path.with_suffix(".json")
        side = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        shards.append(TokenShardSpec(path.name, stream, split, num_tokens, dtype="uint16",
                                     language=languages_by_stream[stream],
                                     sha256=file_sha256(path)))
        slot = inventory.setdefault(f"{stream}/{split}", {"tokens": 0, "utf8_bytes": 0,
                                                          "documents": 0, "byte_fallback_tokens": 0})
        slot["tokens"] += num_tokens
        for key in ("utf8_bytes", "documents", "byte_fallback_tokens"):
            slot[key] += side.get(key, 0)
    if not shards:
        raise ValueError("no packed shards found")
    config = TokenDataConfig(seq_len=seq_len, batch_size=batch_size,
                             total_train_tokens=total_train_tokens,
                             validation_tokens=sum(max(0, v["tokens"] - 1)
                                                   for k, v in inventory.items() if k.endswith("/val")),
                             token_dtype="uint16", source_mix=source_mix)
    provenance = {"normalization": "NFC", "packing": "continuous_eos",
                  "document_dedup": "id_exact_minhash_v1",
                  "document_separator_id": 4, "scope": scope,
                  "licenses": json.loads((output_dir / "licenses.json").read_text())
                  if (output_dir / "licenses.json").exists() else {},
                  "token_inventory": inventory,
                  "tokenized_by": "scripts/tokenize_parallel.py"}
    manifest = TokenManifest(config, shards, contract, provenance)
    manifest_scope(manifest)
    TrainingTokenRouter(manifest, root=output_dir)
    write_json_atomic(output_dir / "manifest.json", asdict(manifest))
    return manifest
