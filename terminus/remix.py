"""Mixture overlays: a new source_mix/phase_source_mixes over prepared shards.

Search trials vary the mixture without re-tokenizing. `remix_manifest` writes
a manifest JSON next to the original (same shard files, new TokenDataConfig
mixture and budget), re-validates it, and returns the report; the resume
contract binds the new fingerprint, so a remixed trial is always a fresh run
directory, never a silent edit of an existing run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from .data import TokenManifest
from .integrity import validate_manifest, write_json_atomic


def remix_manifest(manifest_path, data_root, model_config, *, source_mix=None,
                   phase_source_mixes=None, total_train_tokens=None, batch_size=None, name,
                   development_mode=False):
    manifest_path = Path(manifest_path)
    manifest = TokenManifest.load(manifest_path)
    if not (source_mix or phase_source_mixes or batch_size or total_train_tokens):
        raise ValueError("a remix needs a new mixture, budget, or microbatch size")
    new_mix = (source_mix if source_mix is not None
               else (None if phase_source_mixes else manifest.config.source_mix))
    new_phases = (phase_source_mixes if phase_source_mixes
                  else ([] if source_mix is not None else manifest.config.phase_source_mixes))
    config = replace(manifest.config, source_mix=new_mix, phase_source_mixes=new_phases,
                     total_train_tokens=total_train_tokens or manifest.config.total_train_tokens,
                     batch_size=batch_size or manifest.config.batch_size)
    remixed = TokenManifest(config, manifest.shards, manifest.tokenizer, manifest.provenance)
    report = validate_manifest(remixed, data_root, model_config,
                               development_mode=development_mode)
    target = manifest_path.with_name(f"manifest.{name}.json")
    if target.exists():
        raise FileExistsError(f"remix {name!r} already exists; choose a new name")
    write_json_atomic(target, asdict(remixed))
    return {"path": str(target), **report}
