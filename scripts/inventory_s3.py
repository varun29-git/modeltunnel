"""Live, read-only inventory of the campaign S3 sources.

Lists the campaign-relevant prefixes of s3://nemotron-training-data-us-east-1
(plus explicit presence checks for named-but-maybe-absent sources), samples
the schema and first rows of one Parquet object per source family, and writes
a versioned JSON inventory. Sizes are ListObjects sums; row counts come from
one object's footer and are labeled as such. Nothing is downloaded in full
except one bounded row group per source family.

Usage: .venv/bin/python scripts/inventory_s3.py --output docs/s3_inventory/live_inventory_20260929.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from terminus.s3io import S3RangeReader, list_prefix, s3_client

BUCKET = "nemotron-training-data-us-east-1"

SANGRAHA_LANGS = {"eng": "en", "hin": "hi", "mar": "mr", "kan": "kn", "tel": "te",
                  "tam": "ta", "guj": "gu", "ben": "bn"}
FINEWEB2_LANGS = {"hin_Deva": "hi", "mar_Deva": "mr", "kan_Knda": "kn", "tel_Telu": "te",
                  "tam_Taml": "ta", "guj_Gujr": "gu", "ben_Beng": "bn"}

# (name, prefix, role). Campaign-relevant sub-prefixes only.
SOURCES = [
    ("nemotron-cc-hq", "datasets/nemotron-cc-hq/", "english_web"),
    ("climbmix-400b-shuffle", "datasets/nemotron-climbmix/karpathy-400b-shuffle/", "english_web"),
    ("smollm-corpus-snapshot", "source-snapshots/HuggingFaceTB/smollm-corpus/", "english_synthetic"),
    ("nemotron-cc-math-4plus", "datasets/nemotron-cc-math-v1/4plus/", "math"),
    ("nemotron-cc-math-4plus-mind", "datasets/nemotron-cc-math-v1/4plus_MIND/", "math"),
    ("nemotron-cc-math-3", "datasets/nemotron-cc-math-v1/3/", "math_optional"),
    ("math-textbooks", "datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Math-Textbooks/", "math"),
    ("code-concepts", "datasets/Nemotron-Pretraining-Specialized-v1.1/Nemotron-Pretraining-Code-Concepts/", "code"),
    ("scientific-coding", "datasets/Nemotron-Pretraining-Specialized-v1/Nemotron-Pretraining-Scientific-Coding/", "code"),
]
SANGRAHA_TIERS = ["verified", "synthetic", "unverified"]
TRANSLATION_RUNS = [
    ("cosmopedia-v4", "translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v4-20260905/", "older; keep ids absent from v5"),
    ("cosmopedia-v5", "translations/Cosmopedia-v2/Krutrim/krutrim-cosmopedia-v2-decoupled-v5-20260906/", "preferred"),
    ("hq-synth-v1", "translations/Nemotron-CC-v2.1/High-Quality-Synthetic/krutrim-nemotron-cc-v21-hq-synth-v1-20260906/", "recorded in the 27 Sep inventory; verify presence"),
    ("fact-seeking-r16", "translations/Nemotron-Pretraining-Specialized-v1.2/Fact-Seeking/indictrans2-en-indic-dist-200M-beam1-20260902/fact-seeking-70shard-10lang-r16-20260903/", "committed trainable run"),
]
# Named sources the map says to verify; absence must be explicit.
PRESENCE_CHECKS = [
    "datasets/CodeAlchemy/",
    "datasets/nemotron-code/",
    "datasets/finemath/",
    "datasets/SwallowMath/",
    "translation-staging/",
    "translation-benchmark/",
]


def schema_sample(uri, *, max_row_bytes=600, max_row_group_bytes=64 * 1024 * 1024):
    """Read schema + first rows of one Parquet object via ranged reads.

    Row sampling is skipped when the first row group exceeds
    max_row_group_bytes — footer/schema reads stay cheap regardless.
    """
    with S3RangeReader(uri, cache_bytes=4 * 1024 * 1024) as reader:
        pf = pq.ParquetFile(reader)
        schema = {field.name: str(field.type) for field in pf.schema_arrow}
        rows_total = pf.metadata.num_rows
        first_group_bytes = (pf.metadata.row_group(0).total_byte_size
                             if pf.metadata.num_row_groups else 0)
        result = {"schema": schema, "num_rows_in_sampled_object": rows_total,
                  "sampled_object": uri, "row_groups": pf.metadata.num_row_groups,
                  "first_row_group_bytes": first_group_bytes}
        if first_group_bytes > max_row_group_bytes:
            result["sample_rows_skipped"] = ("first row group exceeds "
                                             f"{max_row_group_bytes} bytes")
            return result
        sample_rows = []
        try:
            batch = next(pf.iter_batches(batch_size=2))
            for row in batch.to_pylist():
                sample_rows.append({key: (value[:max_row_bytes] + f"...[+{len(value) - max_row_bytes} chars]"
                                          if isinstance(value, str) and len(value) > max_row_bytes else value)
                                    for key, value in row.items()})
        except StopIteration:
            pass
        result["sample_rows"] = sample_rows
        return result


def inventory_source(client, name, prefix):
    objects = list_prefix(f"s3://{BUCKET}/{prefix}", client=client)
    entry = {"prefix": prefix, "objects": len(objects),
             "bytes": sum(o["size"] for o in objects),
             "first_key": objects[0]["key"] if objects else None,
             "last_key": objects[-1]["key"] if objects else None,
             "etags_sampled": [o["etag"] for o in objects[:3]]}
    parquet = [o for o in objects if o["key"].endswith(".parquet")]
    if parquet:
        try:
            entry["parquet_sample"] = schema_sample(f"s3://{BUCKET}/{parquet[0]['key']}", max_row_bytes=600)
        except Exception as exc:  # inventory must record, not crash
            entry["parquet_sample_error"] = repr(exc)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    client = s3_client()
    inventory = {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "bucket": BUCKET, "method": "ListObjectsV2 sums; one Parquet footer + first batch per family",
                 "sources": {}, "sangraha": {}, "fineweb2_indic": {}, "translations": {},
                 "presence_checks": {}, "excluded_prefixes": ["translation-staging/", "translation-benchmark/",
                                                              "quark-research-repository-storage TinyLlama packs"]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():  # resume: keep sections already inventoried
        previous = json.loads(out.read_text())
        for key in inventory:
            if key in previous and isinstance(previous[key], dict) and previous[key]:
                inventory[key] = previous[key]
        print(f"resuming; kept sections: {[k for k in inventory if inventory[k]]}", flush=True)

    def flush():
        out.write_text(json.dumps(inventory, indent=1, sort_keys=True, ensure_ascii=False))
        print(f"wrote {out}", flush=True)

    for name, prefix, role in SOURCES:
        if name in inventory["sources"]:
            continue
        print(f"listing {name} {prefix}", flush=True)
        entry = inventory_source(client, name, prefix)
        entry["role"] = role
        inventory["sources"][name] = entry
        flush()

    for tier in SANGRAHA_TIERS:
        for code3, code2 in SANGRAHA_LANGS.items():
            prefix = f"datasets/sangraha/{tier}/{code3}/"
            if f"{tier}/{code3}" in inventory["sangraha"]:
                continue
            print(f"listing {prefix}", flush=True)
            entry = inventory_source(client, f"sangraha-{tier}-{code3}", prefix)
            entry["language"] = code2
            inventory["sangraha"][f"{tier}/{code3}"] = entry
        flush()

    print("listing fineweb2-indic top level", flush=True)
    objects = list_prefix(f"s3://{BUCKET}/datasets/fineweb2-indic/", client=client)
    by_lang = {}
    for obj in objects:
        parts = obj["key"][len("datasets/fineweb2-indic/"):]
        lang_script = parts.split("/", 1)[0]
        slot = by_lang.setdefault(lang_script, {"objects": 0, "bytes": 0, "first_key": obj["key"]})
        slot["objects"] += 1
        slot["bytes"] += obj["size"]
    for lang_script, slot in sorted(by_lang.items()):
        slot["campaign_language"] = FINEWEB2_LANGS.get(lang_script)
        inventory["fineweb2_indic"][lang_script] = slot
    for lang_script in FINEWEB2_LANGS:
        slot = inventory["fineweb2_indic"].get(lang_script)
        if slot:
            candidates = [o["key"] for o in objects
                          if o["key"].startswith(f"datasets/fineweb2-indic/{lang_script}/")
                          and o["key"].endswith(".parquet")]
            if candidates:
                try:
                    slot["parquet_sample"] = schema_sample(f"s3://{BUCKET}/{candidates[0]}", max_row_bytes=600)
                except Exception as exc:
                    slot["parquet_sample_error"] = repr(exc)
    flush()

    for name, prefix, note in TRANSLATION_RUNS:
        if name in inventory["translations"]:
            continue
        print(f"listing {name}", flush=True)
        objects = list_prefix(f"s3://{BUCKET}/{prefix}", client=client)
        translated = [o for o in objects if o["key"].endswith("translated_documents.parquet")]
        languages = {}
        for o in translated:
            for part in o["key"].split("/"):
                if part.startswith("language="):
                    slot = languages.setdefault(part.split("=", 1)[1], {"objects": 0, "bytes": 0})
                    slot["objects"] += 1
                    slot["bytes"] += o["size"]
        entry = {"prefix": prefix, "rule": note, "objects": len(objects),
                 "bytes": sum(o["size"] for o in objects),
                 "translated_documents_parquets": len(translated),
                 "translated_bytes": sum(o["size"] for o in translated),
                 "language_partitions": languages,
                 "first_key": objects[0]["key"] if objects else None}
        if not objects:
            entry["status"] = "missing: prefix has no objects"
        if translated:
            try:
                entry["parquet_sample"] = schema_sample(f"s3://{BUCKET}/{translated[0]['key']}")
            except Exception as exc:
                entry["parquet_sample_error"] = repr(exc)
        inventory["translations"][name] = entry
        flush()

    for prefix in PRESENCE_CHECKS:
        objects = list_prefix(f"s3://{BUCKET}/{prefix}", client=client, max_keys=10)
        inventory["presence_checks"][prefix] = {"present": bool(objects),
                                                "bytes_in_first_10": sum(o["size"] for o in objects)}
    flush()
    print("done")


if __name__ == "__main__":
    main()
