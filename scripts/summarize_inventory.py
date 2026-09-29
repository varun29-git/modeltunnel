"""Render the live S3 inventory JSON as a markdown report with capacity estimates.

Token estimates use measured tokens-per-byte fertility from the bounded sample
receipt when available (pass --fertility), otherwise the freeze note's Gate 3
measurements for en/code/math and a marked assumption for Indic. Every estimate
is labeled with its basis; measured counts stay distinct from guesses.

Usage:
  .venv/bin/python scripts/summarize_inventory.py \
      --inventory docs/s3_inventory/live_inventory_20260929.json \
      --fertility sample/fertility.json --output docs/s3_inventory/live_inventory_20260929.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GIB = 2 ** 30
# Freeze-note Gate 3 (measured on the frozen tokenizer's own held-out files):
GATE3_BYTES_PER_TOKEN = {"en": 2430216382 / 1060557 / (117091 / 82147),  # derived below
                         }
FALLBACK_TOKENS_PER_BYTE = {"en": 1 / 3.55, "code": 1 / 3.44, "math": 1 / 3.42,
                            "indic": 1 / 1.9}  # Indic: ASSUMPTION (see report)


def gib(n):
    return n / GIB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--fertility", help="JSON with measured tokens_per_byte by key")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inv = json.loads(Path(args.inventory).read_text())
    fertility = json.loads(Path(args.fertility).read_text()) if args.fertility else {}
    tpb = dict(FALLBACK_TOKENS_PER_BYTE)
    basis = "freeze-note Gate 3 (en/code/math) + unmeasured assumption (Indic)"
    if fertility:
        tpb.update(fertility["tokens_per_byte"])
        basis = fertility.get("basis", "measured on the bounded sample")

    lines = ["# Live S3 inventory — {}".format(inv["generated_at"]), "",
             "Bucket: `s3://{}` (read-only ListObjects + one Parquet footer/row sample per family).".format(inv["bucket"]),
             "Sizes are ListObjects byte sums; document/token counts are estimates labeled by basis.",
             "Token estimate basis: {}.".format(basis), "",
             "## English / math / code sources", "",
             "| source | prefix | objects | GiB | est. tokens (B) | schema text/id columns |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for name, e in sorted(inv["sources"].items()):
        ps = e.get("parquet_sample") or {}
        schema = ps.get("schema") or {}
        cols = f"text={'text' if 'text' in schema else '?'} id={'id' if 'id' in schema else ('doc_id' if 'doc_id' in schema else '-')}"
        role = e.get("role", "")
        key = "math" if role.startswith("math") else "code" if role == "code" else "en"
        est = e["bytes"] * tpb[key] / 1e9
        lines.append(f"| {name} | `{e['prefix']}` | {e['objects']} | {gib(e['bytes']):.1f} | "
                     f"{est:.1f} ({'measured-fertility' if fertility else 'assumed'}) | {cols} |")
    lines += ["", "## Sangraha (campaign languages only)", "",
              "| tier/lang | objects | GiB | est. tokens (B) |", "| --- | ---: | ---: | ---: |"]
    for key, e in sorted(inv["sangraha"].items()):
        est = e["bytes"] * tpb["indic"] / 1e9
        if e["language"] == "en":
            est = e["bytes"] * tpb["en"] / 1e9
        lines.append(f"| {key} ({e['language']}) | {e['objects']} | {gib(e['bytes']):.1f} | {est:.1f} |")
    lines += ["", "## FineWeb2-Indic (campaign languages)", "",
              "| partition | lang | objects | GiB | est. tokens (B) |", "| --- | --- | ---: | ---: | ---: |"]
    for key, e in sorted(inv["fineweb2_indic"].items()):
        if e.get("campaign_language"):
            lines.append(f"| {key} | {e['campaign_language']} | {e['objects']} | {gib(e['bytes']):.1f} "
                         f"| {e['bytes'] * tpb['indic'] / 1e9:.1f} |")
    lines += ["", "## Committed translations (staging excluded)", "",
              "| run | objects | translated GiB | language partitions (GiB) |",
              "| --- | ---: | ---: | --- |"]
    for name, e in sorted(inv["translations"].items()):
        parts = ", ".join(f"{k}: {gib(v['bytes']):.1f}" for k, v in sorted(e.get("language_partitions", {}).items()))
        status = e.get("status", "")
        lines.append(f"| {name} | {e['translated_documents_parquets']} | "
                     f"{gib(e['translated_bytes']):.1f} | {parts or status} |")
    lines += ["", "## Presence checks (named sources)", ""]
    for prefix, check in sorted(inv["presence_checks"].items()):
        lines.append(f"- `{prefix}`: {'present' if check['present'] else 'ABSENT'}")
    lines += ["", "Excluded from training: `translation-staging/`, `translation-benchmark/`, "
              "TinyLlama-tokenized packs in `quark-research-repository-storage` (incompatible token IDs).", ""]
    Path(args.output).write_text("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
