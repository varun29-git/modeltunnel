#!/bin/bash
# Token-set pipeline for an in-region VM: corpus -> merge -> tokenize -> pack
# -> validate -> 3 nested manifests (3B/6B/10B) -> upload. Tokenization only;
# no training. Run from the repo root on the VM:
#   bash scripts/run_tokenset_vm.sh /data/work s3://quark-research-repository-storage/data/modeltunnelv2_tokensets_20260930
set -euo pipefail
WORK=${1:?work dir}
DEST=${2:?s3 destination prefix}
export PATH="$HOME/test7-cpu-venv/bin:$PATH"
PY=${PYTHON:-$HOME/test7-cpu-venv/bin/python}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

mkdir -p "$WORK"/{corpus_build,merged,packed,untouched,cache}

# 0. evaluation tasks for the contamination guard
$PY scripts/fetch_eval_tasks.py --output-dir eval

# 1. per-source corpus builds in parallel (each single-process, restartable)
SOURCES="climbmix nemotron-cc-math-4plus nemotron-cc-math-4plus-mind code-concepts scientific-coding sangraha-verified-hin sangraha-verified-mar sangraha-verified-kan sangraha-verified-tel sangraha-verified-tam sangraha-verified-guj sangraha-verified-ben fineweb2-hin fineweb2-mar fineweb2-kan fineweb2-tel fineweb2-tam fineweb2-guj fineweb2-ben"
for src in $SOURCES; do
  if [ -f "$WORK/corpus_build/$src/corpus_report.json" ]; then echo "skip $src (done)"; continue; fi
  $PY scripts/build_tokenset_corpus.py --source "$src" \
      --output-dir "$WORK/corpus_build/$src" --cache-dir "$WORK/cache" \
      > "$WORK/corpus_build_$src.log" 2>&1 &
done
wait

# 2. merge streams with cross-source exact dedup
$PY scripts/merge_tokenset_records.py --build-root "$WORK/corpus_build" \
    --output-dir "$WORK/merged" --shuffle-seed terminus-tokensets-20261001-v1

# 3. parallel tokenize (chunks are ordered; output is byte-identical to serial)
$PY scripts/tokenize_parallel.py --records "$WORK/merged/records_map.json" \
    --tokenizer tokenizer/tokenizer-65k.json --output-dir "$WORK/packed" \
    --untouched-dir "$WORK/untouched" --workers "${WORKERS:-56}"

# 4. pack + validate + nested budgets (3B and 6B are exact prefixes of 10B)
cp "$WORK/merged/licenses.json" "$WORK/packed/licenses.json"
cp "$WORK/merged/shuffle_receipt.json" "$WORK/packed/shuffle_receipt.json"
export WORK_DIR="$WORK"
$PY scripts/pack_tokensets.py

# 5. upload manifests + shards + reports (not the raw cache)
aws s3 sync "$WORK/packed" "$DEST/packed"
aws s3 cp "$WORK/merged/records_map.json" "$DEST/records_map.json"
aws s3 cp "$WORK/merged/licenses.json" "$DEST/licenses.json"
echo "uploaded to $DEST"
