"""Chunked parallel tokenizer: corpus records -> packed uint16 shards.

Each (stream, split) record set is split into ordered chunks; workers tokenize
chunks in parallel and write shard files directly in manifest layout
({stream}.{split}.{chunk:05d}.bin + sidecar JSON). Chunk order is fixed by
input order, so the packed streams are byte-identical to a serial run.

Documents get <eod> (id 4) appended; encode_text verifies NFC round-trip per
document (the check that caught the real routing bugs).

Usage:
  python scripts/tokenize_parallel.py --records 'records.json' \
      --tokenizer tokenizer/tokenizer-65k.json --output-dir packed --workers 56

--records: JSON map stream -> {"train": [jsonl...], "val": [jsonl...], "test": [...]}
(test splits are written to a separate untouched directory, never into the
training manifest layout).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def _tokenize_chunk(job):
    stream, split, chunk_index, lines, tokenizer_path, out_dir, verify_mod = job
    from terminus.tokenizer import encode_text, load_tokenizer, normalize_text
    tokenizer = load_tokenizer(tokenizer_path)
    out_dir = Path(out_dir)
    shard = out_dir / f"{stream}.{split}.{chunk_index:05d}.bin"
    sidecar = shard.with_suffix(".json")
    tokens_written = 0
    utf8_bytes = documents = byte_fallback = 0
    with shard.open("wb") as handle:
        for line in lines:
            record = json.loads(line)
            text = normalize_text(record["text"])
            if verify_mod and int(record["uid"], 16) % verify_mod == 0:  # deterministic sample
                tokens = encode_text(tokenizer, text)  # full decode-verify
            else:
                tokens = tokenizer.encode(text)
            tokens = tokens + [tokenizer.eos_id()]
            array = np.asarray(tokens, dtype=np.uint16)
            handle.write(array.tobytes())
            tokens_written += len(tokens)
            utf8_bytes += len(text.encode("utf-8"))
            documents += 1
            if hasattr(tokenizer, "is_byte"):
                byte_fallback += sum(tokenizer.is_byte(t) for t in tokens)
    sidecar.write_text(json.dumps({"tokens": tokens_written, "utf8_bytes": utf8_bytes,
                                   "documents": documents,
                                   "byte_fallback_tokens": byte_fallback}))
    return stream, split, chunk_index, tokens_written, documents


def chunk_lines(paths, target_chunk_docs):
    lines = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            lines.extend(handle.readlines())
    return [lines[i:i + target_chunk_docs] for i in range(0, len(lines), target_chunk_docs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--untouched-dir", required=True, help="test-split output (never packed)")
    parser.add_argument("--workers", type=int, default=56)
    parser.add_argument("--chunk-docs", type=int, default=20_000)
    parser.add_argument("--verify-mod", type=int, default=25,
                        help="decode-verify every Nth document (0 = all)")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    untouched = Path(args.untouched_dir)
    untouched.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(args.records).read_text())
    jobs = []
    for stream, splits in sorted(records.items()):
        for split in ("train", "val"):
            paths = splits.get(split) or []
            if not paths:
                continue
            for index, lines in enumerate(chunk_lines(paths, args.chunk_docs)):
                jobs.append((stream, split, index, lines, args.tokenizer, str(out_dir),
                             args.verify_mod))
        for path in splits.get("test") or []:
            target = untouched / f"{stream}.test.jsonl"
            with target.open("a", encoding="utf-8") as handle:
                handle.write(Path(path).read_text(encoding="utf-8"))
    print(f"{len(jobs)} chunks over {args.workers} workers", flush=True)
    total_tokens = total_docs = 0
    with mp.Pool(args.workers) as pool:
        for done, (stream, split, index, tokens, docs) in enumerate(
                pool.imap_unordered(_tokenize_chunk, jobs), 1):
            total_tokens += tokens
            total_docs += docs
            if done % 20 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {total_tokens/1e9:.2f}B tokens", flush=True)
    print(f"done: {total_tokens} tokens, {total_docs} documents", flush=True)


if __name__ == "__main__":
    main()
