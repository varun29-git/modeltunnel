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
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

_TOKENIZER = None


def _init_tokenizer(path):
    global _TOKENIZER
    from terminus.tokenizer import load_tokenizer

    _TOKENIZER = load_tokenizer(path)


def _tokenize_chunk(job):
    (stream, split, chunk_index, records_path, start, end, records_sha256,
     tokenizer_path, out_dir, verify_mod) = job
    from terminus.tokenizer import encode_text, load_tokenizer, normalize_text
    from terminus.tokenizer65k import BYTE_FALLBACK_BASE, Frozen65kTokenizer
    from terminus.integrity import write_json_atomic
    tokenizer = _TOKENIZER or load_tokenizer(tokenizer_path)
    out_dir = Path(out_dir)
    shard = out_dir / f"{stream}.{split}.{chunk_index:05d}.bin"
    sidecar = shard.with_suffix(".json")
    identity = {"records_sha256": records_sha256, "start_byte": start,
                "end_byte": end, "tokenizer_sha256": tokenizer.sha256}
    if shard.exists() and sidecar.exists():
        prior = json.loads(sidecar.read_text())
        if all(prior.get(key) == value for key, value in identity.items()) and shard.stat().st_size == prior["tokens"] * 2:
            return stream, split, chunk_index, prior["tokens"], prior["documents"]
    tokens_written = 0
    utf8_bytes = documents = byte_fallback = 0
    temporary = shard.with_name(shard.name + f".{os.getpid()}.partial")
    with temporary.open("wb") as handle, Path(records_path).open("rb") as records:
        records.seek(start)
        while records.tell() < end:
            line = records.readline()
            if not line:
                raise EOFError(f"record chunk ended early: {records_path} at {records.tell()}")
            record = json.loads(line)
            text = normalize_text(record["text"])
            if verify_mod == 0 or int(record["uid"], 16) % verify_mod == 0:  # deterministic sample
                tokens = encode_text(tokenizer, text)  # full decode-verify
            else:
                tokens = tokenizer.encode(text)
            tokens = tokens + [tokenizer.eos_id()]
            array = np.asarray(tokens, dtype=np.uint16)
            handle.write(array.tobytes())
            tokens_written += len(tokens)
            utf8_bytes += len(text.encode("utf-8"))
            documents += 1
            if isinstance(tokenizer, Frozen65kTokenizer):
                byte_fallback += int(np.count_nonzero((array >= BYTE_FALLBACK_BASE)
                                                      & (array < BYTE_FALLBACK_BASE + 256)))
            elif hasattr(tokenizer, "is_byte"):
                byte_fallback += sum(tokenizer.is_byte(t) for t in tokens)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, shard)
    write_json_atomic(sidecar, {**identity, "tokens": tokens_written, "utf8_bytes": utf8_bytes,
                                "documents": documents, "byte_fallback_tokens": byte_fallback})
    return stream, split, chunk_index, tokens_written, documents


def chunk_ranges(paths, target_chunk_docs):
    """Scan record files once; pass byte ranges to workers instead of text."""
    chunks = []
    for path in paths:
        path = Path(path)
        digest = hashlib.sha256()
        offsets = [0]
        count = 0
        with path.open("rb") as handle:
            while line := handle.readline():
                digest.update(line)
                count += 1
                if count % target_chunk_docs == 0:
                    offsets.append(handle.tell())
            end = handle.tell()
        if offsets[-1] != end:
            offsets.append(end)
        chunks.extend((str(path), offsets[index], offsets[index + 1], digest.hexdigest())
                      for index in range(len(offsets) - 1))
    return chunks


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
    if args.workers <= 0 or args.chunk_docs <= 0 or args.verify_mod < 0:
        parser.error("workers and chunk-docs must be positive; verify-mod must be nonnegative")
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
            for index, (path, start, end, digest) in enumerate(chunk_ranges(paths, args.chunk_docs)):
                jobs.append((stream, split, index, path, start, end, digest,
                             args.tokenizer, str(out_dir), args.verify_mod))
        test_paths = splits.get("test") or []
        if test_paths:
            target = untouched / f"{stream}.test.jsonl"
            with target.open("wb") as handle:
                for path in test_paths:
                    with Path(path).open("rb") as source:
                        while chunk := source.read(8 * 1024 * 1024):
                            handle.write(chunk)
    print(f"{len(jobs)} chunks over {args.workers} workers", flush=True)
    total_tokens = total_docs = 0
    with mp.Pool(args.workers, initializer=_init_tokenizer, initargs=(args.tokenizer,)) as pool:
        for done, (stream, split, index, tokens, docs) in enumerate(
                pool.imap_unordered(_tokenize_chunk, jobs), 1):
            total_tokens += tokens
            total_docs += docs
            if done % 20 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {total_tokens/1e9:.2f}B tokens", flush=True)
    print(f"done: {total_tokens} tokens, {total_docs} documents", flush=True)


if __name__ == "__main__":
    main()
