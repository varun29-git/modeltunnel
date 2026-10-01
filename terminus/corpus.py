"""Campaign corpus construction: streaming, restartable S3 -> clean JSONL.

This module turns the raw S3 sources into versioned, deduplicated,
split-labelled JSONL record streams ("corpus manifests") that
`prepare_campaign_data` later tokenizes. Design rules:

- Read-only S3 access through `s3io` with a bounded local cache; per-source
  progress cursors make preparation restartable.
- Every record carries a stable `(source, doc_id)` identity, its language,
  its source license, and the object key/ETag it came from.
- Documents are split deterministically (train/dev/test) *before*
  tokenization by hashing the salted document identity, so downstream leakage
  audits are exact.
- Deduplication: stable-ID dedup within a source, exact NFC-text dedup across
  sources, and MinHash-band near-duplicate detection. Translation families
  resolve overlaps by version preference (v5 before v4 before salvage);
  staging and benchmark prefixes are never read.
- Benchmark/evaluation records are kept out of pretraining by the
  contamination guard (hashed shingle overlap against versioned task texts).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from .integrity import write_json_atomic
from .s3io import LocalCache, list_prefix, s3_client, split_s3_uri

CAMPAIGN_SOURCE_LANGS = {"eng": "en", "hin": "hi", "mar": "mr", "kan": "kn", "tel": "te",
                         "tam": "ta", "guj": "gu", "ben": "bn"}
FINEWEB2_SOURCE_LANGS = {"hin_Deva": "hi", "mar_Deva": "mr", "kan_Knda": "kn", "tel_Telu": "te",
                         "tam_Taml": "ta", "guj_Gujr": "gu", "ben_Beng": "bn"}


def normalize_nfc(text):
    if not isinstance(text, str) or "\x00" in text:
        raise ValueError("text must be valid Unicode without NUL")
    return unicodedata.normalize("NFC", text)


def document_uid(source, doc_id, *, salt):
    return hashlib.blake2b(f"{salt}\x00{source}\x00{doc_id}".encode(), digest_size=8).hexdigest()


def split_of_uid(uid, *, dev_per_10000=50, test_per_10000=50):
    bucket = int(uid, 16) % 10000
    if bucket < test_per_10000:
        return "test"
    if bucket < test_per_10000 + dev_per_10000:
        return "dev"
    return "train"


def word_shingles(text, *, n=13, max_shingles=None):
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    count = len(words) - n + 1
    stride = max(1, count // max_shingles) if max_shingles else 1
    return {" ".join(words[i:i + n]) for i in range(0, count, stride)}


def shingle_hashes(text, *, n=13, max_shingles=8192):
    return {hashlib.blake2b(shingle.encode(), digest_size=8).digest()
            for shingle in word_shingles(text, n=n, max_shingles=max_shingles)}


def minhash_signature(text, *, n=5, count=64, max_grams=6144):
    """Deterministic 64x32-bit MinHash over character n-grams.

    Each gram is hashed once with blake2b-64; the 64 signature values use a
    fixed linear-permutation family (a_i * h + b_i mod 2**32), vectorized.
    Long documents are sub-sampled by a deterministic stride so per-document
    cost is bounded (the same stride applies to every document, keeping
    near-duplicate comparisons consistent).
    """
    import numpy as np

    total = max(1, len(text) - n + 1)
    stride = max(1, total // max_grams)
    grams = {text[i:i + n] for i in range(0, total, stride)} or {text}
    hashes = np.frombuffer(b"".join(
        hashlib.blake2b(gram.encode(), digest_size=8).digest() for gram in grams),
        dtype=np.uint64)
    index = np.arange(count, dtype=np.uint64)
    multipliers = (2 * index + 1).astype(np.uint64)
    offsets = ((index * np.uint64(2654435761)) + np.uint64(12345)).astype(np.uint64)
    values = (hashes[:, None] * multipliers[None, :] + offsets[None, :]) & np.uint64(0xFFFFFFFF)
    return [int(v) for v in values.min(axis=0)]


class DedupStore:
    """SQLite-backed dedup: stable IDs, exact NFC text, MinHash near-dups."""

    def __init__(self, path, *, near_duplicate_bands=16, near_duplicate_threshold=0.875):
        self.database = sqlite3.connect(path)
        # WAL + NORMAL: crash-safe across process kills, far fewer fsyncs than
        # the default rollback journal under parallel writers.
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.execute("PRAGMA synchronous=NORMAL")
        self.database.execute("CREATE TABLE IF NOT EXISTS doc_ids "
                              "(source TEXT NOT NULL, doc_id TEXT NOT NULL, "
                              "PRIMARY KEY (source, doc_id))")
        self.database.execute("CREATE TABLE IF NOT EXISTS text_hashes (digest TEXT PRIMARY KEY)")
        self.database.execute("CREATE TABLE IF NOT EXISTS minhash (band INTEGER, key BLOB, uid TEXT)")
        self.database.execute("CREATE INDEX IF NOT EXISTS minhash_band_key ON minhash (band, key)")
        self.database.execute("CREATE TABLE IF NOT EXISTS signatures (uid TEXT PRIMARY KEY, signature BLOB)")
        self.bands = near_duplicate_bands
        self.threshold = near_duplicate_threshold
        self._pending = 0

    def _commit_maybe(self):
        self._pending += 1
        if self._pending % 20000 == 0:
            self.database.commit()

    def seen_id(self, source, doc_id):
        cursor = self.database.execute("INSERT OR IGNORE INTO doc_ids VALUES (?, ?)",
                                       (source, doc_id))
        self._commit_maybe()
        return cursor.rowcount == 0

    def contains_id(self, source, doc_id):
        return self.database.execute("SELECT 1 FROM doc_ids WHERE source = ? AND doc_id = ?",
                                     (source, doc_id)).fetchone() is not None

    def seen_text(self, digest):
        cursor = self.database.execute("INSERT OR IGNORE INTO text_hashes VALUES (?)", (digest,))
        self._commit_maybe()
        return cursor.rowcount == 0

    def near_duplicate(self, uid, signature):
        band_size = len(signature) // self.bands
        candidates = set()
        for band in range(self.bands):
            key = bytes().join(value.to_bytes(4, "little")
                               for value in signature[band * band_size:(band + 1) * band_size])
            for (other_uid,) in self.database.execute(
                    "SELECT uid FROM minhash WHERE band = ? AND key = ?", (band, key)):
                candidates.add(other_uid)
        for other_uid in candidates:
            row = self.database.execute("SELECT signature FROM signatures WHERE uid = ?",
                                        (other_uid,)).fetchone()
            if row is None:
                continue
            stored = [int.from_bytes(row[0][i:i + 4], "little") for i in range(0, len(row[0]), 4)]
            agreement = sum(a == b for a, b in zip(signature, stored)) / len(signature)
            if agreement >= self.threshold:
                return True
        blob = bytes().join(value.to_bytes(4, "little") for value in signature)
        self.database.execute("INSERT OR REPLACE INTO signatures VALUES (?, ?)", (uid, blob))
        self.database.executemany(
            "INSERT INTO minhash VALUES (?, ?, ?)",
            [(band, blob[band * band_size * 4:(band + 1) * band_size * 4], uid)
             for band in range(self.bands)])
        self._commit_maybe()
        return False

    def close(self):
        self.database.commit()
        self.database.close()


class ContaminationGuard:
    """Drops training records sharing a 13-gram word shingle with task texts."""

    def __init__(self, task_texts=(), *, n=13):
        self.n = n
        self.hashes = set()
        for text in task_texts:
            self.hashes |= shingle_hashes(normalize_nfc(text), n=n)
        self.flagged = 0

    def contaminated(self, text):
        if not self.hashes:
            return False
        if shingle_hashes(text, n=self.n) & self.hashes:
            self.flagged += 1
            return True
        return False


@dataclass
class SourceSpec:
    """One raw source: where it lives and how to read it."""
    name: str
    adapter: str                 # adapter registry key
    license: str
    role: str                    # 'language' | 'domain' (math/code) | 'english'
    uris: list = None            # explicit object URIs, or
    prefix: str = None           # one S3 prefix to list
    language: str = None         # fixed language for language sources
    prefer_over: tuple = ()      # translation families: ids here win over these sources
    skip_ids_from: tuple = ()    # drop doc_ids already emitted by these sources
    max_bytes: int = None        # bound for smoke samples
    max_objects: int = None
    max_rows: int = None         # bound rows read per object (streams instead of caching)
    notes: str = ""


@dataclass
class CorpusBuildConfig:
    scope_name: str
    split_salt: str
    dev_per_10000: int = 50
    test_per_10000: int = 50
    near_duplicates: bool = True
    min_text_bytes: int = 32
    max_text_bytes: int = 4 * 1024 * 1024


def sangraha_row(row, *, source, language, **context):
    text = row.get("text")
    doc_id = row.get("doc_id") or hashlib.blake2b(text[:4096].encode(), digest_size=12).hexdigest()
    return [{"doc_id": str(doc_id), "text": text, "language": language}]


def fineweb2_row(row, *, source, language, **context):
    text = row.get("text")
    doc_id = row.get("id") or row.get("doc_id") or \
        hashlib.blake2b(text[:4096].encode(), digest_size=12).hexdigest()
    return [{"doc_id": str(doc_id), "text": text, "language": language}]


def text_id_row(row, *, source, language, **context):
    """Generic adapter: text column plus any of the common id columns."""
    text = row.get("text") or row.get("content")
    for key in ("id", "doc_id", "uid", "url"):
        if row.get(key) is not None:
            doc_id = f"{key}:{row[key]}"
            break
    else:
        doc_id = hashlib.blake2b(text[:4096].encode(), digest_size=12).hexdigest()
    return [{"doc_id": str(doc_id), "text": text, "language": language or "en"}]


def translation_row(row, *, source, language, quarantined=frozenset(), **context):
    """Committed translation chunks: translation text, document_id, language
    (v5 column or the language= hive partition), quarantined ids dropped."""
    if str(row.get("document_id")) in quarantined:
        return []
    text = row.get("translation")
    raw_language = row.get("language") or language
    mapped = FINEWEB2_SOURCE_LANGS.get(raw_language, raw_language)
    return [{"doc_id": str(row.get("document_id")), "text": text, "language": mapped}]


ADAPTERS = {"sangraha": sangraha_row, "fineweb2": fineweb2_row, "text_id": text_id_row,
            "translation": translation_row}


class CorpusBuilder:
    """Streams sources to split-labelled JSONL with a resumable cursor."""

    def __init__(self, config: CorpusBuildConfig, output_dir, *, cache_dir,
                 cache_bytes=2 * 1024**3, client=None, contamination_texts=()):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = client or s3_client()
        self.cache = LocalCache(cache_dir, capacity_bytes=cache_bytes)
        self.dedup = DedupStore(self.output_dir / "dedup.sqlite")
        self.guard = ContaminationGuard(contamination_texts)
        self.cursor_path = self.output_dir / "cursor.json"
        self.cursor = (json.loads(self.cursor_path.read_text())
                       if self.cursor_path.exists() else {"completed_objects": {}, "stats": {}})
        self.stats = self.cursor["stats"]

    def _save_cursor(self):
        write_json_atomic(self.cursor_path, self.cursor)

    def _source_objects(self, spec: SourceSpec):
        if spec.uris:
            objects = []
            for uri in spec.uris:
                bucket, key = split_s3_uri(uri)
                head = self.client.head_object(Bucket=bucket, Key=key)
                objects.append({"key": key, "size": head["ContentLength"],
                                "etag": head["ETag"].strip('"'), "bucket": bucket})
            return objects
        bucket, _ = split_s3_uri(spec.prefix)
        objects = [dict(o, bucket=bucket) for o in list_prefix(spec.prefix, client=self.client)]
        objects = [o for o in objects if o["key"].endswith((".parquet", ".jsonl", ".json"))]
        parquet = [o for o in objects if o["key"].endswith(".parquet")]
        if parquet:
            objects = parquet  # skip SOURCE.json / _parts manifests / checksums
        objects = [o for o in objects if not o["key"].rsplit("/", 1)[-1].startswith("_")]
        if spec.adapter == "translation":
            # only the committed translations; never quarantine manifests or audits
            objects = [o for o in objects if o["key"].endswith("translated_documents.parquet")]
        if spec.max_objects:
            objects = objects[:spec.max_objects]
        if spec.max_bytes:
            bounded, total = [], 0
            for obj in objects:
                if total >= spec.max_bytes:
                    break
                bounded.append(obj)
                total += obj["size"]
            objects = bounded
        return objects

    def build_source(self, spec: SourceSpec, *, task_guard=None):
        adapter = ADAPTERS[spec.adapter]
        completed = self.cursor["completed_objects"].setdefault(spec.name, [])
        objects = self._source_objects(spec)
        manifest_objects = []
        for obj in objects:
            if obj["key"] in completed:
                continue
            self._process_object(spec, adapter, obj)
            completed.append(obj["key"])
            manifest_objects.append({"key": obj["key"], "size": obj["size"], "etag": obj["etag"]})
            self._save_cursor()
        return manifest_objects

    def _writer(self, source, split):
        path = self.output_dir / f"records.{source}.{split}.jsonl"
        return path.open("a", encoding="utf-8")

    def _iter_rows(self, spec, obj):
        """Yield rows, streaming large objects via ranged reads when bounded."""
        uri = f"s3://{obj['bucket']}/{obj['key']}"
        # Full downloads through the bounded cache are far faster than ranged
        # streaming for objects whose row groups are large; stream only when
        # the object cannot reasonably sit in the cache.
        stream = obj["size"] > max(256 * 1024 * 1024, self.cache.capacity_bytes // 2)
        if obj["key"].endswith(".parquet"):
            if stream:
                from .s3io import S3RangeReader

                with S3RangeReader(uri, client=self.client) as reader:
                    pf = pq.ParquetFile(reader)
                    produced = 0
                    for batch in pf.iter_batches(batch_size=2048):
                        for row in batch.to_pylist():
                            yield row
                            produced += 1
                            if spec.max_rows is not None and produced >= spec.max_rows:
                                return
            else:
                local = self.cache.fetch(uri, client=self.client, expected_size=obj["size"],
                                         expected_etag=obj["etag"])
                produced = 0
                for row in pq.read_table(local).to_pylist():
                    yield row
                    produced += 1
                    if spec.max_rows is not None and produced >= spec.max_rows:
                        return
        else:
            local = self.cache.fetch(uri, client=self.client, expected_size=obj["size"],
                                     expected_etag=obj["etag"])
            produced = 0
            with Path(local).open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    yield json.loads(line)
                    produced += 1
                    if spec.max_rows is not None and produced >= spec.max_rows:
                        return

    def _process_object(self, spec, adapter, obj):
        stats = self.stats.setdefault(spec.name, {"documents": 0, "bytes_text": 0,
                                                  "exact_dups": 0, "near_dups": 0,
                                                  "id_dups": 0, "contaminated": 0,
                                                  "superseded_by_version": 0,
                                                  "skipped_empty": 0})
        quarantined = frozenset()
        if spec.adapter == "translation":
            quarantine_key = obj["key"].rsplit("/", 1)[0] + "/quarantine_manifest.jsonl"
            try:
                local = self.cache.fetch(f"s3://{obj['bucket']}/{quarantine_key}",
                                         client=self.client)
                with Path(local).open(encoding="utf-8") as handle:
                    quarantined = frozenset(str(json.loads(line).get("document_id"))
                                            for line in handle if line.strip())
            except Exception as exc:
                stats.setdefault("quarantine_manifest_missing", 0)
                stats["quarantine_manifest_missing"] += 1
        writers = {}
        try:
            for index, row in enumerate(self._iter_rows(spec, obj)):
                docs = adapter(row, source=spec.name, language=spec.language,
                               quarantined=quarantined)
                if not docs and str(row.get("document_id")) in quarantined:
                    stats.setdefault("quarantined", 0)
                    stats["quarantined"] += 1
                for doc in docs:
                    text = doc.get("text")
                    if not isinstance(text, str) or not text.strip():
                        stats["skipped_empty"] += 1
                        continue
                    text = normalize_nfc(text)
                    size = len(text.encode("utf-8"))
                    if not self.config.min_text_bytes <= size <= self.config.max_text_bytes:
                        stats["skipped_empty"] += 1
                        continue
                    doc_id = doc["doc_id"]
                    if spec.skip_ids_from and any(
                            self.dedup.contains_id(other, doc_id) for other in spec.skip_ids_from):
                        stats["superseded_by_version"] += 1
                        continue
                    if self.dedup.seen_id(spec.name, doc_id):
                        stats["id_dups"] += 1
                        continue
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if self.dedup.seen_text(digest):
                        stats["exact_dups"] += 1
                        continue
                    uid = document_uid(spec.name, doc_id, salt=self.config.split_salt)
                    if self.config.near_duplicates:
                        signature = minhash_signature(text)
                        if self.dedup.near_duplicate(uid, signature):
                            stats["near_dups"] += 1
                            continue
                    if self.guard.contaminated(text):
                        stats["contaminated"] += 1
                        continue
                    split = split_of_uid(uid, dev_per_10000=self.config.dev_per_10000,
                                         test_per_10000=self.config.test_per_10000)
                    record = {"uid": uid, "doc_id": doc_id, "source": spec.name,
                              "language": doc.get("language") or spec.language,
                              "role": spec.role, "license": spec.license, "split": split,
                              "text": text, "origin": {"key": obj["key"], "etag": obj["etag"],
                                                       "row": index},
                              "text_sha256": digest}
                    writer = writers.get(split)
                    if writer is None:
                        writer = writers[split] = self._writer(spec.name, split)
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["documents"] += 1
                    stats["bytes_text"] += size
        finally:
            for writer in writers.values():
                writer.close()
        self.dedup.database.commit()

    def finish(self, source_specs, raw_manifest):
        self._save_cursor()
        self.dedup.close()
        report = {"scope": self.config.scope_name, "stats": self.stats,
                  "contamination_flagged": self.guard.flagged,
                  "sources": {spec.name: raw_manifest.get(spec.name) for spec in source_specs}}
        write_json_atomic(self.output_dir / "corpus_report.json", report)
        return report
