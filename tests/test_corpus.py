"""Tests for the campaign corpus pipeline (dedup, splits, contamination, resume)."""

from __future__ import annotations

import io
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from terminus.corpus import (ContaminationGuard, CorpusBuildConfig, CorpusBuilder,
                             DedupStore, SourceSpec, document_uid, minhash_signature,
                             normalize_nfc, split_of_uid)


class FakeBody:
    def __init__(self, data):
        self._io = io.BytesIO(data)

    def read(self, n=-1):
        return self._io.read(n)


class FakeS3Client:
    """Serves parquet/jsonl bytes from a {key: bytes} mapping."""

    def __init__(self, objects):
        self.objects = dict(objects)

    def list_objects_v2(self, Bucket, Prefix, **kwargs):
        contents = [{"Key": key, "Size": len(data), "ETag": f'"etag-{key}"'}
                    for key, data in sorted(self.objects.items()) if key.startswith(Prefix)]
        return {"Contents": contents, "IsTruncated": False}

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key]), "ETag": f'"etag-{Key}"'}

    def get_object(self, Bucket, Key, Range=None):
        data = self.objects[Key]
        if Range:
            span = Range.split("=", 1)[1]
            start, _, end = span.partition("-")
            data = data[int(start): int(end) + 1 if end else len(data)]
        return {"Body": FakeBody(data), "ETag": f'"etag-{Key}"'}


def make_parquet(rows):
    table = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def build_config(**changes):
    base = dict(scope_name="test-scope-v1", split_salt="test-salt", dev_per_10000=50,
                test_per_10000=50, near_duplicates=True)
    base.update(changes)
    return CorpusBuildConfig(**base)


class SmallFunctionTests(unittest.TestCase):
    def test_split_is_deterministic_and_partitioned(self):
        splits = [split_of_uid(document_uid("src", f"doc-{i}", salt="s")) for i in range(20000)]
        self.assertEqual(splits, [split_of_uid(document_uid("src", f"doc-{i}", salt="s"))
                                  for i in range(20000)])
        self.assertLess(abs(splits.count("dev") / 20000 - 0.005), 0.003)
        self.assertLess(abs(splits.count("test") / 20000 - 0.005), 0.003)

    def test_split_salt_changes_assignment(self):
        a = split_of_uid(document_uid("src", "doc", salt="a"))
        b = split_of_uid(document_uid("src", "doc", salt="b"))
        # not a guarantee per-document, but the uid must differ
        self.assertNotEqual(document_uid("src", "doc", salt="a"),
                            document_uid("src", "doc", salt="b"))

    def test_minhash_detects_near_duplicates_only(self):
        text = " ".join(f"sentence {i} has distinct words alpha{i} beta{i} gamma{i}"
                        for i in range(120))
        near = text.replace("alpha60", "omega60")  # a fraction of one percent of shingles
        far = " ".join(f"unrelated topic {i} zeta{i} eta{i} theta{i}" for i in range(120))
        with tempfile.TemporaryDirectory() as root:
            store = DedupStore(Path(root) / "d.sqlite")
            self.assertFalse(store.near_duplicate("a", minhash_signature(text)))
            self.assertTrue(store.near_duplicate("b", minhash_signature(near)))
            self.assertFalse(store.near_duplicate("c", minhash_signature(far)))
            store.close()

    def test_contamination_guard_flags_shingle_overlap(self):
        task = " ".join(f"w{i}" for i in range(100))
        guard = ContaminationGuard([task])
        self.assertTrue(guard.contaminated("prefix " + task + " suffix"))
        self.assertFalse(guard.contaminated("totally different words " * 20))
        self.assertEqual(guard.flagged, 1)

    def test_normalize_rejects_nul(self):
        with self.assertRaises(ValueError):
            normalize_nfc("bad\x00text")


class CorpusBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        def doc_text(i):
            return " ".join(f"doc{i}sentence{j}word{k}unique{i}x{j}y{k}"
                            for j in range(15) for k in range(4))
        rows = [{"doc_id": f"doc-{i}", "text": doc_text(i), "type": "web"} for i in range(12)]
        rows.append({"doc_id": "doc-3", "text": "different text same id " * 30, "type": "web"})
        rows.append({"doc_id": "doc-x", "text": rows[4]["text"], "type": "web"})
        near = dict(rows[6])
        near["doc_id"] = "doc-near"
        near["text"] = near["text"].replace("unique6x10", "altered9x10", 1)  # near-dup of doc-6
        rows.append(near)
        self.objects = {
            "prefix/a.parquet": make_parquet(rows[:7]),
            "prefix/b.parquet": make_parquet(rows[7:]),
        }
        self.client = FakeS3Client(self.objects)
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def make_builder(self, **changes):
        return CorpusBuilder(build_config(**changes), self.root / "out",
                             cache_dir=self.root / "cache", client=self.client)

    def spec(self):
        return SourceSpec(name="fixture", adapter="sangraha", license="CC-BY-4.0",
                          role="language", uris=None,
                          prefix="s3://bucket/prefix/", language="hi")

    def test_build_dedups_and_splits(self):
        builder = self.make_builder()
        objects = builder.build_source(self.spec())
        self.assertEqual(len(objects), 2)
        stats = builder.stats["fixture"]
        self.assertEqual(stats["id_dups"], 1)      # doc-3 repeated id
        self.assertEqual(stats["exact_dups"], 1)   # doc-x repeats doc-4 text
        self.assertEqual(stats["near_dups"], 1)    # doc-near
        self.assertEqual(stats["documents"], 12)
        builder.finish([self.spec()], {"fixture": {"objects": 2}})
        out = self.root / "out"
        splits = set()
        for path in out.glob("records.fixture.*.jsonl"):
            split = path.name.split(".")[2]
            splits.add(split)
            for line in path.read_text().splitlines():
                record = json.loads(line)
                self.assertEqual(record["split"], split)
                self.assertEqual(record["language"], "hi")
                self.assertEqual(record["license"], "CC-BY-4.0")
        self.assertIn("train", splits)
        report = json.loads((out / "corpus_report.json").read_text())
        self.assertEqual(report["stats"]["fixture"]["documents"], 12)

    def test_resume_skips_completed_objects(self):
        spec = self.spec()
        builder = self.make_builder()
        # complete only the first object, simulating an interrupted run
        builder._process_object(spec, __import__("terminus.corpus", fromlist=["ADAPTERS"]).ADAPTERS["sangraha"],
                                {"key": "prefix/a.parquet", "size": len(self.objects["prefix/a.parquet"]),
                                 "etag": "etag-prefix/a.parquet", "bucket": "bucket"})
        builder.cursor["completed_objects"].setdefault("fixture", []).append("prefix/a.parquet")
        builder._save_cursor()
        builder.dedup.close()
        builder2 = self.make_builder()
        builder2.build_source(spec)
        self.assertEqual(builder2.stats["fixture"]["documents"], 12)
        builder2.dedup.close()
        # records were appended once per object, not twice
        total = 0
        for path in (self.root / "out").glob("records.fixture.*.jsonl"):
            total += len(path.read_text().splitlines())
        self.assertEqual(total, 12)

    def test_decoded_text_cap_stops_mid_object_and_resumes_without_duplication(self):
        spec = replace(self.spec(), max_output_text_bytes=100,
                       object_selection_seed="sample-v1")
        builder = self.make_builder(near_duplicates=False)
        selected = [obj["key"] for obj in builder._source_objects(spec)]
        self.assertEqual(selected, sorted(self.objects, key=lambda key: hashlib.blake2b(
            f"sample-v1\0fixture\0{key}".encode(), digest_size=16).digest()))
        builder.build_source(spec)
        self.assertEqual(builder.stats["fixture"]["documents"], 1)
        report = builder.finish([spec], {})
        self.assertEqual(report["partial_objects"]["fixture"]["reason"],
                         "max_output_text_bytes")
        builder2 = self.make_builder(near_duplicates=False)
        builder2.build_source(spec)
        self.assertEqual(builder2.stats["fixture"]["documents"], 1)
        builder2.dedup.close()


if __name__ == "__main__":
    unittest.main()
