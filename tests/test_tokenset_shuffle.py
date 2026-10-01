"""The token-set merge must mix documents without losing reproducibility."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_tokenset_records.py"


def _record(source, index, split="train"):
    text = f"{source} document {index}"
    return {"uid": hashlib.sha256(f"{source}:{index}".encode()).hexdigest()[:16],
            "source": source, "text": text, "split": split,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def _run_merge(build, output, seed):
    subprocess.run([sys.executable, str(SCRIPT), "--build-root", str(build),
                    "--output-dir", str(output), "--allow-missing",
                    "--shuffle-seed", seed], check=True, capture_output=True, text=True)
    return [json.loads(line) for line in (output / "records.math.train.jsonl").read_text().splitlines()]


def test_train_shuffle_is_seeded_lossless_and_cross_source(tmp_path):
    build = tmp_path / "build"
    names = ("nemotron-cc-math-4plus", "nemotron-cc-math-4plus-mind")
    for source in names:
        folder = build / source
        folder.mkdir(parents=True)
        (folder / "corpus_report.json").write_text(json.dumps({
            "sources": {source: {"license": "test"}}, "stats": {source: {"documents": 100}},
        }))
        records = [_record(source, index) for index in range(100)]
        (folder / f"records.{source}.train.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records))
        (folder / f"records.{source}.dev.jsonl").write_text(
            json.dumps(_record(source, 100, "dev")) + "\n")
    first = _run_merge(build, tmp_path / "out1", "seed-1")
    repeated = _run_merge(build, tmp_path / "out2", "seed-1")
    changed = _run_merge(build, tmp_path / "out3", "seed-2")
    assert first == repeated
    assert [record["uid"] for record in first] != [record["uid"] for record in changed]
    assert len(first) == len({record["uid"] for record in first}) == 200
    assert {record["source"] for record in first[:20]} == set(names)
    assert [record["uid"] for record in first] == sorted(
        (record["uid"] for record in first),
        key=lambda uid: hashlib.blake2b(f"seed-1\0math\0{uid}".encode(), digest_size=16).digest())
    assert (tmp_path / "out1" / "records.math.val.jsonl").read_bytes() == (
        tmp_path / "out3" / "records.math.val.jsonl").read_bytes()
    receipt = json.loads((tmp_path / "out1" / "shuffle_receipt.json").read_text())
    assert receipt["train"]["seed"] == "seed-1"
