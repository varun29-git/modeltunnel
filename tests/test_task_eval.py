"""Task-aware evaluation tests: masking, first output token, margins, CIs, sandbox."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from terminus.codeexec import run_program, score_code_item
from terminus.model import BitNetLlama
from terminus.task_eval import (IGNORE_INDEX, TaskItem, aggregate, bootstrap_ci,
                                choice_margins, encode_pair, group_by_length,
                                load_scalingbench_items, reduce_partials, score_item,
                                score_items, shard_items, write_partials)
from tests.helpers import tiny_config


class ToyTokenizer:
    def encode(self, text, out_type=int):
        return [ord(char) % 29 + 3 for char in text]  # ids in 3..31, deterministic

    def get_piece_size(self):
        return 32


class EncodePairTests(unittest.TestCase):
    def test_prompt_masked_and_first_output_token_scored(self):
        tokenizer = ToyTokenizer()
        inputs, targets, meta = encode_pair(tokenizer, "abcd", "xy", max_seq_len=64)
        self.assertEqual(inputs.tolist(), [[3 + (ord(c) % 29) for c in "abcdx"]])
        full = [3 + (ord(c) % 29) for c in "abcdxy"]
        self.assertEqual(inputs.tolist(), [full[:-1]])
        self.assertEqual(targets.tolist()[0][:3], [IGNORE_INDEX] * 3)
        # position 3 predicts tokens[4] = first output token 'x' and is scored
        self.assertEqual(targets.tolist()[0][3], full[4])
        self.assertEqual(targets.tolist()[0][4], full[5])
        self.assertEqual(meta["output_tokens"], 2)

    def test_empty_prompt_scores_all(self):
        tokenizer = ToyTokenizer()
        _, targets, meta = encode_pair(tokenizer, "", "abc", max_seq_len=64)
        self.assertTrue(all(t != IGNORE_INDEX for t in targets.tolist()[0]))

    def test_left_truncation_keeps_reference(self):
        tokenizer = ToyTokenizer()
        inputs, targets, meta = encode_pair(tokenizer, "p" * 100, "r" * 10, max_seq_len=40)
        self.assertEqual(inputs.shape, (1, 40))
        self.assertEqual(meta["output_tokens"], 10)
        self.assertGreater(meta["truncated_prompt_tokens"], 0)
        scored = [t for t in targets.tolist()[0] if t != IGNORE_INDEX]
        self.assertEqual(len(scored), 10)

    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            encode_pair(ToyTokenizer(), "p", "", max_seq_len=64)


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(5)
        cls.model = BitNetLlama(tiny_config(vocab_size=32, max_seq_len=64))
        cls.model.eval()
        cls.tokenizer = ToyTokenizer()

    def test_score_item_matches_manual_cross_entropy(self):
        item = TaskItem(task="toy", item_id="1", prompt="abc", reference="de")
        record = score_item(self.model, self.tokenizer, item, device=torch.device("cpu"),
                            max_seq_len=64)
        inputs, targets, meta = encode_pair(self.tokenizer, "abc", "de", max_seq_len=64)
        with torch.no_grad():
            logits = self.model(inputs)["logits"][0]
        positions = [i for i in range(targets.shape[1]) if targets[0, i] != IGNORE_INDEX]
        manual = torch.nn.functional.cross_entropy(logits[positions].float(),
                                                   targets[0, positions]).item()
        self.assertAlmostEqual(record["loss"], manual, places=6)
        self.assertEqual(record["output_tokens"], 2)
        self.assertIn("bits_per_byte", record)

    def test_answer_only_loss_is_separate(self):
        item = TaskItem(task="toy", item_id="2", prompt="q", reference="long rationale Z",
                        answer="Z")
        record = score_item(self.model, self.tokenizer, item, device=torch.device("cpu"),
                            max_seq_len=64)
        self.assertIn("answer_loss", record)
        self.assertNotEqual(record["answer_loss"], record["loss"])

    def test_choice_margins(self):
        item = TaskItem(task="toy", item_id="3", prompt="pick", choices=("aa", "bb", "cc"),
                        correct_choice=1)
        record = choice_margins(self.model, self.tokenizer, item, device=torch.device("cpu"),
                                max_seq_len=64)
        self.assertEqual(len(record["choice_losses"]), 3)
        self.assertEqual(record["correct_loss"], record["choice_losses"][1])
        self.assertIn(record["correct"], (True, False))

    def test_score_items_deterministic_and_complete(self):
        items = [TaskItem(task="toy", item_id=str(i), prompt=f"p{i}", reference=f"r{i} x")
                 for i in range(7)]
        first = score_items(self.model, self.tokenizer, items, device=torch.device("cpu"),
                            max_seq_len=64)
        second = score_items(self.model, self.tokenizer, items, device=torch.device("cpu"),
                             max_seq_len=64)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        summary = aggregate(first)
        self.assertEqual(summary["items"], 7)
        self.assertEqual(summary["per_task"]["toy"]["n"], 7)
        self.assertLessEqual(summary["per_task"]["toy"]["low"],
                             summary["per_task"]["toy"]["mean"])

    def test_sharding_and_reduction(self):
        items = [TaskItem(task="toy", item_id=f"{i:02d}", prompt=f"p{i}", reference=f"r{i}")
                 for i in range(10)]
        shards = [shard_items(items, rank=rank, world_size=3) for rank in range(3)]
        self.assertEqual(sum(len(shard) for shard in shards), 10)
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for rank, shard in enumerate(shards):
                records = score_items(self.model, self.tokenizer, shard,
                                      device=torch.device("cpu"), max_seq_len=64)
                path = Path(root) / f"part-{rank}.jsonl"
                write_partials(records, path)
                paths.append(path)
            records, summary = reduce_partials(paths)
        self.assertEqual([r["item_id"] for r in records], sorted(f"{i:02d}" for i in range(10)))
        full = score_items(self.model, self.tokenizer, items, device=torch.device("cpu"),
                           max_seq_len=64)
        self.assertEqual([r["loss"] for r in records], [r["loss"] for r in full])

    def test_bootstrap_ci_deterministic(self):
        values = [0.5, 1.0, 1.5, 2.0, 2.5]
        self.assertEqual(bootstrap_ci(values), bootstrap_ci(values))


class LoaderTests(unittest.TestCase):
    def test_scalingbench_loader(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "data.jsonl"
            rows = [{"input": f"prompt {i}", "output": f"output {i}"} for i in range(5)]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            items = load_scalingbench_items(path, task="mmlu", max_items=3)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].prompt, "prompt 0")
        self.assertEqual(items[0].role, "development")
        self.assertEqual(len({item.item_id for item in items}), 3)


class CodeExecTests(unittest.TestCase):
    def test_passing_and_failing_programs(self):
        self.assertEqual(run_program("print('ok')")["status"], "passed")
        self.assertEqual(run_program("raise SystemExit(1)")["status"], "failed")
        self.assertEqual(score_code_item("def f(x):\n    return x + 1\n",
                                         "assert f(1) == 2")["passed"], True)
        self.assertEqual(score_code_item("def f(x):\n    return x\n",
                                         "assert f(1) == 2")["passed"], False)

    def test_timeout_and_output_caps(self):
        result = run_program("while True: pass", timeout_seconds=1, cpu_seconds=1)
        self.assertIn(result["status"], ("timeout", "failed"))
        result = run_program("print('x' * 100000)")
        self.assertLessEqual(len(result["stdout"]), 4096 + 1)


if __name__ == "__main__":
    unittest.main()
