"""Experiment registry, config generator, result ledger, and transfer analysis.

The registry generates exact run configs for the controlled Terminus width
family (~200M / ~700M / ~1.5B at 25 layers, head dim 128) and records every
attempt in an append-only JSONL ledger with full identity: source code hashes,
data/tokenizer/task hashes, seeds, model shape, phase policies, batch
decomposition, optimizer update count, training tokens and raw bytes, realized
mixture, peak memory, sustained tokens/s and GPU-hours. Retries are new
attempts linked by `retry_of`; a failure never rewrites an earlier receipt.

Nothing here launches training; the owner starts each stage.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .integrity import fingerprint, source_hashes, write_json_atomic

WIDTH_FAMILY = {
    "200m": {"dim": 768, "n_heads": 6, "n_kv_heads": 2, "ffn_dim": 1920},
    "700m": {"dim": 1536, "n_heads": 12, "n_kv_heads": 4, "ffn_dim": 3840},
    "1500m": {"dim": 2304, "n_heads": 18, "n_kv_heads": 6, "ffn_dim": 5760},
}
# Measured on the maintained 25-layer / 65,536-vocab Terminus architecture.
EXPECTED_PARAM_COUNTS = {"200m": 200_284_466, "700m": 700_396_082, "1500m": 1_500_334_898}

FAILURE_CLASSES = ("completed", "oom", "nonfinite_loss", "data_depleted",
                   "contract_violation", "infrastructure", "interrupted", "unknown")


def width_model_config(width, **overrides):
    if width not in WIDTH_FAMILY:
        raise ValueError(f"unknown width {width!r}; choose from {sorted(WIDTH_FAMILY)}")
    config = {"vocab_size": 65536, "n_layers": 25, "max_seq_len": 4096,
              "local_attention_window": 1024, "global_attention_every": 5,
              "attention_backend": "fa3", "weight_init": "normal", "pad_token_id": 3,
              "loss_chunk_size": 4096, "residual_init_scale": 2 ** -2.8284271247461903,
              **WIDTH_FAMILY[width], **overrides}
    return config


def count_width_parameters(width):
    from .model import BitNetLlama, BitNetLlamaConfig

    return BitNetLlama(BitNetLlamaConfig(**width_model_config(
        width, attention_backend="sdpa_reference"))).count_parameters()


def make_run_config(width, *, weight_mode, schedule, warmup_tokens, decay_tokens=None,
                    decay_style="cosine", final_lr_fraction=0.1, muon_lr, adamw_lr,
                    norm_lr=2e-4, weight_decay=0.1, muon_lr_rule="sqrt_fan_in",
                    muon_fan_in_base=None, gradient_accumulation_steps=64, seed=1337,
                    linear_backend="deepgemm", fp8_backward="full", fp8_quantizer="triton_split",
                    extra_model=None):
    model = {"quantize_activations": True, "weight_scale_layout": "blockwise",
             "linear_backend": linear_backend, "fp8_backward": fp8_backward,
             "fp8_quantizer": fp8_quantizer, "fp8_cache_weights": True,
             "loss_backend": "triton", "fused_pointwise": True, "fuse_projections": True,
             "residual_dtype": "bf16", "fused_mlp": True, "fused_qkv": True,
             "weight_mode": weight_mode}
    model.update(extra_model or {})
    return {
        "model": width_model_config(width, **model),
        "training": {"device": "cuda", "dtype": "bf16", "development_mode": False,
                     "seed": seed, "gradient_accumulation_steps": gradient_accumulation_steps,
                     "warmup_tokens": warmup_tokens, "final_lr_fraction": final_lr_fraction,
                     "schedule": schedule, "decay_tokens": decay_tokens,
                     "decay_style": decay_style, "grad_clip": 1.0,
                     "checkpoint_interval": 500, "validation_interval": 500,
                     "validation_batches_per_language": 8, "log_interval": 10,
                     "compile": False},
        "optimizer": {"kind": "muon", "muon_lr": muon_lr, "adamw_lr": adamw_lr,
                      "norm_lr": norm_lr, "weight_decay": weight_decay,
                      "muon_lr_rule": muon_lr_rule, "muon_fan_in_base": muon_fan_in_base},
    }


def embedding_fraction_report(widths=("200m", "700m", "1500m"), extra_profiles=None):
    """Embedding share of parameters per profile, with sub-200M hypotheticals.

    The frozen 65,536-token vocabulary makes the tied embedding 151M params at
    width 2304. Small proxies are dominated by the embedding, so width transfer
    measured there reflects embedding scaling more than transformer-trunk
    scaling — the reason the campaign does not add a <200M proxy.
    """
    from .model import BitNetLlama, BitNetLlamaConfig

    profiles = {name: WIDTH_FAMILY[name] for name in widths}
    profiles.update(extra_profiles or {
        "100m_hypothetical": {"dim": 512, "n_heads": 4, "n_kv_heads": 2, "ffn_dim": 1280},
        "50m_hypothetical": {"dim": 384, "n_heads": 3, "n_kv_heads": 1, "ffn_dim": 960},
    })
    report = {}
    for name, shape in profiles.items():
        model = BitNetLlama(BitNetLlamaConfig(vocab_size=65536, n_layers=25,
                                              max_seq_len=4096, local_attention_window=1024,
                                              global_attention_every=5,
                                              attention_backend="sdpa_reference",
                                              pad_token_id=3, **shape))
        counts = model.count_parameters()
        embedding = 65536 * shape["dim"]
        report[name] = {"total": counts["total"], "embedding": embedding,
                        "embedding_fraction": embedding / counts["total"]}
    return report


def git_identity(root=None):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=root).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, cwd=root).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except OSError:
        return {"commit": None, "dirty": None}


class Ledger:
    """Append-only attempt ledger with retry links and failure classes."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records = None

    def records(self):
        if self._records is None:
            self._records = []
            if self.path.exists():
                with self.path.open() as handle:
                    for line in handle:
                        if line.strip():
                            self._records.append(json.loads(line))
        return list(self._records)

    def _append(self, record):
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
        if self._records is not None:
            self._records.append(record)

    def begin_attempt(self, spec):
        """spec: experiment identity. Returns the attempt id."""
        attempt = uuid.uuid4().hex[:16]
        record = {"attempt_id": attempt, "event": "begin", "time_unix": time.time(),
                  "experiment_id": spec["experiment_id"], "stage": spec.get("stage"),
                  "kind": spec.get("kind"), "retry_of": spec.get("retry_of"),
                  "config_sha256": fingerprint(spec["config"]),
                  "config": {"model": spec["config"]["model"],
                             "training": spec["config"]["training"],
                             "optimizer": spec["config"]["optimizer"]},
                  "data_manifest_sha256": spec.get("data_manifest_sha256"),
                  "tokenizer_sha256": spec.get("tokenizer_sha256"),
                  "eval_manifest_sha256": spec.get("eval_manifest_sha256"),
                  "seed": spec["config"]["training"].get("seed"),
                  "git": git_identity(), "source": source_hashes()}
        self._append(record)
        return attempt

    def finish_attempt(self, attempt_id, *, outcome, metrics=None, failure_detail=None,
                       cost=None, realized=None, eval_refs=None):
        if outcome not in FAILURE_CLASSES:
            raise ValueError(f"outcome must be one of {FAILURE_CLASSES}")
        if outcome != "completed" and metrics is not None:
            raise ValueError("failed attempts record failure_detail, not quality metrics")
        begins = [r for r in self.records()
                  if r["attempt_id"] == attempt_id and r["event"] == "begin"]
        if not begins:
            raise ValueError(f"unknown attempt {attempt_id}")
        if any(r["attempt_id"] == attempt_id and r["event"] == "finish"
               for r in self.records()):
            raise ValueError(f"attempt {attempt_id} already finished; retries are new attempts")
        self._append({"attempt_id": attempt_id, "event": "finish", "time_unix": time.time(),
                      "outcome": outcome, "metrics": metrics, "failure_detail": failure_detail,
                      "cost": cost, "realized": realized, "eval_refs": eval_refs})

    def attempts(self):
        """Join begin/finish pairs into attempt records."""
        joined = {}
        for record in self.records():
            slot = joined.setdefault(record["attempt_id"], {"attempt_id": record["attempt_id"]})
            if record["event"] == "begin":
                slot.update(record)
            else:
                slot["outcome"] = record["outcome"]
                for key in ("metrics", "cost", "realized", "eval_refs", "failure_detail",
                            "time_unix"):
                    if record.get(key) is not None:
                        slot[key] = record[key]
        return list(joined.values())


def realized_from_events(events_path):
    """Extract realized training facts from a run's events.jsonl."""
    events = [json.loads(line) for line in Path(events_path).read_text().splitlines()
              if line.strip()]
    trains = [event for event in events if event.get("event") == "train"]
    final = events[-1] if events else {}
    peak_memory = max((event.get("peak_memory_bytes") or 0 for event in trains), default=0)
    tokens_per_second = (sum(e["tokens"] for e in trains) /
                         sum(e["elapsed"] for e in trains)) if trains and all(
                             "tokens" in e and "elapsed" in e for e in trains) else None
    return {"updates": final.get("updates"), "tokens_seen": final.get("tokens_seen"),
            "peak_memory_bytes": peak_memory or None, "sustained_tokens_per_second":
            tokens_per_second, "events": len(events)}


def pareto_frontier(attempts, *, quality="quality", cost="gpu_hours"):
    """Non-dominated (min cost, min quality-as-loss) attempt ids."""
    points = [(a["attempt_id"], a["cost"][cost], a["metrics"][quality])
              for a in attempts if a.get("outcome") == "completed"
              and a.get("cost", {}).get(cost) is not None
              and a.get("metrics", {}).get(quality) is not None]
    frontier = []
    for attempt_id, c, q in points:
        if not any((c2 <= c and q2 < q) or (c2 < c and q2 <= q)
                   for _, c2, q2 in points if _ != attempt_id):
            frontier.append(attempt_id)
    return sorted(frontier)


def spearman_ranks(first, second):
    """Spearman rank correlation between two equal-length score sequences."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("rank correlation needs equal-length sequences of >= 2")
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        index = 0
        while index < len(order):
            end = index
            while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
                end += 1
            for slot in range(index, end + 1):
                result[order[slot]] = (index + end) / 2
            index = end + 1
        return result
    a, b = ranks(first), ranks(second)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    if denominator == 0:
        raise ValueError("rank correlation is undefined for constant sequences")
    return numerator / denominator


def ranking_transfer(proxy_scores, bridge_scores, *, min_rho=0.7):
    """Whether the proxy ranking transfers to the bridge scale.

    proxy_scores/bridge_scores: equal-length loss sequences (same recipes,
    aligned order). Reports the correlation and a pass/fail at min_rho; the
    caller decides what to do with a negative result.
    """
    rho = spearman_ranks(proxy_scores, bridge_scores)
    return {"spearman_rho": rho, "min_rho": min_rho, "transfers": rho >= min_rho}
