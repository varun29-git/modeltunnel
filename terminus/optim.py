from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import metadata

import torch


MUON_COMMIT = "f98f1cacc0263b04290753e32be8d498c1efc806"


@dataclass(frozen=True)
class OptimizerConfig:
    kind: str = "adamw"
    adamw_lr: float = 3e-4
    norm_lr: float = 1e-4
    qk_norm_lr: float = 1e-4
    muon_lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    momentum: float = 0.95
    muon_lr_rule: str = "sqrt_fan_in"   # rule A of the width-transfer study
    muon_fan_in_base: int | None = None  # rule B: extra fan_in_base/fan_in factor

    def __post_init__(self):
        if self.kind not in {"adamw", "muon"}:
            raise ValueError("optimizer must be adamw or muon")
        for name in ("adamw_lr", "norm_lr", "qk_norm_lr", "muon_lr", "eps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= beta < 1 for beta in self.betas) or not 0 <= self.momentum < 1:
            raise ValueError("optimizer moment coefficients must lie in [0, 1)")
        if self.muon_lr_rule not in {"sqrt_fan_in", "sqrt_fan_in_ratio"}:
            raise ValueError("muon_lr_rule must be sqrt_fan_in (rule A) or sqrt_fan_in_ratio (rule B)")
        if self.muon_lr_rule == "sqrt_fan_in_ratio":
            if (isinstance(self.muon_fan_in_base, bool) or not isinstance(self.muon_fan_in_base, int)
                    or self.muon_fan_in_base <= 0):
                raise ValueError("sqrt_fan_in_ratio requires a positive integer muon_fan_in_base")
        elif self.muon_fan_in_base is not None:
            raise ValueError("muon_fan_in_base only applies to sqrt_fan_in_ratio")


def muon_update_scale_for_param(parameter, *, rule="sqrt_fan_in", fan_in_base=None):
    """Per-group Muon LR scale.

    Rule A ("sqrt_fan_in"): 0.2 * sqrt(fan_in) — constant elementwise update
    RMS with upstream's sqrt(max(1, out/in)) factor. Rule B
    ("sqrt_fan_in_ratio"): rule A times fan_in_base/fan_in, the μP-style
    width-ratio correction the width-transfer study compares against rule A.
    Weight decay is divided by the same scale so lr*wd is rule-independent.
    """
    if parameter.ndim != 2:
        raise ValueError("Muon requires a matrix")
    scale = 0.2 * math.sqrt(parameter.shape[1])
    if rule == "sqrt_fan_in_ratio":
        if not fan_in_base:
            raise ValueError("sqrt_fan_in_ratio requires fan_in_base")
        scale *= fan_in_base / parameter.shape[1]
    elif rule != "sqrt_fan_in":
        raise ValueError("unknown Muon LR rule")
    return scale


def logical_muon_rows(name, parameter, model):
    config = model.config
    if name.endswith(".attn.qkv_proj.weight"):
        kv_width = config.n_kv_heads * config.head_dim
        rows = (config.dim, kv_width, kv_width)
    elif name.endswith(".ffn.gate_up_proj.weight"):
        rows = (config.ffn_dim, config.ffn_dim)
    else:
        return None
    if parameter.ndim != 2 or sum(rows) != parameter.shape[0]:
        raise ValueError(f"Muon logical projection rows do not match {name}")
    return rows


def build_optimizer(model, config=None):
    config = config or OptimizerConfig()
    matrices = {id(module.weight) for _, module in model.bitlinear_named_modules()}
    for name in ("embed_rot_in", "embed_rot_out"):
        module = getattr(model, name)
        if hasattr(module, "weight"):
            matrices.add(id(module.weight))
    groups, logical_rows = [], []
    adam_matrices, norms, qk_norms = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if config.kind == "muon" and id(parameter) in matrices:
            scale = muon_update_scale_for_param(parameter, rule=config.muon_lr_rule,
                                                fan_in_base=config.muon_fan_in_base)
            groups.append({"params": [parameter], "lr": config.muon_lr * scale,
                           "weight_decay": config.weight_decay / scale,
                           "momentum": config.momentum, "use_muon": True})
            logical_rows.append(logical_muon_rows(name, parameter, model))
        elif name.endswith((".attn.q_norm.weight", ".attn.k_norm.weight")):
            if parameter.ndim != 1:
                raise ValueError("learned Q/K norm gain must be a channel vector")
            qk_norms.append(parameter)
        elif parameter.ndim < 2:
            norms.append(parameter)
        else:
            adam_matrices.append(parameter)
    for parameters, learning_rate, decay in ((adam_matrices, config.adamw_lr, config.weight_decay),
                                            (norms, config.norm_lr, 0.0),
                                            (qk_norms, config.qk_norm_lr, 0.0)):
        if parameters:
            group = {"params": parameters, "lr": learning_rate, "weight_decay": decay, "betas": tuple(config.betas), "eps": config.eps}
            if parameters is qk_norms:
                group["name"] = "qk_norm"
            if config.kind == "muon":
                group["use_muon"] = False
            groups.append(group)
            logical_rows.append(None)
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(grouped) != len(set(grouped)) or set(grouped) != trainable:
        raise ValueError("optimizer must cover each trainable parameter exactly once")
    if config.kind == "muon":
        distribution = metadata.distribution("muon-optimizer")
        provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
        if provenance.get("vcs_info", {}).get("commit_id") != MUON_COMMIT:
            raise RuntimeError("Muon must match the pinned commit; its shape scaling is part of the optimizer contract")
        from .logical_muon import LogicalMuonWithAuxAdam

        optimizer = LogicalMuonWithAuxAdam(groups, logical_rows)
    else:
        use_cuda = all(parameter.is_cuda for parameter in model.parameters())
        optimizer = torch.optim.AdamW(groups, fused=use_cuda, foreach=not use_cuda)
    for index, group in enumerate(optimizer.param_groups):
        group.setdefault("name", f"{config.kind}_{index}")
        group["base_lr"] = group["lr"]
    return optimizer
