"""Width-transfer diagnostics for the Muon rules (not a μP claim).

The campaign compares these per-role statistics across the 200M/700M/1.5B
family to decide whether LR and recipe rankings transfer. Until a coherent
Muon-compatible μP parameterization is established, results are labeled
"empirical Muon width transfer" with stated limits.

Roles: tied embedding/readout, optional embedding projections, logical
Q/K/V and gate/up slices of fused projections (Muon sees them as separate
matrices), o/down projections, norms. Measured: parameter RMS/absmax,
residual activation RMS per block, final logit scale, gradient RMS per role,
and update-to-weight ratio after one real optimizer step on a clone.
"""

from __future__ import annotations

import copy
import math

import torch

from .logical_muon import *  # noqa: F401,F403  (kept for parity with optim path)
from .optim import build_optimizer


def parameter_slices(model):
    """Yield (role, tensor) pairs; fused projections are split logically."""
    config = model.config
    kv_width = config.n_kv_heads * config.head_dim
    for name, parameter in model.named_parameters():
        if name == "tok_embeddings.weight":
            yield "embedding_readout_tied", parameter
        elif name.startswith("embed_rot"):
            yield "embedding_projection", parameter
        elif name.endswith(".attn.qkv_proj.weight"):
            yield "attn.q", parameter[:config.dim]
            yield "attn.k", parameter[config.dim:config.dim + kv_width]
            yield "attn.v", parameter[config.dim + kv_width:]
        elif name.endswith(".ffn.gate_up_proj.weight"):
            yield "ffn.gate", parameter[:config.ffn_dim]
            yield "ffn.up", parameter[config.ffn_dim:]
        elif name.endswith(".attn.q_proj.weight"):
            yield "attn.q", parameter
        elif name.endswith(".attn.k_proj.weight"):
            yield "attn.k", parameter
        elif name.endswith(".attn.v_proj.weight"):
            yield "attn.v", parameter
        elif name.endswith(".ffn.gate_proj.weight"):
            yield "ffn.gate", parameter
        elif name.endswith(".ffn.up_proj.weight"):
            yield "ffn.up", parameter
        elif name.endswith(".attn.o_proj.weight"):
            yield "attn.o", parameter
        elif name.endswith(".ffn.down_proj.weight"):
            yield "ffn.down", parameter
        elif parameter.ndim < 2:
            yield "norm", parameter
        else:
            yield "other", parameter


def _stats(tensor):
    flat = tensor.detach().float()
    return {"rms": float(flat.square().mean().sqrt()), "absmax": float(flat.abs().max()),
            "elements": flat.numel()}


def parameter_report(model):
    roles = {}
    for role, tensor in parameter_slices(model):
        stats = _stats(tensor)
        if role in roles:
            previous = roles[role]
            total = previous["elements"] + stats["elements"]
            previous["rms"] = math.sqrt((previous["rms"] ** 2 * previous["elements"]
                                         + stats["rms"] ** 2 * stats["elements"]) / total)
            previous["absmax"] = max(previous["absmax"], stats["absmax"])
            previous["elements"] = total
        else:
            roles[role] = stats
    return roles


@torch.no_grad()
def activation_report(model, inputs):
    """Residual RMS after each block plus final logit scale."""
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            activations[name] = float(value.detach().float().square().mean().sqrt())
        return hook

    for name, module in model.named_modules():
        if name.startswith("layers.") and name.count(".") == 1:
            hooks.append(module.register_forward_hook(make_hook(name)))
    try:
        output = model(inputs)
    finally:
        for hook in hooks:
            hook.remove()
    report = {"residual_rms": activations}
    if output.get("logits") is not None:
        logits = output["logits"].detach().float()
        report["logit_rms"] = float(logits.square().mean().sqrt())
        report["logit_absmax"] = float(logits.abs().max())
    return report


def gradient_report(model, inputs, targets):
    """Per-role gradient RMS after one backward (same logical slices)."""
    model.zero_grad(set_to_none=True)
    output = model(inputs, targets=targets)
    output["loss"].backward()
    config = model.config
    kv_width = config.n_kv_heads * config.head_dim

    def slices_for(name, grad):
        if name == "tok_embeddings.weight":
            yield "embedding_readout_tied", grad
        elif name.startswith("embed_rot"):
            yield "embedding_projection", grad
        elif name.endswith(".attn.qkv_proj.weight"):
            yield "attn.q", grad[:config.dim]
            yield "attn.k", grad[config.dim:config.dim + kv_width]
            yield "attn.v", grad[config.dim + kv_width:]
        elif name.endswith(".ffn.gate_up_proj.weight"):
            yield "ffn.gate", grad[:config.ffn_dim]
            yield "ffn.up", grad[config.ffn_dim:]
        elif name.endswith(".attn.q_proj.weight"):
            yield "attn.q", grad
        elif name.endswith(".attn.k_proj.weight"):
            yield "attn.k", grad
        elif name.endswith(".attn.v_proj.weight"):
            yield "attn.v", grad
        elif name.endswith(".ffn.gate_proj.weight"):
            yield "ffn.gate", grad
        elif name.endswith(".ffn.up_proj.weight"):
            yield "ffn.up", grad
        elif name.endswith(".attn.o_proj.weight"):
            yield "attn.o", grad
        elif name.endswith(".ffn.down_proj.weight"):
            yield "ffn.down", grad
        elif grad.ndim < 2:
            yield "norm", grad
        else:
            yield "other", grad

    roles = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        for role, grad in slices_for(name, parameter.grad):
            stats = _stats(grad)
            slot = roles.setdefault(role, {"rms": 0.0, "absmax": 0.0, "elements": 0})
            total = slot["elements"] + stats["elements"]
            slot["rms"] = math.sqrt((slot["rms"] ** 2 * slot["elements"]
                                     + stats["rms"] ** 2 * stats["elements"]) / total)
            slot["absmax"] = max(slot["absmax"], stats["absmax"])
            slot["elements"] = total
    model.zero_grad(set_to_none=True)
    return roles


def update_ratio_report(model, optimizer_config, inputs, targets):
    """One real optimizer step on a clone; per-role |ΔW| rms / |W| rms."""
    clone = copy.deepcopy(model)
    optimizer = build_optimizer(clone, optimizer_config)
    output = clone(inputs, targets=targets)
    output["loss"].backward()
    slices_before = [(role, tensor.detach().clone()) for role, tensor in parameter_slices(clone)]
    slices_after = [(role, tensor) for role, tensor in parameter_slices(clone)]
    optimizer.step()
    ratios = {}
    for (role, base_tensor), (role2, tensor) in zip(slices_before, parameter_slices(clone)):
        assert role == role2
        delta = (tensor.detach() - base_tensor).float()
        base = base_tensor.float()
        slot = ratios.setdefault(role, {"num": 0.0, "den": 0.0, "elements": 0})
        slot["num"] += float(delta.square().sum())
        slot["den"] += float(base.square().sum())
        slot["elements"] += delta.numel()
    return {role: {"update_to_weight": math.sqrt(slot["num"] / max(slot["den"], 1e-30)),
                   "update_rms": math.sqrt(slot["num"] / slot["elements"])}
            for role, slot in ratios.items()}


def diagnostics_snapshot(model, optimizer_config, inputs, targets):
    """Full per-role diagnostic bundle for one checkpoint/model state."""
    return {"parameters": parameter_report(model),
            "activations": activation_report(model, inputs),
            "gradients": gradient_report(model, inputs, targets),
            "update_ratios": update_ratio_report(model, optimizer_config, inputs, targets),
            "label": "empirical Muon width transfer (not established μP)"}


def compare_across_widths(report_by_width):
    """Ratios of per-role statistics between consecutive widths.

    Values near 1 indicate scale-invariant behavior for that role; large
    deviations name the roles where transfer is failing.
    """
    widths = sorted(report_by_width)
    comparisons = {}
    for smaller, larger in zip(widths, widths[1:]):
        per_role = {}
        for role in sorted(report_by_width[smaller]["parameters"]):
            entry = {}
            for section, key in (("parameters", "rms"), ("gradients", "rms"),
                                 ("update_ratios", "update_to_weight")):
                small = report_by_width[smaller][section].get(role, {}).get(key)
                large = report_by_width[larger][section].get(role, {}).get(key)
                if small and large:
                    entry[f"{section}_{key}_ratio"] = large / small
            per_role[role] = entry
        comparisons[f"{smaller}->{larger}"] = per_role
    return comparisons
