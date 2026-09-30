"""Opt-in, sampled numerical telemetry for a single training run.

The probes observe the first microbatch of selected optimizer updates. They do
not change model outputs, gradients, or RNG state. Full-update loss, gradient
norm and memory are recorded separately by the trainer.
"""

from __future__ import annotations

import math

import torch

from .model import BitLinear
from .quantization import dequantize_activation_fp8_e4m3, quantize_activation_fp8_e4m3


def _stats(values):
    values = values.detach().float()
    finite = torch.isfinite(values)
    safe = torch.where(finite, values, torch.zeros_like(values))
    denominator = finite.sum().clamp_min(1)
    return {"rms": (safe.square().sum() / denominator).sqrt(),
            "max_abs": safe.abs().amax(), "nonfinite_fraction": 1 - finite.float().mean()}


def _numbers(value):
    tensors = []

    def collect(item):
        if isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, torch.Tensor):
            tensors.append(item.detach().float().reshape(()))

    collect(value)
    values = iter(torch.stack(tensors).cpu().tolist() if tensors else [])

    def convert(item):
        if isinstance(item, dict):
            return {key: convert(child) for key, child in item.items()}
        if isinstance(item, torch.Tensor):
            item = next(values)
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    return convert(value)


class StabilityMonitor:
    """Sample residuals, Q/K/logits, FP8 inputs, gradients and update sizes."""

    def __init__(self, model, *, max_tokens=16, max_keys=128, max_parameter_values=256):
        self.active = False
        self.records = {}
        self.max_tokens = max_tokens
        self.max_keys = max_keys
        self.max_parameter_values = max_parameter_values
        self.parameter_samples = {}
        self.handles = []
        for index, block in enumerate(model.layers):
            self.handles.append(block.register_forward_hook(self._activation_hook(f"layer_{index:02d}.residual")))
            self.handles.append(block.attn.register_forward_hook(self._activation_hook(f"layer_{index:02d}.attention_output")))
            self.handles.append(block.ffn.register_forward_hook(self._activation_hook(f"layer_{index:02d}.ffn_output")))
            block.attn.stability_probe = self
            block.attn.stability_layer = index
        for name, module in model.named_modules():
            if isinstance(module, BitLinear):
                self.handles.append(module.register_forward_pre_hook(self._fp8_hook(name)))
        self.model = model

    def _activation_hook(self, name):
        @torch.no_grad()
        def hook(_module, _inputs, output):
            if self.active:
                stride = max(1, output.shape[1] // self.max_tokens)
                sample = output[0, ::stride, :][:self.max_tokens]
                self.records[name] = {"values": sample.numel(), **_stats(sample)}
        return hook

    def _fp8_hook(self, name):
        @torch.no_grad()
        def hook(_module, inputs):
            if not self.active:
                return
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])[:self.max_tokens]
            finite = torch.isfinite(values)
            safe = torch.where(finite, values, torch.zeros_like(values))
            codes, scales = quantize_activation_fp8_e4m3(safe)
            recovered = dequantize_activation_fp8_e4m3(codes, scales).float()
            relative_rms = ((recovered - safe.float()).square().mean().sqrt()
                            / safe.float().square().mean().sqrt().clamp_min(1e-12))
            self.records[f"{name}.fp8_input"] = {
                "values": values.numel(), "relative_rms_error": relative_rms,
                "scale_min": scales.amin(), "scale_max": scales.amax(),
                "nonfinite_fraction": 1 - finite.float().mean(),
                "zero_fraction": (codes.float() == 0).float().mean()}
        return hook

    @torch.no_grad()
    def record_attention(self, layer, query, key, is_global):
        if not self.active:
            return
        query_sample = query[0, :, -1, :].detach()
        key_sample = key[0, :, -min(key.shape[2], self.max_keys):, :].detach()
        repeat = query_sample.shape[0] // key_sample.shape[0]
        expanded_key = key_sample.repeat_interleave(repeat, dim=0)
        scores = torch.einsum("hd,hkd->hk", query_sample.float(), expanded_key.float()) / math.sqrt(query_sample.shape[-1])
        probabilities = scores.softmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        self.records[f"layer_{layer:02d}.attention_sample"] = {
            "key_positions": key_sample.shape[1], "query_heads": query_sample.shape[0],
            "global_layer": is_global, "unmasked_last_query": True,
            "query": _stats(query_sample), "key": _stats(key_sample),
            "logits": _stats(scores), "max_probability": probabilities.amax(),
            "mean_entropy": entropy.mean(), "min_entropy": entropy.amin()}

    @torch.no_grad()
    def before_step(self):
        self.parameter_samples = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            values = parameter.detach().reshape(-1)
            stride = max(1, values.numel() // self.max_parameter_values)
            indices = torch.arange(0, values.numel(), stride, device=values.device)[:self.max_parameter_values]
            old = values[indices].clone()
            gradient = parameter.grad
            grad_sample = gradient.detach().reshape(-1)[indices] if gradient is not None else torch.zeros_like(old)
            self.parameter_samples[name] = (parameter, indices, old, _stats(grad_sample))

    @torch.no_grad()
    def after_step(self):
        for name, (parameter, indices, old, grad_stats) in self.parameter_samples.items():
            new = parameter.detach().reshape(-1)[indices]
            delta = new.float() - old.float()
            weight_rms = old.float().square().mean().sqrt()
            self.records[f"{name}.parameter_sample"] = {
                "values": len(indices), "weight": _stats(old), "gradient": grad_stats,
                "update": _stats(delta), "update_to_weight_rms": delta.square().mean().sqrt() / weight_rms.clamp_min(1e-12)}
        self.parameter_samples = {}

    def finish(self):
        result = {"schema": 1, "scope": "first_microbatch_sampled_update",
                  "max_tokens_per_activation": self.max_tokens, "max_keys_per_attention": self.max_keys,
                  "max_parameter_values": self.max_parameter_values, "probes": _numbers(self.records)}
        self.records = {}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        for block in self.model.layers:
            del block.attn.stability_probe
            del block.attn.stability_layer
