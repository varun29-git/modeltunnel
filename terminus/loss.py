"""Bounded-memory linear cross entropy with no logits recomputation.

The scalar loss's input derivatives are computed chunk by chunk during forward.
Backward only applies the upstream scalar. This is a first-order training loss;
the tied embedding gradient still flows through ordinary autograd/DDP.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _torch_loss_gradient(logits, targets, need_gradient, normalizer):
    work = logits.float() if logits.dtype != torch.float64 else logits
    loss = F.cross_entropy(work, targets, ignore_index=-100, reduction="sum")
    if not need_gradient:
        return loss, None
    gradient = work.softmax(-1)
    valid = targets.ne(-100)
    gradient.scatter_add_(1, targets.clamp_min(0)[:, None], -valid[:, None].to(work.dtype))
    gradient.mul_(valid[:, None])
    gradient.div_(normalizer)
    return loss, gradient.to(logits.dtype)


class StreamingLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size, backend):
        ctx.hidden_dtype, ctx.weight_dtype = hidden.dtype, weight.dtype
        ctx.hidden_shape = hidden.shape
        need_hidden, need_weight = ctx.needs_input_grad[:2]
        dtype = (torch.get_autocast_dtype(hidden.device.type)
                 if torch.is_autocast_enabled(hidden.device.type) else hidden.dtype)
        # Cast the 96k x d head once, rather than once per checkpoint replay.
        flat = hidden.reshape(-1, hidden.shape[-1]).to(dtype)
        head = weight.to(dtype)
        targets = targets.reshape(-1)
        normalizer = targets.ne(-100).sum().clamp_min(1).to(torch.float64 if dtype == torch.float64 else torch.float32)
        accumulation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        dh = torch.empty_like(flat) if need_hidden else None
        dw = torch.empty_like(weight, dtype=accumulation_dtype) if need_weight else None
        total = torch.zeros((), dtype=accumulation_dtype, device=hidden.device)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for start in range(0, targets.numel(), chunk_size):
                inputs = flat[start:start + chunk_size]
                logits = inputs @ head.T
                labels = targets[start:start + chunk_size]
                if backend == "triton":
                    from .loss_kernels import cross_entropy_inplace

                    loss, gradient = cross_entropy_inplace(logits, labels, need_hidden or need_weight, normalizer=normalizer)
                else:
                    loss, gradient = _torch_loss_gradient(logits, labels, need_hidden or need_weight, normalizer)
                total.add_(loss)
                if need_hidden:
                    torch.mm(gradient, head, out=dh[start:start + inputs.shape[0]])
                if need_weight:
                    if flat.is_cuda and dtype in {torch.float16, torch.bfloat16}:
                        # cuBLAS accumulates directly into FP32 C. No V x d
                        # temporary/cast/add for each chunk, and no BF16 sum.
                        torch.addmm(dw, gradient.T, inputs, beta=0 if start == 0 else 1,
                                    out_dtype=accumulation_dtype, out=dw)
                    else:
                        torch.addmm(dw, gradient.T.to(accumulation_dtype), inputs.to(accumulation_dtype),
                                    beta=0 if start == 0 else 1, out=dw)
        ctx.save_for_backward(dh, dw, normalizer)
        return total

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        dh, dw, normalizer = ctx.saved_tensors
        # Multiply after promotion: avoid underflow from small token-normalized
        # upstream gradients, including accumulation and externally scaled loss.
        multiplier = grad_output * normalizer
        return ((dh.to(multiplier.dtype) * multiplier).to(ctx.hidden_dtype).reshape(ctx.hidden_shape) if dh is not None else None,
                (dw * multiplier).to(ctx.weight_dtype) if dw is not None else None, None, None, None)


def streaming_cross_entropy_sum(hidden, weight, targets, chunk_size, *, backend="torch"):
    if backend not in {"torch", "triton"} or chunk_size <= 0:
        raise ValueError("invalid streaming loss backend or chunk size")
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("linear loss requires matching hidden/head widths")
    if targets.shape != hidden.shape[:-1] or targets.dtype != torch.long or targets.numel() == 0:
        raise ValueError("linear loss requires nonempty int64 targets matching hidden rows")
    if hidden.device != weight.device or hidden.device != targets.device:
        raise ValueError("linear loss operands must share a device")
    if backend == "triton" and not hidden.is_cuda:
        raise RuntimeError("Triton linear cross entropy requires CUDA")
    if not torch.is_grad_enabled():
        hidden, weight = hidden.detach(), weight.detach()
    return StreamingLinearCrossEntropy.apply(hidden, weight, targets, chunk_size, backend)
