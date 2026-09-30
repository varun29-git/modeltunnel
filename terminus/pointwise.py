"""Optional CUDA pointwise fusions; FP32 RMS statistics and gain gradients."""
import torch


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, weight, eps):
        from .pointwise_kernels import norm_forward

        output, inverse, flat = norm_forward(values, weight, eps)
        ctx.save_for_backward(flat, weight, inverse)
        ctx.shape = values.shape
        return output.reshape(values.shape)

    @staticmethod
    def backward(ctx, gradient):
        from .pointwise_kernels import norm_backward

        values, weight, inverse = ctx.saved_tensors
        dx, dw = norm_backward(gradient, values, weight, inverse)
        return dx.reshape(ctx.shape), dw, None


def rms_norm(values, weight, eps):
    if not values.is_cuda:
        raise RuntimeError("fused RMSNorm requires CUDA")
    return _RMSNorm.apply(values, weight, eps)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        from .pointwise_kernels import swiglu_forward

        ctx.save_for_backward(gate, up)
        return swiglu_forward(gate, up)

    @staticmethod
    def backward(ctx, gradient):
        from .pointwise_kernels import swiglu_backward

        return swiglu_backward(gradient, *ctx.saved_tensors)


def swiglu(gate, up):
    if not gate.is_cuda or gate.stride(-1) != 1 or up.stride(-1) != 1:
        raise ValueError("fused SwiGLU requires CUDA operands with contiguous columns")
    return _SwiGLU.apply(gate, up)


class _PackedSwiGLUNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, weight, eps):
        from .mlp_kernels import packed_forward

        output, inverse = packed_forward(packed, weight, eps)
        ctx.save_for_backward(packed, weight, inverse)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        from .mlp_kernels import packed_backward

        dx, dw = packed_backward(gradient, *ctx.saved_tensors)
        return dx, dw, None


def packed_swiglu_norm(packed, weight=None, eps=1e-5):
    """Packed gate/up -> SwiGLU -> optional scalar RMSNorm, retaining only packed.

    The packed gradient is written directly in the gate/up projection layout,
    eliminating SplitBackward's concatenation. BF16 boundaries match the
    separate operators, including the pre-norm product and norm input gradient.
    """
    import math

    if not packed.is_cuda or not packed.is_contiguous() or packed.ndim < 2 or packed.shape[-1] % 2:
        raise ValueError("packed SwiGLU needs contiguous CUDA [..., 2 * width] inputs")
    if packed.numel() == 0 or packed.dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError("packed SwiGLU needs nonempty BF16/FP32 inputs")
    if weight is not None and (weight.ndim != 0 or weight.dtype != torch.float32 or weight.device != packed.device):
        raise ValueError("packed SwiGLU norm requires a scalar FP32 gain on the input device")
    if not math.isfinite(eps) or eps < 0 or (weight is not None and eps == 0):
        raise ValueError("normalized SwiGLU requires positive finite epsilon")
    return _PackedSwiGLUNorm.apply(packed, weight, eps)


class _QKRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, cos, sin, eps):
        from .pointwise_kernels import qk_rope_forward

        output, inverse = qk_rope_forward(values, cos, sin, eps)
        ctx.save_for_backward(values, cos, sin, inverse)
        return output

    @staticmethod
    def backward(ctx, gradient):
        from .pointwise_kernels import qk_rope_backward

        return qk_rope_backward(gradient, *ctx.saved_tensors), None, None, None


def qk_rope(values, cos, sin, eps):
    return _QKRoPE.apply(values, cos, sin, eps)


class _PackedQKVRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, cos, sin, query_heads, kv_heads, eps, normalize_qk):
        from .qkv_kernels import packed_qkv_forward

        query, key, value, inverse = packed_qkv_forward(packed, cos, sin, query_heads, kv_heads, eps, normalize_qk)
        ctx.save_for_backward(packed, cos, sin, inverse)
        ctx.heads = query_heads, kv_heads
        ctx.normalize_qk = normalize_qk
        return query, key, value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dq, dk, dv):
        from .qkv_kernels import packed_qkv_backward

        return packed_qkv_backward(dq, dk, dv, *ctx.saved_tensors, *ctx.heads, ctx.normalize_qk), None, None, None, None, None, None


def packed_qkv_rope(packed, cos, sin, query_heads, kv_heads, eps, *, normalize_qk=True):
    import math

    if not packed.is_cuda or packed.ndim != 3 or not packed.is_contiguous() or packed.numel() == 0:
        raise ValueError("packed QKV requires nonempty contiguous CUDA [batch, length, width]")
    if any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in (query_heads, kv_heads)):
        raise ValueError("QKV head counts must be positive integers")
    if query_heads % kv_heads or packed.shape[-1] % (query_heads + 2 * kv_heads):
        raise ValueError("invalid packed QKV head dimensions")
    if not isinstance(normalize_qk, bool):
        raise ValueError("normalize_qk must be boolean")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("QKV normalization requires positive finite epsilon")
    dim = packed.shape[-1] // (query_heads + 2 * kv_heads)
    if dim not in {64, 128, 256} or packed.dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError("packed QKV requires BF16/FP32 and head width 64, 128 or 256")
    if any(t.device != packed.device or t.shape != (1, 1, packed.shape[1], dim // 2) or not t.is_contiguous() for t in (cos, sin)):
        raise ValueError("RoPE tables must be contiguous and match sequence/head dimensions")
    return _PackedQKVRoPE.apply(packed, cos, sin, query_heads, kv_heads, eps, normalize_qk)
