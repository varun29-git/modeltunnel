"""SwiGLU + scalar SubLN with packed gradients and no saved product tensor."""
import torch
import triton
import triton.language as tl


@triton.jit
def _packed_swiglu(X, W, R, Y, DY, DX, DW, WIDTH: tl.constexpr, EPS: tl.constexpr,
                   NORM: tl.constexpr, BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    # Packed activations cross 2^31 elements at large microbatches. Keep the
    # row offset in 64-bit arithmetic before multiplying by the row stride.
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    g = tl.load(X + row * (2 * WIDTH) + col, col < WIDTH, other=0).to(tl.float32)
    u = tl.load(X + row * (2 * WIDTH) + WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    sigmoid = 1. / (1. + tl.exp(-g))
    silu = (g * sigmoid).to(X.dtype.element_ty).to(tl.float32)
    product = (silu * u).to(X.dtype.element_ty).to(tl.float32)
    if BACKWARD:
        dy = tl.load(DY + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
        if NORM:
            inverse = tl.load(R + row)
            normalized = product * inverse
            gain = tl.load(W)
            tl.store(DW + row, tl.sum(dy * normalized, 0))
            weighted = dy * gain
            dproduct = (weighted - normalized * (tl.sum(weighted * normalized, 0) / WIDTH)) * inverse
            dy = dproduct.to(X.dtype.element_ty).to(tl.float32)
        dsilu = (dy * u).to(X.dtype.element_ty).to(tl.float32)
        dg = dsilu * sigmoid * (1. + g * (1. - sigmoid))
        du = dy * silu
        tl.store(DX + row * (2 * WIDTH) + col, dg, col < WIDTH)
        tl.store(DX + row * (2 * WIDTH) + WIDTH + col, du, col < WIDTH)
    else:
        if NORM:
            inverse = tl.rsqrt(tl.sum(product * product, 0) / WIDTH + EPS)
            product = (product * inverse) * tl.load(W)
            tl.store(R + row, inverse)
        tl.store(Y + row * WIDTH + col, product, col < WIDTH)


def packed_forward(packed, weight, eps):
    width = packed.shape[-1] // 2
    rows = packed.numel() // (2 * width)
    output = torch.empty((*packed.shape[:-1], width), device=packed.device, dtype=packed.dtype)
    inverse = torch.empty(rows, device=packed.device, dtype=torch.float32) if weight is not None else None
    _packed_swiglu[(rows,)](packed, weight, inverse, output, None, None, None, width, eps,
        weight is not None, False, triton.next_power_of_2(width),
        num_warps=8 if width > 4096 else 4, enable_fp_fusion=False)
    return output, inverse


def packed_backward(gradient, packed, weight, inverse):
    width = packed.shape[-1] // 2
    rows = packed.numel() // (2 * width)
    dx = torch.empty_like(packed)
    partial = torch.empty(rows, device=packed.device, dtype=torch.float32) if weight is not None else None
    _packed_swiglu[(rows,)](packed, weight, inverse, None, gradient.contiguous(), dx, partial,
        width, 0., weight is not None, True, triton.next_power_of_2(width),
        num_warps=8 if width > 4096 else 4, enable_fp_fusion=False)
    return dx, partial.sum().reshape_as(weight) if weight is not None else None
