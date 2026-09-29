import torch
import triton
import triton.language as tl


@triton.jit
def _norm_forward(X, W, Y, R, WIDTH: tl.constexpr, GAIN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
    normalized = x * inverse
    if GAIN:
        w = tl.load(W + (col if GAIN == 2 else tl.full((BLOCK,), 0, tl.int32)), col < WIDTH, other=0).to(tl.float32)
        normalized *= w
    tl.store(Y + row * WIDTH + col, normalized, col < WIDTH)
    tl.store(R + row, inverse)


@triton.jit
def _norm_backward(DY, X, W, R, DX, DW, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                   GAIN: tl.constexpr, PROGRAMS: tl.constexpr, BLOCK: tl.constexpr):
    program = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    if GAIN:
        w = tl.load(W + (col if GAIN == 2 else tl.full((BLOCK,), 0, tl.int32)), col < WIDTH, other=0).to(tl.float32)
    dw = tl.full((BLOCK,), 0., tl.float32)
    for row in range(program, ROWS, PROGRAMS):
        x = tl.load(X + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
        dy = tl.load(DY + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
        inverse = tl.load(R + row)
        normalized = x * inverse
        if GAIN:
            dw += dy * normalized
            dy *= w
        dx = (dy - normalized * (tl.sum(dy * normalized, 0) / WIDTH)) * inverse
        tl.store(DX + row * WIDTH + col, dx, col < WIDTH)
    if GAIN == 2:
        tl.store(DW + program * WIDTH + col, dw, col < WIDTH)
    elif GAIN == 1:
        tl.store(DW + program, tl.sum(dw, 0))


def norm_forward(values, weight, eps):
    flat = values.contiguous().view(-1, values.shape[-1])
    rows, width = flat.shape
    output = torch.empty_like(flat)
    inverse = torch.empty(rows, device=values.device, dtype=torch.float32)
    gain = 0 if weight is None else (1 if weight.ndim == 0 else 2)
    _norm_forward[(rows,)](flat, weight, output, inverse, width, gain, eps, triton.next_power_of_2(width),
                           enable_fp_fusion=False)
    return output, inverse, flat


def norm_backward(gradient, values, weight, inverse, *, programs_override=None):
    rows, width = values.shape
    gain = 0 if weight is None else (1 if weight.ndim == 0 else 2)
    # More CTAs shorten the serialized row loop. Scalar gain needs only one
    # partial per CTA; channel gain must also bound its programs x width
    # reduction workspace. Hopper sweeps favor different grids for the two.
    programs = min(rows, 4096 if gain == 1 else (512 if width <= 4096 else 256)) if gain else rows
    if programs_override is not None:
        programs = min(rows, programs_override)
    dx = torch.empty_like(values)
    partial = torch.empty((programs, width if gain == 2 else 1), device=values.device, dtype=torch.float32) if gain else None
    _norm_backward[(programs,)](gradient.contiguous(), values, weight, inverse, dx, partial,
        rows, width, gain, programs, triton.next_power_of_2(width), enable_fp_fusion=False)
    dw = partial.sum(0).reshape(weight.shape).to(weight.dtype) if gain else None
    return dx, dw


@triton.jit
def _swiglu(G, U, Y, DY, DG, DU, N: tl.constexpr, WIDTH: tl.constexpr,
             G_STRIDE: tl.constexpr, U_STRIDE: tl.constexpr, BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(G + (i // WIDTH) * G_STRIDE + i % WIDTH, i < N, other=0).to(tl.float32)
    up = tl.load(U + (i // WIDTH) * U_STRIDE + i % WIDTH, i < N, other=0).to(tl.float32)
    sigmoid = 1. / (1. + tl.exp(-gate))
    silu = gate * sigmoid
    # Match eager's intermediate BF16 SiLU rounding.
    rounded = silu.to(U.dtype.element_ty).to(tl.float32)
    if BACKWARD:
        dy = tl.load(DY + i, i < N, other=0).to(tl.float32)
        d_silu = (dy * up).to(U.dtype.element_ty).to(tl.float32)
        tl.store(DG + i, d_silu * sigmoid * (1. + gate * (1. - sigmoid)), i < N)
        tl.store(DU + i, dy * rounded, i < N)
    else:
        tl.store(Y + i, rounded * up, i < N)


def swiglu_forward(gate, up):
    output = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
    _swiglu[(triton.cdiv(gate.numel(), 1024),)](gate, up, output, None, None, None, gate.numel(),
        gate.shape[-1], gate.stride(-2), up.stride(-2), False, 1024,
                                             enable_fp_fusion=False)
    return output


def swiglu_backward(gradient, gate, up):
    dg = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
    du = torch.empty_like(dg)
    _swiglu[(triton.cdiv(gate.numel(), 1024),)](gate, up, None, gradient.contiguous(), dg, du, gate.numel(),
        gate.shape[-1], gate.stride(-2), up.stride(-2), True, 1024,
                                             enable_fp_fusion=False)
    return dg, du


@triton.jit
def _qk_rope(X, Cos, Sin, R, Y, DY, DX,
              H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
              XB: tl.constexpr, XH: tl.constexpr, XL: tl.constexpr,
              GB: tl.constexpr, GH: tl.constexpr, GL: tl.constexpr,
              EPS: tl.constexpr, BACKWARD: tl.constexpr):
    row = tl.program_id(0)
    head, position, batch = row % H, (row // H) % L, row // (H * L)
    col = tl.arange(0, D // 2)
    offset = batch * XB + head * XH + position * XL + col
    a = tl.load(X + offset).to(tl.float32)
    b = tl.load(X + offset + D // 2).to(tl.float32)
    cos = tl.load(Cos + position * (D // 2) + col).to(X.dtype.element_ty).to(tl.float32)
    sin = tl.load(Sin + position * (D // 2) + col).to(X.dtype.element_ty).to(tl.float32)
    if BACKWARD:
        inverse = tl.load(R + row)
        go = batch * GB + head * GH + position * GL + col
        da = tl.load(DY + go).to(tl.float32)
        db = tl.load(DY + go + D // 2).to(tl.float32)
        na = ((da * cos).to(X.dtype.element_ty).to(tl.float32)
              + (db * sin).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        nb = ((db * cos).to(X.dtype.element_ty).to(tl.float32)
              - (da * sin).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        ax, bx = a * inverse, b * inverse
        dot = (tl.sum(na * ax, 0) + tl.sum(nb * bx, 0)) / D
        tl.store(DX + row * D + col, (na - ax * dot) * inverse)
        tl.store(DX + row * D + col + D // 2, (nb - bx * dot) * inverse)
    else:
        inverse = tl.rsqrt((tl.sum(a * a, 0) + tl.sum(b * b, 0)) / D + EPS)
        a = (a * inverse).to(X.dtype.element_ty).to(tl.float32)
        b = (b * inverse).to(X.dtype.element_ty).to(tl.float32)
        first = (a * cos).to(X.dtype.element_ty).to(tl.float32) - (b * sin).to(X.dtype.element_ty).to(tl.float32)
        second = (b * cos).to(X.dtype.element_ty).to(tl.float32) + (a * sin).to(X.dtype.element_ty).to(tl.float32)
        tl.store(Y + row * D + col, first)
        tl.store(Y + row * D + col + D // 2, second)
        tl.store(R + row, inverse)


def qk_rope_forward(values, cos, sin, eps):
    batch, heads, length, width = values.shape
    if width not in {64, 128, 256} or values.stride(-1) != 1:
        raise ValueError("fused Q/K RoPE requires a supported contiguous head dimension")
    output = torch.empty((batch, length, heads, width), device=values.device, dtype=values.dtype)
    inverse = torch.empty(batch * heads * length, device=values.device, dtype=torch.float32)
    _qk_rope[(inverse.numel(),)](values, cos, sin, inverse, output, None, None,
        heads, length, width, *values.stride()[:3], 0, 0, 0, eps, False, num_warps=1, enable_fp_fusion=False)
    return output.transpose(1, 2), inverse


def qk_rope_backward(gradient, values, cos, sin, inverse):
    if gradient.stride(-1) != 1:
        gradient = gradient.contiguous()
    batch, heads, length, width = values.shape
    output = torch.empty((batch, length, heads, width), device=values.device, dtype=values.dtype)
    _qk_rope[(inverse.numel(),)](values, cos, sin, inverse, None, gradient, output,
        heads, length, width, *values.stride()[:3], *gradient.stride()[:3], 0., True,
        num_warps=1, enable_fp_fusion=False)
    return output.transpose(1, 2)
