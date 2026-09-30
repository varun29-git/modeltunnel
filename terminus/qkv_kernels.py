"""Normalize/rotate multiple Q/K heads per CTA; write packed QKV gradients."""
import torch
import triton
import triton.language as tl


@triton.jit
def _packed_rope(X, Cos, Sin, R, Y, DQ, DK, DV, DX,
                 Q: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 QB: tl.constexpr, QH: tl.constexpr, QL: tl.constexpr,
                 KB: tl.constexpr, KH: tl.constexpr, KL: tl.constexpr,
                 VB: tl.constexpr, VH: tl.constexpr, VL: tl.constexpr,
                 EPS: tl.constexpr, BACKWARD: tl.constexpr, HEADS: tl.constexpr,
                 NORMALIZE: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
    c = tl.arange(0, D // 2)
    pos, batch = row % L, row // L
    offset = row * ((Q + 2 * KV) * D) + h[:, None] * D + c[None, :]
    a = tl.load(X + offset, h[:, None] < Q + KV, other=0.).to(tl.float32)
    b = tl.load(X + offset + D // 2, h[:, None] < Q + KV, other=0.).to(tl.float32)
    co = tl.load(Cos + pos * (D // 2) + c).to(X.dtype.element_ty).to(tl.float32)
    si = tl.load(Sin + pos * (D // 2) + c).to(X.dtype.element_ty).to(tl.float32)
    if BACKWARD:
        qo = batch * QB + h[:, None] * QH + pos * QL + c[None, :]
        ko = batch * KB + (h[:, None] - Q) * KH + pos * KL + c[None, :]
        vo = batch * VB + (h[:, None] - Q - KV) * VH + pos * VL + c[None, :]
        da = tl.load(DQ + qo, h[:, None] < Q, other=0.).to(tl.float32)
        db = tl.load(DQ + qo + D // 2, h[:, None] < Q, other=0.).to(tl.float32)
        da += tl.load(DK + ko, (h[:, None] >= Q) & (h[:, None] < Q + KV), other=0.).to(tl.float32)
        db += tl.load(DK + ko + D // 2, (h[:, None] >= Q) & (h[:, None] < Q + KV), other=0.).to(tl.float32)
        na = ((da * co[None, :]).to(X.dtype.element_ty).to(tl.float32)
              + (db * si[None, :]).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        nb = ((db * co[None, :]).to(X.dtype.element_ty).to(tl.float32)
              - (da * si[None, :]).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        if NORMALIZE:
            inverse = tl.load(R + row * (Q + KV) + h, h < Q + KV, other=0.)
            ax, bx = a * inverse[:, None], b * inverse[:, None]
            dot = (tl.sum(na * ax, 1) + tl.sum(nb * bx, 1)) / D
            dx_a = (na - ax * dot[:, None]) * inverse[:, None]
            dx_b = (nb - bx * dot[:, None]) * inverse[:, None]
        else:
            dx_a, dx_b = na, nb
        va = tl.load(DV + vo, (h[:, None] >= Q + KV) & (h[:, None] < Q + 2 * KV), other=0.).to(tl.float32)
        vb = tl.load(DV + vo + D // 2, (h[:, None] >= Q + KV) & (h[:, None] < Q + 2 * KV), other=0.).to(tl.float32)
        tl.store(DX + offset, tl.where(h[:, None] < Q + KV, dx_a, va), h[:, None] < Q + 2 * KV)
        tl.store(DX + offset + D // 2, tl.where(h[:, None] < Q + KV, dx_b, vb), h[:, None] < Q + 2 * KV)
    else:
        if NORMALIZE:
            inverse = tl.rsqrt((tl.sum(a * a, 1) + tl.sum(b * b, 1)) / D + EPS)
            a = (a * inverse[:, None]).to(X.dtype.element_ty).to(tl.float32)
            b = (b * inverse[:, None]).to(X.dtype.element_ty).to(tl.float32)
        ya = (a * co[None, :]).to(X.dtype.element_ty).to(tl.float32) - (b * si[None, :]).to(X.dtype.element_ty).to(tl.float32)
        yb = (b * co[None, :]).to(X.dtype.element_ty).to(tl.float32) + (a * si[None, :]).to(X.dtype.element_ty).to(tl.float32)
        out = row * ((Q + KV) * D) + h[:, None] * D + c[None, :]
        tl.store(Y + out, ya, h[:, None] < Q + KV)
        tl.store(Y + out + D // 2, yb, h[:, None] < Q + KV)
        if NORMALIZE:
            tl.store(R + row * (Q + KV) + h, inverse, h < Q + KV)


def packed_qkv_forward(packed, cos, sin, q, kv, eps, normalize_qk=True):
    batch, length, width = packed.shape
    dim = width // (q + 2 * kv)
    out = torch.empty((batch, length, q + kv, dim), device=packed.device, dtype=packed.dtype)
    inverse = torch.empty((batch * length, q + kv) if normalize_qk else (0,), device=packed.device, dtype=torch.float32)
    _packed_rope[(batch * length, triton.cdiv(q + kv, 4))](packed, cos, sin, inverse, out, None, None, None, None,
        q, kv, dim, length, 0, 0, 0, 0, 0, 0, 0, 0, 0, eps, False, 4, normalize_qk,
        num_warps=4, enable_fp_fusion=False)
    value = packed.view(batch, length, q + 2 * kv, dim)[:, :, q + kv:]
    return out[:, :, :q].transpose(1, 2), out[:, :, q:].transpose(1, 2), value.transpose(1, 2), inverse


def packed_qkv_backward(dq, dk, dv, packed, cos, sin, inverse, q, kv, normalize_qk=True):
    batch, length, width = packed.shape
    dim = width // (q + 2 * kv)
    dq, dk, dv = (g if g.stride(-1) == 1 else g.contiguous() for g in (dq, dk, dv))
    output = torch.empty_like(packed)
    _packed_rope[(batch * length, triton.cdiv(q + 2 * kv, 4))](packed, cos, sin, inverse, None, dq, dk, dv, output,
        q, kv, dim, length, *dq.stride()[:3], *dk.stride()[:3], *dv.stride()[:3], 0., True, 4, normalize_qk,
        num_warps=4, enable_fp_fusion=False)
    return output
