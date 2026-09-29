"""Experimental deterministic split-K FP8 wgrad for Hopper.

Consumes the SAME independently quantized transposed operands as DeepGEMM's
1D/1D path. Each 128-token product is accumulated in FP32 then multiplied by
its two per-channel scales. No requantization, atomic sum, or BF16 gradient.
Split partials trade workspace/traffic for shorter reduction dependency chains.
Native preflight and a measured sweep are required before selecting this path.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _wgrad(A, AS, B, BS, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           AS0: tl.constexpr, AS1: tl.constexpr, BS0: tl.constexpr, BS1: tl.constexpr,
           GROUPS_PER_SPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, MAX_ACC: tl.constexpr = 0):
    mi, ni, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = mi * BM + tl.arange(0, BM)
    n = ni * BN + tl.arange(0, BN)
    kk = tl.arange(0, 128)
    accumulator = tl.full((BM, BN), 0., tl.float32)
    for group in range(split * GROUPS_PER_SPLIT, tl.minimum((split + 1) * GROUPS_PER_SPLIT, K // 128)):
        k = group * 128 + kk
        a = tl.load(A + m[:, None] * K + k[None, :], m[:, None] < M, other=0.)
        b = tl.load(B + n[None, :] * K + k[:, None], n[None, :] < N, other=0.)
        # Bound FP8 accumulator error even on architectures with reduced
        # precision internal accumulation. Scales apply AFTER each 128 block.
        product = tl.dot(a, b, out_dtype=tl.float32, max_num_imprecise_acc=MAX_ACC)
        sa = tl.load(AS + m * AS0 + group * AS1, m < M, other=0)
        sb = tl.load(BS + n * BS0 + group * BS1, n < N, other=0)
        accumulator += product * (sa[:, None] * sb[None, :])
    tl.store(C + split * M * N + m[:, None] * N + n[None, :], accumulator,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce_partials(Partials, Output, ELEMENTS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Fixed order, no inter-CTA atomics or uninitialized C reads.
    value = tl.full((BLOCK,), 0., tl.float32)
    for split in range(SPLITS):
        value += tl.load(Partials + split * ELEMENTS + index, index < ELEMENTS, other=0)
    tl.store(Output + index, value, index < ELEMENTS)


def fp8_wgrad(lhs, rhs, *, splits=4):
    a, sa = lhs
    b, sb = rhs
    if isinstance(splits, bool) or not isinstance(splits, int) or splits not in {1, 2, 4, 8, 16}:
        raise ValueError("split-K count must be 1, 2, 4, 8 or 16")
    if not a.is_cuda or torch.cuda.get_device_capability(a.device)[0] != 9:
        raise RuntimeError("split-K FP8 wgrad requires Hopper SM90")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or min(*a.shape, *b.shape) <= 0:
        raise ValueError("wgrad requires nonempty [outputs, tokens] and [inputs, tokens] matrices")
    m, k = a.shape
    n = b.shape[0]
    if k % 128 or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("wgrad codes must be contiguous with a 128-aligned token dimension")
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype or sa.dtype != torch.float32 or sb.dtype != sa.dtype:
        raise ValueError("wgrad requires E4M3 codes and FP32 scales")
    if sa.shape != (m, k // 128) or sb.shape != (n, k // 128):
        raise ValueError("wgrad requires independent 1x128 scales for both operands")
    if len({value.device for value in (a, sa, b, sb)}) != 1:
        raise ValueError("wgrad operands must share a device")
    splits = min(splits, k // 128)
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    partials = torch.empty((splits, m, n), device=a.device, dtype=torch.float32) if splits > 1 else output
    _wgrad[(triton.cdiv(m, 64), triton.cdiv(n, 64), splits)](
        a, sa, b, sb, partials, m, n, k, *sa.stride(), *sb.stride(),
        triton.cdiv(k // 128, splits), 64, 64, num_warps=4, num_stages=3)
    if splits > 1:
        _reduce_partials[(triton.cdiv(m * n, 1024),)](partials, output, m * n, splits, 1024)
    return output
