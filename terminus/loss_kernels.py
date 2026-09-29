"""Tiled FP32 softmax statistics; reuse BF16 logits storage for dlogits."""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial_lse(Z, Stats, V: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(Z + row * V + col, col < V, other=-float("inf")).to(tl.float32)
    maximum = tl.max(z, 0)
    lse = maximum + tl.log(tl.sum(tl.exp(z - maximum), 0))
    tl.store(Stats + row * TILES + tile, lse)


@triton.jit
def _finish_lse(Z, Targets, Stats, LSE, Loss, V: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tiles = tl.arange(0, BLOCK)
    partial = tl.load(Stats + row * TILES + tiles, tiles < TILES, other=-float("inf"))
    maximum = tl.max(partial, 0)
    lse = maximum + tl.log(tl.sum(tl.exp(partial - maximum), 0))
    target = tl.load(Targets + row)
    tl.device_assert((target == -100) | ((target >= 0) & (target < V)), "target outside vocabulary")
    valid = (target >= 0) & (target < V)
    selected = tl.load(Z + row * V + target, valid, other=0).to(tl.float32)
    # Invalid labels poison the update even when device assertions are disabled.
    loss = tl.where(target == -100, 0., tl.where(valid, lse - selected, float("nan")))
    tl.store(LSE + row, lse)
    tl.store(Loss + row, loss)


@triton.jit
def _dlogits(Z, Targets, LSE, Normalizer, V: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0).to(tl.int64), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(Z + row * V + col, col < V, other=0).to(tl.float32)
    target = tl.load(Targets + row)
    lse = tl.load(LSE + row)
    grad = tl.exp(z - lse) - (col == target).to(tl.float32)
    grad = grad / tl.load(Normalizer)
    grad = tl.where(target == -100, 0., grad)
    tl.store(Z + row * V + col, grad, col < V)


def cross_entropy_inplace(logits, targets, need_gradient=True, *, normalizer=None):
    if not logits.is_cuda or not logits.is_contiguous() or logits.dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("fused cross entropy needs contiguous CUDA BF16/FP32 logits")
    rows, vocabulary = logits.shape
    targets = targets.contiguous()
    normalizer = torch.ones((), device=logits.device) if normalizer is None else normalizer
    block = 4096
    tiles = triton.cdiv(vocabulary, block)
    stats = torch.empty((rows, tiles), device=logits.device, dtype=torch.float32)
    lse = torch.empty(rows, device=logits.device, dtype=torch.float32)
    losses = torch.empty_like(lse)
    _partial_lse[(rows, tiles)](logits, stats, vocabulary, tiles, block)
    _finish_lse[(rows,)](logits, targets, stats, lse, losses, vocabulary, tiles, triton.next_power_of_2(tiles))
    if need_gradient:
        _dlogits[(rows, tiles)](logits, targets, lse, normalizer, vocabulary, block)
    return losses.sum(), logits if need_gradient else None
