from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .integrity import fingerprint, source_hashes, write_json_atomic
from .kernels import attention, sdpa_reference_attention
from .model import BitNetLlama
from .fp8_validation import relative_error, run_fp8_preflight
from .train import load_config, runtime_contract


def count_model(config):
    with torch.device("meta"):
        model = BitNetLlama(config)
        counts = model.count_parameters()
    directions = 2 if config.fp8_backward != "bf16" else 1
    cache_bytes = counts["ternary"] * directions + counts["ternary"] // (128 * 128) * 4 * directions if config.fp8_cache_weights else 0
    return {**counts, "fp8_weight_cache_estimated_bytes": cache_bytes, "fp32_weights_and_gradients_bytes": counts["total"] * 8,
            "adamw_fp32_training_state_bytes_excluding_activations": counts["total"] * 16,
            "note": "Latent training state is not 1.58-bit; activations, workspaces and communication buffers are additional."}



def model_numerical_preflight(config, device):
    reduced = replace(config, vocab_size=257, dim=4 * config.head_dim, n_heads=4, n_kv_heads=2,
                      n_layers=config.global_attention_every, ffn_dim=((32 * config.head_dim // 3 + 127) // 128) * 128,
                      max_seq_len=min(35, config.max_seq_len), local_attention_window=min(11, config.local_attention_window),
                      pad_token_id=0, loss_chunk_size=7, gradient_checkpointing=True,
                      embedding_dim=min(config.embedding_dim, 2 * config.head_dim) if config.embedding_dim else None)
    optimized = BitNetLlama(reduced).to(device)
    reference = BitNetLlama(replace(reduced, gradient_checkpointing=False)).to(device)
    reference.load_state_dict(optimized.state_dict())
    inputs = torch.randint(1, reduced.vocab_size, (1, reduced.max_seq_len), device=device)
    targets = torch.randint(1, reduced.vocab_size, inputs.shape, device=device)
    targets[:, 0] = -100
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        actual = optimized(inputs, targets, return_logits=False)["loss"]
        # Compare checkpointed and retained executions of the SAME arithmetic.
        # Changing the head GEMM's chunk shape/rounding then sending the small
        # gradient difference through many FP8 requantizers measures a different
        # surrogate gradient, not checkpoint correctness. The head and pointwise
        # kernels have separate independent forward/backward gates below.
        expected = reference(inputs, targets, return_logits=False)["loss"]
    actual.backward()
    expected.backward()
    gradient_errors = [relative_error(parameter.grad, reference_parameter.grad)
                       for parameter, reference_parameter in zip(optimized.parameters(), reference.parameters())]
    errors = {"loss_relative_error": relative_error(actual, expected), "max_parameter_gradient_relative_error": max(gradient_errors)}
    del reference
    optimized.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        full = optimized(inputs)["logits"]
        cache, chunks = None, []
        for start in range(0, inputs.shape[1], 3):
            output, cache = optimized.forward_cached(inputs[:, start:start + 3], cache=cache)
            chunks.append(output["logits"])
        errors["cached_logits_relative_error"] = relative_error(torch.cat(chunks, dim=1), full)
    if not all(error < 0.03 for error in errors.values()):
        raise AssertionError(f"reduced full-model loss/gradient/cache mismatch: {errors}")
    return {"model": asdict(reduced), "errors": errors,
            "scope": "Reduced full-model composition on the selected backend, not a 1.5B allocation or optimizer soak."}


def optimized_kernel_preflight(config, device):
    from .loss import streaming_cross_entropy_sum
    from .model import RMSNorm, apply_rotary
    from .pointwise import swiglu, qk_rope, packed_swiglu_norm, packed_qkv_rope
    import torch.nn.functional as F

    results = {}

    def check(name, actual, expected, operands, reference_operands):
        gradient = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, operands, gradient)
        expected_grads = torch.autograd.grad(expected, reference_operands, gradient)
        errors = [relative_error(actual, expected)] + [relative_error(a, b) for a, b in zip(actual_grads, expected_grads)]
        if not all(error < .012 for error in errors):
            raise AssertionError(f"optimized {name} differs from independent PyTorch equations: {errors}")
        results[name] = errors

    if config.fused_pointwise:
        for width in sorted({config.head_dim, config.dim, config.ffn_dim}):
            for gain in ("none", "scalar", "channel"):
                values = torch.randn(17, width, dtype=torch.bfloat16, device=device, requires_grad=True)
                native, reference = (RMSNorm(width, config.rms_norm_eps, gain=gain, fused=fused).to(device) for fused in (True, False))
                operands = (values,) if gain == "none" else (values, native.weight)
                references = (values,) if gain == "none" else (values, reference.weight)
                check(f"norm_{width}_{gain}", native(values), reference(values), operands, references)
        gate, up = torch.randn(17, config.ffn_dim * 2, dtype=torch.bfloat16, device=device, requires_grad=True).chunk(2, -1)
        check("swiglu", swiglu(gate, up), F.silu(gate) * up, (gate, up), (gate, up))
        if config.fused_mlp:
            for rows in (1, 17, 257):
                packed = torch.randn(rows, config.ffn_dim * 2, dtype=torch.bfloat16, device=device, requires_grad=True)
                reference = RMSNorm(config.ffn_dim, config.rms_norm_eps, gain="scalar").to(device)
                weight = reference.weight if config.ffn_subln_enabled else None
                operands = (packed, weight) if weight is not None else (packed,)
                gate, up = packed.chunk(2, -1)
                expected = reference(F.silu(gate) * up) if weight is not None else F.silu(gate) * up
                check(f"packed_swiglu_norm_{rows}", packed_swiglu_norm(packed, weight, config.rms_norm_eps),
                      expected, operands, operands)
        values = torch.randn(2, 17, 4, config.head_dim, device=device, dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
        cos, sin = (torch.randn(1, 1, 17, config.head_dim // 2, device=device) for _ in range(2))
        reference = apply_rotary(RMSNorm(config.head_dim, config.rms_norm_eps, gain="none").to(device)(values), cos, sin)
        check("qk_norm_rope", qk_rope(values, cos, sin, config.rms_norm_eps), reference, (values,), (values,))
        if config.qk_norm_mode == "learned_rms":
            native = RMSNorm(config.head_dim, config.rms_norm_eps, gain="channel", fused=True).to(device)
            reference = RMSNorm(config.head_dim, config.rms_norm_eps, gain="channel", fused=False).to(device)
            with torch.no_grad():
                gains = torch.linspace(.5, 1.8, config.head_dim, device=device)
                native.weight.copy_(gains)
                reference.weight.copy_(gains)
            check("learned_qk_norm_before_rope", apply_rotary(native(values), cos, sin),
                  apply_rotary(reference(values), cos, sin),
                  (values, native.weight), (values, reference.weight))
        if config.fused_qkv and config.qk_norm_mode != "learned_rms":
            q, kv, dim = config.n_heads, config.n_kv_heads, config.head_dim
            packed = torch.randn(2, 17, (q + 2 * kv) * dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            normalize_qk = config.qk_norm_mode == "unit_rms"
            actual = packed_qkv_rope(packed, cos, sin, q, kv, config.rms_norm_eps, normalize_qk=normalize_qk)
            expected = []
            for index, part in enumerate(packed.split((q * dim, kv * dim, kv * dim), -1)):
                part = part.view(2, 17, q if index == 0 else kv, dim).transpose(1, 2)
                expected.append((qk_rope(part, cos, sin, config.rms_norm_eps) if normalize_qk
                                 else apply_rotary(part, cos, sin)) if index < 2 else part)
            check("packed_qkv_rope", torch.cat(actual, 1), torch.cat(expected, 1), (packed,), (packed,))
    if config.loss_backend != "checkpoint":
        values = torch.randn(17, 128, device=device, requires_grad=True)
        weight = (torch.randn(config.vocab_size, 128, device=device) * .03).requires_grad_()
        targets = torch.randint(config.vocab_size, (17,), device=device)
        targets[:3] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = streaming_cross_entropy_sum(values, weight, targets, 7,
                        backend="triton" if config.loss_backend == "triton" else "torch") / 14
            expected = F.cross_entropy(F.linear(values, weight).float(), targets, ignore_index=-100)
        check("linear_cross_entropy", actual, expected, (values, weight), (values, weight))
    return results


def graph_numerical_preflight(config, device):
    from .cuda_graph import CapturedMicrostep

    reduced = replace(config, vocab_size=257, dim=4 * config.head_dim, n_heads=4, n_kv_heads=2,
        n_layers=config.global_attention_every, ffn_dim=4 * config.head_dim,
        max_seq_len=16, local_attention_window=min(8, config.local_attention_window), gradient_checkpointing=False,
        embedding_dim=min(config.embedding_dim, 2 * config.head_dim) if config.embedding_dim else None)
    native, reference = BitNetLlama(reduced).to(device), BitNetLlama(reduced).to(device)
    reference.load_state_dict(native.state_dict())
    inputs = torch.randint(1, reduced.vocab_size, (1, 16), device=device)
    targets = torch.randint(0, reduced.vocab_size, inputs.shape, device=device)
    capture = CapturedMicrostep(native, inputs, targets)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001, fused=True) for model in (native, reference)]
    errors = []
    for step in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=False)
        for scale in (.03125, .017):
            inputs.random_(1, reduced.vocab_size)
            targets.random_(0, reduced.vocab_size)
            targets[:, :3] = -100
            actual = capture.run(inputs, targets, scale)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected = reference(inputs, targets, return_logits=False)["loss_sum"]
            (expected * scale).backward()
            errors.append(relative_error(actual, expected))
        errors.extend(relative_error(p.grad, q.grad) for p, q in zip(native.parameters(), reference.parameters()))
        for optimizer in optimizers:
            optimizer.step()
        native.clear_fp8_weight_cache()
        reference.clear_fp8_weight_cache()
    if not all(error < .03 for error in errors):
        raise AssertionError(f"CUDA graph replay/accumulation/weight refresh mismatch: {max(errors)}")
    return {"max_relative_error": max(errors), "optimizer_updates": 2, "microsteps_per_update": 2,
            "changing_inputs_targets_and_loss_scale": True, "scope": "Single-device graph versus identical eager arithmetic."}


def validate_preflight_report(path, config, device, *, cuda_graphs=False):
    if path is None:
        raise ValueError("production requires --preflight-report from this exact code, configuration and GPU runtime")
    report = json.loads(Path(path).read_text())
    expected = {"model_sha256": fingerprint(asdict(config)), "source": source_hashes(), "runtime": runtime_contract(device)}
    if report.get("identity") != expected or report.get("status") != "passed":
        raise ValueError("GPU preflight is missing, stale or incompatible")
    if not report.get("attention") or not report.get("model_numerics") or (config.linear_backend == "deepgemm" and not report.get("deepgemm")):
        raise ValueError("GPU preflight lacks required backend evidence")
    if config.linear_backend == "deepgemm":
        native = report["deepgemm"]
        if (not isinstance(native, dict) or not native.get("cases") or not native.get("quantizers")
                or native.get("backward_mode") != config.fp8_backward
                or native.get("wgrad_splits", 0) != config.fp8_wgrad_splits
                or native.get("quantizer") != config.fp8_quantizer
                or native.get("weight_cache") != config.fp8_cache_weights):
            raise ValueError("GPU preflight does not certify this FP8 backward/quantizer/cache policy")
    if (config.fused_pointwise or config.loss_backend != "checkpoint") and not report.get("optimized_kernels"):
        raise ValueError("GPU preflight lacks optimized loss/pointwise kernel evidence")
    if cuda_graphs and not report.get("graph_microstep"):
        raise ValueError("GPU preflight lacks CUDA graph replay/accumulation evidence")


def run_preflight(config, device, *, cuda_graphs=False):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU preflight requires CUDA; CPU tests do not certify production kernels")
    if config.attention_backend == "sdpa_reference":
        raise ValueError("reference attention cannot pass a production preflight")
    torch.manual_seed(1337)
    cases = []
    lengths = sorted({1, max(1, config.local_attention_window - 1), config.local_attention_window,
                      min(config.max_seq_len, config.local_attention_window + 1), config.max_seq_len})
    for length in lengths:
        for window in (None, config.local_attention_window):
            for query_length in sorted({1, min(3, length), length}):
                shape_query = (1, config.n_heads, query_length, config.head_dim)
                shape_key = (1, config.n_kv_heads, length, config.head_dim)
                operands = [torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
                            for shape in (shape_query, shape_key, shape_key)]
                reference_operands = [operand.detach().clone().requires_grad_() for operand in operands]
                offset = length - query_length
                result = attention(*operands, backend=config.attention_backend, local_window=window, query_position_offset=offset)
                expected = sdpa_reference_attention(*reference_operands, local_window=window, query_position_offset=offset)
                grad_output = torch.randn_like(result)
                gradients = torch.autograd.grad(result, operands, grad_output)
                reference_gradients = torch.autograd.grad(expected, reference_operands, grad_output)
                comparisons = [(result, expected), *zip(gradients, reference_gradients)]
                errors = [relative_error(actual, target) for actual, target in comparisons]
                absolute_errors = [float((actual.detach().float() - target.detach().float()).abs().max()) for actual, target in comparisons]
                # With one visible key, Q/K gradients are analytically zero.
                # Either native or reference SDPA can leave FP32 reduction
                # residuals, so exact count_nonzero is not sufficient here.
                zero_reference = [bool(torch.count_nonzero(target) == 0) for _, target in comparisons]
                if length == 1 or window == 1:
                    for index in (1, 2):
                        actual, target = comparisons[index]
                        if max(actual.detach().abs().max().item(), target.detach().abs().max().item()) >= 2e-5:
                            raise AssertionError("single-key Q/K gradients must be analytically zero")
                        zero_reference[index] = True
                passed = [absolute < 2e-5 if zero else relative < .03
                          for relative, absolute, zero in zip(errors, absolute_errors, zero_reference)]
                if not all(passed):
                    raise AssertionError(f"attention mismatch at {(length, query_length, window)}: relative={errors}, absolute={absolute_errors}")
                cases.append({"length": length, "query_length": query_length, "window": window,
                              "relative_errors": [None if zero else error for error, zero in zip(errors, zero_reference)],
                              "max_absolute_errors": absolute_errors, "zero_reference": zero_reference})
    deepgemm = run_fp8_preflight(config, device) if config.linear_backend == "deepgemm" else None
    optimized_kernels = optimized_kernel_preflight(config, device)
    model_numerics = model_numerical_preflight(config, device)
    graph_microstep = graph_numerical_preflight(config, device) if cuda_graphs else None
    torch.cuda.synchronize(device)
    query = torch.randn(1, config.n_heads, config.max_seq_len, config.head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, config.n_kv_heads, config.max_seq_len, config.head_dim, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        attention(query, key, key, backend=config.attention_backend, local_window=config.local_attention_window)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(10):
        attention(query, key, key, backend=config.attention_backend, local_window=config.local_attention_window)
    torch.cuda.synchronize(device)
    attention_ms = (time.perf_counter() - started) * 100
    return {"status": "passed", "identity": {"model_sha256": fingerprint(asdict(config)), "source": source_hashes(), "runtime": runtime_contract(device)},
            "counts": count_model(config), "attention": cases, "deepgemm": deepgemm, "model_numerics": model_numerics,
            "optimized_kernels": optimized_kernels,
            "graph_microstep": graph_microstep,
            "local_attention_forward_ms": attention_ms,
            "scope": "Kernel and reduced-model numerical gate only; full-scale throughput, quality and multi-node soak tests still required."}


def main():
    parser = argparse.ArgumentParser(description="Count model parameters without allocation, or validate actual GPU kernels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config, training, _ = load_config(args.config)
    result = count_model(config) if args.count_only else run_preflight(config, torch.device("cuda", 0), cuda_graphs=training.cuda_graphs)
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
