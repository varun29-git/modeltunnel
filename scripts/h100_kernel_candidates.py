"""Numerical gates and isolated timings for optional Hopper kernel candidates."""
import argparse
import gc

import torch

from scripts.h100_components import measure
from terminus.fp8_validation import projection_shapes, validate_quantizers
from terminus.integrity import source_hashes, write_json_atomic
from terminus.kernels import deep_gemm_fp8_gemm_nt
from terminus.model import RMSNorm
from terminus.pointwise import packed_swiglu_norm, swiglu
from terminus.quantization import quantize_fp8_directions, dequantize_activation_fp8_e4m3
from terminus.train import load_config, runtime_contract
from terminus.wgrad_kernels import fp8_wgrad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(741)
    config, _, _ = load_config(args.config)
    rows = config.max_seq_len * args.batch_size
    results = {"rows": rows, "source": source_hashes(), "runtime": runtime_contract(torch.device("cuda")),
               "split_quantizer_gate": validate_quantizers("triton_split", torch.device("cuda")), "projections": {}}
    for role, (outputs, width) in projection_shapes(config).items():
        x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn(rows, outputs, device="cuda", dtype=torch.bfloat16)
        _, xc = quantize_fp8_directions(x, transposed=True, backend="triton")
        _, gc = quantize_fp8_directions(dy, transposed=True, eps=0, backend="triton")
        case = {"input_quantizer_ms": {}, "gradient_quantizer_ms": {}, "wgrad_ms": {}}
        for backend in ("triton", "triton_split"):
            case["input_quantizer_ms"][backend] = measure(lambda: quantize_fp8_directions(x, transposed=True, backend=backend))
            case["gradient_quantizer_ms"][backend] = measure(lambda: quantize_fp8_directions(dy, transposed=True, eps=0, backend=backend))
        expected = dequantize_activation_fp8_e4m3(*gc).double() @ dequantize_activation_fp8_e4m3(*xc).double().T
        native = deep_gemm_fp8_gemm_nt(gc, xc, rhs_group_size=1, out_dtype=torch.float32)
        case["deepgemm_relative_error"] = float((native - expected).norm() / expected.norm().clamp_min(1e-30))
        case["wgrad_ms"]["deepgemm"] = measure(lambda: deep_gemm_fp8_gemm_nt(gc, xc, rhs_group_size=1, out_dtype=torch.float32))
        for splits in (1, 2, 4, 8, 16):
            actual = fp8_wgrad(gc, xc, splits=splits)
            error = float((actual - expected).norm() / expected.norm().clamp_min(1e-30))
            if not error < 2e-5:
                raise AssertionError(f"split-K native gate failed for {role}/{splits}: {error}")
            case["wgrad_ms"][str(splits)] = {"ms": measure(lambda: fp8_wgrad(gc, xc, splits=splits)), "relative_error": error}
        results["projections"][role] = case
        print(role, case, flush=True)
        write_json_atomic(args.output, results)
        del x, dy, xc, gc, actual, expected, native
        gc_collect()
    packed = torch.randn(rows, 2 * config.ffn_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    norm = RMSNorm(config.ffn_dim, gain="scalar", fused=True).cuda()
    gradient = torch.randn(rows, config.ffn_dim, device="cuda", dtype=torch.bfloat16)

    def mlp(fused):
        if fused:
            result = packed_swiglu_norm(packed, norm.weight)
        else:
            gate, up = packed.chunk(2, -1)
            result = norm(swiglu(gate, up))
        return torch.autograd.grad(result, (packed, norm.weight), gradient)

    results["mlp_forward_backward_ms"] = {str(fused): measure(lambda: mlp(fused)) for fused in (False, True)}
    results["scope"] = "Isolated CUDA graph timings and numerical checks; not full-model speed or quality evidence."
    write_json_atomic(args.output, results)
    print(results["mlp_forward_backward_ms"], flush=True)


def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
