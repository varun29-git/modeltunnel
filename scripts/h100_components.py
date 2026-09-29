"""CUDA-event component timings; works when confidential mode disables CUPTI."""
import argparse
import json

import torch

from terminus.fp8_validation import projection_shapes
from terminus.integrity import write_json_atomic
from terminus.kernels import deep_gemm_fp8_gemm_nt
from terminus.loss import streaming_cross_entropy_sum
from terminus.quantization import quantize_fp8_directions, quantize_ternary_directions
from terminus.train import load_config


def measure(function, repeats=20):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = function()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config, _, _ = load_config(args.config)
    rows = config.max_seq_len * args.batch_size
    results = {}
    for role, (outputs, width) in projection_shapes(config).items():
        x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(outputs, width, device="cuda")
        dy = torch.randn(rows, outputs, device="cuda", dtype=torch.bfloat16)
        xr, xc = quantize_fp8_directions(x, transposed=True, backend="triton")
        wr, wc = quantize_ternary_directions(w, transposed=True, backend="triton")
        dyr, dyc = quantize_fp8_directions(dy, transposed=True, backend="triton", eps=0)
        result = {
            "input_quantize_ms": measure(lambda: quantize_fp8_directions(x, transposed=True, backend="triton")),
            "gradient_quantize_ms": measure(lambda: quantize_fp8_directions(dy, transposed=True, backend="triton", eps=0)),
            "fp8_forward_ms": measure(lambda: deep_gemm_fp8_gemm_nt(xr, wr)),
            "fp8_dgrad_ms": measure(lambda: deep_gemm_fp8_gemm_nt(dyr, wc)),
            "fp8_wgrad_ms": measure(lambda: deep_gemm_fp8_gemm_nt(dyc, xc, rhs_group_size=1, out_dtype=torch.float32)),
            "bf16_wgrad_ms": measure(lambda: torch.mm(dy.T, x, out_dtype=torch.float32)),
        }
        results[role] = result
        print(role, json.dumps(result), flush=True)
    del x, w, dy, xr, xc, wr, wc, dyr, dyc
    head_width = config.embedding_dim or config.dim
    x = torch.randn(rows, head_width, device="cuda", requires_grad=True)
    w = torch.randn(config.vocab_size, head_width, device="cuda", requires_grad=True)
    targets = torch.randint(config.vocab_size, (rows,), device="cuda")

    def loss():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return streaming_cross_entropy_sum(x, w, targets, config.loss_chunk_size, backend="triton")

    results["head_forward_with_derivatives_ms"] = measure(loss)
    print("head", results["head_forward_with_derivatives_ms"], flush=True)
    write_json_atomic(args.output, {"batch_size": args.batch_size, "rows": rows, "head_width": head_width, "components": results,
        "scope": "Isolated CUDA graph event timings, not full-model throughput; forward head includes its precomputed input and weight derivatives."})


if __name__ == "__main__":
    main()
