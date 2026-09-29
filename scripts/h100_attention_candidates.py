"""Compare exact local/global attention implementations with native gradients."""
import argparse

import torch

from terminus.integrity import source_hashes, write_json_atomic
from terminus.kernels import attention
from terminus.train import load_config, runtime_contract


def measure(function, repeats=20):
    # Match eager training. Autograd stream dependencies from leaf tensors can
    # make an otherwise valid FA3 backward unsuitable for isolated capture.
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(repeats):
        function()
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
    results = {"source": source_hashes(), "runtime": runtime_contract(torch.device("cuda")), "cases": [],
               "scope": "Isolated eager forward/backward CUDA-event timings, including host launch gaps; not model throughput."}
    torch.manual_seed(123)
    q, k, v = (torch.randn(args.batch_size, heads, config.max_seq_len, config.head_dim,
                          device="cuda", dtype=torch.bfloat16, requires_grad=True)
               for heads in (config.n_heads, config.n_kv_heads, config.n_kv_heads))
    dy = torch.randn_like(q)
    for window in (config.local_attention_window, None):
        reference = attention(q, k, v, backend="fa3", local_window=window)
        reference_grads = torch.autograd.grad(reference, (q, k, v), dy)
        for backend in ("fa3", "flex"):
            actual = attention(q, k, v, backend=backend, local_window=window)
            grads = torch.autograd.grad(actual, (q, k, v), dy)
            errors = [float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20))
                      for a, b in zip((actual, *grads), (reference, *reference_grads))]
            if not all(error < .012 for error in errors):
                raise AssertionError(f"attention {backend}/{window} numerical discrepancy: {errors}")

            def step():
                output = attention(q, k, v, backend=backend, local_window=window)
                return torch.autograd.grad(output, (q, k, v), dy)

            case = {"backend": backend, "window": window, "forward_backward_ms": measure(step), "relative_errors": errors}
            results["cases"].append(case)
            write_json_atomic(args.output, results)
            print(case, flush=True)


if __name__ == "__main__":
    main()
