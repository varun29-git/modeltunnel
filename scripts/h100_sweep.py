"""Run preflight and microbatch or fixed-microbatch accumulation sweeps on one H100.

Example: python -m scripts.h100_sweep --config configs/1.5b_h100_optimized.json
Example: python -m scripts.h100_sweep --config configs/terminus_25l_2304d_64k_h100.json --batch-size 8 --accumulations 1,2,4,8
"""
import argparse
import gc
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from terminus.benchmark import training_benchmark
from terminus.integrity import write_json_atomic
from terminus.preflight import run_preflight, validate_preflight_report
from terminus.train import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="artifacts/h100_sweep")
    parser.add_argument("--preflight-report")
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument("--batches", help="microbatches dividing the base accumulation to keep update tokens fixed (default: 1,2,4,8)")
    sweep.add_argument("--accumulations", help="accumulation counts to sweep at a fixed --batch-size")
    parser.add_argument("--batch-size", type=int, default=8, help="fixed microbatch for --accumulations (default: 8)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config, training, optimizer = load_config(args.config)
    if args.accumulations is None:
        batches = [int(value) for value in (args.batches or "1,2,4,8").split(",")]
        if any(batch <= 0 or training.gradient_accumulation_steps % batch for batch in batches):
            parser.error("each microbatch must divide the starting accumulation count to preserve update tokens")
        cases = [(batch, training.gradient_accumulation_steps // batch) for batch in batches]
    else:
        accumulations = [int(value) for value in args.accumulations.split(",")]
        if args.batch_size <= 0 or any(value <= 0 for value in accumulations):
            parser.error("microbatch and accumulation counts must be positive")
        cases = [(args.batch_size, value) for value in accumulations]
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    preflight = Path(args.preflight_report) if args.preflight_report else directory / "preflight.json"
    if args.preflight_report:
        validate_preflight_report(preflight, config, torch.device("cuda", 0), cuda_graphs=training.cuda_graphs)
    else:
        write_json_atomic(preflight, run_preflight(config, torch.device("cuda", 0), cuda_graphs=training.cuda_graphs))
    rows = []
    for batch, accumulation in cases:
        gc.collect()
        torch.cuda.empty_cache()
        case_name = f"batch_{batch}_accum_{accumulation}" if args.accumulations is not None else f"batch_{batch}"
        case_config = directory / f"{case_name}_config.json"
        write_json_atomic(case_config, {"model": asdict(config), "optimizer": asdict(optimizer),
            "training": asdict(replace(training, gradient_accumulation_steps=accumulation))})
        try:
            result = training_benchmark(case_config, batch_size=batch, warmup_steps=args.warmup,
                                         measured_steps=args.steps, preflight_report=preflight)
            write_json_atomic(directory / f"{case_name}.json", result)
            summary = {key: result[key] for key in ("batch_size", "tokens_per_update", "tokens_per_second",
                        "median_step_seconds", "peak_allocated_bytes", "peak_reserved_bytes")}
            summary["status"] = "passed"
        except torch.cuda.OutOfMemoryError:
            summary = {"batch_size": batch, "status": "out_of_memory"}
        summary["gradient_accumulation_steps"] = accumulation
        rows.append(summary)
        write_json_atomic(directory / "summary.json", rows)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
