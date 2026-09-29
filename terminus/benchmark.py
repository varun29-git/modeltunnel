from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import validate_finite_state
from .integrity import fingerprint, source_hashes, write_json_atomic
from .model import BitNetLlama, chunked_cross_entropy_sum, linear_cross_entropy_sum
from .optim import build_optimizer
from .preflight import validate_preflight_report
from .train import load_config, runtime_contract


def loss_memory_benchmark(*, tokens=1024, vocabulary=4096, width=128, chunk_size=64):
    if min(tokens, vocabulary, width, chunk_size) <= 0:
        raise ValueError("benchmark dimensions must be positive")
    torch.manual_seed(1337)
    hidden = torch.randn(tokens, width, requires_grad=True)
    weight = torch.randn(vocabulary, width, requires_grad=True)
    targets = torch.randint(vocabulary, (tokens,))
    excluded = {value.untyped_storage().data_ptr() for value in (hidden, weight, targets)}
    cases, gradients = {}, {}
    for name in ("full", "chunked"):
        saved = {}

        def pack(value):
            storage = value.untyped_storage()
            if storage.data_ptr() not in excluded:
                saved[storage.data_ptr()] = storage.nbytes()
            return value

        started = time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            loss = (linear_cross_entropy_sum(hidden, weight, targets) if name == "full" else
                    chunked_cross_entropy_sum(hidden, weight, targets, chunk_size))
        retained_bytes = sum(saved.values())
        loss.backward()
        elapsed = time.perf_counter() - started
        gradients[name] = (hidden.grad.clone(), weight.grad.clone())
        cases[name] = {"loss": float(loss.detach()), "retained_extra_forward_storage_bytes": retained_bytes,
                       "forward_backward_seconds": elapsed}
        hidden.grad = weight.grad = None
    gradient_errors = [float((actual - expected).abs().max()) for actual, expected in zip(gradients["chunked"], gradients["full"])]
    for actual, expected in zip(gradients["chunked"], gradients["full"]):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    return {"device": "cpu", "torch": str(torch.__version__), "tokens": tokens, "vocabulary": vocabulary,
            "width": width, "chunk_size": chunk_size, "cases": cases, "max_gradient_absolute_errors": gradient_errors,
            "scope": "Unique extra tensor storage retained by autograd after the loss forward, excluding shared inputs/weights/targets; NOT peak process/GPU memory. Chunked backward recomputes logits and trades compute for retention."}


def training_benchmark(config_path, *, batch_size=1, warmup_steps=2, measured_steps=5, preflight_report=None,
                       profile_dir=None, event_profile=False):
    if batch_size <= 0 or warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("invalid benchmark batch or step count")
    config, training, optimizer_config = load_config(config_path)
    if event_profile and (training.device != "cuda" or training.cuda_graphs):
        raise ValueError("event profiling requires eager CUDA training")
    device = torch.device(training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the configured CUDA runtime is unavailable")
    if not training.development_mode:
        validate_preflight_report(preflight_report, config, device, cuda_graphs=training.cuda_graphs)
    torch.manual_seed(training.seed)
    model = BitNetLlama(config).to(device).train()
    optimizer = build_optimizer(model, optimizer_config)
    forward_model = torch.compile(model) if training.compile else model
    # Equal update-token sweeps consume the same random sequences at every
    # microbatch size, rather than repeating one microbatch during accumulation.
    inputs = torch.randint(config.vocab_size,
        (training.gradient_accumulation_steps, batch_size, config.max_seq_len + 1), device=device)
    amp_dtype = torch.bfloat16 if training.dtype == "bf16" else torch.float32
    durations, losses, gradient_norms, step_start_times = [], [], [], []
    tokens_per_update = batch_size * config.max_seq_len * training.gradient_accumulation_steps
    captured = None
    if training.cuda_graphs:
        from .cuda_graph import CapturedMicrostep

        captured = CapturedMicrostep(model, inputs[0, :, :-1], inputs[0, :, 1:], amp_dtype=amp_dtype)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def update(recorder=None):
        from contextlib import nullcontext

        def region(name):
            if recorder is None:
                return nullcontext()
            recorder.phase = name
            return recorder.region("phase/" + name)

        optimizer.zero_grad(set_to_none=not training.cuda_graphs)
        loss_value = torch.zeros((), device=device)
        for micro_index in range(training.gradient_accumulation_steps):
            batch = inputs[micro_index]
            if captured is not None:
                loss_sum = captured.run(batch[:, :-1], batch[:, 1:], 1 / tokens_per_update)
                loss_value += loss_sum.detach() / tokens_per_update
            else:
                with region("forward"):
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                        output = forward_model(batch[:, :-1], batch[:, 1:], return_logits=False)
                        loss = output["loss"] / training.gradient_accumulation_steps
                with region("backward"):
                    loss.backward()
                loss_value += loss.detach()
        with region("clip"):
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip or float("inf"),
                                                           error_if_nonfinite=True, foreach=True)
        with region("optimizer"):
            optimizer.step()
        model.clear_fp8_weight_cache()
        return loss_value, gradient_norm

    for step in range(warmup_steps + measured_steps):
        synchronize()
        if device.type == "cuda" and step == warmup_steps:
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        started_unix = time.time()
        loss_value, gradient_norm = update()
        synchronize()
        duration = time.perf_counter() - started
        if not math.isfinite(float(loss_value)):
            raise FloatingPointError("nonfinite synthetic benchmark loss")
        if step >= warmup_steps:
            step_start_times.append(started_unix)
            durations.append(duration)
            losses.append(float(loss_value))
            gradient_norms.append(float(gradient_norm))
    validate_finite_state((model.state_dict(), optimizer.state_dict()))
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    profile = None
    event_report = None
    if event_profile:
        from .event_profile import EventProfile

        recorder = EventProfile()
        with recorder.instrument(), recorder.region("phase/update"):
            update(recorder)
        event_report = recorder.report()
    profile_cuda_activity = None
    if profile_dir is not None:
        directory = Path(profile_dir)
        directory.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            update()
            synchronize()
        profiler.export_chrome_trace(str(directory / "training_trace.json"))
        averages = profiler.key_averages(group_by_input_shape=True)
        profile_cuda_activity = any(getattr(event, "device_time_total", 0) > 0 for event in profiler.events()) if device.type == "cuda" else False
        (directory / "operators.txt").write_text(averages.table(sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total", row_limit=100))
        profile = str(directory)
    throughput = tokens_per_update * measured_steps / sum(durations)
    physics = None
    if device.type == "cuda" and "H100" in torch.cuda.get_device_name(device) and training.dtype == "bf16":
        from .roofline import training_roofline

        name = torch.cuda.get_device_name(device)
        variant = "nvl" if "NVL" in name else "sxm" if "SXM" in name else "pcie" if "PCIe" in name else None
        if variant:
            physics = training_roofline(config, training, variant=variant, batch_size=batch_size, measured_tokens_s=throughput)
    return {"model_sha256": fingerprint(asdict(config)), "source": source_hashes(), "runtime": runtime_contract(device),
            "training": asdict(training), "optimizer": asdict(optimizer_config),
            "counts": model.count_parameters(), "batch_size": batch_size, "tokens_per_update": tokens_per_update,
            "warmup_steps": warmup_steps, "measured_steps": measured_steps, "step_seconds": durations,
            "step_start_unix": step_start_times,
            "median_step_seconds": statistics.median(durations), "tokens_per_second": throughput,
            "losses": losses, "gradient_norms": gradient_norms,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "profile_dir": profile, "profile_has_cuda_activity": profile_cuda_activity, "physics": physics,
            "event_profile": event_report,
            "scope": "Single-device synthetic full-model forward/backward/optimizer, including gradient accumulation; excludes data IO, distributed communication, evaluation and checkpoint costs. Not a quality or long-soak result."}


def main():
    parser = argparse.ArgumentParser(description="Measure loss retention or rehearse configured full-model optimizer updates")
    parser.add_argument("mode", choices=["loss-memory", "training"])
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--preflight-report")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profile-dir", help="Export one extra untimed optimizer update and CUDA operator totals")
    parser.add_argument("--event-profile", action="store_true", help="CUDA-event phases on an extra update; works without CUPTI")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.mode == "loss-memory":
        result = loss_memory_benchmark()
    else:
        if not args.config:
            parser.error("training requires --config")
        result = training_benchmark(args.config, batch_size=args.batch_size, warmup_steps=args.warmup_steps,
                                    measured_steps=args.steps, preflight_report=args.preflight_report, profile_dir=args.profile_dir,
                                    event_profile=args.event_profile)
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
