"""CUDA-event operator regions for a separate, untimed diagnostic update.

Works without CUPTI. These regions include stream idle time inside a host
call, not just kernel execution. They are not a bandwidth measurement. No
instrumentation is applied to the throughput timing window or CUDA graphs.
"""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import importlib

import torch


class EventProfile:
    def __init__(self):
        self.events = []
        self.phase = "other"

    @contextmanager
    def region(self, name):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, start, end))

    @contextmanager
    def instrument(self):
        targets = {
            "terminus.kernels": ["deep_gemm_fp8_gemm_nt"],
            "terminus.quantization": ["quantize_fp8_directions", "quantize_ternary_directions"],
            "terminus.model": ["attention", "quantize_ternary_directions"],
            "terminus.loss": ["streaming_cross_entropy_sum"],
            "terminus.pointwise_kernels": ["norm_forward", "norm_backward", "swiglu_forward", "swiglu_backward",
                                           "qk_rope_forward", "qk_rope_backward"],
            "terminus.mlp_kernels": ["packed_forward", "packed_backward"],
            "terminus.qkv_kernels": ["packed_qkv_forward", "packed_qkv_backward"],
            "terminus.wgrad_kernels": ["fp8_wgrad"],
        }
        # FA3 backward runs in its custom autograd function, outside the
        # model-level attention call measured during forward.
        try:
            interface = importlib.import_module("flash_attn_interface")
        except ImportError:
            interface = None
        if interface is not None and hasattr(interface, "_flash_attn_backward"):
            targets["flash_attn_interface"] = ["_flash_attn_backward"]
        originals = []

        def wrapper(function, name):
            @wraps(function)
            def timed(*args, **kwargs):
                label = f"{self.phase}/{name}"
                if name == "deep_gemm_fp8_gemm_nt":
                    label += "/wgrad" if kwargs.get("rhs_group_size") == 1 else "/fwd_or_dgrad"
                with self.region(label):
                    return function(*args, **kwargs)
            return timed

        try:
            for module_name, names in targets.items():
                module = importlib.import_module(module_name)
                for name in names:
                    original = getattr(module, name)
                    originals.append((module, name, original))
                    setattr(module, name, wrapper(original, name))
            yield self
        finally:
            for module, name, original in reversed(originals):
                setattr(module, name, original)

    def report(self):
        torch.cuda.synchronize()
        aggregate = defaultdict(lambda: {"calls": 0, "total_ms": 0.})
        for name, start, end in self.events:
            aggregate[name]["calls"] += 1
            aggregate[name]["total_ms"] += start.elapsed_time(end)
        return {
            "regions": dict(sorted(aggregate.items(), key=lambda entry: -entry[1]["total_ms"])),
            "scope": "One separate diagnostic update. CUDA-event regions include host launch gaps. phase/* regions include operator regions: do not sum both. Only explicitly wrapped calls receive operator regions; residuals, embedding gradients and other unwrapped operations remain in phase totals. No achieved bandwidth or production throughput is inferred.",
        }
