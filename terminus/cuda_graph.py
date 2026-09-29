"""Capture a fixed-shape, single-device training microstep.

Optimizer, clipping, finite checks, schedules, data and checkpointing stay outside
the graph. Weight quantization is captured and runs on every replay, so updated
latent weights cannot reuse stale FP8 codes. Gradients accumulate into persistent
FP32 buffers and must be zeroed with set_to_none=False between updates.
"""
import torch


class CapturedMicrostep:
    def __init__(self, model, inputs, targets, *, amp_dtype=torch.bfloat16):
        if not inputs.is_cuda or model.config.dropout or model.config.gradient_checkpointing:
            raise ValueError("CUDA graph microsteps require CUDA, zero dropout and retained activations")
        self.model = model
        self.inputs, self.targets = inputs.contiguous().clone(), targets.contiguous().clone()
        self.scale = torch.ones((), device=inputs.device)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        for p in self.parameters:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        self.gradients = [p.grad for p in self.parameters]
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=inputs.device)
        stream.wait_stream(torch.cuda.current_stream(inputs.device))

        def forward_backward():
            with torch.autocast("cuda", dtype=amp_dtype):
                loss_sum = model(self.inputs, self.targets, return_logits=False)["loss_sum"]
                loss = loss_sum * self.scale
            loss.backward()
            return loss_sum

        # Compile/JIT and initialize all lazy kernel state before capture.
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch._foreach_zero_(self.gradients)
                model.clear_fp8_weight_cache()
                forward_backward()
        torch.cuda.current_stream(inputs.device).wait_stream(stream)
        torch.cuda.synchronize(inputs.device)
        torch._foreach_zero_(self.gradients)
        model.clear_fp8_weight_cache()
        with torch.cuda.graph(self.graph, stream=stream):
            self.loss_sum = forward_backward()
        # Retain operands whose addresses are embedded in captured native calls.
        self.weight_operands = [module._fp8_weight_cache for _, module in model.bitlinear_named_modules()]
        torch._foreach_zero_(self.gradients)

    def matches(self, inputs, targets):
        return (inputs.shape == self.inputs.shape and targets.shape == self.targets.shape
                and inputs.dtype == self.inputs.dtype and targets.dtype == self.targets.dtype
                and inputs.device == self.inputs.device and targets.device == self.targets.device)

    def run(self, inputs, targets, loss_scale):
        if not self.matches(inputs, targets):
            raise ValueError("captured microstep shape, dtype or device changed")
        if any(p.grad is not gradient for p, gradient in zip(self.parameters, self.gradients)):
            raise RuntimeError("captured gradient buffers were replaced; use zero_grad(set_to_none=False)")
        self.inputs.copy_(inputs)
        self.targets.copy_(targets)
        self.scale.fill_(loss_scale)
        self.graph.replay()
        return self.loss_sum
