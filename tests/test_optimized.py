import unittest
from dataclasses import replace

import torch
import torch.nn.functional as F

from terminus.loss import streaming_cross_entropy_sum
from terminus.model import BitNetLlama, RMSNorm, apply_rotary
from tests.helpers import tiny_config


class StreamingLossTests(unittest.TestCase):
    def test_dense_roofline_counts_attention_and_head_recomputation(self):
        from terminus.roofline import causal_pairs, training_roofline
        from terminus.train import TrainConfig

        for length in range(1, 17):
            for window in range(1, 17):
                self.assertEqual(causal_pairs(length, window), sum(min(position + 1, window) for position in range(length)))
        config = tiny_config(attention_backend="fa3", dim=128, n_heads=2, n_kv_heads=1)
        training = TrainConfig()
        checkpointed = training_roofline(config, training)
        streaming = training_roofline(replace(config, loss_backend="streaming"), training)
        difference = checkpointed["recomputation_flops_per_token"]["bf16"] - streaming["recomputation_flops_per_token"]["bf16"]
        self.assertEqual(difference, 2 * config.vocab_size * config.dim)
        self.assertEqual(streaming["hardware"]["bf16_tflops"], 835.5)
        self.assertFalse(streaming["hardware"]["sparse_tensor_cores_assumed"])
        self.assertGreater(streaming["recomputation_flops_per_token"]["bf16"], 0.)

    def test_factorized_embedding_head_and_cached_generation(self):
        config = tiny_config(embedding_dim=16, loss_backend="streaming")
        model = BitNetLlama(config)
        x = torch.randint(1, 32, (1, 11))
        y = torch.randint(0, 32, x.shape)
        result = model(x, y, return_logits=False)
        result["loss"].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(model.tok_embeddings.weight.shape, (32, 16))
        model.eval()
        expected = model(x)["logits"]
        first, cache = model.forward_cached(x[:, :7])
        last, _ = model.forward_cached(x[:, 7:], cache=cache)
        torch.testing.assert_close(torch.cat((first["logits"], last["logits"]), dim=1), expected, rtol=2e-5, atol=2e-6)

    def test_fused_projection_forward_and_gradients(self):
        torch.manual_seed(83)
        reference = BitNetLlama(tiny_config(loss_backend="streaming"))
        actual = BitNetLlama(replace(reference.config, fuse_projections=True))
        source = reference.state_dict()
        state = {}
        for name in actual.state_dict():
            if "qkv_proj.weight" in name:
                state[name] = torch.cat([source[name.replace("qkv_proj", role + "_proj")] for role in ("q", "k", "v")])
            elif "gate_up_proj.weight" in name:
                state[name] = torch.cat([source[name.replace("gate_up_proj", role + "_proj")] for role in ("gate", "up")])
            else:
                state[name] = source[name.replace("qkv_proj", "q_proj").replace("gate_up_proj", "up_proj")]
        actual.load_state_dict(state)
        inputs = torch.randint(1, 32, (2, 11))
        targets = torch.randint(0, 32, inputs.shape)
        a = actual(inputs, targets, return_logits=False)["loss"]
        b = reference(inputs, targets, return_logits=False)["loss"]
        torch.testing.assert_close(a, b)
        a.backward()
        b.backward()
        refs = dict(reference.named_parameters())
        for name, p in actual.named_parameters():
            if "qkv_proj.weight" in name:
                expected = torch.cat([refs[name.replace("qkv_proj", role + "_proj")].grad for role in ("q", "k", "v")])
            elif "gate_up_proj.weight" in name:
                expected = torch.cat([refs[name.replace("gate_up_proj", role + "_proj")].grad for role in ("gate", "up")])
            else:
                expected = refs[name].grad
            torch.testing.assert_close(p.grad, expected, rtol=3e-4, atol=2e-6)
        self.assertEqual(actual.count_parameters(), reference.count_parameters())

    def test_gradcheck_noncontiguous_tail_and_upstream_scale(self):
        torch.manual_seed(71)
        hidden = torch.randn(2, 3, 8, dtype=torch.float64)[..., ::2].requires_grad_()
        weight = torch.randn(11, 4, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[0, 3, -100], [7, 9, 10]])
        function = lambda x, w: streaming_cross_entropy_sum(x, w, targets, 4) * .031
        self.assertTrue(torch.autograd.gradcheck(function, (hidden, weight)))

    def test_tied_head_gradient_accumulation_and_all_ignored(self):
        torch.manual_seed(19)
        config = tiny_config(loss_backend="streaming")
        actual = BitNetLlama(config)
        reference = BitNetLlama(replace(config, loss_backend="checkpoint"))
        reference.load_state_dict(actual.state_dict())
        for ignored in (False, True):
            inputs = torch.randint(1, 32, (2, 11))
            targets = torch.randint(0, 32, inputs.shape)
            targets[:, :3] = -100
            if ignored:
                targets.fill_(-100)
            a = actual(inputs, targets, return_logits=False)["loss"]
            b = reference(inputs, targets, return_logits=True)["loss"]
            torch.testing.assert_close(a, b)
            (a / 2).backward()
            (b / 2).backward()
            for p, q in zip(actual.parameters(), reference.parameters()):
                torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-6)

    def test_frozen_operands_and_no_grad(self):
        for need_hidden, need_weight in ((True, False), (False, True), (True, True)):
            x = torch.randn(7, 8, requires_grad=need_hidden)
            w = torch.randn(17, 8, requires_grad=need_weight)
            y = torch.full((7,), -100)
            loss = streaming_cross_entropy_sum(x, w, y, 3)
            loss.backward()
            for value, need in ((x, need_hidden), (w, need_weight)):
                self.assertEqual(value.grad is not None, need)
                if need:
                    self.assertTrue(torch.equal(value.grad, torch.zeros_like(value)))
            with torch.no_grad():
                self.assertEqual(float(streaming_cross_entropy_sum(x, w, y, 3)), 0.)

    def test_rope_preserves_projection_dtype(self):
        values = torch.randn(2, 4, 7, 16, dtype=torch.bfloat16)
        cos, sin = torch.randn(1, 1, 7, 8), torch.randn(1, 1, 7, 8)
        self.assertEqual(apply_rotary(values, cos, sin).dtype, torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for fused training kernels")
class OptimizedCUDATests(unittest.TestCase):
    def test_fused_qk_rope_strides_forward_and_gradients(self):
        from terminus.pointwise import qk_rope

        for width in (64, 128, 256):
            x = torch.randn(2, 17, 4, width, device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
            cos, sin = (torch.randn(1, 1, 17, width // 2, device="cuda") for _ in range(2))
            a = qk_rope(x, cos, sin, 1e-5)
            b = apply_rotary(RMSNorm(width, gain="none").cuda()(x), cos, sin)
            self.assertEqual(a.transpose(1, 2).is_contiguous(), True)
            for dy in (torch.randn_like(a), torch.ones((), device="cuda", dtype=a.dtype).expand_as(a)):
                ga = torch.autograd.grad(a, x, dy, retain_graph=True)[0]
                gb = torch.autograd.grad(b, x, dy, retain_graph=True)[0]
                self.assertLess(float((ga.float() - gb.float()).norm() / gb.float().norm()), .012)
            self.assertLess(float((a.float() - b.float()).norm() / b.float().norm()), .006)

    def test_captured_microsteps_refresh_weights_accumulate_and_rescale(self):
        from terminus.cuda_graph import CapturedMicrostep

        torch.manual_seed(41)
        config = tiny_config(dim=512, n_heads=4, n_kv_heads=2, ffn_dim=512,
            attention_backend="fa3", linear_backend="deepgemm", weight_scale_layout="blockwise",
            quantize_activations=True, fp8_backward="full", fp8_quantizer="triton", fp8_cache_weights=True,
            fused_pointwise=True, fuse_projections=True, loss_backend="triton")
        actual, reference = BitNetLlama(config).cuda(), BitNetLlama(config).cuda()
        reference.load_state_dict(actual.state_dict())
        x = torch.randint(1, config.vocab_size, (1, 16), device="cuda")
        y = torch.randint(0, config.vocab_size, x.shape, device="cuda")
        captured = CapturedMicrostep(actual, x, y)
        optimizers = [torch.optim.AdamW(model.parameters(), lr=.001, fused=True) for model in (actual, reference)]
        for step in range(2):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=False)
            for scale in (.03125, .017):
                x.random_(1, config.vocab_size)
                y.random_(0, config.vocab_size)
                y[:, :3] = -100
                a = captured.run(x, y, scale)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    b = reference(x, y, return_logits=False)["loss_sum"]
                (b * scale).backward()
                torch.testing.assert_close(a, b, rtol=.002, atol=.002)
            for p, q in zip(actual.parameters(), reference.parameters()):
                self.assertLess(float((p.grad - q.grad).norm() / q.grad.norm().clamp_min(1e-8)), .03)
            for optimizer in optimizers:
                optimizer.step()
            actual.clear_fp8_weight_cache()
            reference.clear_fp8_weight_cache()

    def test_loss_vocabulary_tails_ignored_and_fp32_accumulation(self):
        for vocabulary in (257, 96000):
            torch.manual_seed(31)
            x = torch.randn(17, 128, device="cuda", requires_grad=True)
            w = (torch.randn(vocabulary, 128, device="cuda") * .03).requires_grad_()
            y = torch.randint(vocabulary, (17,), device="cuda")
            y[:3] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a = streaming_cross_entropy_sum(x, w, y, 7, backend="triton")
                b = F.cross_entropy(F.linear(x, w).float(), y, ignore_index=-100, reduction="sum")
            ga = torch.autograd.grad(a * 1e-7, (x, w))
            gb = torch.autograd.grad(b * 1e-7, (x, w))
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-4)
            for p, q in zip(ga, gb):
                self.assertEqual(p.dtype, torch.float32)
                self.assertLess(float((p - q).norm() / q.norm()), .012)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                empty = streaming_cross_entropy_sum(x, w, torch.full_like(y, -100), 7, backend="triton")
            self.assertEqual(float(empty), 0.)
            for g in torch.autograd.grad(empty, (x, w)):
                self.assertEqual(float(g.abs().max()), 0.)

    def test_norm_forward_backward_gain_modes_and_noncontiguous(self):
        for dtype in (torch.bfloat16, torch.float32):
            for width in (128, 2048, 5376):
                for gain in ("none", "scalar", "channel"):
                    x = torch.randn(2, 5, width, device="cuda", dtype=dtype).transpose(0, 1).requires_grad_()
                    reference = RMSNorm(width, gain=gain).cuda()
                    native = RMSNorm(width, gain=gain, fused=True).cuda()
                    native.load_state_dict(reference.state_dict())
                    y, expected = native(x), reference(x)
                    dy = torch.randn_like(y)
                    operands = (x,) if gain == "none" else (x, native.weight)
                    ref_operands = (x,) if gain == "none" else (x, reference.weight)
                    grads = torch.autograd.grad(y, operands, dy)
                    refs = torch.autograd.grad(expected, ref_operands, dy)
                    torch.testing.assert_close(y, expected, rtol=.01, atol=1e-6)
                    for g, r in zip(grads, refs):
                        self.assertLess(float((g.float() - r.float()).norm() / r.float().norm()), .008)

    def test_swiglu_forward_backward(self):
        from terminus.pointwise import swiglu

        for dtype in (torch.bfloat16, torch.float32):
            gate, up = torch.randn(2, 11, 2 * 5376, device="cuda", dtype=dtype, requires_grad=True).chunk(2, -1)
            actual, expected = swiglu(gate, up), F.silu(gate) * up
            dy = torch.randn_like(actual)
            torch.testing.assert_close(actual, expected, rtol=.01, atol=1e-6)
            for a, b in zip(torch.autograd.grad(actual, (gate, up), dy), torch.autograd.grad(expected, (gate, up), dy)):
                self.assertLess(float((a.float() - b.float()).norm() / b.float().norm()), .006)


if __name__ == "__main__":
    unittest.main()
