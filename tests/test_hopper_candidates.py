import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from terminus.model import BitNetLlama, RMSNorm
from terminus.quantization import deepgemm_bitlinear, quantize_fp8_directions, dequantize_activation_fp8_e4m3
from tests.helpers import tiny_config
from tests.test_fp8 import emulated_fp8_gemm, error_ratio


class CandidatePolicyTests(unittest.TestCase):
    def test_candidates_keep_parameters_and_checkpoint_schema(self):
        base = tiny_config(fuse_projections=True, fused_pointwise=True)
        old, new = BitNetLlama(base), BitNetLlama(replace(base, fused_mlp=True))
        self.assertEqual(old.count_parameters(), new.count_parameters())
        new.load_state_dict(old.state_dict(), strict=True)
        for policy in ({"fp8_wgrad_splits": 4}, {"fp8_wgrad_splits": True}, {"fp8_wgrad_splits": 3},
                       {"fused_mlp": True, "fuse_projections": False},
                       {"fused_mlp": True, "ffn_activation": "relu2"}):
            with self.assertRaises(ValueError):
                replace(base, **policy)

    def test_component_physics_matches_total_and_update_budget(self):
        from terminus.roofline import training_roofline
        from terminus.train import TrainConfig

        for checkpointed in (False, True):
            config = tiny_config(gradient_checkpointing=checkpointed, loss_backend="streaming")
            result = training_roofline(config, TrainConfig(), batch_size=4, measured_tokens_s=45000)
            expected = result["ideal_compute_us_per_token"] * result["tokens_per_update"] / 1000
            self.assertAlmostEqual(sum(result["ideal_component_compute_ms_per_update"].values()), expected)
            self.assertAlmostEqual(result["required_time_reduction_fraction"], 1 - 45000 / 67000)


@unittest.skipUnless(torch.cuda.is_available(), "native CUDA validation required")
class HopperCandidateTests(unittest.TestCase):
    def test_split_quantizer_matches_existing_codes_and_scales_exactly(self):
        for rows in (1, 7, 257):
            x = torch.randn(rows, 768, device="cuda", dtype=torch.bfloat16)[:, ::2]
            for rowwise, transposed in ((True, True), (False, True), (True, False)):
                for eps in (0., 1e-5):
                    actual = quantize_fp8_directions(x, rowwise=rowwise, transposed=transposed, eps=eps, backend="triton_split")
                    expected = quantize_fp8_directions(x, rowwise=rowwise, transposed=transposed, eps=eps, backend="triton")
                    for a, b in zip(actual, expected):
                        if a is not None:
                            self.assertTrue(torch.equal(a[0].float(), b[0].float()))
                            self.assertTrue(torch.equal(a[1], b[1]))

    def test_norm_grid_large_rows_matches_reference(self):
        for gain in ("scalar", "channel"):
            x = torch.randn(4097, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            dy = torch.randn_like(x)
            fused, reference = (RMSNorm(2048, gain=gain, fused=value).cuda() for value in (True, False))
            actual, expected = fused(x), reference(x)
            ga = torch.autograd.grad(actual, (x, fused.weight), dy)
            gb = torch.autograd.grad(expected, (x, reference.weight), dy)
            for a, b in zip(ga, gb):
                self.assertLess(error_ratio(a, b), .012)

    def test_full_vocabulary_loss_offsets_above_int32(self):
        from terminus.loss_kernels import cross_entropy_inplace

        # 32k tokens x 96k classes exceeds 2^31 elements even in BF16.
        # Exercise both reduction reads and in-place derivative writes.
        if torch.cuda.mem_get_info()[0] < 10_000_000_000:
            self.skipTest("large-index regression needs 10 GB free CUDA memory")
        rows, vocab = 32768, 96000
        logits = torch.zeros(rows, vocab, device="cuda", dtype=torch.bfloat16)
        targets = torch.zeros(rows, device="cuda", dtype=torch.long)
        loss, gradient = cross_entropy_inplace(logits, targets)
        torch.testing.assert_close(loss, torch.tensor(rows * math.log(vocab), device="cuda"), rtol=2e-6, atol=0)
        indices = torch.tensor([0, 22370, rows - 1], device="cuda")
        torch.testing.assert_close(gradient[indices, 0], torch.full((3,), 1 / vocab - 1, device="cuda", dtype=logits.dtype))
        torch.testing.assert_close(gradient[indices, -1], torch.full((3,), 1 / vocab, device="cuda", dtype=logits.dtype))

    def test_packed_qkv_equations_strided_gradients_and_value_only(self):
        from terminus.pointwise import packed_qkv_rope, qk_rope
        from terminus.model import apply_rotary

        for q, kv, dim, normalize_qk in ((16, 4, 128, True), (4, 1, 64, True),
                                          (6, 2, 256, True), (5, 1, 128, False)):
            packed = torch.randn(2, 17, (q + 2 * kv) * dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            cos, sin = (torch.randn(1, 1, 17, dim // 2, device="cuda") for _ in range(2))
            actual = packed_qkv_rope(packed, cos, sin, q, kv, 1e-5, normalize_qk=normalize_qk)
            expected = []
            for index, part in enumerate(packed.split((q * dim, kv * dim, kv * dim), -1)):
                part = part.view(2, 17, q if index == 0 else kv, dim).transpose(1, 2)
                expected.append((qk_rope(part, cos, sin, 1e-5) if normalize_qk
                                 else apply_rotary(part, cos, sin)) if index < 2 else part)
            for a, b in zip(actual, expected):
                self.assertLess(error_ratio(a, b), .006)
            for broadcast in (False, True):
                dy = [torch.ones((), device="cuda", dtype=a.dtype).expand_as(a) if broadcast else torch.randn_like(a) for a in actual]
                ga = torch.autograd.grad(actual, packed, dy, retain_graph=True)[0]
                gb = torch.autograd.grad(expected, packed, dy, retain_graph=True)[0]
                self.assertTrue(ga.is_contiguous())
                self.assertLess(error_ratio(ga, gb), .012)
            ga = torch.autograd.grad(actual[2].sum(), packed, retain_graph=True)[0]
            gb = torch.autograd.grad(expected[2].sum(), packed)[0]
            torch.testing.assert_close(ga, gb, rtol=0, atol=0)

    def test_packed_mlp_matches_independent_equations(self):
        from terminus.pointwise import packed_swiglu_norm

        torch.manual_seed(943)
        for dtype in (torch.bfloat16, torch.float32):
            for rows, width in ((1, 128), (17, 384), (257, 5376)):
                for normalized in (False, True):
                    packed = torch.randn(rows, 2 * width, device="cuda", dtype=dtype, requires_grad=True)
                    reference = RMSNorm(width, gain="scalar").cuda()
                    with torch.no_grad():
                        reference.weight.fill_(.71)
                    gain = reference.weight if normalized else None
                    operands = (packed, gain) if normalized else (packed,)
                    actual = packed_swiglu_norm(packed, gain)
                    gate, up = packed.chunk(2, -1)
                    expected = F.silu(gate) * up
                    if normalized:
                        expected = reference(expected)
                    threshold = .012 if dtype == torch.bfloat16 else 2e-5
                    self.assertLess(error_ratio(actual, expected), threshold)
                    for dy in (torch.randn_like(actual), torch.ones((), device="cuda", dtype=dtype).expand_as(actual)):
                        ga = torch.autograd.grad(actual, operands, dy, retain_graph=True)
                        gb = torch.autograd.grad(expected, operands, dy, retain_graph=True)
                        for a, b in zip(ga, gb):
                            torch.testing.assert_close(a, b, rtol=threshold, atol=.02 if dtype == torch.bfloat16 else 1e-4)
                        self.assertTrue(ga[0].is_contiguous())

    def test_packed_mlp_retains_no_product_and_handles_zero(self):
        from terminus.pointwise import packed_swiglu_norm

        packed = torch.zeros(3, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        gain = torch.tensor(1., device="cuda", requires_grad=True)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            result = packed_swiglu_norm(packed, gain)
        self.assertEqual([t.numel() for t in saved], [packed.numel(), 1, 3])
        result.sum().backward()
        self.assertTrue(torch.equal(packed.grad, torch.zeros_like(packed)))
        self.assertEqual(float(gain.grad), 0.)

    def test_split_wgrad_scales_tails_and_determinism(self):
        from terminus.wgrad_kernels import fp8_wgrad

        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("Hopper required")
        torch.manual_seed(12)
        for tokens in (1, 257, 4096):
            for magnitude in (0., 1e-12, 1.):
                a = (torch.randn(tokens, 256, device="cuda") * magnitude).bfloat16()
                b = torch.randn(tokens, 384, device="cuda", dtype=torch.bfloat16)
                _, ac = quantize_fp8_directions(a, rowwise=False, transposed=True, eps=0, backend="triton")
                _, bc = quantize_fp8_directions(b, rowwise=False, transposed=True, backend="triton")
                # FP64 oracle measures reduction error independently of TF32
                # and the implementation's block-dot accumulation strategy.
                expected = dequantize_activation_fp8_e4m3(*ac).double() @ dequantize_activation_fp8_e4m3(*bc).double().T
                for splits in (1, 2, 4, 8, 16):
                    actual = fp8_wgrad(ac, bc, splits=splits)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertLessEqual(error_ratio(actual, expected), 2e-5)
                    self.assertTrue(torch.equal(actual, fp8_wgrad(ac, bc, splits=splits)))

    def test_split_wgrad_end_to_end_and_requested_gradients(self):
        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("Hopper required")
        for need_input, need_weight in ((True, True), (False, True), (True, False)):
            x = torch.randn(257, 256, device="cuda", dtype=torch.bfloat16, requires_grad=need_input)
            w = torch.randn(384, 256, device="cuda", requires_grad=need_weight)
            dy = torch.randn(257, 384, device="cuda", dtype=torch.bfloat16)
            # Isolate the new wgrad kernel from optional DeepGEMM installation.
            with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm):
                result = deepgemm_bitlinear(x, w, backward_mode="full", quantizer="triton", wgrad_splits=4)
                result.backward(dy)
            self.assertEqual(x.grad is not None, need_input)
            self.assertEqual(w.grad is not None, need_weight)
            if need_weight:
                _, gc = quantize_fp8_directions(dy, rowwise=False, transposed=True, eps=0, backend="triton")
                _, xc = quantize_fp8_directions(x.detach(), rowwise=False, transposed=True, backend="triton")
                expected = emulated_fp8_gemm(gc, xc, rhs_group_size=1, out_dtype=torch.float32)
                self.assertLess(error_ratio(w.grad, expected), 2e-5)
