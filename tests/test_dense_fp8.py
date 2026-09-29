"""Dense-weight FP8 stage: numerics, config validation, and WSD schedules.

The dense stage keeps FP32 master weights and quantizes projection weights to
dense FP8 E4M3 operands (blockwise 128x128 absmax/448 FP32 scales) instead of
ternary codes. CPU tests emulate the DeepGEMM GEMM by dequantizing operands,
exactly like the ternary FP8 tests; native SM90 numerics are covered by the
GPU preflight, which is not run in this environment.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from terminus.model import BitLinear
from terminus.quantization import (deepgemm_bitlinear, dense_weight_scales,
                                   dequantize_weight_fp8_e4m3_2d,
                                   quantize_weight_dense_fp8_e4m3_2d,
                                   quantize_weight_directions,
                                   quantize_weight_fp8_e4m3_2d)
from terminus.train import TrainConfig, schedule_scale
from tests.helpers import tiny_config
from tests.test_fp8 import emulated_fp8_gemm, error_ratio


class DenseQuantizerTests(unittest.TestCase):
    def test_roundtrip_close_and_not_ternary(self):
        torch.manual_seed(7)
        weight = torch.randn(256, 384) * 0.05
        codes, scales = quantize_weight_dense_fp8_e4m3_2d(weight)
        dequantized = dequantize_weight_fp8_e4m3_2d(codes, scales)
        self.assertLess(error_ratio(dequantized, weight), 0.04)  # E4M3 has 3 mantissa bits
        self.assertEqual(codes.dtype, torch.float8_e4m3fn)
        # Dense codes are not ternary: more than three distinct values.
        self.assertGreater(len(torch.unique(codes.float())), 3)
        ternary_codes, _ = quantize_weight_fp8_e4m3_2d(weight)
        self.assertLessEqual(set(torch.unique(ternary_codes.float()).tolist()), {-1.0, 0.0, 1.0})

    def test_blockwise_absmax_scale_layout(self):
        torch.manual_seed(7)
        weight = torch.zeros(128, 128)
        weight[3, 5] = 2.24
        scales = dense_weight_scales(weight)
        self.assertEqual(scales.shape, (1, 1))
        self.assertAlmostEqual(float(scales[0, 0]), 2.24 / 448.0, places=7)
        codes, _ = quantize_weight_dense_fp8_e4m3_2d(weight)
        self.assertEqual(float(codes[3, 5].float()), 448.0)

    def test_zero_block_and_nonfinite_propagation(self):
        weight = torch.zeros(256, 128)
        codes, scales = quantize_weight_dense_fp8_e4m3_2d(weight)
        self.assertTrue(torch.equal(codes.float(), torch.zeros_like(codes.float())))
        weight[0, 0] = float("inf")
        codes, _ = quantize_weight_dense_fp8_e4m3_2d(weight)
        self.assertTrue(torch.isnan(codes[0, 0].float()))
        weight[0, 0] = float("nan")
        codes, _ = quantize_weight_dense_fp8_e4m3_2d(weight)
        self.assertTrue(torch.isnan(codes[0, 0].float()))

    def test_directional_transpose_is_exact(self):
        torch.manual_seed(7)
        weight = torch.randn(256, 384)
        (codes, scales), transposed = quantize_weight_directions(
            weight, weight_mode="dense", transposed=True, backend="torch")
        self.assertTrue(torch.equal(transposed[0].float(), codes.float().T))
        self.assertTrue(torch.equal(transposed[1], scales.T))
        with self.assertRaises(ValueError):
            quantize_weight_directions(weight, weight_mode="fp16", backend="torch")


class DenseDeepGEMMTests(unittest.TestCase):
    def _linear(self, mode, weight_mode):
        config = tiny_config(dim=256, n_heads=4, n_kv_heads=2, ffn_dim=512,
                             linear_backend="deepgemm", weight_scale_layout="blockwise",
                             quantize_activations=True, fp8_backward=mode,
                             fp8_cache_weights=False, weight_mode=weight_mode)
        return BitLinear(256, 512, alpha_group_size=128, weight_scale_layout="blockwise",
                         linear_backend="deepgemm", quantize_activations=True,
                         fp8_backward=mode, weight_mode=weight_mode)

    def test_dense_forward_matches_emulated_dense_fp8(self):
        torch.manual_seed(11)
        linear = self._linear("full", "dense")
        values = torch.randn(17, 256, dtype=torch.bfloat16, requires_grad=True)
        with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm):
            output = linear(values)
        codes, scales = quantize_weight_dense_fp8_e4m3_2d(linear.weight)
        from terminus.quantization import quantize_activation_fp8_e4m3, dequantize_activation_fp8_e4m3
        reference = (dequantize_activation_fp8_e4m3(*quantize_activation_fp8_e4m3(values)).float() @
                     dequantize_weight_fp8_e4m3_2d(codes, scales).T).to(torch.bfloat16)
        self.assertLess(error_ratio(output.float(), reference.float()), 1e-5)
        self.assertNotEqual(float((output.float() - values.float() @ linear.weight.T).abs().max()), 0.0)

    def test_dense_and_ternary_forwards_differ(self):
        torch.manual_seed(11)
        dense = self._linear("bf16", "dense")
        ternary = self._linear("bf16", "ternary")
        ternary.weight.data.copy_(dense.weight.data)
        values = torch.randn(9, 256, dtype=torch.bfloat16)
        with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm):
            dense_out = dense(values)
            ternary_out = ternary(values)
        self.assertGreater(error_ratio(dense_out.float(), ternary_out.float()), 0.05)

    def test_dense_backward_modes_produce_fp32_weight_gradient(self):
        for mode in ("bf16", "dgrad", "full"):
            with self.subTest(mode=mode):
                torch.manual_seed(13)
                linear = self._linear(mode, "dense")
                values = torch.randn(17, 256, dtype=torch.bfloat16, requires_grad=True)
                with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm):
                    loss = linear(values).square().mean()
                    loss.backward()
                self.assertEqual(linear.weight.grad.dtype, torch.float32)
                self.assertTrue(torch.isfinite(linear.weight.grad).all())
                self.assertIsNotNone(values.grad)

    def test_dense_weight_gradient_tracks_reference(self):
        # bf16 backward on emulated operands: dW ~= g^T @ qa(x) against the
        # dense FP8 forward, not the ternary one.
        torch.manual_seed(13)
        linear = self._linear("bf16", "dense")
        values = torch.randn(33, 256, dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(33, 512, dtype=torch.bfloat16)
        with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm):
            output = linear(values)
            output.backward(grad)
        from terminus.quantization import quantize_activation_fp8_e4m3, dequantize_activation_fp8_e4m3
        quantized_inputs = dequantize_activation_fp8_e4m3(
            *quantize_activation_fp8_e4m3(values), out_dtype=grad.dtype)
        reference = (grad.T @ quantized_inputs).float()  # mirrors the bf16 backward recipe
        self.assertLess(error_ratio(linear.weight.grad, reference), 1e-4)


class DenseTorchBackendTests(unittest.TestCase):
    def test_dense_torch_backend_is_exact_linear(self):
        torch.manual_seed(3)
        linear = BitLinear(32, 64, linear_backend="torch", weight_mode="dense")
        values = torch.randn(5, 32, dtype=torch.float32, requires_grad=True)
        output = linear(values)
        self.assertTrue(torch.equal(output, values @ linear.weight.T))
        output.sum().backward()
        self.assertIsNotNone(linear.weight.grad)
        # dense reference: the weight gradient is exact, not a surrogate
        torch.testing.assert_close(linear.weight.grad,
                                   values.detach().sum(dim=0).expand(64, 32))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            tiny_config(weight_mode="fp16")
        tiny_config(weight_mode="dense")  # valid
        tiny_config(dim=256, n_heads=4, n_kv_heads=2, ffn_dim=512, linear_backend="deepgemm",
                    weight_scale_layout="blockwise", quantize_activations=True,
                    weight_mode="dense")  # dense DeepGEMM is valid
        with self.assertRaises(ValueError):
            BitLinear(32, 64, weight_mode="fp4")


class WSDScheduleTests(unittest.TestCase):
    def test_wsd_phases(self):
        kwargs = dict(schedule="wsd", decay_tokens=20, decay_style="linear")
        self.assertEqual(schedule_scale(10, 100, 20, 0.1, **kwargs), 0.5)   # warmup midpoint
        self.assertEqual(schedule_scale(20, 100, 20, 0.1, **kwargs), 1.0)   # warmup end
        self.assertEqual(schedule_scale(50, 100, 20, 0.1, **kwargs), 1.0)   # stable
        self.assertEqual(schedule_scale(80, 100, 20, 0.1, **kwargs), 1.0)   # decay start
        self.assertAlmostEqual(schedule_scale(90, 100, 20, 0.1, **kwargs), 0.55)
        self.assertAlmostEqual(schedule_scale(100, 100, 20, 0.1, **kwargs), 0.1)

    def test_wsd_cosine_decay_matches_cosine_in_window(self):
        wsd = schedule_scale(95, 100, 10, 0.1, schedule="wsd", decay_tokens=20, decay_style="cosine")
        cosine = schedule_scale(25, 30, 10, 0.1)  # same 75% position in a pure cosine window
        self.assertAlmostEqual(wsd, cosine, places=12)

    def test_wsd_validation(self):
        with self.assertRaises(ValueError):
            schedule_scale(10, 100, 20, 0.1, schedule="wsd", decay_tokens=None)
        with self.assertRaises(ValueError):
            schedule_scale(10, 100, 20, 0.1, schedule="wsd", decay_tokens=85)  # warmup+decay > budget
        with self.assertRaises(ValueError):
            schedule_scale(10, 100, 20, 0.1, schedule="bogus")
        with self.assertRaises(ValueError):
            TrainConfig(schedule="wsd")  # missing decay budget
        with self.assertRaises(ValueError):
            TrainConfig(schedule="cosine", decay_tokens=10)
        TrainConfig(schedule="wsd", decay_tokens=10)  # valid


if __name__ == "__main__":
    unittest.main()
