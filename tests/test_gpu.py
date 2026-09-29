import unittest
from dataclasses import replace

import torch

from terminus.preflight import run_preflight
from tests.helpers import tiny_config


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: these are mandatory hardware gates, not certified passes")
class GPUPreflightTests(unittest.TestCase):
    def test_flex_forward_backward_and_cached_offsets(self):
        config = tiny_config(dim=128, n_heads=2, n_kv_heads=1, max_seq_len=32, attention_backend="flex")
        result = run_preflight(config, torch.device("cuda", 0))
        self.assertEqual(result["status"], "passed")

    def test_fa3_forward_backward_and_cached_offsets(self):
        # The pinned production FA3 build enables BF16 head dimension 128.
        config = tiny_config(dim=256, n_heads=2, n_kv_heads=1, max_seq_len=32, attention_backend="fa3")
        result = run_preflight(config, torch.device("cuda", 0))
        self.assertEqual(result["status"], "passed")

    def test_deepgemm_full_fp8_backward_and_fused_cache(self):
        config = tiny_config(dim=512, n_heads=4, n_kv_heads=2, ffn_dim=1024, max_seq_len=32,
                             attention_backend="fa3", linear_backend="deepgemm", quantize_activations=True,
                             weight_scale_layout="blockwise", fp8_backward="full", fp8_quantizer="triton",
                             fp8_cache_weights=True)
        result = run_preflight(config, torch.device("cuda", 0))
        self.assertEqual(result["deepgemm"]["backward_mode"], "full")
        self.assertTrue(result["deepgemm"]["quantizers"])

    def test_deepgemm_native_forward_and_ste(self):
        config = tiny_config(dim=512, n_heads=4, n_kv_heads=2, ffn_dim=1024, max_seq_len=32,
                             attention_backend="fa3", linear_backend="deepgemm", quantize_activations=True,
                             weight_scale_layout="blockwise")
        result = run_preflight(config, torch.device("cuda", 0))
        self.assertTrue(result["deepgemm"])

    def test_large_packed_activation_offsets(self):
        # The first packed row past 2^31 elements used to wrap 32-bit Triton
        # offsets in SwiGLU and FP8 activation/gradient quantization.
        if torch.cuda.mem_get_info()[0] < 24 * (1 << 30):
            self.skipTest("large-offset regression needs at least 24 GiB free GPU memory")
        from terminus.fp8_kernels import quantize_directions
        from terminus.mlp_kernels import packed_backward, packed_forward

        width = 5760
        rows = (1 << 31) // (2 * width) + 2
        packed = torch.zeros((rows, 2 * width), device="cuda", dtype=torch.bfloat16)
        gain = torch.ones((), device="cuda")
        output, inverse = packed_forward(packed, gain, 1e-5)
        self.assertEqual(float(output[-1, -1]), 0.)
        gradient, gain_gradient = packed_backward(torch.zeros_like(output), packed, gain, inverse)
        self.assertEqual(float(gradient[-1, -1]), 0.)
        self.assertEqual(float(gain_gradient), 0.)
        del gradient, gain_gradient, output, inverse
        rowwise, transposed = quantize_directions(packed, rowwise=True, transposed=True, separate=True)
        self.assertEqual(float(rowwise[0][-1, -1]), 0.)
        self.assertEqual(float(transposed[0][-1, -1]), 0.)
