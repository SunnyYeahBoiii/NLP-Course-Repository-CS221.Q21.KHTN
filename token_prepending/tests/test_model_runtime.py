import os
import sys
import unittest
from unittest import mock


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_runtime import build_loading_kwargs, resolve_torch_device


class ModelRuntimeTest(unittest.TestCase):
    @mock.patch("torch.backends.mps.is_available", return_value=True)
    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_auto_prefers_mps_on_apple_silicon(self, _cuda_available, _mps_available):
        self.assertEqual(resolve_torch_device("auto").type, "mps")

    @mock.patch("torch.backends.mps.is_available", return_value=False)
    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_auto_uses_cuda_when_mps_unavailable(self, _cuda_available, _mps_available):
        self.assertEqual(resolve_torch_device("auto").type, "cuda")

    def test_build_loading_kwargs_uses_cache_and_mps_device_map(self):
        kwargs = build_loading_kwargs(
            device_type="mps",
            cache_dir=".cache/huggingface",
            output_hidden_states=True,
        )

        self.assertEqual(kwargs["cache_dir"], ".cache/huggingface")
        self.assertEqual(kwargs["device_map"], "mps")
        self.assertTrue(kwargs["output_hidden_states"])
        self.assertTrue(kwargs["trust_remote_code"])


if __name__ == "__main__":
    unittest.main()
