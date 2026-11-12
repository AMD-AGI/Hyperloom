from __future__ import annotations

import argparse
import unittest

from vllm_launcher import apply_runtime_defaults, validate_model_input


def launcher_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "turing_compat": False,
        "attention_backend": None,
        "kv_cache_dtype": None,
        "block_size": None,
        "enforce_eager": False,
        "gpu_memory_utilization": None,
        "max_num_seqs": None,
        "max_num_batched_tokens": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class LauncherConfigurationTests(unittest.TestCase):
    def test_general_defaults_remain_unchanged(self) -> None:
        args = launcher_args()

        apply_runtime_defaults(args)

        self.assertEqual(args.gpu_memory_utilization, 0.6)
        self.assertEqual(args.max_num_seqs, 64)
        self.assertIsNone(args.max_num_batched_tokens)
        self.assertFalse(args.enforce_eager)

    def test_turing_profile_applies_reporter_confirmed_values(self) -> None:
        args = launcher_args(turing_compat=True)

        apply_runtime_defaults(args)

        self.assertEqual(args.attention_backend, "TRITON_ATTN")
        self.assertEqual(args.kv_cache_dtype, "float16")
        self.assertEqual(args.block_size, 32)
        self.assertTrue(args.enforce_eager)
        self.assertEqual(args.gpu_memory_utilization, 0.89)
        self.assertEqual(args.max_num_seqs, 1)
        self.assertEqual(args.max_num_batched_tokens, 2048)

    def test_turing_profile_preserves_explicit_overrides(self) -> None:
        args = launcher_args(
            turing_compat=True,
            attention_backend="CUSTOM_ATTN",
            kv_cache_dtype="auto",
            block_size=16,
            gpu_memory_utilization=0.8,
            max_num_seqs=2,
            max_num_batched_tokens=1024,
        )

        apply_runtime_defaults(args)

        self.assertEqual(args.attention_backend, "CUSTOM_ATTN")
        self.assertEqual(args.kv_cache_dtype, "auto")
        self.assertEqual(args.block_size, 16)
        self.assertEqual(args.gpu_memory_utilization, 0.8)
        self.assertEqual(args.max_num_seqs, 2)
        self.assertEqual(args.max_num_batched_tokens, 1024)
        self.assertTrue(args.enforce_eager)

    def test_direct_gguf_is_rejected_before_model_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "Direct GGUF files are not supported"):
            validate_model_input(r"E:\models\model.GGUF")

        validate_model_input("cyankiwi/Qwen3.5-9B-AWQ-4bit")


if __name__ == "__main__":
    unittest.main()
