#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for :mod:`kernel-agent.tools._collective_names` (locks r24-regression collective detection)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from _collective_names import (  # type: ignore[import-not-found]
    _normalise_kernel_name,
    kernel_name_implies_multigpu,
)


class CollectiveNameDetectionTests(unittest.TestCase):
    def test_custom_allreduce_variants(self) -> None:
        # r24 stall names plus framework-dispatcher variants.
        for name in [
            "custom_allreduce",
            "CustomAllReduce",
            "custom_all_reduce",
            "rccl_AllReduce",
            "vllm_custom_allreduce_kernel",
        ]:
            self.assertTrue(kernel_name_implies_multigpu(name), msg=name)

    def test_other_collectives_match(self) -> None:
        for name in [
            "nccl_all_gather",
            "triton_all_gather_fwd",
            "all_gather_along_first_dim",
            "ReduceScatterFusion",
            "reduce_scatter_grad",
            "all_to_all_dispatch",
            "rccl_broadcast",
            "broadcast_kernel",
            "nccl_send",   # NCCL p2p still routes through NCCL collectives
            "rccl_AllToAll",
        ]:
            self.assertTrue(kernel_name_implies_multigpu(name), msg=name)

    def test_non_collectives_do_not_match(self) -> None:
        # Names containing reduce/all/broadcast as a substring but not collectives.
        for name in [
            "rms_norm",
            "rotary_embedding",
            "gemm_a8w8_blockscale",
            "fused_moe",
            "flash_attention_v3",
            "reduce_max",       # reduce only, not reduce_scatter
            "reduce_sum_kernel",
            "reduce_kernel",
            "smallreduce",      # NOT a word-bounded "all_reduce"
            "tall_gemm",        # NOT a bounded "all_gather"
            "broadcastable_check",  # boundary: "broadcast" only matches when followed by "_"
            "",
        ]:
            with self.subTest(name=name):
                if name == "broadcastable_check":
                    # broadcast matches only when followed by "_"; this has none, so no match.
                    self.assertFalse(kernel_name_implies_multigpu(name))
                else:
                    self.assertFalse(kernel_name_implies_multigpu(name))

    def test_normalisation_handles_camel_case_and_delims(self) -> None:
        self.assertEqual(_normalise_kernel_name("CustomAllReduce"),
                         "custom_all_reduce")
        self.assertEqual(_normalise_kernel_name("rccl.AllGather"),
                         "rccl_all_gather")
        self.assertEqual(_normalise_kernel_name("triton-all-to-all-fwd"),
                         "triton_all_to_all_fwd")
        self.assertEqual(_normalise_kernel_name("__leading_underscores"),
                         "leading_underscores")
        self.assertEqual(_normalise_kernel_name(""), "")

    def test_returns_false_on_empty_or_none_like(self) -> None:
        self.assertFalse(kernel_name_implies_multigpu(""))
        self.assertFalse(kernel_name_implies_multigpu("   "))


if __name__ == "__main__":
    unittest.main()
