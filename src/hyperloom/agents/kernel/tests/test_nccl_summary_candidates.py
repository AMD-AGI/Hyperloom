#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for :mod:`_nccl_summary_candidates` collective candidate injection."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from _nccl_summary_candidates import (  # type: ignore[import-not-found]
    _itanium_components,
    _prorated_totals,
    collective_symbol,
    extract_collective_candidates,
    locate_device_symbol,
)


AITER_2STAGE = (
    "_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataES2_"
    "NS_11RankSignalsEPNS_6SignalEPT_ii"
)


class ItaniumParsingTests(unittest.TestCase):
    def test_nested_symbol_splits_into_components(self) -> None:
        self.assertEqual(
            _itanium_components(AITER_2STAGE),
            ["aiter", "cross_device_reduce_2stage"],
        )

    def test_non_nested_name_yields_nothing(self) -> None:
        self.assertEqual(_itanium_components("ncclDevKernel_Generic_1"), [])

    def test_truncated_length_prefix_does_not_overrun(self) -> None:
        # Length 99 exceeds what remains, so parsing stops instead of slicing
        # past the end of the string.
        self.assertEqual(_itanium_components("_ZN99short"), [])

    def test_zero_length_component_stops_parsing(self) -> None:
        self.assertEqual(_itanium_components("_ZN0abc"), [])


class CollectiveSymbolTests(unittest.TestCase):
    def test_mangled_name_yields_device_function(self) -> None:
        self.assertEqual(collective_symbol(AITER_2STAGE), "cross_device_reduce_2stage")

    def test_demangled_name_drops_namespace_and_template(self) -> None:
        self.assertEqual(
            collective_symbol("aiter::cross_device_reduce_1stage<__hip_bfloat16, 8>"),
            "cross_device_reduce_1stage",
        )

    def test_plain_name_passes_through(self) -> None:
        self.assertEqual(collective_symbol("ncclDevKernel_Generic_1"), "ncclDevKernel_Generic_1")

    def test_empty_name_yields_empty(self) -> None:
        self.assertEqual(collective_symbol(""), "")
        self.assertEqual(collective_symbol("   "), "")


class SymbolLocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, relpath: str, text: str) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_exact_symbol_wins_over_suffixed_neighbour(self) -> None:
        # A prefix match on cross_device_reduce_2stage_naive must not be
        # mistaken for the real kernel.
        self._write(
            "include/custom_all_reduce.cuh",
            "__global__ void cross_device_reduce_2stage_naive(RankData* d) {}\n"
            "__global__ void cross_device_reduce_2stage(RankData* d) {}\n",
        )
        located = locate_device_symbol("cross_device_reduce_2stage", [str(self.root)])
        self.assertIsNotNone(located)
        assert located is not None
        self.assertEqual(located[1], 2)
        self.assertEqual(located[2], "cross_device_reduce_2stage")

    def test_global_qualifier_on_preceding_line_is_seen(self) -> None:
        self._write(
            "include/k.cuh",
            "__global__ void __launch_bounds__(512, 1)\n    my_collective(int* p) {}\n",
        )
        located = locate_device_symbol("my_collective", [str(self.root)])
        self.assertIsNotNone(located)
        assert located is not None
        self.assertEqual(located[1], 2)

    def test_global_definition_preferred_over_call_site(self) -> None:
        self._write("a/call.cu", "void host() { my_collective(p); }\n")
        self._write("b/def.cuh", "__global__ void my_collective(int* p) {}\n")
        located = locate_device_symbol("my_collective", [str(self.root)])
        self.assertIsNotNone(located)
        assert located is not None
        self.assertTrue(located[0].endswith("def.cuh"))

    def test_call_site_used_as_fallback_when_no_global(self) -> None:
        self._write("a/call.cu", "void host() { my_collective(p); }\n")
        located = locate_device_symbol("my_collective", [str(self.root)])
        self.assertIsNotNone(located)
        assert located is not None
        self.assertTrue(located[0].endswith("call.cu"))

    def test_unknown_symbol_returns_none(self) -> None:
        self._write("a/def.cuh", "__global__ void other(int* p) {}\n")
        self.assertIsNone(locate_device_symbol("missing_kernel", [str(self.root)]))

    def test_empty_symbol_returns_none(self) -> None:
        self.assertIsNone(locate_device_symbol("", [str(self.root)]))

    def test_missing_root_is_skipped(self) -> None:
        self.assertIsNone(locate_device_symbol("x", [str(self.root / "nope")]))


class ProratedTotalsTests(unittest.TestCase):
    def test_single_name_absorbs_whole_total(self) -> None:
        ops = [{"name": "k", "duration_us": 100.0, "stream": 4}] * 3
        out = _prorated_totals(ops, total_time_ms=10.0, total_count=900)
        self.assertEqual(list(out), ["k"])
        duration_us, calls, stream = out["k"]
        self.assertAlmostEqual(duration_us, 10_000.0)
        self.assertEqual(calls, 900)
        self.assertEqual(stream, 4)

    def test_two_names_split_by_sampled_weight(self) -> None:
        ops = [
            {"name": "a", "duration_us": 300.0, "stream": 4},
            {"name": "b", "duration_us": 100.0, "stream": 5},
        ]
        out = _prorated_totals(ops, total_time_ms=8.0, total_count=400)
        self.assertEqual(list(out), ["a", "b"])  # ordered by descending weight
        self.assertAlmostEqual(out["a"][0], 6000.0)
        self.assertAlmostEqual(out["b"][0], 2000.0)
        self.assertEqual(out["a"][1], 300)
        self.assertEqual(out["b"][1], 100)

    def test_zero_weight_sample_yields_nothing(self) -> None:
        ops = [{"name": "a", "duration_us": 0.0}]
        self.assertEqual(_prorated_totals(ops, 5.0, 10), {})

    def test_malformed_entries_are_ignored(self) -> None:
        ops = [
            "not-a-dict",
            {"name": "", "duration_us": 5.0},
            {"name": "a", "duration_us": "bad", "stream": "bad"},
            {"name": "a", "duration_us": 10.0},
        ]
        out = _prorated_totals(ops, total_time_ms=1.0, total_count=2)
        self.assertEqual(list(out), ["a"])
        self.assertEqual(out["a"][2], 0)

    def test_call_count_never_rounds_to_zero(self) -> None:
        ops = [
            {"name": "big", "duration_us": 10_000.0},
            {"name": "tiny", "duration_us": 1.0},
        ]
        out = _prorated_totals(ops, total_time_ms=1.0, total_count=10)
        self.assertGreaterEqual(out["tiny"][1], 1)


class ExtractCollectiveCandidatesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tl = Path(self._tmp.name) / "tracelens"
        (self.tl / "category_data").mkdir(parents=True)
        self.src_root = Path(self._tmp.name) / "csrc"
        (self.src_root / "include").mkdir(parents=True)
        (self.src_root / "include" / "custom_all_reduce.cuh").write_text(
            "__global__ void cross_device_reduce_2stage(RankData* d) {}\n"
        )
        self.addCleanup(self._tmp.cleanup)

    def _write_metrics(self, payload: object) -> None:
        (self.tl / "category_data" / "multi_kernel_metrics.json").write_text(json.dumps(payload))

    def _summary(self, **over: object) -> dict:
        base = {
            "nccl_summary": {
                "total_count": 8960,
                "total_time_ms": 478.846,
                "top_ops": [{"name": AITER_2STAGE, "duration_us": 3925.43, "stream": 4}],
            }
        }
        base["nccl_summary"].update(over)  # type: ignore[union-attr]
        return base

    def test_resolvable_collective_becomes_candidate(self) -> None:
        self._write_metrics(self._summary())
        out = extract_collective_candidates(self.tl, [str(self.src_root)])
        self.assertEqual(len(out), 1)
        cand = out[0]
        self.assertEqual(cand["name"], AITER_2STAGE)
        self.assertTrue(cand["source_file"].endswith("custom_all_reduce.cuh"))
        self.assertEqual(cand["source_line"], 1)
        self.assertEqual(cand["source_function"], "cross_device_reduce_2stage")
        self.assertEqual(cand["source_resolution_method"], "nccl_summary_symbol_lookup")
        self.assertEqual(cand["candidate_source"], "nccl_summary")
        self.assertTrue(cand["is_multigpu"])
        self.assertEqual(cand["tracelens_category"], "collective")
        self.assertEqual(cand["bound_type"], "communication")
        self.assertAlmostEqual(cand["duration_us"], 478_846.0, places=1)
        self.assertEqual(cand["call_count"], 8960)
        self.assertEqual(cand["collective_stream"], 4)
        self.assertIn("prorated", cand["duration_provenance"])

    def test_unresolvable_symbol_is_dropped_and_logged(self) -> None:
        self._write_metrics(
            self._summary(top_ops=[{"name": "ncclDevKernel_Generic_1", "duration_us": 10.0}])
        )
        messages: list[str] = []
        out = extract_collective_candidates(
            self.tl, [str(self.src_root)], log_fn=messages.append
        )
        self.assertEqual(out, [])
        self.assertEqual(len(messages), 1)
        self.assertIn("ncclDevKernel_Generic_1", messages[0])

    def test_candidates_sorted_by_duration(self) -> None:
        (self.src_root / "include" / "other.cuh").write_text(
            "__global__ void small_collective(int* p) {}\n"
        )
        self._write_metrics(
            self._summary(
                top_ops=[
                    {"name": "small_collective", "duration_us": 10.0},
                    {"name": AITER_2STAGE, "duration_us": 90.0},
                ]
            )
        )
        out = extract_collective_candidates(self.tl, [str(self.src_root)])
        self.assertEqual([c["name"] for c in out], [AITER_2STAGE, "small_collective"])

    def test_missing_metrics_file_yields_nothing(self) -> None:
        self.assertEqual(extract_collective_candidates(self.tl, [str(self.src_root)]), [])

    def test_corrupt_metrics_file_yields_nothing(self) -> None:
        (self.tl / "category_data" / "multi_kernel_metrics.json").write_text("{not json")
        self.assertEqual(extract_collective_candidates(self.tl, [str(self.src_root)]), [])

    def test_absent_nccl_summary_yields_nothing(self) -> None:
        self._write_metrics({"status": "OK"})
        self.assertEqual(extract_collective_candidates(self.tl, [str(self.src_root)]), [])

    def test_empty_top_ops_yields_nothing(self) -> None:
        self._write_metrics(self._summary(top_ops=[]))
        self.assertEqual(extract_collective_candidates(self.tl, [str(self.src_root)]), [])

    def test_zero_total_time_yields_nothing(self) -> None:
        self._write_metrics(self._summary(total_time_ms=0.0))
        self.assertEqual(extract_collective_candidates(self.tl, [str(self.src_root)]), [])

    def test_no_source_roots_drops_everything(self) -> None:
        self._write_metrics(self._summary())
        self.assertEqual(extract_collective_candidates(self.tl, [""]), [])


class CollectiveContractTests(unittest.TestCase):
    """The contract must name all-reduce semantics for aiter's custom kernels."""

    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        from tracelens_analysis import _enrich_kernel_contract  # type: ignore[import-not-found]

        self.enrich = _enrich_kernel_contract

    def test_cross_device_reduce_maps_to_all_reduce(self) -> None:
        # A bare "reduce" match would pick torch.distributed.reduce, whose
        # result only lands on rank 0, silently voiding the parity gate.
        item = {"name": AITER_2STAGE}
        self.enrich(item, {"TP_SIZE": 8})
        contract = item["kernel_contract"]
        self.assertEqual(contract["kind"], "collective")
        self.assertEqual(contract["collective_op"], "all_reduce")
        self.assertEqual(contract["reference"], "torch.distributed.all_reduce")
        self.assertEqual(contract["tp_size"], 8)

    def test_genuine_reduce_still_maps_to_reduce(self) -> None:
        item = {"name": "my_reduce_kernel"}
        self.enrich(item, {})
        self.assertEqual(item["kernel_contract"]["collective_op"], "reduce")

    def test_reduce_scatter_unaffected(self) -> None:
        item = {"name": "aiter_reduce_scatter_bf16"}
        self.enrich(item, {})
        self.assertEqual(item["kernel_contract"]["collective_op"], "reduce_scatter")


if __name__ == "__main__":
    unittest.main()
