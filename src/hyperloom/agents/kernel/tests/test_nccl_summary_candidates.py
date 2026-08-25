#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for :mod:`_nccl_summary_candidates` collective candidate injection."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from _nccl_summary_candidates import (  # type: ignore[import-not-found]
    _itanium_components,
    _prorated_totals,
    collective_symbol,
    extract_collective_candidates,
    index_device_symbols,
    locate_device_symbol,
)


AITER_2STAGE = (
    "_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPNS_6SignalEPT_ii"
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

    def test_call_site_is_not_a_device_definition(self) -> None:
        self._write("a/call.cu", "void host() { my_collective(p); }\n")
        self.assertIsNone(locate_device_symbol("my_collective", [str(self.root)]))

    def test_unknown_symbol_returns_none(self) -> None:
        self._write("a/def.cuh", "__global__ void other(int* p) {}\n")
        self.assertIsNone(locate_device_symbol("missing_kernel", [str(self.root)]))

    def test_empty_symbol_returns_none(self) -> None:
        self.assertIsNone(locate_device_symbol("", [str(self.root)]))

    def test_missing_root_is_skipped(self) -> None:
        self.assertIsNone(locate_device_symbol("x", [str(self.root / "nope")]))

    def test_multiple_symbols_read_each_source_once(self) -> None:
        """One index pass must not reread a file for each requested symbol."""
        source = self._write(
            "include/two.cuh",
            "__global__ void first_collective(int* p) {}\n__global__ void second_collective(int* p) {}\n",
        )
        original_read_text = Path.read_text
        source_reads = 0

        def counted_read_text(path: Path, *args: object, **kwargs: object) -> str:
            """Count reads of the source shared by both requested symbols."""
            nonlocal source_reads
            if path == source:
                source_reads += 1
            return original_read_text(path, *args, **kwargs)

        with unittest.mock.patch.object(Path, "read_text", counted_read_text):
            index, truncated = index_device_symbols(
                ["first_collective", "second_collective"],
                [str(self.root)],
            )

        self.assertFalse(truncated)
        self.assertEqual(set(index), {"first_collective", "second_collective"})
        self.assertEqual(source_reads, 1)

    def test_scan_cap_bounds_non_source_traversal(self) -> None:
        """Unrelated files must count toward the source-tree traversal cap."""
        self._write("include/00_notes.txt", "not device source\n")
        self._write(
            "include/01_collective.cuh",
            "__global__ void late_collective(int* p) {}\n",
        )

        with unittest.mock.patch(
            f"{index_device_symbols.__module__}._MAX_SCANNED_FILES",
            1,
        ):
            index, truncated = index_device_symbols(
                ["late_collective"],
                [str(self.root)],
            )

        self.assertEqual(index, {})
        self.assertTrue(truncated)


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

    def test_zero_weight_sample_is_rejected(self) -> None:
        ops = [{"name": "a", "duration_us": 0.0}]
        with self.assertRaises(ValueError):
            _prorated_totals(ops, 5.0, 10)

    def test_malformed_entries_are_rejected(self) -> None:
        for op in (
            "not-a-dict",
            {"name": "", "duration_us": 5.0},
            {"name": "a", "duration_us": "bad", "stream": "bad"},
        ):
            with self.subTest(op=op), self.assertRaises(ValueError):
                _prorated_totals([op], total_time_ms=1.0, total_count=2)

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
        self._write_metrics(self._summary(top_ops=[{"name": "ncclDevKernel_Generic_1", "duration_us": 10.0}]))
        messages: list[str] = []
        out = extract_collective_candidates(self.tl, [str(self.src_root)], log_fn=messages.append)
        self.assertEqual(out, [])
        self.assertEqual(len(messages), 1)
        self.assertIn("ncclDevKernel_Generic_1", messages[0])

    def test_candidates_sorted_by_duration(self) -> None:
        (self.src_root / "include" / "other.cuh").write_text("__global__ void small_collective(int* p) {}\n")
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

    def test_corrupt_metrics_file_is_rejected(self) -> None:
        (self.tl / "category_data" / "multi_kernel_metrics.json").write_text("{not json")
        with self.assertRaises(ValueError):
            extract_collective_candidates(self.tl, [str(self.src_root)])

    def test_absent_nccl_summary_yields_nothing(self) -> None:
        self._write_metrics({"status": "OK"})
        self.assertEqual(extract_collective_candidates(self.tl, [str(self.src_root)]), [])

    def test_empty_top_ops_yields_nothing(self) -> None:
        self._write_metrics(self._summary(top_ops=[]))
        self.assertEqual(extract_collective_candidates(self.tl, [str(self.src_root)]), [])

    def test_zero_total_time_is_rejected(self) -> None:
        self._write_metrics(self._summary(total_time_ms=0.0))
        with self.assertRaises(ValueError):
            extract_collective_candidates(self.tl, [str(self.src_root)])

    def test_no_source_roots_drops_everything(self) -> None:
        self._write_metrics(self._summary())
        self.assertEqual(extract_collective_candidates(self.tl, [""]), [])

    def test_main_flow_injection_skips_candidate_without_workload(self) -> None:
        """A source-only summary row is not a runnable candidate."""
        from tracelens_analysis import _inject_collective_candidates

        self._write_metrics(self._summary())
        existing = [{"name": "compute_kernel", "duration_us": 1000.0}]
        out = _inject_collective_candidates(
            self.tl,
            existing,
            source_roots=[str(self.src_root)],
        )
        self.assertEqual(out, existing)

    def test_nonmatching_workload_shapes_are_not_borrowed_by_default(self) -> None:
        """A different all-reduce row must not supply an unobserved workload."""
        from tracelens_analysis import _inject_collective_candidates

        self._write_metrics(self._summary())
        donor = {
            "name": "sgl_kernel::qr_all_reduce",
            "duration_us": 1000.0,
            "input_shapes": [{"shape": "(4096, 7168)"}],
            "input_dtypes": ["bf16"],
        }

        with unittest.mock.patch.dict(
            os.environ,
            {"HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES": ""},
        ):
            out = _inject_collective_candidates(
                self.tl,
                [donor],
                source_roots=[str(self.src_root)],
            )

        self.assertEqual(out, [donor])

    def test_truncated_source_scan_is_a_health_warning_after_partial_resolution(
        self,
    ) -> None:
        """A resolved row must not hide that another symbol exceeded the cap."""
        from tracelens_analysis import _inject_collective_candidates

        (self.src_root / "include" / "late.cuh").write_text("__global__ void late_collective(int* p) {}\n")
        self._write_metrics(
            self._summary(
                top_ops=[
                    {"name": AITER_2STAGE, "duration_us": 90.0},
                    {"name": "late_collective", "duration_us": 10.0},
                ]
            )
        )
        exact = {
            "name": AITER_2STAGE,
            "duration_us": 1000.0,
            "input_shapes": [{"shape": "(4096, 7168)"}],
            "input_dtypes": ["bf16"],
        }
        warnings: list[dict] = []

        with unittest.mock.patch(
            f"{index_device_symbols.__module__}._MAX_SCANNED_FILES",
            1,
        ):
            out = _inject_collective_candidates(
                self.tl,
                [exact],
                source_roots=[str(self.src_root)],
                health_warnings=warnings,
            )

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["candidate_source"], "nccl_summary")
        self.assertEqual(
            [warning["code"] for warning in warnings],
            ["collective_source_scan_truncated"],
        )
        self.assertEqual(warnings[0]["scanned_file_limit"], 1)

    def test_a_skipped_injection_is_reported_as_a_trace_health_warning(self) -> None:
        """A dirty summary must not look like a workload with no collective.

        Injection is the only path a collective has into the candidate pool, so
        a skip silently disables the entire lane unless it reaches the report.
        """
        from tracelens_analysis import _inject_collective_candidates

        self._write_metrics(self._summary(total_time_ms="not-a-number"))
        warnings: list[dict] = []

        out = _inject_collective_candidates(
            self.tl,
            [{"name": "compute_kernel", "duration_us": 1000.0}],
            source_roots=[str(self.src_root)],
            health_warnings=warnings,
        )

        self.assertEqual(len(out), 1)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["code"], "collective_summary_unusable")
        self.assertIn("collective optimization lane cannot run", warnings[0]["message"])

    def test_injection_without_a_source_root_is_also_reported(self) -> None:
        """The other skip path needs the same visibility."""
        import tracelens_analysis as tla

        self._write_metrics(self._summary())
        warnings: list[dict] = []

        with unittest.mock.patch.object(tla, "_aiter_csrc_root", return_value=""):
            tla._inject_collective_candidates(
                self.tl,
                [],
                source_roots=[],
                health_warnings=warnings,
            )

        self.assertEqual(
            [w["code"] for w in warnings],
            ["collective_source_root_missing"],
        )

    def test_opt_in_attaches_unique_all_reduce_workload(self) -> None:
        """The compatibility flag permits explicitly requested shape inference."""
        from tracelens_analysis import _inject_collective_candidates

        self._write_metrics(self._summary())
        donor = {
            "name": "sgl_kernel::qr_all_reduce",
            "duration_us": 1000.0,
            "input_shapes": [{"shape": "(4096, 7168)"}],
            "input_dtypes": ["bf16"],
        }
        with unittest.mock.patch.dict(
            os.environ,
            {"HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES": "1"},
        ):
            out = _inject_collective_candidates(
                self.tl,
                [donor],
                source_roots=[str(self.src_root)],
            )

        self.assertEqual(len(out), 2)
        self.assertEqual(out[1]["input_shapes"], donor["input_shapes"])
        self.assertEqual(out[1]["input_dtypes"], donor["input_dtypes"])
        self.assertEqual(
            out[1]["shape_provenance"],
            "borrowed_sole_all_reduce_family",
        )
        self.assertEqual(
            out[1]["workload_source_kernel"],
            donor["name"],
        )

    def test_opt_in_merges_one_profiled_workload_family(self) -> None:
        """Opted-in prefill and decode rows remain separate driver cases."""
        from tracelens_analysis import _inject_collective_candidates

        self._write_metrics(self._summary())
        donors = [
            {
                "name": ("sglang_profiler::tensor_model_parallel_allreduce->_Z_prefill (Synthetic Op) (prefill)"),
                "duration_us": 900.0,
                "call_count": 40,
                "shapes": ["(1024,5120) bf16", "(5120,) bf16"],
            },
            {
                "name": ("sglang_profiler::tensor_model_parallel_allreduce->_Z_decode (Synthetic Op) (decode)"),
                "duration_us": 100.0,
                "call_count": 40,
                "shapes": ["(64,5120) bf16", "(5120,) bf16"],
            },
        ]

        with unittest.mock.patch.dict(
            os.environ,
            {"HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES": "true"},
        ):
            out = _inject_collective_candidates(
                self.tl,
                donors,
                source_roots=[str(self.src_root)],
            )

        self.assertEqual(len(out), 3)
        candidate = out[-1]
        self.assertEqual(
            candidate["input_shapes"],
            [
                {"call_num": 40, "shape": "(1024,5120) bf16"},
                {"call_num": 40, "shape": "(64,5120) bf16"},
            ],
        )
        self.assertEqual(candidate["input_dtypes"], ["bf16", "bf16"])
        self.assertEqual(candidate["workload_source_kernel"], donors[0]["name"])
        self.assertEqual(
            candidate["workload_source_kernels"],
            [donor["name"] for donor in donors],
        )

    def test_main_flow_injection_rejects_ambiguous_workload_families(self) -> None:
        """Unrelated all-reduce wrappers must not donate an arbitrary shape."""
        from tracelens_analysis import _inject_collective_candidates

        self._write_metrics(self._summary())
        donors = [
            {
                "name": "sglang_profiler::first_allreduce->_Z_first",
                "duration_us": 900.0,
                "shapes": ["(1024,5120) bf16"],
            },
            {
                "name": "sglang_profiler::second_allreduce->_Z_second",
                "duration_us": 100.0,
                "shapes": ["(64,5120) bf16"],
            },
        ]

        with unittest.mock.patch.dict(
            os.environ,
            {"HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES": "yes"},
        ):
            out = _inject_collective_candidates(
                self.tl,
                donors,
                source_roots=[str(self.src_root)],
            )

        self.assertEqual(out, donors)

    def test_main_flow_isolates_invalid_summary(self) -> None:
        """Invalid NCCL metrics must not abort the full trace analysis."""
        from tracelens_analysis import _inject_collective_candidates

        (self.tl / "category_data" / "multi_kernel_metrics.json").write_text("{not json")
        existing = [{"name": "compute_kernel", "duration_us": 1000.0}]
        log_path = self.tl / "analysis.log"

        out = _inject_collective_candidates(
            self.tl,
            existing,
            source_roots=[str(self.src_root)],
            log_path=log_path,
        )

        self.assertEqual(out, existing)
        self.assertIn("invalid TraceLens metrics file", log_path.read_text())

    def test_main_flow_injection_merges_exact_candidate(self) -> None:
        """Exact trace rows receive source metadata without duplication."""
        from tracelens_analysis import _inject_collective_candidates

        self._write_metrics(self._summary())
        existing = [{"name": AITER_2STAGE, "duration_us": 1000.0}]
        out = _inject_collective_candidates(
            self.tl,
            existing,
            source_roots=[str(self.src_root)],
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], AITER_2STAGE)
        self.assertEqual(out[0]["candidate_source"], "nccl_summary")
        self.assertTrue(out[0]["source_file"].endswith("custom_all_reduce.cuh"))


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
