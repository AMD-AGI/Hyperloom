#!/usr/bin/env python3
"""Tests for `tools/unittest_agent.py`.

Covers:
  * Generation of the AgentKernelArena task layout from a synthetic
    candidate (Triton kernel from this repo's fixtures).
  * AST validity of the rendered ``task_runner.py``.
  * Self-verify success on the unmodified kernel (compile + correctness).
  * Self-verify failure when the live source is mutated (the snapshot
    correctly catches the regression).
  * Performance mode emits the AgentKernelArena schema (test_case_id +
    execution_time_ms + params).
  * Degraded path when ``input_shapes`` is empty (no CUDA execution).

These tests use the same kernel fixtures the Hyperloom optimizer hands
to GEAK (real Triton from sgl-workspace / AgentKernelArena), and only
exercise GPU-dependent paths when CUDA is actually available.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import unittest_agent as ua  # noqa: E402


# Pick a fixture: the AgentKernelArena triton_bmm kernel is self-contained
# (single host launcher + single jit body), small enough to compile in
# < 5s, and exercises the most common shape pattern. If the AgentKernelArena
# checkout is not mounted, we skip everything that needs the fixture.
AKA_BMM = Path(
    "/wekafs/zihao/2026/geak_cc/AgentKernelArena/tasks/triton2triton/vllm"
    "/triton_bmm/source/triton_bmm.py"
)


def _has_cuda() -> bool:
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(AKA_BMM.is_file(), "AgentKernelArena bmm fixture not mounted")
class UnittestAgentGenerationTests(unittest.TestCase):
    """Pure-Python generation paths (no CUDA required)."""

    def _bmm_candidate(self) -> dict:
        return {
            "kernel_id": "triton_bmm_kernel",
            "name": "bmm_kernel",
            "kernel_name": "bmm_kernel",
            "source_file": str(AKA_BMM),
            "kernel_url": str(AKA_BMM),
            "input_shapes": [[2, 64, 32], [2, 32, 64]],
            "input_dtypes": ["float16", "float16"],
            "env_vars": {"TRITON_PRINT_AUTOTUNING": "0"},
            "kernel_repo": "/sgl-workspace/vllm",
            "gpu_pct": "12.3%",
            "bound_type": "compute-bound",
        }

    def test_generates_aka_layout(self):
        cand = self._bmm_candidate()
        with tempfile.TemporaryDirectory(prefix="ua_layout_") as td:
            out = Path(td) / "task"
            m = ua.generate_unittest(cand, out_dir=out, self_verify=False)
            self.assertIn(m["status"], ("ok", "degraded"))
            self.assertTrue((out / "config.yaml").is_file())
            self.assertTrue((out / "scripts" / "task_runner.py").is_file())
            self.assertTrue((out / "source" / "triton_bmm.py").exists())
            self.assertTrue((out / "source" / "_baseline_snapshot" / "triton_bmm.py").is_file())
            # config carries the AgentKernelArena keys
            cfg = (out / "config.yaml").read_text()
            for key in ("source_file_path", "target_kernel_functions",
                        "compile_command", "correctness_command",
                        "performance_command", "task_type"):
                self.assertIn(key, cfg, f"config missing {key}")
            self.assertIn("source/triton_bmm.py", cfg)
            self.assertIn("triton2triton", cfg)
            # task_runner.py is a valid Python file
            ast.parse((out / "scripts" / "task_runner.py").read_text())

    def test_host_entry_resolution_prefers_launcher_over_jit(self):
        cand = self._bmm_candidate()
        with tempfile.TemporaryDirectory(prefix="ua_entry_") as td:
            m = ua.generate_unittest(cand, out_dir=Path(td) / "task", self_verify=False)
            # bmm_kernel is the @triton.jit body; bmm_triton is the host
            # launcher — the latter is the only one callable from torch
            # without preparation, so the picker MUST prefer it.
            self.assertEqual(m["host_entry"], "bmm_triton")
            # The original kernel symbol is still in target_kernels so
            # config.yaml's correctness check still imports it.
            self.assertIn("bmm_kernel", m["target_kernels"])

    def test_host_entry_picker_picks_arg_count_match_over_first_public(self):
        """When the source file has many small helpers (``num_programs``,
        ``block_size``, ``use_blocked`` ...) plus the real launcher (with
        N tensor args), the picker MUST prefer the launcher based on the
        candidate's ``input_shapes`` arg count. Regression test for the
        live aiter `rms_norm` case (rmsnorm.py has 16 public functions;
        the first by file order is the 1-arg helper ``num_programs``)."""
        with tempfile.TemporaryDirectory(prefix="ua_picker_") as td:
            src = Path(td) / "kernel_mod.py"
            src.write_text(
                "import triton\n"
                "import triton.language as tl\n"
                "@triton.jit\n"
                "def _foo_kernel(x):\n    pass\n"
                "def num_programs(x):\n    return 1\n"
                "def block_size(x):\n    return 1\n"
                "def use_blocked(x):\n    return False\n"
                "def foo(input, weight, epsilon):\n"
                "    return input\n"
                "def helper(a, b):\n"
                "    return a + b\n"
            )
            cand = {
                "kernel_id": "foo_k", "name": "_foo_kernel", "kernel_name": "_foo_kernel",
                "source_file": str(src),
                "input_shapes": [[2, 4096], [4096]],
                "input_dtypes": ["bfloat16", "bfloat16"],
            }
            m = ua.generate_unittest(cand, out_dir=Path(td) / "task",
                                     self_verify=False)
            self.assertEqual(m["host_entry"], "foo",
                             f"picker missed the 3-arg launcher; chose {m['host_entry']!r}")

    def test_env_vars_redact_secrets(self):
        cand = self._bmm_candidate()
        cand["env_vars"] = {
            "SGLANG_OPT_USE_TILELANG_INDEXER": "true",
            "MY_API_KEY": "should-not-appear",
            "AUTH_TOKEN": "should-not-appear",
            "AITER_USE_TRITON": "1",
        }
        with tempfile.TemporaryDirectory(prefix="ua_secret_") as td:
            out = Path(td) / "task"
            ua.generate_unittest(cand, out_dir=out, self_verify=False)
            runner_text = (out / "scripts" / "task_runner.py").read_text()
            self.assertIn("SGLANG_OPT_USE_TILELANG_INDEXER", runner_text)
            self.assertIn("AITER_USE_TRITON", runner_text)
            self.assertNotIn("should-not-appear", runner_text)
            self.assertNotIn("MY_API_KEY", runner_text)
            self.assertNotIn("AUTH_TOKEN", runner_text)

    def test_degraded_when_no_shapes(self):
        cand = self._bmm_candidate()
        cand.pop("input_shapes")
        with tempfile.TemporaryDirectory(prefix="ua_noshape_") as td:
            m = ua.generate_unittest(cand, out_dir=Path(td) / "task", self_verify=False)
            self.assertEqual(m["num_shapes"], 0)
            self.assertTrue(any("no input shapes" in w for w in m["warnings"]))

    def test_skips_hip_kernel(self):
        # Make a tiny fake .cu source so we don't need to ship one
        with tempfile.TemporaryDirectory(prefix="ua_hip_") as td:
            src = Path(td) / "kernel.cu"
            src.write_text("__global__ void k() {}\n")
            cand = {
                "kernel_id": "k", "name": "k", "kernel_name": "k",
                "source_file": str(src), "input_shapes": [[8]],
            }
            m = ua.generate_unittest(cand, out_dir=Path(td) / "task", self_verify=False)
            self.assertEqual(m["status"], "skipped")
            self.assertIn("only generates Python/Triton", m["error"])

    def test_missing_source_file(self):
        cand = {"kernel_id": "x", "name": "x",
                "source_file": "/this/path/does/not/exist.py"}
        with tempfile.TemporaryDirectory() as td:
            m = ua.generate_unittest(cand, out_dir=Path(td) / "task", self_verify=False)
            self.assertEqual(m["status"], "failed")
            self.assertIn("does not exist", m["error"])


AITER_RMSNORM = Path("/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py")


@unittest.skipUnless(AITER_RMSNORM.is_file() and _has_cuda(),
                     "needs /sgl-workspace/aiter + CUDA-capable GPU")
class UnittestAgentLiveAiterTests(unittest.TestCase):
    """End-to-end self-verify against the LIVE aiter rms_norm kernel —
    the production target of the Hyperloom optimizer. Captures the
    exact scenario the optimizer hands to GEAK: a candidate with
    `source_file` pointing at `/sgl-workspace/aiter/...`, real input
    shapes (bf16 [2, 4096] x [4096]) from a Qwen3 decode trace, and
    SGLang/aiter env vars from the live serving process.

    This test pins the contract that:
      1. The picker resolves `_rms_norm_kernel` → `rms_norm` (the public
         host launcher) — NOT `num_programs` or another small helper.
      2. The extra scalar arg (`epsilon`) gets auto-filled to `1e-6`.
      3. Self-verify returns status="ok" (compile + correctness pass on
         the unmodified live source).
    """

    def test_live_aiter_rmsnorm_passes_self_verify(self):
        cand = {
            "kernel_id": "qwen3_rms_norm_aiter",
            "name": "_rms_norm_kernel",
            "kernel_name": "_rms_norm_kernel",
            "source_file": str(AITER_RMSNORM),
            "input_shapes": [[2, 4096], [4096]],
            "input_dtypes": ["bfloat16", "bfloat16"],
            "kernel_repo": "/sgl-workspace/aiter",
            "env_vars": {"AITER_USE_TRITON": "1"},
            "bound_type": "memory-bound",
        }
        with tempfile.TemporaryDirectory(prefix="ua_live_") as td:
            m = ua.generate_unittest(
                cand, out_dir=Path(td) / "task",
                target_platform="mi300x", self_verify=True,
            )
            self.assertEqual(m["host_entry"], "rms_norm",
                             f"picker chose wrong entry: {m['host_entry']!r}; "
                             "expected `rms_norm` (the public 3-arg launcher).")
            self.assertEqual(m["status"], "ok",
                             f"live aiter rmsnorm self_verify failed: "
                             f"{m['self_verify']}")
            self.assertTrue(any("scalar arg" in w for w in m["warnings"]),
                            "expected warning about auto-filled epsilon")


@unittest.skipUnless(AKA_BMM.is_file() and _has_cuda(),
                     "needs AgentKernelArena fixture + CUDA-capable GPU")
class UnittestAgentSelfVerifyTests(unittest.TestCase):
    """End-to-end self-verify paths (require GPU)."""

    def _bmm_candidate(self) -> dict:
        return {
            "kernel_id": "triton_bmm_kernel",
            "name": "bmm_kernel",
            "kernel_name": "bmm_kernel",
            "source_file": str(AKA_BMM),
            "input_shapes": [[2, 64, 32], [2, 32, 64]],
            "input_dtypes": ["float16", "float16"],
            "env_vars": {},
        }

    def test_self_verify_passes_on_unmodified_source(self):
        with tempfile.TemporaryDirectory(prefix="ua_sv_ok_") as td:
            out = Path(td) / "task"
            m = ua.generate_unittest(
                self._bmm_candidate(), out_dir=out, self_verify=True,
            )
            self.assertEqual(m["status"], "ok",
                             f"unexpected status {m['status']}: {m['self_verify']}")
            self.assertEqual(m["self_verify"]["compile"], "ok")
            self.assertEqual(m["self_verify"]["correctness"], "ok")

    def test_correctness_detects_mutation(self):
        with tempfile.TemporaryDirectory(prefix="ua_sv_mut_") as td:
            out = Path(td) / "task"
            m = ua.generate_unittest(
                self._bmm_candidate(), out_dir=out, self_verify=False,
            )
            runner = Path(m["task_runner"])
            live = Path(m["source_file"])
            # Detach the symlink so we can mutate without touching the
            # /sgl-workspace original.
            if live.is_symlink():
                actual = live.resolve()
                live.unlink()
                shutil.copy2(actual, live)
            text = live.read_text()
            broken = text.replace(
                "tl.store(c_ptrs, c, mask=c_mask)",
                "tl.store(c_ptrs, c + 1.0, mask=c_mask)",
            )
            self.assertNotEqual(broken, text, "could not inject regression")
            live.write_text(broken)
            proc = subprocess.run(
                [sys.executable, str(runner), "correctness"],
                cwd=str(out), capture_output=True, text=True, timeout=300,
            )
            self.assertNotEqual(proc.returncode, 0,
                                "correctness should have failed for mutated kernel; "
                                f"stdout={proc.stdout}")
            self.assertIn("FAIL", proc.stdout)

    def test_performance_emits_aka_schema(self):
        with tempfile.TemporaryDirectory(prefix="ua_sv_perf_") as td:
            out = Path(td) / "task"
            m = ua.generate_unittest(
                self._bmm_candidate(), out_dir=out, self_verify=False,
            )
            runner = Path(m["task_runner"])
            proc = subprocess.run(
                [sys.executable, str(runner), "performance"],
                cwd=str(out), capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads(
                (out / "build" / "performance_report.json").read_text()
            )
            self.assertIsInstance(report, list)
            self.assertGreater(len(report), 0)
            for case in report:
                self.assertIn("test_case_id", case)
                self.assertIn("execution_time_ms", case)
                self.assertIn("params", case)


if __name__ == "__main__":
    unittest.main()
