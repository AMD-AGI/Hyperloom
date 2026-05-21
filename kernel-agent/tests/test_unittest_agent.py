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

These tests generate all portable fixtures at runtime and only exercise
GPU-dependent paths when CUDA is actually available.
"""

from __future__ import annotations

import ast
import gzip
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


def _write_bmm_fixture(path: Path) -> None:
    """Write a tiny self-contained launcher fixture for unittest generation.

    The dummy ``triton.jit`` decorator gives the host-entry picker the same AST
    shape as a real Triton source without requiring a local AgentKernelArena
    checkout or importing Triton during self-verify.
    """
    path.write_text(
        "import torch\n\n"
        "class _DummyTriton:\n"
        "    def jit(self, fn):\n"
        "        return fn\n\n"
        "triton = _DummyTriton()\n\n"
        "@triton.jit\n"
        "def bmm_kernel(a, b, c):\n"
        "    return c\n\n"
        "def bmm_triton(a, b):\n"
        "    return torch.bmm(a, b)\n",
        encoding="utf-8",
    )


def _has_cuda() -> bool:
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class UnittestAgentGenerationTests(unittest.TestCase):
    """Pure-Python generation paths (no CUDA required)."""

    def _bmm_candidate(self, root: Path) -> dict:
        source = root / "triton_bmm.py"
        _write_bmm_fixture(source)
        return {
            "kernel_id": "triton_bmm_kernel",
            "name": "bmm_kernel",
            "kernel_name": "bmm_kernel",
            "source_file": str(source),
            "kernel_url": str(source),
            "input_shapes": [[2, 64, 32], [2, 32, 64]],
            "input_dtypes": ["float16", "float16"],
            "env_vars": {"TRITON_PRINT_AUTOTUNING": "0"},
            "kernel_repo": "/sgl-workspace/vllm",
            "gpu_pct": "12.3%",
            "bound_type": "compute-bound",
        }

    def test_generates_aka_layout(self):
        with tempfile.TemporaryDirectory(prefix="ua_layout_") as td:
            cand = self._bmm_candidate(Path(td))
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
        with tempfile.TemporaryDirectory(prefix="ua_entry_") as td:
            cand = self._bmm_candidate(Path(td))
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
        with tempfile.TemporaryDirectory(prefix="ua_secret_") as td:
            cand = self._bmm_candidate(Path(td))
            cand["env_vars"] = {
                "SGLANG_OPT_USE_TILELANG_INDEXER": "true",
                "MY_API_KEY": "should-not-appear",
                "AUTH_TOKEN": "should-not-appear",
                "AITER_USE_TRITON": "1",
            }
            out = Path(td) / "task"
            ua.generate_unittest(cand, out_dir=out, self_verify=False)
            runner_text = (out / "scripts" / "task_runner.py").read_text()
            self.assertIn("SGLANG_OPT_USE_TILELANG_INDEXER", runner_text)
            self.assertIn("AITER_USE_TRITON", runner_text)
            self.assertNotIn("should-not-appear", runner_text)
            self.assertNotIn("MY_API_KEY", runner_text)
            self.assertNotIn("AUTH_TOKEN", runner_text)

    def test_degraded_when_no_shapes(self):
        with tempfile.TemporaryDirectory(prefix="ua_noshape_") as td:
            cand = self._bmm_candidate(Path(td))
            cand.pop("input_shapes")
            m = ua.generate_unittest(cand, out_dir=Path(td) / "task", self_verify=False)
            self.assertEqual(m["num_shapes"], 0)
            self.assertTrue(any("no input shapes" in w for w in m["warnings"]))

    def test_generates_hip_harness_for_existing_benchmark(self):
        # Make a tiny fake .cu source and a tiny benchmark so we do not need
        # hipcc/GPU for the generation path.  Correctness is deferred to GEAK
        # runtime for HIP/C++ kernels, but the harness should still be created
        # and exposed as a test command.
        with tempfile.TemporaryDirectory(prefix="ua_hip_") as td:
            src = Path(td) / "kernel.cu"
            src.write_text("__global__ void k() {}\n")
            bench = Path(td) / "bench.py"
            bench.write_text("print('Correctness: PASS')\n")
            cand = {
                "kernel_id": "k", "name": "k", "kernel_name": "k",
                "source_file": str(src), "input_shapes": [[8]],
                "benchmark_files": [str(bench)], "kernel_repo": str(Path(td)),
            }
            m = ua.generate_unittest(cand, out_dir=Path(td) / "task", self_verify=False)
            self.assertEqual(m["status"], "ok")
            self.assertEqual(m["task_type"], "hip2hip")
            self.assertTrue(Path(m["task_runner"]).is_file())
            self.assertIn("bench.py", m["benchmark_commands"][0])
            ast.parse(Path(m["task_runner"]).read_text())

    def test_hip_harness_compile_self_verify_is_lightweight(self):
        with tempfile.TemporaryDirectory(prefix="ua_hip_sv_") as td:
            src = Path(td) / "kernel.cu"
            src.write_text("__global__ void k() {}\n")
            bench = Path(td) / "bench.py"
            bench.write_text("raise SystemExit(0)\n")
            cand = {
                "kernel_id": "k", "name": "k", "kernel_name": "k",
                "source_file": str(src), "input_shapes": [[8]],
                "benchmark_files": [str(bench)], "kernel_repo": str(Path(td)),
            }
            m = ua.generate_unittest(cand, out_dir=Path(td) / "task", self_verify=True)
            self.assertEqual(m["status"], "ok")
            self.assertEqual(m["self_verify"]["compile"], "ok")
            self.assertEqual(m["self_verify"]["correctness"], "skipped")

    def test_hip_shape_cases_fall_back_to_profile_trace(self):
        with tempfile.TemporaryDirectory(prefix="ua_hip_profile_") as td:
            root = Path(td)
            src = root / "rmsnorm_quant_kernels.cu"
            src.write_text("__global__ void k() {}\n")
            bench = root / "bench.py"
            bench.write_text("raise SystemExit(0)\n")
            trace_dir = root / "runs" / "profile" / "task" / "torch_trace"
            trace_dir.mkdir(parents=True)
            trace = trace_dir / "merged.trace.json.gz"
            payload = {
                "traceEvents": [
                    {
                        "name": "aiter::rmsnorm",
                        "args": {
                            "Input Dims": [[1024, 4096], [1024, 4096], [4096], []],
                            "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16", "Scalar"],
                        },
                    },
                    {
                        "name": "aiter::rmsnorm",
                        "args": {
                            "Input Dims": [[32768, 128], [32768, 128], [128], []],
                            "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16", "Scalar"],
                        },
                    },
                ]
            }
            with gzip.open(trace, "wt") as fh:
                json.dump(payload, fh)
            old_user_data = os.environ.get("USER_DATA_PATH")
            os.environ["USER_DATA_PATH"] = str(root)
            try:
                cand = {
                    "kernel_id": "k009",
                    "name": "add_rmsnorm_quant_kernel",
                    "kernel_name": "add_rmsnorm_quant_kernel",
                    "source_file": str(src),
                    "benchmark_files": [str(bench)],
                    "kernel_repo": str(root),
                    "kernel_params": {"HEAD_SIZE": 128},
                }
                m = ua.generate_unittest(cand, out_dir=root / "task", self_verify=False)
            finally:
                if old_user_data is None:
                    os.environ.pop("USER_DATA_PATH", None)
                else:
                    os.environ["USER_DATA_PATH"] = old_user_data
            self.assertEqual(m["status"], "ok")
            self.assertEqual(m["shape_cases"][0]["input_dims"], [[32768, 128], [32768, 128], [128], []])
            self.assertNotIn([1024, 4096], m["shapes"])

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


@unittest.skipUnless(_has_cuda(), "needs CUDA-capable GPU")
class UnittestAgentSelfVerifyTests(unittest.TestCase):
    """End-to-end self-verify paths (require GPU)."""

    def _bmm_candidate(self, root: Path) -> dict:
        source = root / "triton_bmm.py"
        _write_bmm_fixture(source)
        return {
            "kernel_id": "triton_bmm_kernel",
            "name": "bmm_kernel",
            "kernel_name": "bmm_kernel",
            "source_file": str(source),
            "input_shapes": [[2, 64, 32], [2, 32, 64]],
            "input_dtypes": ["float16", "float16"],
            "env_vars": {},
        }

    def test_self_verify_passes_on_unmodified_source(self):
        with tempfile.TemporaryDirectory(prefix="ua_sv_ok_") as td:
            out = Path(td) / "task"
            m = ua.generate_unittest(
                self._bmm_candidate(Path(td)), out_dir=out, self_verify=True,
            )
            self.assertEqual(m["status"], "ok",
                             f"unexpected status {m['status']}: {m['self_verify']}")
            self.assertEqual(m["self_verify"]["compile"], "ok")
            self.assertEqual(m["self_verify"]["correctness"], "ok")

    def test_correctness_detects_mutation(self):
        with tempfile.TemporaryDirectory(prefix="ua_sv_mut_") as td:
            out = Path(td) / "task"
            m = ua.generate_unittest(
                self._bmm_candidate(Path(td)), out_dir=out, self_verify=False,
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
                "return torch.bmm(a, b)",
                "return torch.bmm(a, b) + 1.0",
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
                self._bmm_candidate(Path(td)), out_dir=out, self_verify=False,
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


class TimeoutBudgetTests(unittest.TestCase):
    """Regression coverage for ``_compute_hip_timeout_budget``.

    Anchors the 2026-05-20 silu_and_mul k010 incident where the inner
    900s default tripped on baseline JIT recompile + 4 shape benchmarks
    and forced GEAK's select_agent to fall back to the unmodified baseline.
    """

    def test_budget_floor_at_30_minutes(self):
        b = ua._compute_hip_timeout_budget([])
        self.assertGreaterEqual(b["correctness"], 1800)
        self.assertGreaterEqual(b["performance"], 1800)
        self.assertGreaterEqual(b["per_shape"], 300)

    def test_budget_scales_with_shape_count(self):
        small = ua._compute_hip_timeout_budget([{}, {}])
        large = ua._compute_hip_timeout_budget([{} for _ in range(8)])
        self.assertGreaterEqual(large["correctness"], small["correctness"])
        self.assertGreater(large["correctness"], 1800)

    def test_budget_respects_env_override(self):
        old = os.environ.get("UNITTEST_HIP_PER_SHAPE_TIMEOUT_SEC")
        try:
            os.environ["UNITTEST_HIP_PER_SHAPE_TIMEOUT_SEC"] = "600"
            b = ua._compute_hip_timeout_budget([{}, {}])
            self.assertEqual(b["per_shape"], 600)
            self.assertGreaterEqual(b["correctness"], 2 * 600)
        finally:
            if old is None:
                os.environ.pop("UNITTEST_HIP_PER_SHAPE_TIMEOUT_SEC", None)
            else:
                os.environ["UNITTEST_HIP_PER_SHAPE_TIMEOUT_SEC"] = old


class HipManifestTimeoutTests(unittest.TestCase):
    """The HIP harness must write timeouts to ``unittest_meta.json`` so the
    orchestrator + GEAK env injector can mirror them, and must hardcode the
    same defaults into the generated task_runner.py (still env-overridable)."""

    def test_meta_carries_harness_timeouts(self):
        with tempfile.TemporaryDirectory(prefix="ua_hip_tmo_") as td:
            src = Path(td) / "kernel.cu"
            src.write_text("__global__ void k() {}\n")
            bench = Path(td) / "bench.py"
            bench.write_text("raise SystemExit(0)\n")
            m = ua.generate_unittest(
                {"kernel_id": "k", "name": "k", "kernel_name": "k",
                 "source_file": str(src), "benchmark_files": [str(bench)],
                 "kernel_repo": str(Path(td))},
                out_dir=Path(td) / "task", self_verify=False,
            )
            self.assertGreaterEqual(m["harness_timeout_correctness_sec"], 1800)
            self.assertGreaterEqual(m["harness_timeout_performance_sec"], 1800)
            self.assertGreaterEqual(m["harness_per_shape_timeout_sec"], 300)
            runner_text = Path(m["task_runner"]).read_text()
            self.assertIn("_DEFAULT_CORRECTNESS_TIMEOUT_SEC", runner_text)
            self.assertIn(str(m["harness_timeout_correctness_sec"]), runner_text)


class HipPerformanceExitCodeTests(unittest.TestCase):
    """``performance`` must return non-zero when every shape errors out so
    GEAK select_agent can distinguish a real measurement from a silent fail.

    Without this fix (which was the case before 2026-05-20), a harness that
    failed every shape still returned exit 0 with ``-1.0`` ms values,
    making patches look "successful" to GEAK's per-task parser.
    """

    def test_zero_good_cases_exits_nonzero(self):
        # Isolate the harness from any USER_DATA_PATH the surrounding pytest
        # process inherited (otherwise ``_shape_cases`` finds a real trace
        # and the runner ends up exercising whatever live shape was last
        # captured by the optimizer — see 2026-05-20 silu_and_mul k010).
        with tempfile.TemporaryDirectory(prefix="ua_hip_exit_") as td:
            src = Path(td) / "kernel.cu"
            src.write_text("__global__ void k() {}\n")
            bench = Path(td) / "bench.py"
            bench.write_text("import sys; sys.exit(7)\n")
            sandbox_data = Path(td) / "data"
            sandbox_data.mkdir()
            saved_udp = os.environ.get("USER_DATA_PATH")
            os.environ["USER_DATA_PATH"] = str(sandbox_data)
            try:
                m = ua.generate_unittest(
                    {"kernel_id": "k", "name": "k", "kernel_name": "k",
                     "source_file": str(src), "benchmark_files": [str(bench)],
                     "kernel_repo": str(Path(td))},
                    out_dir=Path(td) / "task", self_verify=False,
                )
                self.assertEqual(
                    m.get("shape_cases"), [],
                    "USER_DATA_PATH leaked real shapes into the harness",
                )
                proc = subprocess.run(
                    [sys.executable, m["task_runner"], "performance"],
                    capture_output=True, text=True, timeout=120,
                )
            finally:
                if saved_udp is None:
                    os.environ.pop("USER_DATA_PATH", None)
                else:
                    os.environ["USER_DATA_PATH"] = saved_udp
            self.assertNotEqual(
                proc.returncode, 0,
                f"perf with no good case must exit non-zero; got rc=0 stdout={proc.stdout!r}",
            )


class OverlayLockTests(unittest.TestCase):
    """``_OverlayLiveSource`` acquires an fcntl lock so two parallel
    kernel_opt runs don't race on the shared aiter JIT cache.

    The template embeds a small helper, ``_overlay_lockfile``, that we
    smoke-test by rendering the HIP template and parsing it back as AST
    plus a string-presence check.
    """

    def test_template_embeds_fcntl_lock(self):
        budget = ua._compute_hip_timeout_budget([{}])
        text = ua._HIP_TASK_RUNNER_TEMPLATE.format(
            kernel_name="foo", task_name="t/k", source_basename="x.cu",
            live_source="/x", kernel_repo="/r",
            benchmark_commands=["python /x.py"], target_kernels=["foo"],
            env_vars={}, jit_roots=["/jit"], jit_match_tokens=["foo"],
            shape_cases=[],
            default_correctness_timeout=budget["correctness"],
            default_performance_timeout=budget["performance"],
            default_per_shape_timeout=budget["per_shape"],
        )
        self.assertIn("fcntl.flock", text)
        self.assertIn("_overlay_lockfile", text)
        ast.parse(text)

    def test_template_embeds_stale_aiter_baton_purge(self):
        """Without this purge, a SIGKILL'd compile leaves
        ``/sgl-workspace/aiter/aiter/jit/build/lock_module_<name>`` on disk and
        the next ``import aiter.jit.module_<name>`` spins forever in
        ``FileBaton.wait()`` (no timeout, no detection). We embed
        ``_purge_stale_aiter_batons`` in the generated runner and call it
        from inside ``_OverlayLiveSource.__enter__`` after the overlay
        file-lock is held — so every fresh GEAK attempt has a clean slate.
        """
        budget = ua._compute_hip_timeout_budget([{}])
        text = ua._HIP_TASK_RUNNER_TEMPLATE.format(
            kernel_name="foo", task_name="t/k", source_basename="x.cu",
            live_source="/x", kernel_repo="/r",
            benchmark_commands=["python /x.py"], target_kernels=["foo"],
            env_vars={}, jit_roots=["/jit"], jit_match_tokens=["foo"],
            shape_cases=[],
            default_correctness_timeout=budget["correctness"],
            default_performance_timeout=budget["performance"],
            default_per_shape_timeout=budget["per_shape"],
        )
        self.assertIn("_purge_stale_aiter_batons", text)
        # The purge is gated on overlay-lock ownership; verify the call
        # happens after fcntl.flock acquisition rather than at module import
        # time (where it would race with a still-alive sibling compile).
        flock_idx = text.find("fcntl.flock")
        purge_idx = text.find("_purge_stale_aiter_batons()")
        self.assertGreater(
            flock_idx, 0, "fcntl.flock acquisition must appear in template"
        )
        self.assertGreater(
            purge_idx, flock_idx,
            "_purge_stale_aiter_batons must be invoked after overlay lock is held",
        )
        # The helper itself must compile under f-string template substitution.
        ast.parse(text)


class EnvSubsetDeviceSelectionTests(unittest.TestCase):
    """Profile traces capture ``ROCR_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES``
    that were valid at trace time on physical GPU0. The generated harness must
    NOT bake those into RUNTIME_ENV — when GEAK later runs the harness on
    GPU2/3 (manual handoff or multi-tenant ray scheduling), the embedded
    ``ROCR=0`` overrides the caller's ``HIP=2`` and torch reports ``No HIP
    GPUs are available`` (every patch is then incorrectly rejected as
    failing baseline).
    """

    def test_device_selection_vars_stripped_from_candidate(self):
        cand = {"env_vars": {
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "AMD_VISIBLE_DEVICES": "0",
            "GPU_DEVICE_ORDINAL": "0",
            "NVIDIA_VISIBLE_DEVICES": "0",
            "SGLANG_USE_AITER": "1",
            "AITER_COMMIT": "v0.1.10",
        }}
        result = ua._env_subset_for_runtime(cand)
        for var in (
            "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES", "AMD_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL", "NVIDIA_VISIBLE_DEVICES",
        ):
            self.assertNotIn(
                var, result,
                f"{var} must not leak into harness RUNTIME_ENV from "
                f"candidate env (got {result.get(var)!r})",
            )
        # Workload prefix vars should still pass through.
        self.assertEqual(result.get("SGLANG_USE_AITER"), "1")
        self.assertEqual(result.get("AITER_COMMIT"), "v0.1.10")

    def test_device_selection_vars_stripped_from_os_environ(self):
        cand = {"env_vars": {}}
        saved = {}
        for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            saved[var] = os.environ.get(var)
            os.environ[var] = "0"
        try:
            result = ua._env_subset_for_runtime(cand)
            self.assertNotIn("HIP_VISIBLE_DEVICES", result)
            self.assertNotIn("ROCR_VISIBLE_DEVICES", result)
        finally:
            for var, val in saved.items():
                if val is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = val


class HipShapeSubprocessBudgetTests(unittest.TestCase):
    """``_shape_subprocess_budget`` divides the outer correctness timeout
    across the inner ``save`` / ``compare`` subprocesses; it must respect
    the caller's cap (so we never starve the remaining work) while still
    giving each shape a non-trivial floor.
    """

    def test_budget_helper_visible_in_runner(self):
        budget = ua._compute_hip_timeout_budget([{}])
        text = ua._HIP_TASK_RUNNER_TEMPLATE.format(
            kernel_name="foo", task_name="t/k", source_basename="x.cu",
            live_source="/x", kernel_repo="/r",
            benchmark_commands=["python /x.py"], target_kernels=["foo"],
            env_vars={}, jit_roots=["/jit"], jit_match_tokens=["foo"],
            shape_cases=[],
            default_correctness_timeout=budget["correctness"],
            default_performance_timeout=budget["performance"],
            default_per_shape_timeout=budget["per_shape"],
        )
        self.assertIn("_shape_subprocess_budget", text)
        # The compare phase must reuse the remaining budget so a slow save
        # doesn't starve compare. Pin the contract through a substring.
        self.assertIn("Recompute remaining budget", text)


class HipShapeScriptPerCaseTests(unittest.TestCase):
    """The compare-mode shape sub-script must emit per-case diagnostics
    instead of aborting on the first failure (without these, GEAK only ever
    sees the first failing shape and can't tell whether others would have
    regressed too).
    """

    def test_compare_emits_per_case_records(self):
        # ``_shape_case_script`` is defined inside the HIP task_runner
        # template (it runs in the worker subprocess, not in the agent),
        # so we render the template once and assert against its source.
        budget = ua._compute_hip_timeout_budget([{}])
        text = ua._HIP_TASK_RUNNER_TEMPLATE.format(
            kernel_name="foo", task_name="t/k", source_basename="x.cu",
            live_source="/x", kernel_repo="/r",
            benchmark_commands=["python /x.py"], target_kernels=["foo"],
            env_vars={}, jit_roots=["/jit"], jit_match_tokens=["foo"],
            shape_cases=[{"op": "silu_and_mul", "input_dims": [[1, 8]]}],
            default_correctness_timeout=budget["correctness"],
            default_performance_timeout=budget["performance"],
            default_per_shape_timeout=budget["per_shape"],
        )
        self.assertIn("per_case", text)
        self.assertIn("case_idx", text)
        self.assertIn("torch.cuda.empty_cache", text)


class PythonCorrectnessSummaryTests(unittest.TestCase):
    """``run_correctness`` for Python/Triton harnesses now collects per-shape
    results instead of bailing out on the first failure."""

    def test_python_runner_template_collects_per_shape(self):
        text = ua._TASK_RUNNER_TEMPLATE
        self.assertIn("per_shape", text)
        self.assertIn("num_pass", text)
        self.assertIn("num_fail", text)
        # Per-shape ``(shape): X ms`` lines for GEAK select_agent's
        # parse_shape_latencies_ms regex. The template uses doubled braces
        # (``{{...}}``) because it's a ``.format`` string, so we match
        # against the post-format-escaping marker instead.
        self.assertIn("({{list(shape)}}): {{c['execution_time_ms']:.4f}} ms",
                      text)


if __name__ == "__main__":
    unittest.main()
