#!/usr/bin/env python3
"""Local tests for Kernel Agent tools.

The tests generate all fixtures at runtime so the repository does not carry
large trace files.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_TOOL = ROOT / "tools" / "tracelens_analysis.py"
OPT_TOOL = ROOT / "tools" / "kernel_optimization.py"
INSTALL_SCRIPT = ROOT / "scripts" / "install.sh"
RAY_RUNTIME = ROOT / "tools" / "backends" / "ray_runtime.py"


def run_json(cmd: list[str], *, workspace: Path) -> dict:
    env = {**os.environ, "USER_DATA_PATH": str(workspace)}
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=60)
    if proc.returncode != 0:
        raise AssertionError(f"command failed\ncmd={cmd}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return json.loads(proc.stdout)


def run_json_allow_fail(cmd: list[str], *, workspace: Path) -> tuple[int, dict]:
    env = {**os.environ, "USER_DATA_PATH": str(workspace)}
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=60)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"non-json stdout\ncmd={cmd}\nstdout={proc.stdout}\nstderr={proc.stderr}") from exc
    return proc.returncode, payload


def write_trace(
    path: Path,
    *,
    kernel_name: str = "triton_rmsnorm_kernel",
    source_file: str = "/src/kernels/rmsnorm.py",
) -> None:
    payload = {
        "traceEvents": [
            {
                "name": kernel_name,
                "cat": "kernel",
                "dur": 7000,
                "args": {
                    "source_file": source_file,
                    "shape": {"M": 64, "N": 4096},
                },
            },
            {
                "name": "aiter_moe_gemm_kernel",
                "cat": "kernel",
                "dur": 3000,
                "args": {
                    "source_file": "/src/kernels/moe.hip",
                    "shape": {"M": 128, "N": 4096, "K": 4096},
                },
            },
            {"name": "python_function", "cat": "cpu", "dur": 100000},
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_vendor_trace(path: Path) -> None:
    payload = {
        "traceEvents": [
            {
                "name": "hipblasLt_Cijk_Ailk_Bljk_kernel",
                "cat": "kernel",
                "dur": 5000,
                "args": {"shape": {"M": 128, "N": 128, "K": 128}},
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class KernelAgentToolTests(unittest.TestCase):
    def test_install_help_and_dry_run_backend_flags(self) -> None:
        help_proc = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--help"],
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(help_proc.returncode, 0)
        self.assertIn("--all-backends", help_proc.stdout)
        dry_proc = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--dry-run", "--all-backends"],
            text=True,
            capture_output=True,
            timeout=30,
            env={
                **os.environ,
                "TRACELENS_ROOT": str(ROOT / "missing-tracelens"),
                "HYPERLOOM_BUNDLE": str(ROOT / "missing-bundle"),
            },
        )
        self.assertEqual(dry_proc.returncode, 0)
        self.assertIn("TraceLens root not found", dry_proc.stderr)
        self.assertIn("ensuring Node.js/npm for claude/codex CLIs", dry_proc.stdout)
        self.assertIn("ensuring ray[default]==2.44.1", dry_proc.stdout)

    def test_new_environment_defaults_are_documented_in_tools(self) -> None:
        install_text = INSTALL_SCRIPT.read_text(encoding="utf-8")
        trace_tool_text = TRACE_TOOL.read_text(encoding="utf-8")
        skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        ray_runtime_text = RAY_RUNTIME.read_text(encoding="utf-8")

        self.assertIn('TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"', install_text)
        self.assertIn('GEAK_REF="${GEAK_REF:-v3.1.0}"', install_text)
        self.assertIn('python3 -m pip install -q --no-cache-dir "${HYPERLOOM_ROOT}/geak"', install_text)
        self.assertIn('python3 -m pip install -q --no-cache-dir "${HYPERLOOM_ROOT}/geak/mcp_tools/rag-mcp"', install_text)
        # GEAK_RAG_INDEX_DEVICE_VAL is now auto-detected: cuda when rocm-smi
        # or torch.cuda.is_available() succeeds, else cpu. The explicit env
        # override still wins. Regression guard against accidental removal of
        # either branch (a CPU-only fallback caused 1.5h+ embedder builds and
        # zombie installers in pre-fix runs — observed 2026-05-20).
        self.assertIn('if [ -z "${GEAK_RAG_INDEX_DEVICE:-}" ]; then', install_text)
        self.assertIn('GEAK_RAG_INDEX_DEVICE_VAL="cuda"', install_text)
        self.assertIn('GEAK_RAG_INDEX_DEVICE_VAL="cpu"', install_text)
        self.assertIn('GEAK_RAG_INDEX_DEVICE_VAL="${GEAK_RAG_INDEX_DEVICE}"', install_text)
        self.assertIn("python3 scripts/build_index.py --force --device", install_text)
        self.assertIn("ensure_node()", install_text)
        self.assertIn("installing Node.js 20 from NodeSource", install_text)
        self.assertIn("ensure_node", install_text)
        self.assertIn("GEAK_RAG_INDEX_DEVICE=cuda", skill_text)
        self.assertIn("tools:", install_text)
        self.assertIn("  rag: true", install_text)
        self.assertIn("GEAK_MEMORY_STORE_PATH", install_text)
        self.assertNotIn("GEAK_MEMORY_KB_PATH", install_text)
        self.assertIn('"click<8.3.0" "ray[default]==2.44.1"', install_text)
        self.assertIn('chmod 600 "$env_file"', install_text)
        self.assertIn("GEAK_MEMORY_STORE_PATH", ray_runtime_text)
        self.assertIn("GEAK_SAVE_TO_KNOWLEDGE_BASE", ray_runtime_text)
        self.assertIn('DEFAULT_TRACELENS_ROOT = "/wekafs/hyperloom/TraceLens-internal"', trace_tool_text)
        self.assertNotIn('TRACELENS_ROOT="${TRACELENS_ROOT:-/hyperloom/TraceLens-internal}"', install_text)
        self.assertNotIn("Executor asks", skill_text)
        # Read-only TRACELENS_ROOT must trigger a writable mirror under
        # ${HYPERLOOM_ROOT}/TraceLens-internal (parallel to GEAK / OOB),
        # and write_env_file() must export the resolved TRACELENS_ROOT so
        # CLI subprocesses inherit the mirror instead of falling back to
        # the read-only /wekafs default. Regression guard for the
        # tracelens-oob-mirror change.
        self.assertIn(
            'TRACELENS_MIRROR_DIR="${TRACELENS_MIRROR_DIR:-${HYPERLOOM_ROOT}/TraceLens-internal}"',
            install_text,
        )
        self.assertIn('cp -r "$TRACELENS_ROOT" "$TRACELENS_MIRROR_DIR"', install_text)
        self.assertIn('export TRACELENS_ROOT', install_text)
        self.assertIn(
            "echo \"export TRACELENS_ROOT='${TRACELENS_ROOT}'\"",
            install_text,
        )
        self.assertIn("MAGPIE_PYTHON", install_text)
        self.assertIn("PYTHONPATH", install_text)
        self.assertIn("echo \"export MAGPIE_PYTHON='${MAGPIE_PYTHON}'\"", install_text)
        self.assertIn("echo \"export PYTHONPATH='${PYTHONPATH}'\"", install_text)

    def test_trace_file_analysis_writes_report_logs_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            write_trace(
                trace,
                kernel_name="triton_unmapped_kernel",
                source_file="/src/kernels/unmapped.py",
            )

            result = run_json([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", str(trace),
                "--session-id", "s1",
                "--model-name", "test-model",
                "--framework", "sglang",
                "--dry-run",
            ], workspace=workspace)

            self.assertEqual(result["trace_input_type"], "file")
            self.assertEqual(result["hot_kernels"][0]["kernel_id"], "k001")
            self.assertTrue(Path(result["trace_report_path"]).exists())
            self.assertTrue(Path(result["cli_log_path"]).exists())
            self.assertTrue(Path(result["status_path"]).exists())
            status = json.loads(Path(result["status_path"]).read_text(encoding="utf-8"))
            self.assertIn("last_lines", status)
            self.assertGreater(status["offset_bytes"], 0)

    def test_capture_directory_analysis_supports_gzip_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            capture = workspace / "capture"
            capture.mkdir()
            raw_trace = workspace / "trace.json"
            write_trace(raw_trace)
            gz_trace = capture / "rank0.trace.json.gz"
            with gzip.open(gz_trace, "wt", encoding="utf-8") as fh:
                fh.write(raw_trace.read_text(encoding="utf-8"))

            result = run_json([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", str(capture),
                "--session-id", "s2",
                "--model-name", "test-model",
                "--framework", "sglang",
                "--dry-run",
            ], workspace=workspace)

            self.assertEqual(result["trace_input_type"], "capture_dir")
            self.assertGreaterEqual(len(result["hot_kernels"]), 2)

    def test_default_backends_include_geak_without_benchmark(self) -> None:
        """Policy change (#144 last comment Layer 1, broadened): every
        kernel Claude/Codex can rewrite, GEAK can rewrite too. Auto-pick
        MUST include GEAK in the ladder even when no benchmark is
        present — the previous "skip GEAK" behaviour was over-conservative
        and starved GEAK of high-priority kernels on runs that hadn't
        registered a harness yet. ``geak_without_benchmark`` flags the
        reduced verification confidence so downstream KEEP gates audit
        appropriately; the decision is still ``NEEDS_REVIEW`` because
        E2E evidence is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            write_trace(
                trace,
                kernel_name="triton_unmapped_kernel",
                source_file="/src/kernels/unmapped.py",
            )
            run_json([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", str(trace),
                "--session-id", "s3",
                "--model-name", "test-model",
                "--framework", "sglang",
                "--dry-run",
            ], workspace=workspace)

            result = run_json([
                sys.executable, str(OPT_TOOL),
                "--kernel-id", "k001",
                "--session-id", "s3",
                "--dry-run",
            ], workspace=workspace)

            # GEAK is now in the ladder (FIRST, the high-priority handoff
            # the policy intended).
            self.assertIn("geak", result["selected_backends"])
            self.assertEqual(result["selected_backends"][0], "geak")
            # No bench → flagged for downstream verification gates.
            self.assertTrue(
                result["backend_selection"]["geak_without_benchmark"]
            )
            # Decision unchanged: E2E evidence still missing, still
            # NEEDS_REVIEW. The change is that GEAK got a swing at the
            # rewrite, not that we lowered the KEEP bar.
            self.assertEqual(result["proposal"]["decision"], "NEEDS_REVIEW")
            self.assertIn("E2E evidence missing", result["proposal"]["reasons"])

    def test_user_specified_geak_without_benchmark_is_marked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            write_trace(
                trace,
                kernel_name="triton_unmapped_kernel",
                source_file="/src/kernels/unmapped.py",
            )
            run_json([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", str(trace),
                "--session-id", "s4",
                "--model-name", "test-model",
                "--framework", "sglang",
                "--dry-run",
            ], workspace=workspace)

            result = run_json([
                sys.executable, str(OPT_TOOL),
                "--kernel-id", "k001",
                "--session-id", "s4",
                "--backends", "geak",
                "--disable-rag",
                "--disable-xs-memory",
                "--dry-run",
            ], workspace=workspace)

            self.assertEqual(result["selected_backends"], ["geak"])
            self.assertTrue(result["backend_selection"]["geak_without_benchmark"])
            self.assertFalse(result["backend_selection"]["rag_enabled"])
            self.assertFalse(result["backend_selection"]["xs_memory_enabled"])
            self.assertEqual(result["rag_hits"], [])
            self.assertEqual(result["xs_memory_hits"], [])

    def test_keep_requires_benchmark_e2e_and_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            harness = workspace / "bench.py"
            write_trace(trace)
            harness.write_text("print('bench')\n", encoding="utf-8")
            run_json([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", str(trace),
                "--session-id", "s5",
                "--model-name", "test-model",
                "--framework", "sglang",
                "--dry-run",
            ], workspace=workspace)

            result = run_json([
                sys.executable, str(OPT_TOOL),
                "--kernel-id", "k001",
                "--session-id", "s5",
                "--test-harness-path", str(harness),
                "--e2e-gain-pct", "1.5",
                "--accuracy-passed", "true",
                # Need >= 1.50x microbench to clear KEEP threshold; the
                # dry-run placeholder of 1.05 only earns NEEDS_REVIEW now.
                "--micro-speedup", "1.6",
                "--dry-run",
            ], workspace=workspace)

            self.assertIn("geak", result["selected_backends"])
            self.assertEqual(result["proposal"]["decision"], "KEEP")
            self.assertIn("rag_hits", result)
            self.assertIn("xs_memory_hits", result)

    def test_missing_trace_input_fails_with_status_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            code, result = run_json_allow_fail([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", str(workspace / "missing.json"),
                "--session-id", "s6",
                "--dry-run",
            ], workspace=workspace)

            self.assertNotEqual(code, 0)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(Path(result["status_path"]).exists())
            self.assertTrue(Path(result["cli_log_path"]).exists())

    def test_vendor_binary_candidate_selects_no_default_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "vendor_trace.json"
            write_vendor_trace(trace)
            run_json([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", str(trace),
                "--session-id", "s7",
                "--model-name", "test-model",
                "--framework", "sglang",
                "--dry-run",
            ], workspace=workspace)

            result = run_json([
                sys.executable, str(OPT_TOOL),
                "--kernel-id", "k001",
                "--session-id", "s7",
                "--dry-run",
            ], workspace=workspace)

            self.assertEqual(result["selected_backends"], [])
            self.assertEqual(result["proposal"]["decision"], "REVERT")
            self.assertIn("compile failed", result["proposal"]["reasons"])

    def test_unknown_kernel_id_fails_without_partial_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            write_trace(trace)
            run_json([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", str(trace),
                "--session-id", "s8",
                "--dry-run",
            ], workspace=workspace)

            code, result = run_json_allow_fail([
                sys.executable, str(OPT_TOOL),
                "--kernel-id", "missing-kernel",
                "--session-id", "s8",
                "--dry-run",
            ], workspace=workspace)

            self.assertNotEqual(code, 0)
            self.assertEqual(result["status"], "failed")
            self.assertIn("kernel not found", result["error"])

    def test_invalid_backend_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            write_trace(trace)
            run_json([
                sys.executable, str(TRACE_TOOL),
                "--trace-input", str(trace),
                "--session-id", "s9",
                "--dry-run",
            ], workspace=workspace)

            code, result = run_json_allow_fail([
                sys.executable, str(OPT_TOOL),
                "--kernel-id", "k001",
                "--session-id", "s9",
                "--backends", "badbackend",
                "--dry-run",
            ], workspace=workspace)

            self.assertNotEqual(code, 0)
            self.assertIn("unsupported backend", result["error"])


class AuthFailureDetectionTests(unittest.TestCase):
    """Cover :func:`_count_auth_failures` and the partial-promotion guard.

    Regression for the r24 custom_allreduce loop where GEAK's inner
    SelectPatchAgent kept hitting 401 against the wrong gateway
    (``https://llm-api.amd.com/Anthropic`` expecting an
    ``AMD_LLM_API_KEY`` distinct from ``SAFE_API_KEY``), left an empty
    ``optimized_versions/`` directory on disk, got promoted to "partial"
    by the evidence scanner, and shipped back ``decision=PARTIAL``. The
    matching SharedState fix only retires kernels whose run_optimization
    returns >= max_partial PARTIAL outcomes; this test fixture pins the
    upstream half — when stdout shows a persistent 401 loop, the attempt
    must NOT be promoted to partial in the first place, so make_proposal
    returns REVERT and SharedState retires the kernel immediately.
    """

    def test_count_auth_failures_recognises_401_loop(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        try:
            import kernel_optimization as ko  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)
        log = (
            "INFO calling https://llm-api.amd.com/Anthropic/v1/messages\n"
            "HTTP/1.1 401 Unauthorized\n"
            "INFO retry 1\n"
            "HTTP/1.1 401 Unauthorized\n"
            "AuthenticationError: Invalid API key (Subscription-Key not present)\n"
        )
        self.assertGreaterEqual(ko._count_auth_failures(log), 3)

    def test_count_auth_failures_clean_logs(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        try:
            import kernel_optimization as ko  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)
        self.assertEqual(ko._count_auth_failures(""), 0)
        self.assertEqual(ko._count_auth_failures("speedup: 1.32x faster"), 0)
        # A single 401 in a long run is recoverable; below threshold.
        self.assertLess(
            ko._count_auth_failures(
                "WARN: HTTP/1.1 401 Unauthorized; retrying.\n"
                "INFO retry succeeded\n"
                "speedup: 1.21x"
            ),
            3,
        )

    def test_count_auth_failures_primus_token_pattern(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        try:
            import kernel_optimization as ko  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)
        log = (
            "Primus.00009 token not present\n"
            "Primus.00009 token not present\n"
            "Primus.00009 token not present\n"
        )
        self.assertEqual(ko._count_auth_failures(log), 3)


class GeakConfigEnvTimeoutInjectionTests(unittest.TestCase):
    """A custom GEAK config (e.g. install.sh-generated ``local.yaml``)
    must end up with ``env.timeout`` >= the unittest harness's own
    advertised budget before it reaches mini-swe-agent, or every
    ``save_and_test`` call inside the auto-generated unittest harness
    silently dies at the 30s default ``LocalEnvironmentConfig.timeout``
    with ``Test command timed out``.

    Before ``unittest_agent`` was added, GEAK's ``--test-command`` was a
    fast ``python bench_<kernel>.py`` script that finished in seconds and
    the 30s default was fine. After ``unittest_agent`` started writing
    ``task_runner.py correctness`` test commands (which trigger an aiter
    JIT recompile + multi-shape benchmark and routinely take minutes),
    nothing was propagating the new budget to GEAK's outer timeout, so
    every patch silently fell back to baseline.

    Hyperloom kernel_optimization fixes this at three layers:
      * ``scripts/install.sh`` writes ``env.timeout: 3600`` into newly
        generated ``local.yaml`` files (covered by the install.sh diff).
      * ``_geak_config_for_run`` defensively rewrites any pre-existing
        config to a timeout no smaller than the unittest_agent's own
        ``harness_timeout_*_sec`` advertisement (or 3600s when no
        manifest is available).
      * ``_harness_outer_timeout`` is the single source of truth that
        ties the harness's internal budget to GEAK's outer
        ``subprocess.run(timeout=...)``.
    """

    def _import_ko(self):
        sys.path.insert(0, str(ROOT / "tools"))
        try:
            import kernel_optimization as ko  # type: ignore[import-not-found]
            return ko
        finally:
            sys.path.pop(0)

    def test_injects_env_timeout_when_missing(self) -> None:
        ko = self._import_ko()
        base = "model:\n  model_class: litellm\ntools:\n  rag: true\n"
        injected = ko._ensure_yaml_env_timeout(base)
        self.assertIn("env:", injected)
        self.assertRegex(injected, r"timeout:\s*3600")
        # Idempotent: applying twice does not stack duplicate env blocks.
        self.assertEqual(injected, ko._ensure_yaml_env_timeout(injected))

    def test_preserves_existing_env_block_when_already_large_enough(self) -> None:
        ko = self._import_ko()
        base = (
            "model:\n  model_class: litellm\n"
            "env:\n"
            "  env:\n    PAGER: cat\n"
            "  timeout: 7200\n"  # already >= the 3600s default
            "tools:\n  rag: true\n"
        )
        self.assertEqual(ko._ensure_yaml_env_timeout(base), base)

    def test_upgrades_too_small_explicit_timeout(self) -> None:
        """A copy-pasted ``env.timeout: 30`` (or any value < harness budget)
        must be rewritten to the harness budget rather than silently
        defeating the entire safety net."""
        ko = self._import_ko()
        base = (
            "model:\n  model_class: litellm\n"
            "env:\n"
            "  env:\n    PAGER: cat\n"
            "  timeout: 30\n"
            "tools:\n  rag: true\n"
        )
        out = ko._ensure_yaml_env_timeout(base, timeout=2100)
        self.assertRegex(out, r"timeout:\s*2100")
        self.assertNotRegex(out, r"timeout:\s*30\b")
        # Still only one env block / one timeout line.
        self.assertEqual(out.count("\nenv:\n"), 1)
        self.assertEqual(out.count("timeout:"), 1)

    def test_appends_timeout_to_existing_env_without_one(self) -> None:
        ko = self._import_ko()
        base = (
            "model:\n  model_class: litellm\n"
            "env:\n"
            "  env:\n    PAGER: cat\n"
            "tools:\n  rag: true\n"
        )
        out = ko._ensure_yaml_env_timeout(base, timeout=2100)
        self.assertRegex(out, r"timeout:\s*2100")
        # Did not duplicate the env block.
        self.assertEqual(out.count("\nenv:\n"), 1)

    def test_harness_outer_timeout_uses_manifest_budget_plus_buffer(self) -> None:
        ko = self._import_ko()
        manifest = {
            "status": "ok",
            "harness_timeout_correctness_sec": 1800,
            "harness_timeout_performance_sec": 1800,
            "harness_per_shape_timeout_sec": 300,
        }
        expected = 1800 + ko._DEFAULT_HARNESS_OUTER_BUFFER_SEC
        self.assertEqual(ko._harness_outer_timeout(manifest), expected)

    def test_harness_outer_timeout_returns_none_for_bad_manifest(self) -> None:
        ko = self._import_ko()
        self.assertIsNone(ko._harness_outer_timeout(None))
        self.assertIsNone(ko._harness_outer_timeout({"status": "degraded"}))
        self.assertIsNone(ko._harness_outer_timeout({"status": "ok"}))

    def test_geak_config_for_run_uses_manifest_timeout(self) -> None:
        """End-to-end glue: when a unittest manifest declares its own
        budget, the rewritten GEAK config must adopt it (the root-cause
        fix the user spotted: previously *only* ``--test-command``
        flowed through, ``env.timeout`` did not)."""
        ko = self._import_ko()
        import tempfile, argparse

        manifest = {
            "status": "ok",
            "harness_timeout_correctness_sec": 1800,
            "harness_timeout_performance_sec": 1800,
            "harness_per_shape_timeout_sec": 300,
        }
        expected = 1800 + ko._DEFAULT_HARNESS_OUTER_BUFFER_SEC

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base = tmp / "local.yaml"
            base.write_text(
                "model:\n  model_class: litellm\ntools:\n  rag: true\n",
                encoding="utf-8",
            )
            prompt = tmp / "prompt.md"
            prompt.write_text("placeholder", encoding="utf-8")
            previous = os.environ.get("GEAK_CONFIG")
            os.environ["GEAK_CONFIG"] = str(base)
            try:
                args = argparse.Namespace(disable_rag=False)
                override = ko._geak_config_for_run(
                    args, prompt, unittest_manifest=manifest,
                )
            finally:
                if previous is None:
                    os.environ.pop("GEAK_CONFIG", None)
                else:
                    os.environ["GEAK_CONFIG"] = previous

            self.assertNotEqual(override, str(base))
            text = Path(override).read_text(encoding="utf-8")
            self.assertRegex(text, rf"timeout:\s*{expected}")
            self.assertIn("model_class: litellm", text)

    def test_geak_config_for_run_falls_back_to_3600_without_manifest(self) -> None:
        ko = self._import_ko()
        import tempfile, argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base = tmp / "local.yaml"
            base.write_text(
                "model:\n  model_class: litellm\ntools:\n  rag: true\n",
                encoding="utf-8",
            )
            prompt = tmp / "prompt.md"
            prompt.write_text("placeholder", encoding="utf-8")
            previous = os.environ.get("GEAK_CONFIG")
            os.environ["GEAK_CONFIG"] = str(base)
            try:
                args = argparse.Namespace(disable_rag=False)
                override = ko._geak_config_for_run(args, prompt)
            finally:
                if previous is None:
                    os.environ.pop("GEAK_CONFIG", None)
                else:
                    os.environ["GEAK_CONFIG"] = previous

            text = Path(override).read_text(encoding="utf-8")
            self.assertRegex(text, r"timeout:\s*3600")


if __name__ == "__main__":
    unittest.main()
