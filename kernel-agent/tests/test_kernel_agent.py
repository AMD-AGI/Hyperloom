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


class UpdateStatusTimingTests(unittest.TestCase):
    """Hyperloom P2-3: ``update_status`` writes ``ended_at`` +
    ``duration_seconds`` once the run reaches a terminal state, so
    downstream session-breakdown collectors can fill the timeline
    event with a real wall-clock duration.
    """

    def _import_module(self):
        # The tracelens_analysis module imports ``tracelens_skill_runner``
        # as a sibling module, so we need the tools directory on
        # ``sys.path`` for the import to resolve.
        tools_dir = str(TRACE_TOOL.parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tracelens_analysis_under_test", TRACE_TOOL,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def test_running_state_omits_ended_at_and_duration(self) -> None:
        mod = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmpd = Path(tmp)
            log_path = tmpd / "log.txt"
            log_path.write_text("hello\n", encoding="utf-8")
            status_path = tmpd / "status.json"
            mod.update_status(
                status_path,
                state="running",
                current_step="discover",
                log_path=log_path,
                artifact_paths={},
                run_id="tl-run-test",
                started_at="2026-05-22T01:00:00+00:00",
            )
            data = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(data["state"], "running")
            self.assertNotIn("ended_at", data)
            self.assertNotIn("duration_seconds", data)
            self.assertEqual(data["started_at"], "2026-05-22T01:00:00+00:00")

    def test_succeeded_state_writes_ended_at_and_duration(self) -> None:
        mod = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmpd = Path(tmp)
            log_path = tmpd / "log.txt"
            log_path.write_text("done\n", encoding="utf-8")
            status_path = tmpd / "status.json"
            mod.update_status(
                status_path,
                state="succeeded",
                current_step="done",
                log_path=log_path,
                artifact_paths={},
                run_id="tl-run-test",
                started_at="2026-05-22T01:00:00+00:00",
            )
            data = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(data["state"], "succeeded")
            self.assertIn("ended_at", data)
            self.assertIn("duration_seconds", data)
            self.assertEqual(data["ended_at"], data["updated_at"])
            self.assertIsNotNone(data["duration_seconds"])
            self.assertGreaterEqual(data["duration_seconds"], 0.0)

    def test_failed_state_writes_ended_at_and_duration(self) -> None:
        mod = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmpd = Path(tmp)
            log_path = tmpd / "log.txt"
            log_path.write_text("err\n", encoding="utf-8")
            status_path = tmpd / "status.json"
            mod.update_status(
                status_path,
                state="failed",
                current_step="failed",
                log_path=log_path,
                artifact_paths={},
                run_id="tl-run-test",
                started_at="2026-05-22T01:00:00+00:00",
                error="RuntimeError: boom",
            )
            data = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(data["state"], "failed")
            self.assertEqual(data["error"], "RuntimeError: boom")
            self.assertIn("ended_at", data)
            self.assertIn("duration_seconds", data)


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
        self.assertIn('GEAK_REF="${GEAK_REF:-v3.2.0}"', install_text)
        self.assertIn("ensure_rocm_torch_for_geak()", install_text)
        self.assertIn("KERNEL_AGENT_SKIP_TORCH_GATE", install_text)
        self.assertIn("rocm-smi --showid", install_text)
        # pip flags are factored into `_PIP_FLAGS`; assert the core flags
        # survive (prefix match allows future additions) and the install line
        # references the variable.
        self.assertIn('_PIP_FLAGS="-q --no-cache-dir', install_text)
        self.assertIn(
            'python3 -m pip install ${_PIP_FLAGS} ${_PIP_CONSTRAINT_ARGS} "${HYPERLOOM_ROOT}/geak"',
            install_text,
        )
        # GEAK v3.2.0 ships 4 MCP tool folders; minisweagent imports
        # profiler_mcp / cross_session_memory_mcp / automated_test_discovery
        # in addition to rag-mcp. Metrix is consumed transitively as a
        # PyPI dependency of profiler-mcp (no standalone metrix-mcp folder
        # in v3.2.0). Regression-guard: install.sh must NOT pip-install a
        # ``mcp_tools/metrix-mcp`` path (it does not exist in v3.2.0 and
        # was causing install to fail with "File ... does not exist").
        # We assert on the path form to allow human-readable comments that
        # explain the v3.1.0 -> v3.2.0 removal to keep mentioning the name.
        for _mcp in (
            "rag-mcp",
            "profiler-mcp",
            "cross-session-memory-mcp",
            "automated-test-discovery",
        ):
            self.assertIn(_mcp, install_text)
        # The actual install loop iterates over hyphenated names; v3.1.0
        # had ``rag-mcp profiler-mcp metrix-mcp ...``. Pin the v3.2.0
        # ordering so accidental re-adding of ``metrix-mcp`` between
        # ``profiler-mcp`` and ``cross-session-memory-mcp`` regresses
        # this test (the comment block above is allowed to mention
        # ``metrix-mcp`` for human readers).
        self.assertIn(
            "for _geak_mcp in rag-mcp profiler-mcp \\\n"
            "                    cross-session-memory-mcp automated-test-discovery; do",
            install_text,
        )
        self.assertIn(
            'python3 -m pip install ${_PIP_FLAGS} ${_PIP_CONSTRAINT_ARGS} \\\n'
            '        "${HYPERLOOM_ROOT}/geak/mcp_tools/${_geak_mcp}"',
            install_text,
        )
        # GEAK_RAG_INDEX_DEVICE_VAL was refactored from a single-line `:-cuda`
        # default into an auto-detect block (rocm-smi / torch.cuda) with an
        # explicit env override. Assert the two semantic invariants instead of
        # the old literal: cuda remains the preferred default, and the env var
        # can still override.
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

    def test_unittest_skill_is_self_contained(self) -> None:
        kernel_skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        inference_skill = (ROOT.parent / "inference_optimizer" / "SKILL.md").read_text(
            encoding="utf-8",
        )
        combined = kernel_skill + "\n" + inference_skill

        self.assertIn("unittest skill", combined)
        self.assertIn("--test-command", combined)
        self.assertIn("validate_harness", combined)
        self.assertNotIn("/wekafs/" + "zihao", combined)
        self.assertNotIn("geak_cc/" + "AgentKernelArena", combined)

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

    GEAK's ``--test-command`` harness (generated by the unittest skill or
    discovered from benchmark files) can take minutes (e.g. aiter JIT
    recompile + multi-shape benchmark). mini-swe-agent's
    ``LocalEnvironmentConfig.timeout`` defaults to 30s, so if the GEAK
    config doesn't override it, every patch test gets SIGKILLed.

    Hyperloom kernel_optimization fixes this at two layers:
      * ``scripts/install.sh`` writes ``env.timeout: 3600`` into newly
        generated ``local.yaml`` files (covered by the install.sh diff).
      * ``_ensure_yaml_env_timeout`` defensively rewrites any pre-existing
        config to a timeout of at least 3600s.
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

    def test_geak_config_for_run_falls_back_to_3600(self) -> None:
        ko = self._import_ko()
        import tempfile
        import argparse

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
class GeakCostLimitDefaultTests(unittest.TestCase):
    """Lock in the GEAK cost-limit contract: Hyperloom must pass 0.0
    (= unlimited) so GEAK's sub-agent spawn path does not silently
    fall back to ``AgentConfig.cost_limit = 3.0``
    (``minisweagent/agents/default.py``), which on 2026-05-15 killed
    every Qwen3-32B GEAK sub-agent at $3.08 after ~50 steps.

    The only externally addressable lever is GEAK's
    ``-l/--cost-limit`` CLI option (``minisweagent/run/mini.py:194``),
    which writes ``config["agent"]["cost_limit"]`` and is honoured by
    every child agent spawned from that config. These tests guard the
    full propagation chain:

      kernel_optimization.py --geak-cost-limit (default 0.0)
        → geak_submit.submit(cost_limit=0.0)
        → ``geak ... --cost-limit 0.0``

    If anyone reverts the default to ``None`` or drops the
    propagation, every GEAK attempt will silently die at $3 again.
    """

    def setUp(self) -> None:
        import tempfile
        # _resolve_geak_config() requires GEAK_CONFIG to point at a file
        # containing "model_class: litellm". Create a minimal stub so tests
        # that exercise _build_cmd() can run without a real install.
        self._cfg_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        self._cfg_file.write("model:\n  model_class: litellm\n")
        self._cfg_file.flush()
        self._cfg_file.close()
        self._prev_geak_config = os.environ.get("GEAK_CONFIG")
        os.environ["GEAK_CONFIG"] = self._cfg_file.name

    def tearDown(self) -> None:
        if self._prev_geak_config is None:
            os.environ.pop("GEAK_CONFIG", None)
        else:
            os.environ["GEAK_CONFIG"] = self._prev_geak_config
        Path(self._cfg_file.name).unlink(missing_ok=True)

    # Source-text match: ArgumentParser does not surface ``default=...``
    # in ``--help`` output, so the most direct way to lock the contract
    # is to assert the exact ``add_argument`` expression. If anyone
    # refactors the expression they must update this assertion at the
    # same time — which is the point.
    _EXPECTED_DEFAULT_EXPR = (
        'default=float(os.environ.get("HYPERLOOM_GEAK_COST_LIMIT", "0.0"))'
    )

    def test_kernel_optimization_default_is_zero(self) -> None:
        src = OPT_TOOL.read_text(encoding="utf-8")
        self.assertIn('"--geak-cost-limit"', src)
        self.assertIn(self._EXPECTED_DEFAULT_EXPR, src,
                      "kernel_optimization.py --geak-cost-limit default "
                      "must be 0.0 (matching GEAK geak.yaml `cost_limit: 0.`)")

    def test_parallel_e2e_runner_default_is_zero(self) -> None:
        tool = ROOT / "tools" / "parallel_e2e_runner.py"
        src = tool.read_text(encoding="utf-8")
        self.assertIn('"--geak-cost-limit"', src)
        self.assertIn(self._EXPECTED_DEFAULT_EXPR, src,
                      "parallel_e2e_runner.py --geak-cost-limit default "
                      "must mirror kernel_optimization.py (0.0 / env-overridable)")

    def test_parallel_e2e_runner_run_json_invokes_subprocess(self) -> None:
        """run_json must have its subprocess dependency imported."""
        import importlib.util
        import tempfile

        tool = ROOT / "tools" / "parallel_e2e_runner.py"
        spec = importlib.util.spec_from_file_location("parallel_e2e_runner", tool)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            result = module.run_json(
                [
                    sys.executable,
                    "-c",
                    'import json; print(json.dumps({"ok": True}))',
                ],
                env=os.environ.copy(),
                timeout_s=10,
                log_path=Path(td) / "run.log",
            )
        self.assertEqual(result, {"ok": True})

    def test_env_var_overrides_default(self) -> None:
        """End-to-end smoke: HYPERLOOM_GEAK_COST_LIMIT must reach
        kernel_optimization.py's argparse default at import time.
        We exercise this by invoking the tool with ``--help`` and a
        bogus required arg so the parser instantiation completes."""
        env = {**os.environ, "HYPERLOOM_GEAK_COST_LIMIT": "12.5"}
        # Inject a probe right before ``main()`` runs so we can read the
        # resolved default without executing the tool body.
        probe = (
            "import sys, re, runpy\n"
            f"sys.argv = ['kernel_optimization.py', '--help']\n"
            "try:\n"
            f"    runpy.run_path(r'{OPT_TOOL}', run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
        )
        # Simpler / more direct: import the parser construction as a
        # subprocess and have argparse error out on missing --kernel-id,
        # which prints the help line including the resolved default — no,
        # argparse still does not show defaults. Fall back to AST: load
        # the source, evaluate the default expression in a controlled
        # namespace with the env var set.
        import ast
        src = OPT_TOOL.read_text(encoding="utf-8")
        tree = ast.parse(src)
        default_node = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--geak-cost-limit"
            ):
                for kw in node.keywords:
                    if kw.arg == "default":
                        default_node = kw.value
                        break
                break
        self.assertIsNotNone(default_node,
                             "--geak-cost-limit add_argument not found")
        # Evaluate the default expression with our env var set.
        os.environ["HYPERLOOM_GEAK_COST_LIMIT"] = "12.5"
        try:
            value = eval(  # noqa: S307 — controlled expression from our own source
                compile(ast.Expression(default_node), filename="<test>", mode="eval"),
                {"os": os, "float": float, "int": int, "str": str},
            )
        finally:
            os.environ.pop("HYPERLOOM_GEAK_COST_LIMIT", None)
        self.assertEqual(value, 12.5,
                         "HYPERLOOM_GEAK_COST_LIMIT env var must override "
                         f"--geak-cost-limit default; got {value!r}")

    def _import_geak_submit(self):
        # geak_submit imports its sibling ``ray_runtime`` with a bare
        # ``from ray_runtime import ...``, so the ``backends/`` directory
        # must be on sys.path BEFORE the import (not the package root).
        backends_dir = ROOT / "tools" / "backends"
        sys.path.insert(0, str(backends_dir))
        try:
            import importlib
            if "geak_submit" in sys.modules:
                return importlib.reload(sys.modules["geak_submit"])
            return importlib.import_module("geak_submit")
        finally:
            sys.path.pop(0)

    def test_geak_submit_build_cmd_propagates_zero(self) -> None:
        """``_build_cmd(cost_limit=0.0)`` must emit ``--cost-limit 0.0``;
        the ``is not None`` check is what makes 0.0 reach GEAK's mini.py
        and override the dataclass $3 default."""
        geak_submit = self._import_geak_submit()
        cmd = geak_submit._build_cmd(
            prompt_file=Path("/tmp/p.md"),
            output_dir=Path("/tmp/out"),
            kernel_path="/tmp/k.cu",
            gpu_ids="0",
            cost_limit=0.0,
        )
        self.assertIn("--cost-limit", cmd,
                      "cost_limit=0.0 must emit --cost-limit (without it "
                      "GEAK falls back to AgentConfig.cost_limit = 3.0)")
        idx = cmd.index("--cost-limit")
        self.assertEqual(cmd[idx + 1], "0.0",
                         f"--cost-limit must carry the explicit 0.0 value: {cmd}")

    def test_geak_submit_build_cmd_omits_when_none(self) -> None:
        """``cost_limit=None`` (direct CLI users of geak_submit) must NOT
        add the flag, so GEAK falls through to its config-file value."""
        geak_submit = self._import_geak_submit()
        cmd = geak_submit._build_cmd(
            prompt_file=Path("/tmp/p.md"),
            output_dir=Path("/tmp/out"),
            kernel_path="/tmp/k.cu",
            gpu_ids="0",
            cost_limit=None,
        )
        self.assertNotIn("--cost-limit", cmd,
                         f"cost_limit=None must omit --cost-limit: {cmd}")


if __name__ == "__main__":
    unittest.main()
