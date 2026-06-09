#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Local tests for Kernel Agent tools (fixtures generated at runtime; no large trace files in repo)."""

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


# ``classify_patchability`` only routes a candidate to optimization
# backends when its ``source_file`` resolves under a reusable framework
# root (sglang / vllm / aiter / FlyDSL checkouts) — see
# ``tracelens_analysis.classify_patchability``. The synthetic fixtures
# below therefore live under ``/sgl-workspace/sglang/...`` so the trace
# tool classifies them as routable; arbitrary paths like
# ``/src/kernels/rmsnorm.py`` are (correctly) rejected as "source not
# under a reusable framework root" and short-circuit to a skipped REVERT.
# The files need not exist on disk: the gate keys off the path prefix +
# source_type, not file contents.
_FRAMEWORK_ROOT = "/sgl-workspace/sglang/python/sglang/srt/layers"


def write_trace(
    path: Path,
    *,
    kernel_name: str = "triton_rmsnorm_kernel",
    source_file: str = f"{_FRAMEWORK_ROOT}/rmsnorm.py",
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
                    "source_file": f"{_FRAMEWORK_ROOT}/moe.py",
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
    """Hyperloom P2-3: ``update_status`` writes ``ended_at``/``duration_seconds`` on terminal states."""

    def _import_module(self):
        # tools dir must be on sys.path so the sibling import resolves.
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
        self.assertIn("--check-only", help_proc.stdout)
        dry_proc = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--dry-run"],
            text=True,
            capture_output=True,
            timeout=30,
            env={
                **os.environ,
                # Missing paths make the "not found" warning deterministic.
                "TRACELENS_ROOT": str(ROOT / "missing-tracelens-root"),
                "TRACELENS_INTERNAL_ROOT": str(ROOT / "missing-tracelens"),
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

        self.assertIn('TRACELENS_ROOT="${TRACELENS_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors/TraceLens}"', install_text)
        # Internal extension is opt-in: no default path.
        self.assertIn('TRACELENS_INTERNAL_ROOT="${TRACELENS_INTERNAL_ROOT:-}"', install_text)
        # Assert the override pattern, not the exact pin, so ref bumps don't break this.
        self.assertIn('GEAK_REF="${GEAK_REF:-', install_text)
        self.assertIn("ensure_rocm_torch_for_geak()", install_text)
        self.assertIn("KERNEL_AGENT_SKIP_TORCH_GATE", install_text)
        self.assertIn("rocm-smi --showid", install_text)
        # Prefix match on `_PIP_FLAGS` allows future flag additions.
        self.assertIn('_PIP_FLAGS="-q --no-cache-dir', install_text)
        self.assertIn(
            'python3 -m pip install ${_PIP_FLAGS} ${_PIP_CONSTRAINT_ARGS} "${HYPERLOOM_ROOT}/geak"',
            install_text,
        )
        # GEAK v3.2.0 ships 4 MCP tool folders (metrix-mcp removed; transitive via profiler-mcp).
        for _mcp in (
            "rag-mcp",
            "profiler-mcp",
            "cross-session-memory-mcp",
            "automated-test-discovery",
        ):
            self.assertIn(_mcp, install_text)
        # Pin the v3.2.0 install-loop ordering so re-adding metrix-mcp regresses this.
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
        # Auto-detect block: cuda stays the preferred default, env var still overrides.
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
        # No hard-coded /wekafs TRACELENS_ROOT fallback; tool fails loudly when missing.
        self.assertNotIn("DEFAULT_TRACELENS_ROOT", trace_tool_text)
        self.assertIn(
            'parser.add_argument("--tracelens-root", default=os.environ.get("TRACELENS_ROOT", "")',
            trace_tool_text,
        )
        self.assertIn(
            "TraceLens root not provided: set TRACELENS_ROOT in env",
            trace_tool_text,
        )
        self.assertNotIn('TRACELENS_ROOT="${TRACELENS_ROOT:-/hyperloom/TraceLens}"', install_text)
        self.assertNotIn('TRACELENS_INTERNAL_ROOT="${TRACELENS_INTERNAL_ROOT:-/hyperloom/TraceLens-internal}"', install_text)
        self.assertNotIn("Executor asks", skill_text)
        # Public TraceLens is cloned into the runtime tree by default; TRACELENS_ROOT overrides.
        self.assertIn(
            'TRACELENS_REPO="https://github.com/AMD-AGI/TraceLens.git"',
            install_text,
        )
        self.assertIn('TRACELENS_REF="0ebaa7109992b98b8f747a0fc0973e0f3b65d5d9"', install_text)
        self.assertIn('TRACELENS_ROOT="${TRACELENS_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors/TraceLens}"', install_text)
        self.assertIn('git clone --depth 1 "$TRACELENS_REPO" "$TRACELENS_ROOT"', install_text)
        self.assertIn('git -C "$TRACELENS_ROOT" fetch --depth 1 origin "$TRACELENS_REF"', install_text)
        self.assertNotIn('TRACELENS_PUBLIC_MIRROR_DIR="${TRACELENS_PUBLIC_MIRROR_DIR:-${HYPERLOOM_ROOT}/TraceLens}"', install_text)
        # Read-only internal root may trigger a writable mirror (optional extension).
        self.assertIn(
            'TRACELENS_MIRROR_DIR="${TRACELENS_MIRROR_DIR:-${HYPERLOOM_ROOT}/TraceLens-internal}"',
            install_text,
        )
        self.assertIn('export TRACELENS_INTERNAL_ROOT', install_text)
        self.assertIn(
            "echo \"export TRACELENS_INTERNAL_ROOT='${TRACELENS_INTERNAL_ROOT}'\"",
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
                source_file=f"{_FRAMEWORK_ROOT}/unmapped.py",
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
        """Policy change (#144): auto-pick includes GEAK even with no benchmark; decision stays NEEDS_REVIEW."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            write_trace(
                trace,
                kernel_name="triton_unmapped_kernel",
                source_file=f"{_FRAMEWORK_ROOT}/unmapped.py",
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

            # GEAK is now first in the ladder.
            self.assertIn("geak", result["selected_backends"])
            self.assertEqual(result["selected_backends"][0], "geak")
            # No bench → flagged for downstream verification gates.
            self.assertTrue(
                result["backend_selection"]["geak_without_benchmark"]
            )
            # E2E evidence still missing, so still NEEDS_REVIEW.
            self.assertEqual(result["proposal"]["decision"], "NEEDS_REVIEW")
            self.assertIn("E2E evidence missing", result["proposal"]["reasons"])

    def test_user_specified_geak_without_benchmark_is_marked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            write_trace(
                trace,
                kernel_name="triton_unmapped_kernel",
                source_file=f"{_FRAMEWORK_ROOT}/unmapped.py",
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
                # Need >= 1.50x microbench to clear KEEP threshold.
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

            # A vendor BLAS binary (hipblasLt) carries no rewritable source,
            # so ``classify_patchability`` marks it non-routable. The
            # optimizer now short-circuits *before* backend selection
            # (PR #314 "filter non-routable kernels") into a skipped/REVERT
            # result instead of running an empty backend ladder. The
            # invariant remains: no backend is dispatched and the proposal
            # is REVERT.
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["reason"], "non_routable_candidate")
            self.assertEqual(result["error_class"], "missing_native_source")
            self.assertEqual(result["decision"], "REVERT")
            self.assertEqual(result["proposal"]["decision"], "REVERT")
            self.assertNotIn("selected_backends", result)

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

            # A hallucinated kernel_id is now a *graceful skip* rather than
            # a hard failure (commit bdeac3ce "survive hallucinated
            # kernel_id"): the optimizer exits 0 with status=skipped so the
            # orchestrator moves to the next decision instead of burning the
            # whole run. The invariant this test pins is that no fabricated /
            # partial optimization result is emitted for an unknown id.
            self.assertEqual(code, 0)
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["reason"], "kernel_id_not_in_candidates")
            self.assertEqual(result["error_class"], "invalid_kernel_id")
            self.assertNotIn("proposal", result)
            self.assertNotIn("verification", result)

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
    """Cover :func:`_count_auth_failures` and the partial-promotion guard (r24 custom_allreduce 401-loop regression)."""

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
        # A single 401 is recoverable; below threshold.
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
    """GEAK config ``env.timeout`` must be >= harness budget (else SIGKILL at mini-swe-agent's 30s default)."""

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
        # Idempotent.
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
        """A too-small env.timeout must be rewritten up to the harness budget."""
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
        self.assertEqual(out.count("\nenv:\n"), 1)

    def test_geak_config_for_run_falls_back_to_3600(self) -> None:
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
class GeakCostLimitDefaultTests(unittest.TestCase):
    """Lock in the GEAK cost-limit contract: Hyperloom passes 0.0 (unlimited) so GEAK skips its 3.0 fallback."""

    def setUp(self) -> None:
        import tempfile
        # _resolve_geak_config() needs GEAK_CONFIG pointing at a litellm stub.
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

    # Assert the exact add_argument expression since --help doesn't show defaults.
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
        """HYPERLOOM_GEAK_COST_LIMIT must override the argparse default."""
        env = {**os.environ, "HYPERLOOM_GEAK_COST_LIMIT": "12.5"}
        probe = (
            "import sys, re, runpy\n"
            f"sys.argv = ['kernel_optimization.py', '--help']\n"
            "try:\n"
            f"    runpy.run_path(r'{OPT_TOOL}', run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
        )
        # AST fallback: evaluate the default expression in a controlled namespace.
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
        # backends/ must be on sys.path for geak_submit's bare sibling import.
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
        """``_build_cmd(cost_limit=0.0)`` must emit ``--cost-limit 0.0`` to override the $3 default."""
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
        """``cost_limit=None`` must NOT add the flag, so GEAK uses its config-file value."""
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
