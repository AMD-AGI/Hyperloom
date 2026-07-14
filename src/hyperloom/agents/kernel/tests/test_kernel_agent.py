#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Local tests for Kernel-agent tools (fixtures generated at runtime; no large trace files in repo)."""

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
    """``update_status`` writes ``ended_at``/``duration_seconds`` on terminal states."""

    def _import_module(self):
        # tools dir must be on sys.path so the sibling import resolves.
        tools_dir = str(TRACE_TOOL.parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "tracelens_analysis_under_test",
            TRACE_TOOL,
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
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
        self.assertIn("ensuring ray[default]==2.44.1", dry_proc.stdout)

    def test_new_environment_defaults_are_documented_in_tools(self) -> None:
        install_text = INSTALL_SCRIPT.read_text(encoding="utf-8")
        trace_tool_text = TRACE_TOOL.read_text(encoding="utf-8")
        skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")

        # Open-source deps default to a pod-local base, decoupled from USER_DATA_PATH,
        # with HYPERLOOM_OPEN_SOURCE_ROOT as an optional override.
        self.assertIn(
            '_open_source_root="${HYPERLOOM_OPEN_SOURCE_ROOT:-/opt/hyperloom/open-source-repos}"',
            install_text,
        )
        self.assertIn('MAGPIE_PATH="${MAGPIE_PATH:-${_open_source_root}/Magpie}"', install_text)
        self.assertIn('TRACELENS_ROOT="${TRACELENS_ROOT:-${_open_source_root}/TraceLens}"', install_text)
        # Internal extension is opt-in: no default path.
        self.assertIn('TRACELENS_INTERNAL_ROOT="${TRACELENS_INTERNAL_ROOT:-}"', install_text)
        # GEAK has a dedicated root so moving it to pod-local storage does not
        # implicitly move TraceLens via HYPERLOOM_ROOT.
        self.assertIn('GEAK_ROOT="${GEAK_ROOT:-${_open_source_root}/GEAK}"', install_text)
        # Override is read but never written into a generated env file.
        self.assertNotIn("export HYPERLOOM_OPEN_SOURCE_ROOT", install_text)
        # Assert the override pattern, not the exact pin, so ref bumps don't break this.
        self.assertIn('GEAK_REF="${GEAK_REF:-', install_text)
        # Prefix match on `_PIP_FLAGS` allows future flag additions.
        self.assertIn('_PIP_FLAGS="-q --no-cache-dir', install_text)
        self.assertNotIn("GEAK_MEMORY_KB_PATH", install_text)
        self.assertIn('"click<8.3.0" "ray[default]==2.44.1"', install_text)
        self.assertIn('chmod 600 "$env_file"', install_text)
        # No hard-coded /wekafs TRACELENS_ROOT fallback; tool fails loudly when missing.
        self.assertNotIn("DEFAULT_TRACELENS_ROOT", trace_tool_text)
        # Robust to the add_argument(...) call being split across lines: assert
        # the flag and its TRACELENS_ROOT env default independently.
        self.assertIn('"--tracelens-root"', trace_tool_text)
        self.assertIn('default=os.environ.get("TRACELENS_ROOT", "")', trace_tool_text)
        self.assertIn(
            "TraceLens root not provided: set TRACELENS_ROOT in env",
            trace_tool_text,
        )
        self.assertNotIn('TRACELENS_ROOT="${TRACELENS_ROOT:-/hyperloom/TraceLens}"', install_text)
        self.assertNotIn(
            'TRACELENS_INTERNAL_ROOT="${TRACELENS_INTERNAL_ROOT:-/hyperloom/TraceLens-internal}"', install_text
        )
        self.assertNotIn("Executor asks", skill_text)
        # Public TraceLens is cloned into the pod-local repo tree by default; TRACELENS_ROOT overrides.
        self.assertIn(
            'TRACELENS_REPO="https://github.com/AMD-AGI/TraceLens.git"',
            install_text,
        )

    def test_forge_cli_install_is_not_backend_order_gated(self) -> None:
        install_text = INSTALL_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("should_install_forge_backend()", install_text)
        self.assertNotIn("if should_install_forge_backend; then", install_text)
        self.assertIn("ensure_forge_claude_cli", install_text)
        self.assertNotIn("skipping forge claude CLI", install_text)
        with tempfile.TemporaryDirectory() as td:
            env = {
                **os.environ,
                "USER_DATA_PATH": td,
                "KERNEL_OPT_BACKEND_ORDER": "forge",
                "TRACELENS_ROOT": str(ROOT / "missing-tracelens-root"),
                "TRACELENS_INTERNAL_ROOT": str(ROOT / "missing-tracelens-internal"),
            }
            proc = subprocess.run(
                ["bash", str(INSTALL_SCRIPT), "--dry-run"],
                text=True,
                capture_output=True,
                timeout=30,
                env=env,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertIn("would install Node.js/npm + @anthropic-ai/claude-code", proc.stdout)
        self.assertIn('TRACELENS_REF="48f7cf6d1cc7c6d3e0aaee06c9689639021d11e3"', install_text)
        self.assertIn('TRACELENS_ROOT="${TRACELENS_ROOT:-${_open_source_root}/TraceLens}"', install_text)
 # clone AND pin the ref inside the temp sibling, then atomically
        # rename — never publish an unpinned/half-cloned $TRACELENS_ROOT.
        self.assertIn('git clone --depth 1 "$TRACELENS_REPO" "$_tl_tmp"', install_text)
        self.assertIn('git -C "$_tl_tmp" fetch --depth 1 origin "$TRACELENS_REF"', install_text)
        self.assertIn('git -C "$_tl_tmp" checkout -q FETCH_HEAD', install_text)
        self.assertIn('mv "$_tl_tmp" "$TRACELENS_ROOT"', install_text)
        self.assertNotIn(
            'TRACELENS_PUBLIC_MIRROR_DIR="${TRACELENS_PUBLIC_MIRROR_DIR:-${HYPERLOOM_ROOT}/TraceLens}"', install_text
        )
        # Read-only internal root may trigger a writable mirror (optional extension).
        self.assertIn(
            'TRACELENS_MIRROR_DIR="${TRACELENS_MIRROR_DIR:-${_open_source_root}/TraceLens-internal}"',
            install_text,
        )
        self.assertIn("export TRACELENS_INTERNAL_ROOT", install_text)
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
        inference_skill = (ROOT.parents[1] / "inference_optimizer" / "SKILL.md").read_text(
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

            result = run_json(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "--trace-input",
                    str(trace),
                    "--session-id",
                    "s1",
                    "--model-name",
                    "test-model",
                    "--framework",
                    "sglang",
                    "--dry-run",
                ],
                workspace=workspace,
            )

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

            result = run_json(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "--trace-input",
                    str(capture),
                    "--session-id",
                    "s2",
                    "--model-name",
                    "test-model",
                    "--framework",
                    "sglang",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            self.assertEqual(result["trace_input_type"], "capture_dir")
            self.assertGreaterEqual(len(result["hot_kernels"]), 2)

    def test_default_backends_forge_without_benchmark(self) -> None:
        """Auto-pick is forge even with no benchmark; decision stays NEEDS_REVIEW."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            write_trace(
                trace,
                kernel_name="triton_unmapped_kernel",
                source_file=f"{_FRAMEWORK_ROOT}/unmapped.py",
            )
            run_json(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "--trace-input",
                    str(trace),
                    "--session-id",
                    "s3",
                    "--model-name",
                    "test-model",
                    "--framework",
                    "sglang",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            result = run_json(
                [
                    sys.executable,
                    str(OPT_TOOL),
                    "--kernel-id",
                    "k001",
                    "--session-id",
                    "s3",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            # Default ladder converged to forge-only; legacy per-kernel backends removed.
            self.assertEqual(result["selected_backends"], ["forge"])
            # E2E evidence still missing, so still NEEDS_REVIEW.
            self.assertEqual(result["proposal"]["decision"], "NEEDS_REVIEW")
            self.assertIn("E2E evidence missing", result["proposal"]["reasons"])

    def test_user_specified_forge_disables_rag_and_xs_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            write_trace(
                trace,
                kernel_name="triton_unmapped_kernel",
                source_file=f"{_FRAMEWORK_ROOT}/unmapped.py",
            )
            run_json(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "--trace-input",
                    str(trace),
                    "--session-id",
                    "s4",
                    "--model-name",
                    "test-model",
                    "--framework",
                    "sglang",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            result = run_json(
                [
                    sys.executable,
                    str(OPT_TOOL),
                    "--kernel-id",
                    "k001",
                    "--session-id",
                    "s4",
                    "--backends",
                    "forge",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            self.assertEqual(result["selected_backends"], ["forge"])
            self.assertEqual(result["rag_hits"], [])
            self.assertEqual(result["xs_memory_hits"], [])

    def test_keep_requires_benchmark_e2e_and_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "trace.json"
            harness = workspace / "bench.py"
            write_trace(trace)
            harness.write_text("print('bench')\n", encoding="utf-8")
            run_json(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "--trace-input",
                    str(trace),
                    "--session-id",
                    "s5",
                    "--model-name",
                    "test-model",
                    "--framework",
                    "sglang",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            result = run_json(
                [
                    sys.executable,
                    str(OPT_TOOL),
                    "--kernel-id",
                    "k001",
                    "--session-id",
                    "s5",
                    "--test-harness-path",
                    str(harness),
                    "--e2e-gain-pct",
                    "1.5",
                    "--accuracy-passed",
                    "true",
                    # Well above the 1.05x KEEP threshold.
                    "--micro-speedup",
                    "1.6",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            self.assertIn("forge", result["selected_backends"])
            self.assertEqual(result["proposal"]["decision"], "KEEP")
            self.assertIn("rag_hits", result)
            self.assertIn("xs_memory_hits", result)

    def test_missing_trace_input_fails_with_status_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            code, result = run_json_allow_fail(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "--trace-input",
                    str(workspace / "missing.json"),
                    "--session-id",
                    "s6",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            self.assertNotEqual(code, 0)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(Path(result["status_path"]).exists())
            self.assertTrue(Path(result["cli_log_path"]).exists())

    def test_vendor_binary_candidate_selects_no_default_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            trace = workspace / "vendor_trace.json"
            write_vendor_trace(trace)
            run_json(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "--trace-input",
                    str(trace),
                    "--session-id",
                    "s7",
                    "--model-name",
                    "test-model",
                    "--framework",
                    "sglang",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            result = run_json(
                [
                    sys.executable,
                    str(OPT_TOOL),
                    "--kernel-id",
                    "k001",
                    "--session-id",
                    "s7",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            # A vendor BLAS binary (hipblasLt) carries no rewritable source,
            # so ``classify_patchability`` marks it non-routable. The
            # optimizer now short-circuits *before* backend selection
            # into a skipped/REVERT
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
            run_json(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "--trace-input",
                    str(trace),
                    "--session-id",
                    "s8",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            code, result = run_json_allow_fail(
                [
                    sys.executable,
                    str(OPT_TOOL),
                    "--kernel-id",
                    "missing-kernel",
                    "--session-id",
                    "s8",
                    "--dry-run",
                ],
                workspace=workspace,
            )

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
            run_json(
                [
                    sys.executable,
                    str(TRACE_TOOL),
                    "--trace-input",
                    str(trace),
                    "--session-id",
                    "s9",
                    "--dry-run",
                ],
                workspace=workspace,
            )

            code, result = run_json_allow_fail(
                [
                    sys.executable,
                    str(OPT_TOOL),
                    "--kernel-id",
                    "k001",
                    "--session-id",
                    "s9",
                    "--backends",
                    "badbackend",
                    "--dry-run",
                ],
                workspace=workspace,
            )

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
            ko._count_auth_failures("WARN: HTTP/1.1 401 Unauthorized; retrying.\nINFO retry succeeded\nspeedup: 1.21x"),
            3,
        )

    def test_count_auth_failures_primus_token_pattern(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        try:
            import kernel_optimization as ko  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)
        log = "Primus.00009 token not present\nPrimus.00009 token not present\nPrimus.00009 token not present\n"
        self.assertEqual(ko._count_auth_failures(log), 3)


class ParallelE2ERunnerTests(unittest.TestCase):
    """Smoke tests for the whole-pipeline e2e runner helper."""

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


if __name__ == "__main__":
    unittest.main()
