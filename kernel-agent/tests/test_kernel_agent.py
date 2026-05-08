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
    env = {**os.environ, "WORKSPACE_PATH": str(workspace)}
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=60)
    if proc.returncode != 0:
        raise AssertionError(f"command failed\ncmd={cmd}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return json.loads(proc.stdout)


def run_json_allow_fail(cmd: list[str], *, workspace: Path) -> tuple[int, dict]:
    env = {**os.environ, "WORKSPACE_PATH": str(workspace)}
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
        self.assertIn("ensuring ray[default]==2.44.1", dry_proc.stdout)

    def test_new_environment_defaults_are_documented_in_tools(self) -> None:
        install_text = INSTALL_SCRIPT.read_text(encoding="utf-8")
        trace_tool_text = TRACE_TOOL.read_text(encoding="utf-8")
        skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        ray_runtime_text = RAY_RUNTIME.read_text(encoding="utf-8")

        self.assertIn('TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"', install_text)
        self.assertIn('GEAK_REF="${GEAK_REF:-v3.1.0}"', install_text)
        self.assertIn('python3 -m pip install -q --no-cache-dir -e "${HYPERLOOM_ROOT}/geak/mcp_tools/rag-mcp"', install_text)
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

    def test_default_backends_skip_geak_without_benchmark(self) -> None:
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

            self.assertNotIn("geak", result["selected_backends"])
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


if __name__ == "__main__":
    unittest.main()
