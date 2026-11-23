# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the in-session gate decision logic (workflows/forge_loop/gate/insession_gate.py)."""

from __future__ import annotations

import asyncio
import dataclasses
import subprocess
from pathlib import Path

import pytest

from kernelforge.evaluation.scoring import KEEP_MEASUREMENT_COUNT
from kernelforge.utils.git import GitError
from kernelforge.workflows.forge_loop.gate.gate_settings import GateMeasurement, GateScope
from kernelforge.workflows.forge_loop.gate.insession_gate import InSessionGate

_MEASUREMENT_FIELDS = [field.name for field in dataclasses.fields(GateMeasurement)]
_SCOPE_FIELDS = [field.name for field in dataclasses.fields(GateScope)]


def _run(coro):
    return asyncio.run(coro)


def test_insession_gate_has_no_duplicate_module_defs():
    """Guard the F811 blind spot: ruff/Pyflakes does NOT flag redefinition of *annotated* module-level functions, so a duplicate (e.g. a bad merge) can silently shadow the real one."""
    import ast
    import collections

    from kernelforge.workflows.forge_loop.gate import insession_gate

    tree = ast.parse(Path(insession_gate.__file__).read_text())
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    dupes = [name for name, c in collections.Counter(names).items() if c > 1]
    assert not dupes, f"duplicate module-level defs shadow each other: {dupes}"


def _gate(tmp_path: Path, **overrides) -> tuple[InSessionGate, Path]:
    workspace = tmp_path / "ws"
    source = workspace / "aiter" / "csrc"
    source.mkdir(parents=True)
    (workspace / "forge_driver.py").write_text("print('driver')\n")
    kernel = source / "kernel.cu"
    kernel.write_text("__global__ void kernel() {}\n")
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=KernelForge Tests",
            "-c",
            "user.email=tests@example.com",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=workspace,
        check=True,
    )
    kwargs = dict(
        driver_script=str(workspace / "forge_driver.py"),
        snr_threshold=30.0,
        baseline_case_times={"case": 1.0},
        best_mean_case_speedup=1.0,
        kernel_file=str(kernel),
        target_files=[str(kernel)],
    )
    kwargs.update(overrides)
    measurement = {name: kwargs.pop(name) for name in _MEASUREMENT_FIELDS if name in kwargs}
    scope = {name: kwargs.pop(name) for name in _SCOPE_FIELDS if name in kwargs}
    return InSessionGate(GateMeasurement(**measurement), GateScope(**scope), **kwargs), workspace


# ── constructor bookkeeping ────────────────────────────────────────────────────


def test_findings_blob_joins(tmp_path):
    gate, _ = _gate(tmp_path)
    gate.findings = ["a", "b"]
    assert gate.findings_blob() == "a\n---\nb"


def test_infer_workspace_root_prefers_existing(tmp_path):
    gate, workspace = _gate(tmp_path)
    assert gate.workspace_root == (workspace).resolve()


def test_infer_workspace_root_none_when_nothing_exists():
    assert InSessionGate._infer_workspace_root(None, "", "") is None
    assert InSessionGate._infer_workspace_root(None, "/no/such/x.py", "") is None
    # A declared workspace that does not exist is no better than none.
    assert InSessionGate._infer_workspace_root("/no/such/ws", "", "") is None


def test_extra_protected_globs_merged_and_deduped(tmp_path):
    gate, _ = _gate(tmp_path, extra_protected_globs=["*harness*.py", "ref_*.py"])
    assert "ref_*.py" in gate.protected_globs
    assert gate.protected_globs.count("*harness*.py") == 1


# ── path-classification helpers ────────────────────────────────────────────────


def test_is_protected_detects_test_dir(tmp_path):
    gate, workspace = _gate(tmp_path)
    assert gate._is_protected(str(workspace / "tests" / "ref.py")) is True
    assert gate._is_protected(str(workspace / "aiter" / "kernel.cu")) is False


def test_protected_changes_reports_added_and_deleted(tmp_path):
    gate, workspace = _gate(tmp_path)
    driver = workspace / "forge_driver.py"
    driver.unlink()
    assert "deleted" in gate._protected_changes()


# ── make_agent_hooks ───────────────────────────────────────────────────────────


def test_the_stop_hook_outlasts_correctness_and_every_bench_measurement(tmp_path):
    """Under a multi-rank driver each stage is a full launch; a ceiling sized for one loses the verdict mid-bench."""
    gate, _ = _gate(tmp_path)

    (stop_hook,) = gate.make_agent_hooks().stop

    assert stop_hook.timeout_sec > gate.stage_timeout_sec + KEEP_MEASUREMENT_COUNT * gate.bench_timeout_sec


# ── PreToolUse edit deny ───────────────────────────────────────────────────────


def test_on_pre_edit_denies_protected(tmp_path):
    gate, workspace = _gate(tmp_path)
    out = _run(
        gate._on_pre_edit(
            {"tool_name": "Edit", "tool_input": {"file_path": str(workspace / "forge_driver.py")}}, None, None
        )
    )
    dec = out["hookSpecificOutput"]
    assert dec["permissionDecision"] == "deny"
    assert any("protected measurement file" in f for f in gate.findings)


def test_on_pre_edit_allows_target_kernel(tmp_path):
    gate, workspace = _gate(tmp_path)
    kernel = workspace / "aiter" / "csrc" / "kernel.cu"
    out = _run(gate._on_pre_edit({"tool_name": "Edit", "tool_input": {"file_path": str(kernel)}}, None, None))
    assert out == {}


def test_on_pre_edit_ignores_non_edit_tool(tmp_path):
    gate, _ = _gate(tmp_path)
    out = _run(gate._on_pre_edit({"tool_name": "Read", "tool_input": {}}, None, None))
    assert out == {}


# ── PreToolUse bash deny ───────────────────────────────────────────────────────


def test_on_pre_bash_denies_protected_write(tmp_path):
    gate, _ = _gate(tmp_path)
    out = _run(
        gate._on_pre_bash({"tool_name": "Bash", "tool_input": {"command": "echo x > forge_driver.py"}}, None, None)
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize(
    "command",
    [
        "pkill -f arena_forge_driver",
        "pkill -9 -f 'g.py 1 b9'",
        "killall python3",
        "timeout 60 python3 sweep.py || pkill -f sweep.py",
        "kill -9 $(pgrep -f rocprofv3) 2>/dev/null",
        'kill -9 "$(pgrep -f bench.py)"',
        "kill -9 `pgrep -f bench.py`",
        'kill -9 "`pidof rocprofv3`"',
        "for p in $(pgrep -f bench.py); do kill $p; done",
        'pids="$(pgrep -f bench.py)"; kill $pids',
        "while read p; do kill $p; done < <(pgrep -f bench.py)",
        "if pgrep -f bench.py > /dev/null; then kill $(cat bench.pid); fi",
        'timeout 30 bash -c "kill -9 $(pidof rocprofv3)"',
        "pgrep -f bench | xargs -r kill",
        "ps aux | grep sweep | awk '{print $2}' | xargs kill -9",
        "bash -c 'pkill python'",
        "kill -9 -1",
        "python3 -c \"import subprocess; subprocess.run(['pkill', '-f', 'bench'])\"",
        'python3 -c "import os, subprocess; [os.kill(int(p), 9) for p in '
        "subprocess.check_output(['pgrep', '-f', 'bench.py']).split()]\"",
        "python3 - <<'PY'\nimport os, signal, subprocess\n"
        "for pid in subprocess.run(['ps', '-eo', 'pid,cmd'], capture_output=True, text=True).stdout.split():\n"
        "    os.kill(int(pid), signal.SIGKILL)\nPY",
    ],
)
def test_on_pre_bash_denies_a_kill_that_chooses_its_targets_by_name(tmp_path, command):
    """Other campaigns on the host run the same tools, so a kill by name is never scoped to this session."""
    gate, _ = _gate(tmp_path)

    out = _run(gate._on_pre_bash({"tool_name": "Bash", "tool_input": {"command": command}}, None, None))

    decision = out["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert "kernelforge.utils.session_kill '<pattern>'" in decision["permissionDecisionReason"]
    assert any("DENIED process kill" in finding for finding in gate.findings)


@pytest.mark.parametrize(
    "command",
    [
        "kill 12345",
        "kill -9 $!",
        "kill %1",
        "kill -TERM $(cat sweep.pid)",
        'kill "$(cat sweep.pid)"',
        "kill `cat sweep.pid`",
        "kill $(jobs -p)",
        "python3 bench.py & pid=$!; sleep 2; kill $pid; wait",
        "echo 'kill $(pgrep bench) is denied'",
        "grep -n pgrep notes.txt; kill 4242",
        "python3 - <<'PY'\nimport subprocess\nrun = subprocess.Popen(['python3', 'bench.py'])\nrun.terminate()\nPY",
        "kill -l",
        "pgrep -f sweep.py",
        "ps aux | grep rocprofv3",
        "python3 -m kernelforge.utils.session_kill sweep.py",
    ],
)
def test_on_pre_bash_allows_a_kill_of_named_processes_and_a_search_that_kills_nothing(tmp_path, command):
    gate, _ = _gate(tmp_path)

    assert _run(gate._on_pre_bash({"tool_name": "Bash", "tool_input": {"command": command}}, None, None)) == {}


def test_on_pre_bash_allows_readonly(tmp_path):
    gate, _ = _gate(tmp_path)
    out = _run(gate._on_pre_bash({"tool_name": "Bash", "tool_input": {"command": "ls -la 2>/dev/null"}}, None, None))
    assert out == {}


def test_on_pre_bash_ignores_non_bash(tmp_path):
    gate, _ = _gate(tmp_path)
    out = _run(gate._on_pre_bash({"tool_name": "Edit", "tool_input": {}}, None, None))
    assert out == {}


# ── Stop hook decisions ────────────────────────────────────────────────────────

import kernelforge.workflows.forge_loop.gate.insession_gate as gate_module


def _patch_canonical(monkeypatch, *, correct=True, wall_ms=0.5):
    """Stub the gate's canonical correctness + bench with in-process fakes."""

    async def _corr(*a, **k):
        return {"passed": correct, "message": "" if correct else "SNR too low"}

    async def _bench(*a, **k):
        return {
            "success": True,
            "median_ms": wall_ms,
            "case_times": {"case": wall_ms},
            "measurements": [
                {
                    "success": True,
                    "case_times": {"case": wall_ms},
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
        }

    monkeypatch.setattr(gate_module, "test_correctness", _corr)
    monkeypatch.setattr(gate_module, "measure_wallclock", _bench)


def _edit_kernel(workspace: Path, body: str = "__global__ void kernel() { /* faster */ }\n") -> None:
    """Change the committed kernel on disk, as any tool would: no hook sees the write."""
    (workspace / "aiter" / "csrc" / "kernel.cu").write_text(body)


def _count_canonical_calls(monkeypatch, *, wall_ms=0.5) -> dict[str, int]:
    """Stub the canonical checks as correct (and faster at 0.5 ms), counting each call."""
    calls = {"correctness": 0, "bench": 0}
    _patch_canonical(monkeypatch, correct=True, wall_ms=wall_ms)
    correctness, bench = gate_module.test_correctness, gate_module.measure_wallclock

    async def counted_correctness(*a, **k):
        calls["correctness"] += 1
        return await correctness(*a, **k)

    async def counted_bench(*a, **k):
        calls["bench"] += 1
        return await bench(*a, **k)

    monkeypatch.setattr(gate_module, "test_correctness", counted_correctness)
    monkeypatch.setattr(gate_module, "measure_wallclock", counted_bench)
    return calls


def test_stop_benches_a_candidate_written_without_an_edit_tool(tmp_path, monkeypatch):
    """A kernel rewritten from Bash (heredoc, sed -i) is as much a candidate as one written with Edit."""
    gate, workspace = _gate(tmp_path)
    calls = _count_canonical_calls(monkeypatch)
    _edit_kernel(workspace)

    assert _run(gate._on_stop({}, None, None)) == {}
    assert calls == {"correctness": 1, "bench": 1}
    assert gate.edit_count == 1
    assert gate.end_reason == "converged"


def test_stop_after_edit_then_revert_allows_without_validation_or_bench(tmp_path, monkeypatch):
    """A tree back at HEAD is the incumbent: re-benching it can only block on noise against a ratcheted bar."""
    gate, workspace = _gate(tmp_path)
    calls = _count_canonical_calls(monkeypatch, wall_ms=1.0)
    gate.last_bench_result = {"success": True, "median_ms": 9.0}
    gate.last_mean_case_speedup = 9.0
    gate.last_wall_ms = 9.0
    _edit_kernel(workspace)
    subprocess.run(["git", "checkout", "--", "aiter/csrc/kernel.cu"], cwd=workspace, check=True)

    assert _run(gate._on_stop({}, None, None)) == {}
    assert calls == {"correctness": 0, "bench": 0}
    assert gate.end_reason == "no_changes"
    assert (gate.block_count, gate.edit_count, gate.passed) == (0, 0, False)
    assert gate.findings == []
    # No measurement of this stop exists, so none may be reported for it.
    assert (gate.last_bench_result, gate.last_mean_case_speedup, gate.last_wall_ms) == (None, None, None)


def test_stop_with_only_untracked_files_allows_without_validation_or_bench(tmp_path, monkeypatch):
    """A new file no tracked source changed to use cannot change the build, and the outer loop calls this NO_CHANGES."""
    gate, workspace = _gate(tmp_path)
    calls = _count_canonical_calls(monkeypatch)
    (workspace / "aiter" / "csrc" / "scratch.cu").write_text("__global__ void unused() {}\n")

    assert _run(gate._on_stop({}, None, None)) == {}
    assert calls == {"correctness": 0, "bench": 0}
    assert gate.end_reason == "no_changes"
    assert (gate.block_count, gate.edit_count) == (0, 0)


def test_stop_in_a_workspace_git_cannot_diff_leaves_the_candidate_to_the_outer_loop(tmp_path, monkeypatch):
    gate, workspace = _gate(tmp_path)
    calls = _count_canonical_calls(monkeypatch)
    _edit_kernel(workspace)

    def broken_git_bytes(*args, **kwargs):
        raise GitError(128, ["git", *args], stderr=b"fatal: bad object HEAD")

    monkeypatch.setattr(gate_module, "git_bytes", broken_git_bytes)

    assert _run(gate._on_stop({}, None, None)) == {}
    assert calls == {"correctness": 0, "bench": 0}
    assert gate.end_reason == "jit_rebuild_unavailable"
    assert gate.block_count == 0
    assert "bad object HEAD" in gate.findings[-1]


def test_stop_benches_an_edit_tool_change_as_before(tmp_path, monkeypatch):
    gate, workspace = _gate(tmp_path)
    calls = _count_canonical_calls(monkeypatch)
    kernel = str(workspace / "aiter" / "csrc" / "kernel.cu")
    assert _run(gate._on_pre_edit({"tool_name": "Edit", "tool_input": {"file_path": kernel}}, None, None)) == {}
    _edit_kernel(workspace)

    assert _run(gate._on_stop({}, None, None)) == {}
    assert calls == {"correctness": 1, "bench": 1}
    assert gate.edit_count == 1
    assert gate.passed is True
    assert gate.end_reason == "converged"
    assert gate.last_wall_ms == 0.5


def test_session_end_counts_the_candidate_diff_without_a_stop(tmp_path):
    """A lane session runs no Stop hook; its edit count still comes from the tree it leaves."""
    gate, workspace = _gate(tmp_path)
    _edit_kernel(workspace)

    assert gate.finalize_integrity() == ""
    assert gate.edit_count == 1


def test_stop_blocks_when_protected_changed(tmp_path):
    gate, workspace = _gate(tmp_path)
    (workspace / "forge_driver.py").write_text("print('driver')\nhacked=1\n")
    out = _run(gate._on_stop({}, None, None))
    assert out["decision"] == "block"
    assert "Protected benchmark harness" in out["reason"]
    assert gate.block_count == 0
    assert gate.harness_block_count == 1


def test_stop_allows_when_correct_and_faster(tmp_path, monkeypatch):
    # best_ms=1.0; a 0.5ms candidate beats it by > noise floor -> converged.
    gate, workspace = _gate(tmp_path)
    _edit_kernel(workspace)
    _patch_canonical(monkeypatch, correct=True, wall_ms=0.5)
    out = _run(gate._on_stop({}, None, None))
    assert out == {}
    assert gate.passed is True
    assert gate.last_wall_ms == 0.5
    assert gate.end_reason == "converged"


def test_stop_gate_invokes_driver_without_shape_selectors(tmp_path, monkeypatch):
    gate, workspace = _gate(tmp_path)
    _edit_kernel(workspace)
    calls = []

    async def correctness(**kwargs):
        calls.append(
            (
                "correctness",
                kwargs["driver_args"],
                kwargs["timeout_sec"],
            )
        )
        return {"passed": True, "message": ""}

    async def benchmark(run, **kwargs):
        calls.append(
            (
                "benchmark",
                run.driver_args,
                kwargs["timeout_sec"],
                kwargs["measurements"],
            )
        )
        return {
            "success": True,
            "median_ms": 0.5,
            "case_times": {"case": 0.5},
            "measurements": [
                {
                    "success": True,
                    "case_times": {"case": 0.5},
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
        }

    monkeypatch.setattr(gate_module, "test_correctness", correctness)
    monkeypatch.setattr(gate_module, "measure_wallclock", benchmark)

    assert _run(gate._on_stop({}, None, None)) == {}
    assert calls == [
        ("correctness", [], 1800),
        ("benchmark", [], 300, 3),
    ]


def test_stop_hands_validation_timeout_to_outer_loop(tmp_path, monkeypatch):
    gate, workspace = _gate(tmp_path)
    _edit_kernel(workspace)
    bench_calls = {"count": 0}

    async def correctness(**_kwargs):
        return {
            "passed": False,
            "outcome": "timeout",
            "message": "TIMEOUT after 1800s",
        }

    async def benchmark(_run, **_kwargs):
        bench_calls["count"] += 1
        return {"median_ms": 0.5}

    monkeypatch.setattr(gate_module, "test_correctness", correctness)
    monkeypatch.setattr(gate_module, "measure_wallclock", benchmark)

    assert _run(gate._on_stop({}, None, None)) == {}
    assert gate.end_reason == "validation_timeout"
    assert gate.block_count == 0
    assert gate.passed is False
    assert bench_calls["count"] == 0
    assert "outer validation" in gate.findings_blob()


def test_stop_blocks_when_incorrect(tmp_path, monkeypatch):
    gate, workspace = _gate(tmp_path)
    _edit_kernel(workspace)
    _patch_canonical(monkeypatch, correct=False)
    out = _run(gate._on_stop({}, None, None))
    assert out["decision"] == "block"
    assert "fails correctness" in out["reason"]
    assert gate.passed is False


def test_stop_blocks_when_correct_but_not_faster(tmp_path, monkeypatch):
    # best_ms=1.0; a 1.0ms candidate does not beat it -> block, keep optimizing.
    gate, workspace = _gate(tmp_path)
    _edit_kernel(workspace)
    _patch_canonical(monkeypatch, correct=True, wall_ms=1.0)
    out = _run(gate._on_stop({}, None, None))
    assert out["decision"] == "block"
    assert "NOT faster" in out["reason"]
    assert gate.passed is False


def test_performance_verdict_clears_stale_bench_state_on_coverage_failure(tmp_path, monkeypatch):
    gate, _ = _gate(tmp_path)
    gate.last_bench_result = {"success": True, "median_ms": 9.0}
    gate.last_mean_case_speedup = 9.0
    gate.last_wall_ms = 9.0

    async def benchmark(*_args, **_kwargs):
        return {
            "success": True,
            "median_ms": 0.5,
            "measurements": [
                {
                    "success": True,
                    "case_times": {},
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
        }

    monkeypatch.setattr(gate_module, "measure_wallclock", benchmark)

    out = _run(gate._stop_performance_verdict())

    assert out["decision"] == "block"
    assert (gate.last_bench_result, gate.last_mean_case_speedup, gate.last_wall_ms) == (None, None, None)


def test_performance_verdict_keeps_current_wall_when_scoring_is_unavailable(tmp_path, monkeypatch):
    gate, _ = _gate(tmp_path, best_mean_case_speedup=None)
    gate.last_bench_result = {"success": True, "median_ms": 9.0}
    gate.last_mean_case_speedup = 9.0
    gate.last_wall_ms = 9.0
    _patch_canonical(monkeypatch, wall_ms=0.5)

    out = _run(gate._stop_performance_verdict())

    assert out["decision"] == "block"
    assert (gate.last_bench_result, gate.last_mean_case_speedup, gate.last_wall_ms) == (None, None, 0.5)


def test_stop_clears_stale_bench_state_when_benchmark_raises(tmp_path, monkeypatch):
    gate, workspace = _gate(tmp_path)
    _edit_kernel(workspace)
    gate.last_bench_result = {"success": True, "median_ms": 9.0}
    gate.last_mean_case_speedup = 9.0
    gate.last_wall_ms = 9.0

    async def correctness(**_kwargs):
        return {"passed": True, "message": ""}

    async def benchmark(*_args, **_kwargs):
        raise RuntimeError("benchmark crashed")

    monkeypatch.setattr(gate_module, "test_correctness", correctness)
    monkeypatch.setattr(gate_module, "measure_wallclock", benchmark)

    assert _run(gate._on_stop({}, None, None)) == {}
    assert gate.end_reason == "gate_error"
    assert (gate.last_bench_result, gate.last_mean_case_speedup, gate.last_wall_ms) == (None, None, None)


def test_correctness_only_allows_without_ever_consulting_the_perf_gate(tmp_path, monkeypatch):
    """PORT-mode contract: with correctness_only=True the gate allows a CORRECT kernel and MUST NOT run the benchmark / perf gate at all."""
    gate, workspace = _gate(
        tmp_path,
        correctness_only=True,
    )
    _edit_kernel(workspace)
    bench_calls = {"n": 0}

    async def _corr(*a, **k):
        return {"passed": True, "message": ""}

    async def _bench(*a, **k):
        bench_calls["n"] += 1
        return {"median_ms": 999.0}  # would fail the perf gate if it were consulted

    monkeypatch.setattr(gate_module, "test_correctness", _corr)
    monkeypatch.setattr(gate_module, "measure_wallclock", _bench)

    out = _run(gate._on_stop({}, None, None))
    assert out == {}  # allowed
    assert gate.passed is True
    assert gate.end_reason == "converged"
    assert bench_calls["n"] == 0  # perf gate never consulted in PORT mode


def test_stop_hands_off_when_block_budget_exhausted(tmp_path, monkeypatch):
    # Budget is checked BEFORE the canonical validation, so an exhausted session hands off immediately (the fakes
    # would otherwise report correct+faster).
    gate, workspace = _gate(tmp_path, max_blocks=2)
    _edit_kernel(workspace)
    _patch_canonical(monkeypatch, correct=True, wall_ms=0.5)
    gate.block_count = gate.max_blocks
    out = _run(gate._on_stop({}, None, None))
    assert out == {}
    assert gate.end_reason == "block_budget_exhausted"
    assert gate.passed is False  # never ran the canonical pass


def test_stop_block_cap_hands_off_as_harness_tampered(tmp_path):
    # An agent that never restores a tampered harness must not block forever.
    gate, workspace = _gate(tmp_path, max_stop_blocks=2)
    (workspace / "forge_driver.py").write_text("print('driver')\nhacked=1\n")

    for _ in range(gate.max_stop_blocks):
        out = _run(gate._on_stop({}, None, None))
        assert out["decision"] == "block"
    assert gate.harness_block_count == gate.max_stop_blocks
    assert gate.block_count == 0

    # Cap reached: the next stop is ALLOWED and tagged for the outer force-REVERT.
    out = _run(gate._on_stop({}, None, None))
    assert out == {}
    assert gate.end_reason == "harness_tampered"
    assert gate.harness_block_count == gate.max_stop_blocks
    assert gate.block_count == 0


def test_stop_fails_open_on_exception(tmp_path, monkeypatch):
    gate, _ = _gate(tmp_path)

    # Force the protection check to raise; the gate must fail OPEN (allow stop) so a hook crash can never hang the
    # session — the outer loop re-validates.
    def boom():
        raise RuntimeError("gate crash")

    monkeypatch.setattr(gate, "_protected_changes", boom)
    out = _run(gate._on_stop({}, None, None))
    assert out == {}
    assert gate.end_reason == "gate_error"
