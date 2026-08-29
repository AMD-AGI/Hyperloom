"""Measurement-fidelity guards for the keep/revert decision."""

from __future__ import annotations

import ast
import asyncio
import json
import pathlib

from kernelforge.loop.insession_gate import InSessionGate
from kernelforge.loop.scoring import (
    keep_score,
    passes_keep_threshold,
    required_keep_speedup,
)
from kernelforge.mcp_server.tools import bench as bench_module
from kernelforge.resources import resource_path
from kernelforge.mcp_server.tools.bench import (
    bench_wallclock,
    measure_wallclock,
)

# Records the argv it was invoked with so tests can assert on the exact flags
# bench_wallclock chose to pass.
_ARGV_DRIVER = """
import json, pathlib, sys
pathlib.Path(sys.argv[0] + ".argv").write_text(json.dumps(sys.argv[1:]))
print("mean_ms: 5.0")
"""


def _run_bench(tmp_path, **kwargs) -> list[str]:
    """Run bench_wallclock against the argv-recording driver, return its argv."""
    drv = tmp_path / "drv.py"
    drv.write_text(_ARGV_DRIVER)
    res = asyncio.run(bench_wallclock(driver_script=str(drv), **kwargs))
    assert res["success"], res
    return json.loads(pathlib.Path(str(drv) + ".argv").read_text())


def test_repeat_one_omits_the_flag(tmp_path):
    """Single-GPU drivers predate --repeat; passing it would crash argparse."""
    argv = _run_bench(tmp_path)
    assert "--repeat" not in argv


def test_repeat_above_one_passes_the_flag(tmp_path):
    argv = _run_bench(tmp_path, repeat=3)
    assert argv[argv.index("--repeat") + 1] == "3"


def test_three_measurements_aggregate_diagnostics_by_per_case_median(monkeypatch):
    """Keep case medians while reporting only the final bandwidth snapshot."""
    results = iter(
        [
            {
                "success": True,
                "median_ms": 10.0,
                "case_times": {"a": 1.0, "b": 8.0},
                "case_bandwidth": {
                    "a": {"bytes": 64, "algbw_gbs": 1.0, "busbw_gbs": 0.8},
                    "stale": {
                        "bytes": 128,
                        "algbw_gbs": 2.0,
                        "busbw_gbs": 1.6,
                    },
                },
            },
            {
                "success": True,
                "median_ms": 12.0,
                "case_times": {"a": 3.0, "b": 6.0},
                "case_bandwidth": {
                    "a": {"bytes": 64, "algbw_gbs": 1.5, "busbw_gbs": 1.2},
                },
            },
            {
                "success": True,
                "median_ms": 11.0,
                "case_times": {"a": 2.0, "b": 7.0},
                "case_bandwidth": {
                    "a": {"bytes": 64, "algbw_gbs": 1.8, "busbw_gbs": 1.4},
                },
            },
        ]
    )
    calls = []

    async def fake_bench(**kwargs):
        calls.append(kwargs)
        return next(results)

    monkeypatch.setattr(bench_module, "bench_wallclock", fake_bench)
    measured = asyncio.run(
        measure_wallclock(
            driver_script="driver.py",
            measurements=3,
            timeout_sec=45,
        )
    )

    assert len(calls) == 3
    assert measured["case_times"] == {"a": 2.0, "b": 7.0}
    assert measured["median_ms"] == 11.0
    assert measured["measurement_count"] == 3
    assert measured["case_bandwidth"] == {
        "a": {"bytes": 64, "algbw_gbs": 1.8, "busbw_gbs": 1.4},
    }


def test_the_threshold_is_inclusive():
    """A score landing exactly on the bar is a KEEP, and one below it is not."""
    required = required_keep_speedup(1.0, [1.0006, 1.0007, 1.00065])

    assert passes_keep_threshold(
        [required] * 3,
        best_mean_case_speedup=1.0,
    )
    assert not passes_keep_threshold(
        [required, required, required - 1e-6],
        best_mean_case_speedup=1.0,
    )


def test_the_published_pristine_score_is_monotonic_across_keeps():
    current_best = 1.25
    scores = [1.257, 1.2565, 1.25625]

    assert passes_keep_threshold(
        scores,
        best_mean_case_speedup=current_best,
    )
    assert keep_score(scores) >= required_keep_speedup(current_best, scores)


def _gate(**kwargs) -> InSessionGate:
    return InSessionGate(
        driver_script="/tmp/drv.py",
        snr_threshold=30.0,
        baseline_case_times={"case": 1.0},
        best_mean_case_speedup=1.0,
        **kwargs,
    )


def test_stop_hook_timeout_covers_both_stages():
    """The hook runs correctness THEN bench; a shorter timeout truncates it.

    Regression: the timeout was stage_timeout + 120, which is under the 240+300
    worst case, so a slow (e.g. multi-rank) driver lost the verdict entirely.
    """
    gate = _gate(stage_timeout_sec=240, bench_timeout_sec=300)
    hook = gate.make_agent_hooks().stop[0]
    assert hook.timeout_sec >= 240 + 3 * 300


def test_gate_forwards_measurement_settings():
    gate = _gate(bench_timeout_sec=450, bench_repeat=3)
    assert gate.bench_timeout_sec == 450
    assert gate.bench_repeat == 3


def test_gate_measurement_defaults_are_legacy():
    """Single-GPU tasks must see byte-identical behavior."""
    gate = _gate()
    assert gate.bench_repeat == 1
    assert gate.bench_timeout_sec == 300


def test_warmstart_baseline_uses_the_same_repeat_as_the_loop(tmp_path):
    """A single-shot baseline vs repeat-and-median candidates is a free win.

    Regression: warm start seeded the keep threshold from its own bench, which
    ignored bench_repeat. On the TP4 all-reduce suite that offset measured 3.7% --
    above the 2% gate -- so an unchanged kernel cleared it.
    """
    from kernelforge.knowledge import experience_integration as ei

    drv = tmp_path / "drv.py"
    drv.write_text(_ARGV_DRIVER)
    ei._bench_once(str(drv), bench_repeat=3)
    argv = json.loads(pathlib.Path(str(drv) + ".argv").read_text())
    assert argv[argv.index("--repeat") + 1] == "3"


def test_warmstart_baseline_defaults_to_single_shot(tmp_path):
    """Unchanged for tasks that don't configure repeats."""
    from kernelforge.knowledge import experience_integration as ei

    drv = tmp_path / "drv.py"
    drv.write_text(_ARGV_DRIVER)
    ei._bench_once(str(drv))
    argv = json.loads(pathlib.Path(str(drv) + ".argv").read_text())
    assert "--repeat" not in argv


def _load_driver_module():
    """Import the all-reduce driver without its torch/torchrun dependencies."""
    import sys
    import types
    import importlib.util

    path = resource_path("examples") / "aiter-allreduce-forge-loop" / "driver.py"
    # The driver imports torch at module scope purely for dtype/element_size; the
    # suite definitions under test need none of it.
    stub = types.ModuleType("torch")
    stub.bfloat16 = "bfloat16"
    stub.float16 = "float16"
    stub.float32 = "float32"
    stub.tensor = lambda *a, **k: types.SimpleNamespace(element_size=lambda: 2)
    dist = types.ModuleType("torch.distributed")
    stub.distributed = dist
    saved = {k: sys.modules.get(k) for k in ("torch", "torch.distributed")}
    sys.modules["torch"] = stub
    sys.modules["torch.distributed"] = dist
    try:
        spec = importlib.util.spec_from_file_location("_ar_driver_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        # dataclass resolves field types via sys.modules[cls.__module__], so the
        # module has to be registered before its body executes.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# Measured over 5 runs: excluding these reduced raw-case noise from 0.90% to
# 0.50% and fused-case noise from 2.00% to 0.44%.
_NOISY_CASES = {"raw_bf16_4x8192", "fused_bf16_64x8192"}


def test_wide_suite_excludes_the_noisy_cases_from_scoring():
    """Keep noisy cases measured as guards but out of the KEEP score."""
    mod = _load_driver_module()
    cases = mod._suite_tp4_wide("bf16")
    by_id = {c.case_id: c for c in cases}
    for cid in _NOISY_CASES:
        assert cid in by_id, f"{cid} must remain in the suite as a regression guard"
        assert not by_id[cid].sensitive, f"{cid} must not feed the KEEP score"


def test_wide_suite_still_scores_every_other_case():
    """The wide suite's point is that ordinary cases all count."""
    mod = _load_driver_module()
    cases = mod._suite_tp4_wide("bf16")
    scored = {c.case_id for c in cases if c.sensitive}
    assert scored == {c.case_id for c in cases} - _NOISY_CASES


def _forge_loop_cmd():
    """The forge-loop click command, however its name is registered."""
    from kernelforge.cli import main

    for name, cmd in main.commands.items():
        if name.replace("_", "-") == "forge-loop":
            return cmd
    raise AssertionError(f"forge-loop not among {list(main.commands)}")


def test_gate_and_warm_start_are_not_configurable():
    """Neither is a knob: they are unconditional loop behaviour.

    Both were briefly exposed as CLI switches while debugging a collective
    task. They are unrelated to collective profiling and turning either off
    changes campaign semantics, so the loop keeps them fixed on.
    """
    names = {p.name for p in _forge_loop_cmd().params}
    assert "gate" not in names
    assert "warm_start" not in names


def test_driver_aggregates_repeats_by_median():
    """Reduce each sample across ranks before taking either median.

    The driver needs torch+torchrun to execute, so this asserts on the source
    of the aggregation step rather than running it.
    """
    src = resource_path("examples") / "aiter-allreduce-forge-loop" / "driver.py"
    text = src.read_text()
    assert "statistics.median(" in text
    assert "import statistics" in text
    assert "min(r[case.case_id] for r in rounds)" not in text

    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    bench_case = functions["bench_case"]
    sample_loop = next(
        node
        for node in ast.walk(bench_case)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Constant)
        and node.iter.args[0].value == 5
    )
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_reduce_max"
        for node in ast.walk(sample_loop)
    ), "each timing sample must be reduced across ranks"

    worker_main = functions["worker_main"]
    round_assignment = next(
        node
        for node in ast.walk(worker_main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "this_round"
            for target in node.targets
        )
    )
    round_calls = {
        node.func.id
        for node in ast.walk(round_assignment.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "bench_case" in round_calls
    assert "_reduce_max" not in round_calls, "worker must not reduce a case median again"


def test_cli_can_actually_call_kb_warmstart():
    """The CLI's call site must match the function it calls.

    Regression: the CLI passed bench_repeat while kb_warmstart did not accept
    it, so forge-loop raised TypeError three minutes into a campaign -- after
    the workspace and caches were already set up, and only on the warm-start
    path that no test exercised. An 8-hour run produced nothing.
    """
    import inspect
    import re

    from kernelforge import cli
    from kernelforge.knowledge.experience_integration import kb_warmstart

    accepted = set(inspect.signature(kb_warmstart).parameters)
    src = inspect.getsource(cli.forge_loop.callback)
    call = src[src.index("kb_warmstart(") + len("kb_warmstart(") :]
    call = call[: call.index(")\n")]
    passed = set(re.findall(r"(?:^|[\s,(])([a-z_][a-z0-9_]*)\s*=", call))
    assert passed, "could not read the CLI call site"
    unknown = passed - accepted
    assert not unknown, f"CLI passes keywords kb_warmstart rejects: {sorted(unknown)}"
