# SPDX-License-Identifier: MIT
"""Guards for defects that were shipped once as unreachable code.

Every case here covers a mechanism that existed, was correct, and was never
called -- or was called with the wrong input. Unit-testing the helper in
isolation would have passed in each instance, so these assert the wiring: that
the verdict actually changes.
"""

from __future__ import annotations

import ast

import pytest

from kernelforge.resources import resource_path

from kernelforge.conftest import PACKAGE_ROOT

RUNNER = PACKAGE_ROOT / "loop" / "runner.py"


def _calls_in_runner(name: str) -> int:
    """How many times ``name`` is called in runner.py, ignoring its own def."""
    tree = ast.parse(RUNNER.read_text())
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == name)
            or (isinstance(node.func, ast.Name) and node.func.id == name)
        )
    )


@pytest.mark.parametrize(
    "helper",
    ["_promote_best"],
)
def test_scoring_helpers_are_reachable(helper):
    """A guard nobody calls protects nothing."""
    assert _calls_in_runner(helper) > 0, f"{helper} has no call site"


def _load_example_driver(monkeypatch, ranks: str = "8"):
    """Import the example driver with a known rank count."""
    import importlib.util
    import sys

    for var in ("WORLD_SIZE", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FORGE_NPROC_PER_NODE", ranks)
    path = resource_path("examples") / "aiter-allreduce-forge-loop" / "driver.py"
    spec = importlib.util.spec_from_file_location("_example_driver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_example_driver"] = module
    spec.loader.exec_module(module)
    return module


def _install_parallel_state_stub(monkeypatch):
    """Install the minimal aiter parallel-state module used by driver tests."""
    import sys
    import types

    aiter = types.ModuleType("aiter")
    aiter_dist = types.ModuleType("aiter.dist")
    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.destroy_distributed_environment = lambda: None
    parallel_state.destroy_model_parallel = lambda: None
    aiter.dist = aiter_dist
    aiter_dist.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.dist", aiter_dist)
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", parallel_state)
    return parallel_state


def test_empty_shape_measures_the_whole_suite(monkeypatch):
    """Validation and benchmarking pass no shape, and they decide KEEP.

    Mapping their empty string onto the preflight's single probe case scored
    the campaign on one 1x7168 all-reduce and dropped the crossover sweep, the
    production row count and the fused regression guard.
    """
    torch = pytest.importorskip("torch")
    assert torch  # the driver imports it at module scope
    driver = _load_example_driver(monkeypatch)

    scored, _ = driver.parse_shape("")
    probe, _ = driver.parse_shape("default")

    assert len(probe) == 1
    assert len(scored) > 1
    assert any(c.target == "fused" for c in scored), "fused guard missing"
    assert any(c.rows == 64 for c in scored), "production row count missing"


def test_unspecified_mode_runs_the_full_correctness_matrix(monkeypatch):
    """Smoke alone cannot see the known publish-path race.

    Unit-scale inputs still sum plausibly when a rank reads a half-updated
    buffer; the stability scale is what moves the result far enough to fail.
    """
    torch = pytest.importorskip("torch")
    assert torch
    driver = _load_example_driver(monkeypatch)
    parser = driver.build_parser()

    default_modes = parser.parse_args([]).mode
    assert default_modes is None, "an explicit default hides the caller's intent"
    assert parser.parse_args(["--mode", "smoke"]).mode == "smoke"


def test_formal_correctness_checks_eager_and_graph_for_each_case(monkeypatch):
    """Formal validation must cover both paths without duplicate benchmark ids."""
    import types

    torch = pytest.importorskip("torch")
    assert torch
    driver = _load_example_driver(monkeypatch)
    cases, _ = driver.parse_shape("")
    case_ids = [case.case_id for case in cases]
    assert len(case_ids) == len(set(case_ids)), "benchmark case ids must be unique"
    assert not any(case.graph for case in cases), "graph coverage must not duplicate cases"

    calls = []

    def fake_check_case(case, _ctx, _seed, mode):
        """Record one eager correctness check."""
        calls.append(("eager", case.case_id, mode))
        return {"snr_db": 200.0, "max_diff": 0.0, "finite": True}

    def fake_check_graph_case(case, _ctx, _seed, mode):
        """Record one graph correctness check."""
        calls.append(("graph", case.case_id, mode))
        return {"snr_db": 200.0, "max_diff": 0.0, "finite": True}

    def identity_reduce(value, _ctx):
        """Return a single-process stand-in for a distributed reduction."""
        return value

    parallel_state = _install_parallel_state_stub(monkeypatch)
    ctx = types.SimpleNamespace(rank=0, device=torch.device("cpu"))
    monkeypatch.setattr(driver, "_quick_reduce_guard", lambda: None)
    monkeypatch.setattr(driver, "parse_shape", lambda _shape: (cases, {"tp": "8"}))
    monkeypatch.setattr(driver, "init_worker", lambda _tp: ctx)
    monkeypatch.setattr(driver, "check_case", fake_check_case)
    monkeypatch.setattr(driver, "check_graph_case", fake_check_graph_case)
    monkeypatch.setattr(driver, "_reduce_min", identity_reduce)
    monkeypatch.setattr(driver, "_reduce_max", identity_reduce)
    monkeypatch.setattr(driver.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(driver.torch.cuda, "empty_cache", lambda: None)
    assert parallel_state

    args = driver.build_parser().parse_args([])
    assert driver.worker_main(args) == 0
    assert calls == [
        (path, case.case_id, mode) for mode in ("smoke", "stability") for case in cases for path in ("eager", "graph")
    ]


def test_graph_replays_validate_distinct_inputs(monkeypatch):
    """A stale first output must fail the second replay's comparison."""
    import contextlib
    import types

    torch = pytest.importorskip("torch")
    assert torch
    driver = _load_example_driver(monkeypatch)
    parallel_state = _install_parallel_state_stub(monkeypatch)
    seeds = []
    graphs = []

    @contextlib.contextmanager
    def graph_capture():
        """Provide the stream handle expected by the graph capture block."""
        yield types.SimpleNamespace(stream=None)

    class NoOpGraph:
        """Model a broken graph whose replay leaves its output stale."""

        def __init__(self):
            self.replays = 0
            graphs.append(self)

        def replay(self):
            """Count a replay without refreshing the captured output."""
            self.replays += 1

    def fake_cuda_graph(_graph, stream=None):
        """Provide a no-op CUDA graph capture context."""
        assert stream is None
        return contextlib.nullcontext()

    def fake_inputs(_case, _ctx, seed, _mode):
        """Return a seed-identifiable tensor for each replay."""
        seeds.append(seed)
        return {"x": torch.tensor([float(seed)])}

    def fake_reference(_case, _ctx, inp):
        """Return the expected output for the current replay input."""
        return inp["x"] * 2

    def fake_candidate(_case, _ctx, inp):
        """Return the capture-time output that broken replays leave stale."""
        return inp["x"] * 2

    monkeypatch.setattr(parallel_state, "graph_capture", graph_capture, raising=False)
    monkeypatch.setattr(driver.torch.cuda, "CUDAGraph", NoOpGraph)
    monkeypatch.setattr(driver.torch.cuda, "graph", fake_cuda_graph)
    monkeypatch.setattr(driver.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(driver, "_make_inputs", fake_inputs)
    monkeypatch.setattr(driver, "_assert_custom_ar", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(driver, "run_reference", fake_reference)
    monkeypatch.setattr(driver, "run_candidate", fake_candidate)

    case = driver.Case("raw", 1, 1)
    result = driver.check_graph_case(case, types.SimpleNamespace(), seed=7)

    assert len(seeds) == 2 and len(set(seeds)) == 2
    assert graphs[0].replays == 2
    assert result["max_diff"] > 0.0


def test_named_suite_reaches_the_scored_run(monkeypatch):
    """forge passes no --shape, so a named suite arrives by environment.

    Without this the operator's SUITE only affects the launcher's self-check
    while the campaign scores the derived default.
    """
    torch = pytest.importorskip("torch")
    assert torch
    monkeypatch.setenv("FORGE_COLLECTIVE_SUITE", "tp8_k3")
    driver = _load_example_driver(monkeypatch)
    cases, kv = driver.parse_shape("")
    assert kv.get("suite") == "tp8_k3"

    monkeypatch.setenv("FORGE_COLLECTIVE_SUITE", "default")
    default_driver = _load_example_driver(monkeypatch)
    default_cases, _ = default_driver.parse_shape("")
    assert len(cases) != len(default_cases), "named suite collapsed to default"


def test_unknown_suite_name_is_rejected(monkeypatch):
    """A typo must fail loudly, not silently score a different workload."""
    torch = pytest.importorskip("torch")
    assert torch
    monkeypatch.setenv("FORGE_COLLECTIVE_SUITE", "tp8_k4")
    driver = _load_example_driver(monkeypatch)
    with pytest.raises(ValueError, match="unknown suite"):
        driver.parse_shape("")


def test_a_self_relaunching_driver_stays_in_the_callers_group():
    """The AITER driver must not put its torchrun in a second session.

    SIGKILL is neither catchable nor deliverable across a session boundary, so a
    detached launcher would survive the caller's group kill with its GPUs still
    allocated -- and no handler in the driver would ever run to release them.
    """
    driver = resource_path("examples") / "aiter-allreduce-forge-loop" / "driver.py"
    src = driver.read_text()
    start = src.index("def self_launch(")
    launch = src[start : start + 2000]
    # Check the CALL, not the prose: the comment above it names the flag it is
    # deliberately not passing.
    popen_line = next(line for line in launch.splitlines() if "subprocess.Popen(" in line)
    assert "start_new_session" not in popen_line
    assert "cmd" in popen_line and "env=env" in popen_line
