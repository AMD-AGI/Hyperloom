# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for grid compatibility and multi-node variant filters."""

from __future__ import annotations

import importlib.util
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors import _benchmark_interpreter as bi
from hyperloom.orchestrator.actions.executors import _grid_variant_filter as vf
from hyperloom.orchestrator.actions.executors import _multi_node_env as mn
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_variant_filter import (
    apply_multi_node_invalid_variants,
)


def _v(name: str, *, args: str = "") -> GridVariant:
    return GridVariant(name=name, extra_server_args=args)


@pytest.fixture()
def _multi_node(monkeypatch):
    monkeypatch.setattr(mn, "is_multi_node", lambda: True)


def test_single_node_is_a_strict_noop(monkeypatch):
    monkeypatch.setattr(mn, "is_multi_node", lambda: False)
    monkeypatch.setenv("CONC", "64")
    grid = [
        _v("low-graph", args="--cuda-graph-max-bs 8"),
        _v("keep", args="--cuda-graph-max-bs 64"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    # Identity is the production contract: single-node returns the input list (`return grid, []`), not a copy.
    assert kept is grid
    assert dropped == []


def test_multi_node_drops_cuda_graph_max_bs_below_conc(_multi_node, monkeypatch):
    monkeypatch.setenv("CONC", "64")
    grid = [
        _v("space-form", args="--cuda-graph-max-bs 32"),
        _v("equals-form", args="--cuda_graph_max_bs=8"),
        _v("at-threshold", args="--cuda-graph-max-bs 64"),
        _v("above", args="--cuda-graph-max-bs 128"),
        _v("no-flag", args="--chunked-prefill-size 8192"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["at-threshold", "above", "no-flag"]
    assert [row["name"] for row in dropped] == ["space-form", "equals-form"]
    assert all(row["source"] == "multi_node_invalid" for row in dropped)
    assert "CONC=64" in dropped[0]["reason"]
    assert "cuda_graph_max_bs=32" in dropped[0]["reason"]


def test_conc_zero_does_not_drop(_multi_node, monkeypatch):
    # Documents observable behaviour: CONC=0 never drops anything.
    monkeypatch.setenv("CONC", "0")
    grid = [_v("low-graph", args="--cuda-graph-max-bs 1")]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["low-graph"]
    assert dropped == []


def test_multi_node_flag_in_non_leading_position_is_detected(_multi_node, monkeypatch):
    """The filter uses re.search(), not re.match() — flag anywhere in the string must fire."""
    monkeypatch.setenv("CONC", "64")
    grid = [
        _v("multi-flag-drop", args="--tp 8 --cuda-graph-max-bs 32"),
        _v("multi-flag-keep", args="--tp 8 --cuda-graph-max-bs 128"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["multi-flag-keep"]
    assert [row["name"] for row in dropped] == ["multi-flag-drop"]
    assert "cuda_graph_max_bs=32" in dropped[0]["reason"]
    assert "CONC=64" in dropped[0]["reason"]


def test_conc_unset_defaults_to_64(_multi_node, monkeypatch):
    """CONC env var absent → the os.environ.get default of '64' applies."""
    monkeypatch.delenv("CONC", raising=False)
    grid = [
        _v("below-default", args="--cuda-graph-max-bs 32"),
        _v("at-default", args="--cuda-graph-max-bs 64"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["at-default"]
    assert [row["name"] for row in dropped] == ["below-default"]
    assert "CONC=64" in dropped[0]["reason"]
    assert "cuda_graph_max_bs=32" in dropped[0]["reason"]


def test_conc_empty_string_defaults_to_64(_multi_node, monkeypatch):
    """CONC='' → the `or 64` branch applies (empty string is falsy)."""
    monkeypatch.setenv("CONC", "")
    grid = [
        _v("below-default", args="--cuda-graph-max-bs 32"),
        _v("at-default", args="--cuda-graph-max-bs 64"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["at-default"]
    assert [row["name"] for row in dropped] == ["below-default"]
    assert "CONC=64" in dropped[0]["reason"]
    assert "cuda_graph_max_bs=32" in dropped[0]["reason"]


def test_unparseable_conc_falls_back_to_64(_multi_node, monkeypatch):
    monkeypatch.setenv("CONC", "not-an-int")
    grid = [
        _v("below-default", args="--cuda-graph-max-bs 32"),
        _v("at-default", args="--cuda-graph-max-bs 64"),
    ]
    kept, dropped = apply_multi_node_invalid_variants(grid)
    assert [v.name for v in kept] == ["at-default"]
    assert [row["name"] for row in dropped] == ["below-default"]
    assert "CONC=64" in dropped[0]["reason"]
    assert "cuda_graph_max_bs=32" in dropped[0]["reason"]


def test_none_extra_server_args_is_treated_as_empty(_multi_node, monkeypatch):
    """`v.extra_server_args or ""` must not raise or match when args is None."""
    monkeypatch.setenv("CONC", "64")
    v = GridVariant(name="none-args", extra_server_args=None)
    kept, dropped = apply_multi_node_invalid_variants([v])
    assert kept == [v]
    assert dropped == []


@pytest.fixture
def package_probe(tmp_path, monkeypatch):
    """Use real child Python and a tiny local parser package, not a serving stack."""
    state = SimpleNamespace(root=tmp_path / "packages", executable=sys.executable, calls=0, clock=1000.0)
    state.root.mkdir()
    state.trace = tmp_path / "imports.txt"
    real_run = subprocess.run
    monkeypatch.setenv("PYTHONPATH", str(state.root))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(bi, "_resolve_probe_python", lambda _framework: state.executable)
    monkeypatch.setattr(vf, "_HELP_PROBE_FAILURES", {})
    monkeypatch.setattr(vf.time, "monotonic", lambda: state.clock)

    def write_package(root, flag):
        package = root / "sglang"
        (package / "srt").mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "srt" / "__init__.py").write_text("", encoding="utf-8")
        source = package / "srt" / "server_args.py"
        source.write_text(
            "from pathlib import Path\n"
            f"with Path({str(state.trace)!r}).open('a') as trace:\n    trace.write('import\\n')\n"
            "class ServerArgs:\n"
            "    @staticmethod\n"
            "    def add_cli_args(parser):\n"
            f"        parser.add_argument({flag!r})\n",
            encoding="utf-8",
        )
        return source

    def run(cmd, **kwargs):
        state.calls += 1
        return real_run(cmd, **kwargs)

    state.source = write_package(state.root, "--before")
    state.write_package = write_package
    state.import_count = lambda: len(state.trace.read_text().splitlines()) if state.trace.exists() else 0
    monkeypatch.setattr(vf.subprocess, "run", run)
    return state


def test_missing_interpreter_failure_has_bounded_identity_scoped_cooldown(package_probe, tmp_path):
    state = package_probe
    state.executable = str(tmp_path / "missing-python")
    assert vf._probe_server_help_text("sglang") == ""
    assert vf._probe_server_help_text("sglang") == ""
    assert state.calls == 1
    state.clock += 300
    assert vf._probe_server_help_text("sglang") == ""
    assert state.calls == 2
    state.executable = sys.executable
    assert "--before" in vf._probe_server_help_text("sglang")
    assert state.calls == 3


def test_timeout_before_identity_has_bounded_cooldown(package_probe, monkeypatch):
    calls = []

    def timeout(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, timeout=30)

    monkeypatch.setattr(vf.subprocess, "run", timeout)
    assert vf._probe_server_help_text("sglang") == ""
    assert vf._probe_server_help_text("sglang") == ""
    assert len(calls) == 1
    package_probe.clock += 300
    assert vf._probe_server_help_text("sglang") == ""
    assert len(calls) == 2


def test_successful_help_cache_tracks_environment_selected_package_path(package_probe, monkeypatch, tmp_path):
    state = package_probe
    alternate = tmp_path / "alternate"
    state.write_package(alternate, "--redirected")
    (state.root / "sglang" / "__init__.py").write_text(
        "import os\nif os.environ.get('FAKE_PARSER_PATH'):\n    __path__ = [os.environ['FAKE_PARSER_PATH']]\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FAKE_PARSER_PATH", raising=False)
    assert "--before" in vf._probe_server_help_text("sglang")
    monkeypatch.setenv("FAKE_PARSER_PATH", str(alternate / "sglang"))

    assert "--redirected" in vf._probe_server_help_text("sglang")
    assert state.import_count() == 2


def test_successful_help_cache_tracks_imported_parser_dependency(package_probe):
    state = package_probe
    dependency = state.source.with_name("flags.py")
    dependency.write_text("FLAG = '--before'\n", encoding="utf-8")
    state.source.write_text(
        "from .flags import FLAG\n" + state.source.read_text().replace("'--before'", "FLAG"), encoding="utf-8"
    )
    assert "--before" in vf._probe_server_help_text("sglang")
    dependency.write_text("FLAG = '--dependency-updated'\n", encoding="utf-8")

    assert "--dependency-updated" in vf._probe_server_help_text("sglang")
    assert state.import_count() == 2


def test_help_source_identity_does_not_claim_stale_timestamp_bytecode(package_probe):
    state = package_probe
    py_compile.compile(str(state.source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    stat = state.source.stat()
    state.source.write_text(state.source.read_text().replace("--before", "--afterx"), encoding="utf-8")
    os.utime(state.source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert state.source.stat().st_size == stat.st_size
    assert "--before" in vf._probe_server_help_text("sglang")
    assert "--before" in vf._probe_server_help_text("sglang")
    assert state.import_count() == 2

    # Removing stale bytecode changes what the unmodified interpreter executes.
    Path(importlib.util.cache_from_source(str(state.source))).unlink()
    assert "--afterx" in vf._probe_server_help_text("sglang")
    assert state.import_count() == 3


def test_successful_help_observes_same_stat_source_edits(package_probe):
    state = package_probe
    assert "--before" in vf._probe_server_help_text(" SGLANG ")
    stat = state.source.stat()
    state.source.write_text(state.source.read_text().replace("--before", "--afterx"), encoding="utf-8")
    os.utime(state.source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert state.source.stat().st_size == stat.st_size

    assert "--afterx" in vf._probe_server_help_text("sglang")
    assert state.calls == state.import_count() == 2


def _env_variant(index: int) -> GridVariant:
    return GridVariant(name=f"env-{index}", extra_envs={"SGLANG_MOE_A2A_BACKEND": "deepep"})


@pytest.mark.parametrize(
    "grid",
    [
        pytest.param([], id="empty-grid"),
        pytest.param([_env_variant(0)], id="one-env-only-variant"),
        pytest.param([_env_variant(i) for i in range(32)], id="many-env-only-variants"),
        pytest.param([_v("unrelated-flag", args="--mem-fraction-static 0.9")], id="no-rule-flag"),
    ],
)
def test_no_compatibility_flag_in_batch_starts_no_subprocess(package_probe, grid):
    """Help text is only consumed by a flag rule, so a batch without one must not fork an interpreter."""
    kept, dropped = vf.apply_compatibility_filter(grid, framework="sglang", model_path="/models/deepseek-v3")

    assert kept == list(grid)
    assert dropped == []
    assert package_probe.calls == 0
    assert package_probe.import_count() == 0


def test_a_batch_that_needs_the_help_check_probes_once(package_probe):
    before = {key: value for key, value in sys.modules.items() if key == "sglang" or key.startswith("sglang.")}
    needs_help = _v("deepep", args="--enable-deepep-moe")
    grid = [*(_env_variant(i) for i in range(10)), needs_help, *(_env_variant(i) for i in range(10, 20))]

    kept, dropped = vf.apply_compatibility_filter(grid, framework="sglang", model_path="/models/deepseek-v3")

    assert package_probe.calls == package_probe.import_count() == 1
    assert [v.name for v in kept] == [v.name for v in grid if v is not needs_help]
    assert [row["name"] for row in dropped] == ["deepep"]
    assert "--help" in dropped[0]["reason"]
    after = {key: value for key, value in sys.modules.items() if key == "sglang" or key.startswith("sglang.")}
    assert after == before


def test_parser_failure_retries_source_changes_after_cooldown(package_probe):
    state = package_probe
    source = state.source.read_text()
    state.source.write_text("raise RuntimeError('fake parser failure')\n", encoding="utf-8")
    assert vf._probe_server_help_text("sglang") == ""
    state.source.write_text(source, encoding="utf-8")
    state.clock += 299
    assert vf._probe_server_help_text("sglang") == ""
    assert state.calls == 1
    state.clock += 1
    assert "--before" in vf._probe_server_help_text("sglang")
    assert state.calls == 2


@pytest.mark.parametrize("change", ["executable", "command"])
def test_failed_launch_retries_when_observable_identity_changes(package_probe, monkeypatch, tmp_path, change):
    state = package_probe
    executable = tmp_path / "not-an-executable"
    executable.write_text("invalid executable", encoding="utf-8")
    state.executable = str(executable)
    assert vf._probe_server_help_text("sglang") == ""
    assert vf._probe_server_help_text("sglang") == ""
    assert state.calls == 1
    if change == "executable":
        executable.write_text("replaced invalid executable", encoding="utf-8")
    else:
        monkeypatch.setitem(vf._HELP_PROBE_COMMANDS, "sglang", ("-c", "print('different parser')"))

    assert vf._probe_server_help_text("sglang") == ""
    assert state.calls == 2


@pytest.mark.parametrize("churn", ["env", "cwd"])
def test_failed_launch_cooldown_survives_ambient_churn(package_probe, monkeypatch, tmp_path, churn):
    """Neither of these changes what the parser prints, and a round rewrites them constantly.

    Folding them into the launch identity would expire the cooldown every round
    and re-pay a multi-second probe for a framework already known to be broken.
    """
    state = package_probe
    executable = tmp_path / "not-an-executable"
    executable.write_text("invalid executable", encoding="utf-8")
    state.executable = str(executable)
    assert vf._probe_server_help_text("sglang") == ""
    assert state.calls == 1
    if churn == "env":
        monkeypatch.setenv("FAKE_RUNTIME_ENV", "changed")
    else:
        monkeypatch.chdir(tmp_path)

    assert vf._probe_server_help_text("sglang") == ""
    assert state.calls == 1


def test_failed_stdout_and_stderr_never_become_help(package_probe, monkeypatch):
    monkeypatch.setattr(
        vf.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "--invalid", "Traceback: --invalid"),
    )

    assert vf._probe_server_help_text("sglang") == ""
    assert vf._HELP_PROBE_FAILURES["sglang"][1] == package_probe.clock + 300
