# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the ``run_grid`` per-variant failure branches (yaml build error, magpie timeout, server-dead / overtime sentinels, missing workspace, invalid measurement)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    run_grid,
)


@pytest.fixture(autouse=True)
def _single_node(monkeypatch):
    """Default every test in this module to single-node mode."""
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: False)


def _write_base_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "benchmark_script": "sglang_mi300x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


@pytest.mark.asyncio
async def test_run_grid_yaml_build_error_branch(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _boom(*_a, **_k):
        raise ValueError("bad yaml render")

    monkeypatch.setattr(gr, "_build_variant_yaml", _boom)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
    )
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].error_class == "yaml_build_error"


@pytest.mark.asyncio
async def test_run_grid_magpie_timeout_branch(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="magpie", timeout=5)

    monkeypatch.setattr(gr, "_run_magpie", _timeout)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
    )
    assert results[0].status == "failed"
    assert results[0].error_class == "magpie_timeout"


@pytest.mark.asyncio
async def test_run_grid_server_dead_branch(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _dead(*_a, **_k):
        return gr.SERVER_DEAD_RETURNCODE, "", "engine crashed"

    monkeypatch.setattr(gr, "_run_magpie", _dead)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
    )
    assert results[0].status == "failed"
    assert results[0].error_class == "server_init_dead"
    assert results[0].returncode == gr.SERVER_DEAD_RETURNCODE


@pytest.mark.asyncio
async def test_run_grid_no_workspace_branch_stops_on_failure(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _no_ws(*_a, **_k):
        return 1, "stdout", "boom stderr"  # nonzero, no benchmark_* dir

    monkeypatch.setattr(gr, "_run_magpie", _no_ws)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA"), GridVariant("vB")],
        output_root=tmp_path / "out",
        keep_going_on_failure=False,
    )
    assert len(results) == 1
    assert results[0].error_class == "no_benchmark_workspace"


@pytest.mark.asyncio
async def test_agentx_preflight_abort_keeps_its_own_error_class(tmp_path, monkeypatch):
    """An AgentX preflight abort must not be filed as a missing workspace."""
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        AGENTX_PREFLIGHT_ERROR_CLASS,
        AGENTX_PREFLIGHT_RETURNCODE,
    )

    base = tmp_path / "base.yaml"
    _write_base_yaml(base)
    diagnosis = "AgentX preflight failed: HYPERLOOM_AGENTX is on but aiperf was not found."

    monkeypatch.setattr(gr, "_run_magpie", lambda *_a, **_k: (AGENTX_PREFLIGHT_RETURNCODE, "", diagnosis))
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
        keep_going_on_failure=False,
    )
    assert len(results) == 1
    assert results[0].error_class == AGENTX_PREFLIGHT_ERROR_CLASS
    # The diagnosis itself has to survive too: it names the fix.
    assert "aiperf was not found" in (results[0].error or "")


@pytest.mark.asyncio
async def test_agentx_preflight_abort_abandons_the_rest_of_the_grid(tmp_path, monkeypatch):
    """The client is missing for the whole grid, not for one variant."""
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        AGENTX_PREFLIGHT_RETURNCODE,
    )

    base = tmp_path / "base.yaml"
    _write_base_yaml(base)
    monkeypatch.setattr(gr, "_run_magpie", lambda *_a, **_k: (AGENTX_PREFLIGHT_RETURNCODE, "", "aiperf was not found"))
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA"), GridVariant("vB"), GridVariant("vC")],
        output_root=tmp_path / "out",
        keep_going_on_failure=True,  # would otherwise walk every point
    )
    assert [r.status for r in results] == ["failed", "skipped", "skipped"], (
        "the grid either kept benchmarking after an environment abort, or dropped "
        "the abandoned points instead of recording why they never ran"
    )
    assert [r.name for r in results[1:]] == ["vB", "vC"]
    assert all(r.error_class == "agentx_preflight" for r in results)


@pytest.mark.asyncio
async def test_agentx_preflight_abort_never_reports_an_empty_error(tmp_path, monkeypatch):
    """A blank stderr must not become a blank `error`, the way the sibling branch's non-empty fallback already prevents."""
    from hyperloom.orchestrator.actions.executors._subprocess_kill import (
        AGENTX_PREFLIGHT_RETURNCODE,
    )

    base = tmp_path / "base.yaml"
    _write_base_yaml(base)
    monkeypatch.setattr(gr, "_run_magpie", lambda *_a, **_k: (AGENTX_PREFLIGHT_RETURNCODE, "", "   "))
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
        keep_going_on_failure=False,
    )
    assert (results[0].error or "").strip(), "an empty diagnosis reached the result"


@pytest.mark.asyncio
async def test_run_grid_invalid_measurement_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "REPORT_SETTLE_SECONDS", 0.0)
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _empty_report(magpie_python, config_path, output_dir, **_k):
        ws = Path(output_dir) / "benchmark_sglang_20260101_000000"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "benchmark_report.json").write_text(
            '{"success": false, "framework": "sglang"}',
        )
        return 0, "ok", ""

    monkeypatch.setattr(gr, "_run_magpie", _empty_report)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
    )
    assert results[0].status == "failed"
    assert results[0].error_class in {
        "benchmark_report_invalid_metric",
        "benchmark_report_missing",
    }


@pytest.mark.asyncio
async def test_run_grid_nonzero_rc_with_valid_measurement_fails(tmp_path, monkeypatch):
    """A parseable measurement must not launder a non-zero exit code into success."""
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _nonzero_but_valid(magpie_python, config_path, output_dir, **_k):
        ws = Path(output_dir) / "benchmark_sglang_20260101_000000"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "benchmark_report.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "framework": "sglang",
                    "throughput": {
                        "output_throughput": 1200.0,
                        "request_throughput": 120.0,
                        "completed_requests": 640,
                        "duration_seconds": 120.0,
                    },
                }
            )
        )
        return 1, "stdout tail", "server exited 1"

    monkeypatch.setattr(gr, "_run_magpie", _nonzero_but_valid)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA")],
        output_root=tmp_path / "out",
    )
    r = results[0]
    assert r.status == "failed"
    assert r.error_class == "magpie_nonzero_after_valid_measurement"
    assert r.returncode == 1
    markers = list((tmp_path / "out").rglob("abort_reason.json"))
    assert len(markers) == 1
    assert json.loads(markers[0].read_text())["error_class"] == "magpie_nonzero_after_valid_measurement"


def _valid_report_body(completed: int, requested: int) -> str:
    """A parseable Magpie report for ``completed`` of ``requested`` requests."""
    return json.dumps(
        {
            "success": True,
            "framework": "sglang",
            "throughput": {
                "output_throughput": 1200.0,
                "request_throughput": 120.0,
                "completed_requests": completed,
                "num_prompts": requested,
                "duration_seconds": 120.0,
            },
        }
    )


def _magpie_writing(completed: int, requested: int, rc: int, stderr: str):
    """A fake ``_run_magpie`` that writes one report and exits ``rc``."""

    def _run(magpie_python, config_path, output_dir, **_k):
        ws = Path(output_dir) / "benchmark_sglang_20260101_000000"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "benchmark_report.json").write_text(_valid_report_body(completed, requested))
        return rc, "", stderr

    return _run


_STALE_HANDLE = "benchmarks/sglang_mi355x.sh: error reading input file: Stale file handle"


@pytest.mark.asyncio
async def test_run_grid_keeps_a_measurement_that_served_every_request(tmp_path, monkeypatch):
    """A wrapper that exits non-zero after serving the whole protocol keeps its measurement.

    The bash wrapper reads its own script off the InferenceX checkout for the
    whole round, so a mount flap at the tail exits non-zero on a benchmark that
    already ran to completion. Nothing about that measurement is short.

    The variant carries no ``NUM_PROMPTS`` of its own, which is the shape every
    caller but conc_sweep uses: the count is read back off the run's own
    result, not off the variant's env layer.
    """
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)
    monkeypatch.setattr(gr, "_run_magpie", _magpie_writing(192, 192, 2, _STALE_HANDLE))

    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA", extra_server_args="--foo")],
        output_root=tmp_path / "out",
    )
    r = results[0]
    assert r.status == "succeeded"
    assert r.error_class == ""
    assert r.output_throughput == pytest.approx(1200.0)
    assert r.completed_requests == 192
    assert r.returncode == 2
    assert "nonzero_rc_after_complete_protocol:2" in r.nonfatal_warnings
    # The cause of the non-zero exit stays on the record, not only in the log.
    assert "Stale file handle" in (r.error or "")
    assert list((tmp_path / "out").rglob("abort_reason.json")) == []


@pytest.mark.asyncio
async def test_run_grid_fails_a_measurement_that_served_short(tmp_path, monkeypatch):
    """A server that died mid-protocol still fails, however parseable its report.

    This is the case the nonzero-rc branch exists for: fewer requests were
    served than were asked for, so the throughput is not the protocol's.
    """
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)
    monkeypatch.setattr(gr, "_run_magpie", _magpie_writing(120, 192, 1, "server exited 1"))

    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA", extra_server_args="--foo")],
        output_root=tmp_path / "out",
    )
    r = results[0]
    assert r.status == "failed"
    assert r.error_class == "magpie_nonzero_after_valid_measurement"
    markers = list((tmp_path / "out").rglob("abort_reason.json"))
    assert len(markers) == 1


@pytest.mark.asyncio
async def test_run_grid_fails_when_the_run_recorded_no_request_count(tmp_path, monkeypatch):
    """A report with no ``num_prompts`` cannot be judged complete, so rc wins.

    An unpatched InferenceX checkout serves ``max_concurrency`` prompts and may
    record no requested count at all. That is not evidence of a whole protocol.
    """
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)

    def _no_request_count(magpie_python, config_path, output_dir, **_k):
        ws = Path(output_dir) / "benchmark_sglang_20260101_000000"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "benchmark_report.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "framework": "sglang",
                    "throughput": {
                        "output_throughput": 1200.0,
                        "completed_requests": 192,
                        "duration_seconds": 120.0,
                    },
                }
            )
        )
        return 2, "", _STALE_HANDLE

    monkeypatch.setattr(gr, "_run_magpie", _no_request_count)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("vA", extra_server_args="--foo")],
        output_root=tmp_path / "out",
    )
    assert results[0].status == "failed"
    assert results[0].error_class == "magpie_nonzero_after_valid_measurement"


@pytest.mark.asyncio
async def test_server_dead_surfaces_log_excerpt(tmp_path, monkeypatch):
    """server_log_death_excerpt is used when a seeded server.log exists."""
    base = tmp_path / "base.yaml"
    _write_base_yaml(base)
    out_root = tmp_path / "out"

    def _dead(magpie_python, config_path, output_dir, **_k):
        slog = Path(output_dir) / "server.log"
        slog.parent.mkdir(parents=True, exist_ok=True)
        slog.write_text(
            "Worker init started\n"
            "mla_gluon[bh16bn128] requires batch_size=1, got 512\n"
            "Engine core initialization failed.\n"
            "Traceback follows\n"
        )
        return gr.SERVER_DEAD_RETURNCODE, "", ""

    monkeypatch.setattr(gr, "_run_magpie", _dead)
    results = await run_grid(
        base_yaml_path=base,
        base_extra_args="",
        grid=[GridVariant("fp8_kv")],
        output_root=out_root,
    )
    r = results[0]
    assert r.status == "failed"
    assert r.error_class == "server_init_dead"
    assert "mla_gluon" in (r.error or ""), "excerpt should mention mla_gluon"
    assert r.server_log_path is not None
    assert r.server_log_path.endswith("server.log")
