# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Behavior-lock tests for ``run_grid``: variant-boundary progress, ``keep_going_on_failure`` asymmetry, and
auto-warmup teardown timing.
"""

from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _grid_runner as gr
from hyperloom.orchestrator.actions.executors import _multi_node_env as mne
from hyperloom.orchestrator.actions.executors._grid_runner import (
    GridVariant,
    run_grid,
    GridLaunch,
    GridRecipe,
)
from hyperloom.inference_optimizer.trace.task_progress import progress_scope

from hyperloom.orchestrator.actions.executors._workload_envs import BenchmarkTarget


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root_behavior_lock")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))
    # Multi-node client warmup is orthogonal noise; keep it off so the mocked ``run_with_session_kill`` call counts
    # are unambiguous.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BENCH_WARMUP", "0")


@pytest.fixture(autouse=True)
def _single_node_default(monkeypatch):
    monkeypatch.setattr(mne, "is_multi_node", lambda: False)


# Variant-boundary progress matrix


# keep_going_on_failure asymmetry


# aborting a variant mid-grid


# auto-warmup teardown timing


# the server inference_suite records in a variant's slot


@pytest.fixture
def client_servers():
    """Stand-ins for the server recorded in a variant's slot; whatever the run leaves of them dies at teardown."""
    servers: list[subprocess.Popen] = []
    yield servers
    for server in servers:
        if server.poll() is None:
            server.kill()
        server.wait(timeout=5)


# _resolve_mn_effective_server_args: prefer the materialized variant YAML; fall back to recomposing from the base YAML
# when the variant read fails.
class TestResolveMnEffectiveServerArgs:
    def test_reads_the_materialized_variant_yaml(self, tmp_path):
        cfg = tmp_path / "variant.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {"benchmark": {"framework": "sglang", "envs": {"EXTRA_SGLANG_ARGS": "--tp 8 --chunked-prefill 4096"}}}
            ),
            encoding="utf-8",
        )

        assert gr._resolve_mn_effective_server_args(cfg) == "--tp 8 --chunked-prefill 4096"

    def test_variant_env_absent_returns_empty(self, tmp_path):
        cfg = tmp_path / "variant.yaml"
        cfg.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {}}}), encoding="utf-8")

        assert gr._resolve_mn_effective_server_args(cfg) == ""


# Per-variant progress heartbeat


class TestVariantHeartbeat:
    """A grid that runs for hours must be distinguishable from one that hung."""

    def _run_capture_progress(self, run_side_effect, base, out, *, grid_n=2, notes=None, sink=None):
        notes = [] if notes is None else notes

        async def _collect(**note):
            notes.append(note)

        with (
            progress_scope(sink or _collect),
            patch(
                "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
                side_effect=run_side_effect,
            ),
        ):
            results = asyncio.run(
                run_grid(
                    grid=[GridVariant(name=f"c{i}") for i in range(grid_n)],
                    output_root=out,
                    recipe=GridRecipe(base_yaml_path=base, target=BenchmarkTarget(gpu_type="mi300x")),
                    launch=GridLaunch(),
                )
            )
        return results, notes

    def test_the_note_names_the_variant_that_ran_not_the_last_row(self):
        """The tail of ``results`` is not always the variant that just reported."""
        grid = [GridVariant(name=f"c{i}") for i in range(3)]
        stopped = gr.VariantResult(
            name="c0",
            extra_server_args="",
            extra_envs={},
            status="skipped",
            error_class="orchestrator_cancelled",
        )
        never_ran = [
            gr.VariantResult(
                name=variant.name,
                extra_server_args="",
                extra_envs={},
                status="not_run",
            )
            for variant in grid[1:]
        ]

        assert gr._variant_progress_note(grid, [stopped, *never_ran], 0) == {
            "unit": "variant",
            "label": "c0",
            "index": 1,
            "total": 3,
            "status": "skipped",
            "output_throughput": None,
        }
