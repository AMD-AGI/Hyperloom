# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the roofline-comparison breakdown renderer."""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.reporters._renderers import roofline as rl


# ---- _snapshot_kv ----


def test_snapshot_kv_empty():
    assert rl._snapshot_kv("Baseline", None) == ""
    assert rl._snapshot_kv("Baseline", {}) == ""


def test_snapshot_kv_full():
    snap = {
        "snapshot_id": "s1",
        "ts": "t0",
        "compute_pct": 70,
        "idle_pct": 10,
        "comm_pct": 5,
        "top_bottleneck": "compute",
        "top_kernel": {"name": "gemm", "gpu_pct": 40, "efficiency_pct": 80, "bound_type": "compute"},
    }
    out = rl._snapshot_kv("Baseline", snap)
    assert "**Baseline**" in out
    assert "gemm" in out


def test_snapshot_kv_non_dict_top_kernel():
    out = rl._snapshot_kv("Latest", {"snapshot_id": "s", "top_kernel": "bad"})
    assert "**Latest**" in out


# ---- _delta_block ----


def test_delta_block_empty():
    assert rl._delta_block(None) == ""
    assert rl._delta_block({}) == ""


def test_delta_block_table():
    out = rl._delta_block({"compute_pct": 5, "idle_pct": -2})
    assert "**Delta**" in out
    assert "compute_pct" in out


# ---- render ----


def _event(*snapshots: dict | None) -> dict:
    """A ``roofline`` timeline event holding one action per snapshot."""
    return {
        "type": "roofline",
        "ext": {"actions": [{"outcome": {"snapshot": snap}} for snap in snapshots]},
    }


def test_render_absent():
    assert rl.render({}).skipped is True


def test_render_skips_a_session_whose_roofline_never_ran():
    assert rl.render({"timeline": []}).skipped is True


def test_render_skips_a_run_that_recorded_no_snapshot():
    """A failed roofline action closes with no conclusion to compare."""
    assert rl.render({"timeline": [_event(None)]}).skipped is True


def test_render_reports_the_baseline_top_kernel():
    bd = {
        "timeline": [
            _event(
                {
                    "snapshot_id": 1,
                    "compute_pct": 60,
                    "top_kernel": {
                        "name": "attn",
                        "gpu_pct": 50,
                        "efficiency_pct": 60,
                        "bound_type": "memory",
                    },
                }
            )
        ]
    }
    sec = rl.render(bd)

    assert sec.skipped is False
    assert "Roofline snapshots recorded: 1" in sec.key_facts[0]
    assert "attn" in " ".join(sec.key_facts)
    assert "**Baseline**" in sec.markdown_block


def test_render_compares_the_first_snapshot_against_the_last():
    """Two snapshots are a before/after, and the delta is what moved."""
    bd = {
        "timeline": [
            _event({"snapshot_id": 1, "compute_pct": 60.0, "idle_pct": 30.0, "comm_pct": 10.0}),
            _event({"snapshot_id": 2, "compute_pct": 72.0, "idle_pct": 18.0, "comm_pct": 10.0}),
        ]
    }
    sec = rl.render(bd)

    assert sec.skipped is False
    assert "Roofline snapshots recorded: 2" in sec.key_facts[0]
    assert "before_after" in sec.key_facts[0]
    assert "**Delta**" in sec.markdown_block


def test_render_gathers_snapshots_from_every_action_on_one_event():
    """A phase can dispatch roofline twice in a cycle; both land on one event."""
    bd = {"timeline": [_event({"snapshot_id": 1}, {"snapshot_id": 2})]}

    assert "recorded: 2" in rl.render(bd).key_facts[0]
