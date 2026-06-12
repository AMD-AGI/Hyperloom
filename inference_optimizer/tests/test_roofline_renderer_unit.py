# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the roofline-comparison breakdown renderer."""

from __future__ import annotations

from inference_optimizer.breakdown.reporters._renderers import roofline as rl


# ---- _snapshot_kv ----

def test_snapshot_kv_empty():
    assert rl._snapshot_kv("Baseline", None) == ""
    assert rl._snapshot_kv("Baseline", {}) == ""


def test_snapshot_kv_full():
    snap = {
        "snapshot_id": "s1", "ts": "t0", "compute_pct": 70,
        "idle_pct": 10, "comm_pct": 5, "top_bottleneck": "compute",
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

def test_render_absent():
    assert rl.render({}).skipped is True


def test_render_empty_list():
    assert rl.render({"roofline": []}).skipped is True


def test_render_non_list():
    assert rl.render({"roofline": "bad"}).skipped is True


def test_render_full_entry():
    bd = {
        "roofline": [
            {
                "source_path": "/x/final.json",
                "mode": "vs_baseline",
                "baseline": {
                    "snapshot_id": "b", "top_kernel": {
                        "name": "attn", "gpu_pct": 50, "efficiency_pct": 60, "bound_type": "memory",
                    },
                },
                "latest": {"snapshot_id": "l"},
                "delta": {"compute_pct": 3},
            },
        ]
    }
    sec = rl.render(bd)
    assert sec.skipped is False
    assert "final.json files surfaced: 1" in sec.key_facts[0]
    assert "attn" in " ".join(sec.key_facts)
    assert "Roofline #1" in sec.markdown_block


def test_render_skips_non_dict_entries():
    sec = rl.render({"roofline": ["bad", 1]})
    assert sec.skipped is False
    assert "surfaced: 2" in sec.key_facts[0]
