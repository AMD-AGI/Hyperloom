"""D2 — multi-source server-log tailing tests."""

from __future__ import annotations

import pytest

from robustness_agent.sources.local_probe import (
    LocalProbeConfig,
    LocalProbeSource,
    _tail_logs,
)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_tail_logs_returns_primary_when_no_extras(tmp_path):
    p = tmp_path / "server.log"
    _write(p, "line-1\nline-2\nline-3\n")
    out = _tail_logs(p, None, (), 0, max_lines=10)
    assert out == ["line-1", "line-2", "line-3"]


def test_tail_logs_picks_up_runs_glob(tmp_path):
    primary = tmp_path / "primary.log"
    _write(primary, "PRIMARY-A\n")
    # Two grid variants under runs/.
    _write(tmp_path / "runs" / "backends" / "t1" / "server.log",
           "VARIANT-1A\nVARIANT-1B\n")
    _write(tmp_path / "runs" / "params" / "t2" / "server.log",
           "VARIANT-2A\n")
    out = _tail_logs(
        primary,
        tmp_path,
        ("runs/*/*/server.log",),
        5,
        max_lines=20,
    )
    # Primary lines first, then per-file tagged extras.
    assert "PRIMARY-A" in out
    # Each variant line is prefixed with ``[server.log]``.
    assert any("[server.log] VARIANT-1A" in line for line in out)
    assert any("[server.log] VARIANT-2A" in line for line in out)


def test_tail_logs_dedup_when_primary_matches_glob(tmp_path):
    primary = tmp_path / "runs" / "backends" / "t1" / "server.log"
    _write(primary, "ONLY-LINE\n")
    out = _tail_logs(
        primary, tmp_path, ("runs/*/*/server.log",), 5, max_lines=10,
    )
    # ``primary`` already covers the file; should not appear twice.
    matches = [line for line in out if "ONLY-LINE" in line]
    assert len(matches) == 1


def test_tail_logs_cap_extras_by_mtime(tmp_path):
    primary = None
    # 5 variants — only the 3 most-recent should be tailed.
    for i in range(5):
        path = tmp_path / "runs" / "backends" / f"t{i}" / "server.log"
        _write(path, f"variant-{i}\n")
        # Force ordered mtimes.
        import os
        os.utime(path, (1000.0 + i, 1000.0 + i))
    out = _tail_logs(
        primary, tmp_path, ("runs/*/*/server.log",), 3, max_lines=10,
    )
    body = "\n".join(out)
    # Most recent 3 (t2/t3/t4) should be present; older (t0/t1) absent.
    assert "variant-4" in body
    assert "variant-3" in body
    assert "variant-2" in body
    assert "variant-0" not in body
    assert "variant-1" not in body


def test_tail_logs_empty_when_no_max_lines(tmp_path):
    _write(tmp_path / "server.log", "x\n")
    assert _tail_logs(tmp_path / "server.log", None, (), 0, max_lines=0) == []


@pytest.mark.asyncio
async def test_local_probe_picks_up_grid_variant_logs(tmp_path):
    """End-to-end: LocalProbe sees a grid variant log under runs/."""
    _write(tmp_path / "runs" / "backends" / "t1" / "server.log",
           "CUDA out of memory at allocator.cc:42\n")
    cfg = LocalProbeConfig(
        session_dir=tmp_path,
        disk_mountpoints=(),
        process_patterns=(),
        ray_probe_enabled=False, fd_probe_enabled=False,
        decision_audit_enabled=False, preflight_enabled=False,
        critic_health_enabled=False,
    )
    data = await LocalProbeSource(cfg).fetch(ctx=None)
    # The OOM pattern should have surfaced from the grid variant log.
    assert any(
        h.get("pattern") == r"CUDA out of memory" for h in data.local_log_errors
    )
