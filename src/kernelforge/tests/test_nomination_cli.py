# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The --auto invocation rules and the envelope a nominated run emits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pytest

from kernelforge import cli
from kernelforge import nomination as nom


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _request_file(tmp_path: Path, *, max_kernels: int = 1, rows: list[dict[str, Any]] | None = None) -> Path:
    trace = _write(tmp_path / "decode.trace.json", {})
    candidates = _write(
        tmp_path / "kernel_candidates.json",
        {
            "manifest_version": nom.MANIFEST_VERSION,
            "hot_kernels": rows if rows is not None else [{"kernel_name": "hot", "source_file": "/repo/hot.py"}],
        },
    )
    return _write(
        tmp_path / "nomination.json",
        {
            "protocol_version": nom.PROTOCOL_VERSION,
            "lane": nom.LANE_REWRITE,
            "trace_path": str(trace),
            "candidates_path": str(candidates),
            "lane_budget_sec": 6000,
            "max_kernels": max_kernels,
        },
    )


def test_without_auto_the_nomination_path_is_not_taken() -> None:
    """A run that does not ask for nomination must behave exactly as before."""
    assert cli._resolve_nomination(auto=False, nomination_input="", kernel="/repo/k.py", resume=False) is None


def test_nomination_input_without_auto_is_refused(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException, match="requires --auto"):
        cli._resolve_nomination(auto=False, nomination_input=str(tmp_path / "x.json"), kernel=None, resume=False)


def test_auto_without_input_is_refused() -> None:
    with pytest.raises(click.ClickException, match="requires --nomination-input"):
        cli._resolve_nomination(auto=True, nomination_input="", kernel=None, resume=False)


def test_auto_with_explicit_kernel_is_refused(tmp_path: Path) -> None:
    """Two sources for the same decision is a caller bug, not a preference."""
    path = _request_file(tmp_path)
    with pytest.raises(click.ClickException, match="do not pass --kernel"):
        cli._resolve_nomination(auto=True, nomination_input=str(path), kernel="/repo/k.py", resume=False)


def test_auto_with_resume_is_refused(tmp_path: Path) -> None:
    path = _request_file(tmp_path)
    with pytest.raises(click.ClickException, match="cannot be combined with --resume"):
        cli._resolve_nomination(auto=True, nomination_input=str(path), kernel=None, resume=True)


def test_auto_resolves_a_single_target(tmp_path: Path) -> None:
    resolution = cli._resolve_nomination(
        auto=True, nomination_input=str(_request_file(tmp_path)), kernel=None, resume=False
    )
    assert resolution is not None
    assert [target.kernel_name for target in resolution.targets] == ["hot"]
    assert resolution.summary.to_dict() == {"candidates_seen": 1, "resolved": 1, "selected": 1}


def test_multi_target_nomination_is_refused_loudly(tmp_path: Path) -> None:
    """Executing several targets needs per-target base commits, which is pending."""
    path = _request_file(
        tmp_path,
        max_kernels=2,
        rows=[
            {"kernel_name": "a", "source_file": "/repo/a.py", "gpu_pct": 9.0},
            {"kernel_name": "b", "source_file": "/repo/b.py", "gpu_pct": 8.0},
        ],
    )
    with pytest.raises(click.ClickException, match="multi-target execution is not implemented"):
        cli._resolve_nomination(auto=True, nomination_input=str(path), kernel=None, resume=False)


def test_malformed_request_becomes_a_cli_error(tmp_path: Path) -> None:
    path = tmp_path / "nomination.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(click.ClickException, match="could not read nomination request"):
        cli._resolve_nomination(auto=True, nomination_input=str(path), kernel=None, resume=False)


def test_empty_nomination_emits_a_clean_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Nothing eligible is an answer; the phase latch needs a clean exit."""
    path = _request_file(tmp_path, rows=[{"kernel_name": "blind", "source_file": ""}])
    resolution = cli._resolve_nomination(auto=True, nomination_input=str(path), kernel=None, resume=False)
    assert resolution is not None and not resolution.targets
    cli._emit_empty_nomination(resolution)
    emitted = capsys.readouterr().out
    payload = json.loads(emitted.split("__FORGE_RESULT__")[1])
    assert payload["patches"] == []
    assert payload["improved"] is False
    assert payload["nomination"] == {"candidates_seen": 1, "resolved": 0, "selected": 0}


def _resolution(tmp_path: Path) -> Any:
    return cli._resolve_nomination(auto=True, nomination_input=str(_request_file(tmp_path)), kernel=None, resume=False)


def _publish(tmp_path: Path, *, with_snapshot: bool) -> Path:
    """Lay out a published best bundle the way ``BestResultPublisher`` does."""
    campaign = tmp_path / "forge_experiments"
    published = campaign / "best" / "iter_007"
    published.mkdir(parents=True)
    (published / "forge.patch").write_text("diff", encoding="utf-8")
    if with_snapshot:
        (published / "files" / "repo").mkdir(parents=True)
        (published / "files" / "repo" / "hot.py").write_text("patched", encoding="utf-8")
    _write(
        campaign / "best" / "manifest.json",
        {"patch_path": "best/iter_007/forge.patch", "artifact_dir": "best/iter_007"},
    )
    return campaign


def test_patches_read_the_published_path_from_the_manifest(tmp_path: Path) -> None:
    """The published directory is versioned, so the path cannot be assumed."""
    resolution = _resolution(tmp_path)
    campaign = _publish(tmp_path, with_snapshot=True)
    entries = cli._nominated_patches(
        resolution,
        campaign_root=campaign,
        workspace_dir=str(tmp_path / "repo"),
        best_commit="abc123",
        micro_speedup=1.4,
    )
    assert entries == [
        {
            "kernel_name": "hot",
            "patch_path": str(campaign / "best" / "iter_007" / "forge.patch"),
            "target_file": "/repo/hot.py",
            "micro_speedup": 1.4,
            "base_commit": "abc123",
            "kernel_repo": str(tmp_path / "repo"),
            "snapshot_dir": str(campaign / "best" / "iter_007" / "files"),
        }
    ]


def test_a_bundle_without_a_snapshot_reports_no_snapshot_dir(tmp_path: Path) -> None:
    """A patch touching nothing publishable has no ``files/`` tree, and naming
    one that does not exist would fail the atomic apply it is meant to enable."""
    campaign = _publish(tmp_path, with_snapshot=False)
    entries = cli._nominated_patches(
        _resolution(tmp_path),
        campaign_root=campaign,
        workspace_dir=str(tmp_path / "repo"),
        best_commit="abc123",
        micro_speedup=1.4,
    )
    assert entries == [
        {
            "kernel_name": "hot",
            "patch_path": str(campaign / "best" / "iter_007" / "forge.patch"),
            "target_file": "/repo/hot.py",
            "micro_speedup": 1.4,
            "base_commit": "abc123",
            "kernel_repo": str(tmp_path / "repo"),
        }
    ]


def test_hyperloom_reads_back_the_repo_and_snapshot_forge_emitted(tmp_path: Path) -> None:
    """The two fields exist for the consumer's atomic multi-file apply, so the
    envelope is proved through the consumer that reads them."""
    from hyperloom.orchestrator.kernel.nomination_result import parse_outcome

    campaign = _publish(tmp_path, with_snapshot=True)
    entries = cli._nominated_patches(
        _resolution(tmp_path),
        campaign_root=campaign,
        workspace_dir=str(tmp_path / "repo"),
        best_commit="abc123",
        micro_speedup=1.4,
    )
    patch = parse_outcome({"patches": entries}).patches[0]
    assert patch.kernel_repo == str(tmp_path / "repo")
    assert patch.snapshot_dir == str(campaign / "best" / "iter_007" / "files")


def test_no_best_commit_yields_no_patches(tmp_path: Path) -> None:
    entries = cli._nominated_patches(
        _resolution(tmp_path),
        campaign_root=tmp_path,
        workspace_dir=str(tmp_path / "repo"),
        best_commit="",
        micro_speedup=1.4,
    )
    assert entries == []


def test_absent_manifest_yields_no_patches(tmp_path: Path) -> None:
    entries = cli._nominated_patches(
        _resolution(tmp_path),
        campaign_root=tmp_path,
        workspace_dir=str(tmp_path / "repo"),
        best_commit="abc123",
        micro_speedup=1.4,
    )
    assert entries == []


def test_manifest_without_patch_path_yields_no_patches(tmp_path: Path) -> None:
    campaign = tmp_path / "forge_experiments"
    (campaign / "best").mkdir(parents=True)
    _write(campaign / "best" / "manifest.json", {"artifact_dir": "best/iter_001"})
    entries = cli._nominated_patches(
        _resolution(tmp_path),
        campaign_root=campaign,
        workspace_dir=str(tmp_path / "repo"),
        best_commit="abc123",
        micro_speedup=1.4,
    )
    assert entries == []


def test_manifest_pointing_at_a_missing_patch_yields_no_patches(tmp_path: Path) -> None:
    campaign = tmp_path / "forge_experiments"
    (campaign / "best").mkdir(parents=True)
    _write(campaign / "best" / "manifest.json", {"patch_path": "best/iter_001/forge.patch"})
    entries = cli._nominated_patches(
        _resolution(tmp_path),
        campaign_root=campaign,
        workspace_dir=str(tmp_path / "repo"),
        best_commit="abc123",
        micro_speedup=1.4,
    )
    assert entries == []


def test_forge_loop_declares_both_new_options() -> None:
    names = {param.name for param in cli.forge_loop.params}
    assert {"auto", "nomination_input"} <= names
