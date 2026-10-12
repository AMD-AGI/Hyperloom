# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the enablement round artifact snapshot."""

from __future__ import annotations

import difflib
import json
import os
import subprocess
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown.round_archive import (
    ROLE_ARTIFACT_PREIMAGE,
    ROLE_ARTIFACT_SOURCE,
    ROLE_LAUNCH_CONFIG,
    ROLE_PATCH,
    ROLE_PATCH_EVIDENCE,
    ROLE_PROMPT,
    ROLE_SERVER_LOG,
    ROLE_SPECIALIST_RESULT,
)
from hyperloom.orchestrator.enablement.artifacts import (
    _FILE_SIZE_LIMIT,
    _LOG_TRUNCATION_NOTE,
    _SERVER_LOG_TAIL_LIMIT,
    snapshot_round,
    write_setting_script,
)
from hyperloom.orchestrator.specialists.patch_safety import vet_patches
from hyperloom.orchestrator.specialists.subprocess_ import SpecialistSubprocessDispatcher
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound


def _res(**kw):
    base = {
        "status": "kept",
        "specialist_task_id": "abc123",
        "patches_applied": [],
        "extra_envs_applied": {},
        "extra_server_args_applied": "",
        "setup_commands_applied": [],
        "after_signature": {},
        "enablement_accepted_config_path": "",
        "enablement_effective_config": {},
        "enablement_launch_log": "",
    }
    base.update(kw)
    return base


def test_applied_patch_is_copied(tmp_path):
    src = tmp_path / "runs" / "specialist" / "abc123" / "patches"
    src.mkdir(parents=True)
    (src / "001_fix.patch").write_text("diff --git a/f b/f\n", encoding="utf-8")
    archive = snapshot_round(tmp_path, _res(patches_applied=[str(src / "001_fix.patch")]))
    dest = tmp_path / "reports" / "enablement" / "abc123" / "patches" / "001_fix.patch"
    assert dest.read_text() == "diff --git a/f b/f\n"
    assert archive.to_list() == [{"path": "reports/enablement/abc123/patches/001_fix.patch", "role": ROLE_PATCH}]


def test_every_patch_of_a_round_is_reported(tmp_path):
    """A round applies any number of patches, and every one of them is reported."""
    src = tmp_path / "runs" / "specialist" / "abc123" / "patches"
    src.mkdir(parents=True)
    for name in ("001_a.patch", "002_b.patch", "003_c.patch"):
        (src / name).write_text("diff\n", encoding="utf-8")
    archive = snapshot_round(tmp_path, _res(patches_applied=[str(p) for p in sorted(src.iterdir())]))
    assert archive.paths_for(ROLE_PATCH) == tuple(
        f"reports/enablement/abc123/patches/{n}" for n in ("001_a.patch", "002_b.patch", "003_c.patch")
    )


def test_unapplied_workspace_patch_is_preserved_as_evidence(tmp_path):
    """An unapplied attempt explains what was tried, not what passed integration."""
    src = tmp_path / "runs" / "specialist" / "abc123" / "worktree" / "patches"
    src.mkdir(parents=True)
    evidence = b"diff\r\n"
    (src / "002_try.diff").write_bytes(evidence)
    archive = snapshot_round(tmp_path, _res())
    assert archive.to_list() == [
        {"path": "reports/enablement/abc123/attempted_patches/002_try.diff", "role": ROLE_PATCH_EVIDENCE}
    ]
    assert (tmp_path / archive.path_for(ROLE_PATCH_EVIDENCE)).read_bytes() == evidence
    assert archive.paths_for(ROLE_PATCH) == ()


def test_specialist_result_and_prompt_are_copied(tmp_path):
    ws = tmp_path / "runs" / "specialist" / "abc123"
    ws.mkdir(parents=True)
    (ws / "specialist_done.json").write_text('{"summary": "ok"}', encoding="utf-8")
    (ws / "prompt.md").write_text("# prompt", encoding="utf-8")
    archive = snapshot_round(tmp_path, _res())
    out = tmp_path / "reports" / "enablement" / "abc123"
    assert (out / "specialist_done.json").is_file()
    assert (out / "prompt.md").is_file()
    assert archive.path_for(ROLE_SPECIALIST_RESULT) == "reports/enablement/abc123/specialist_done.json"
    assert archive.path_for(ROLE_PROMPT) == "reports/enablement/abc123/prompt.md"


def _launch_config(tmp_path, body=b"tp: 8\n"):
    """The materialized config a bench launched from, where integrate_patch leaves it."""
    cfg = tmp_path / "runs" / "integrate_patch" / "t1" / "integrate_patch.with_envs.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_bytes(body)
    return cfg


def test_launch_config_is_copied(tmp_path):
    cfg = _launch_config(tmp_path)
    snapshot_round(tmp_path, _res(enablement_accepted_config_path=str(cfg)))
    assert (tmp_path / "reports" / "enablement" / "abc123" / "launch_config.yaml").is_file()


def test_recorded_config_path_is_the_archived_copy(tmp_path):
    """The runs/ original is dropped by the collector, so only the copy resolves."""
    cfg = _launch_config(tmp_path)
    archive = snapshot_round(tmp_path, _res(enablement_accepted_config_path=str(cfg)))
    assert archive.path_for(ROLE_LAUNCH_CONFIG) == "reports/enablement/abc123/launch_config.yaml"


def test_config_the_copy_refused_is_not_recorded(tmp_path):
    """Naming a file the archive does not hold is worse than naming none."""
    cfg = _launch_config(tmp_path, body=b"x" * (_FILE_SIZE_LIMIT + 1))
    archive = snapshot_round(tmp_path, _res(enablement_accepted_config_path=str(cfg)))
    assert archive.path_for(ROLE_LAUNCH_CONFIG) == ""


def test_server_log_is_archived(tmp_path):
    """The bench error string cannot cluster failures; the server's own log can."""
    log = tmp_path / "runs" / "integrate_patch" / "t1" / "server.log"
    log.parent.mkdir(parents=True)
    log.write_text("booting\nImportError: no module named aiter\n", encoding="utf-8")
    archive = snapshot_round(tmp_path, _res(bench_result={"server_log_path": str(log)}))
    dest = tmp_path / "reports" / "enablement" / "abc123" / "server.log"
    assert dest.read_text() == "booting\nImportError: no module named aiter\n"
    assert archive.path_for(ROLE_SERVER_LOG) == "reports/enablement/abc123/server.log"


def test_oversized_server_log_keeps_its_tail(tmp_path):
    """A crash is written at the end, and the largest logs are the ones worth reading."""
    log = tmp_path / "server.log"
    log.write_bytes(b"START OF LOG\n" + b"x" * _SERVER_LOG_TAIL_LIMIT + b"\nSIGSEGV in fused_moe\n")
    snapshot_round(tmp_path, _res(bench_result={"server_log_path": str(log)}))
    text = (tmp_path / "reports" / "enablement" / "abc123" / "server.log").read_text(encoding="utf-8")
    assert "SIGSEGV in fused_moe" in text
    assert "START OF LOG" not in text
    assert text.startswith("[hyperloom] truncated")


def test_server_log_tail_survives_a_split_codepoint(tmp_path):
    """A byte offset lands mid-character; the decode must not cost the whole log.

    The euro sign is three bytes in UTF-8, so the seek can land inside one.
    """
    log = tmp_path / "server.log"
    log.write_bytes("€".encode() * (_SERVER_LOG_TAIL_LIMIT // 3 + 1))
    snapshot_round(tmp_path, _res(bench_result={"server_log_path": str(log)}))
    text = (tmp_path / "reports" / "enablement" / "abc123" / "server.log").read_text(encoding="utf-8")
    body = text[len(_LOG_TRUNCATION_NOTE.format(dropped=2)) :]
    # Only the character the seek landed inside is lost, and nothing is replaced.
    assert set(body) == {"€"}
    assert len(body) == _SERVER_LOG_TAIL_LIMIT // 3


def test_a_log_still_growing_cannot_exceed_the_bound(tmp_path, monkeypatch):
    """The seek offset comes from a stat taken before the read; the read must not trust it."""
    log = tmp_path / "server.log"
    log.write_bytes(b"y" * (_SERVER_LOG_TAIL_LIMIT * 3))
    real_stat = Path.stat

    def _stale_size(self, *args, **kwargs):
        st = real_stat(self, *args, **kwargs)
        if self != log:
            return st
        fields = list(st)
        fields[6] = _SERVER_LOG_TAIL_LIMIT  # st_size, as of a moment ago
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "stat", _stale_size)
    snapshot_round(tmp_path, _res(bench_result={"server_log_path": str(log)}))
    dest = tmp_path / "reports" / "enablement" / "abc123" / "server.log"
    assert real_stat(dest).st_size == _SERVER_LOG_TAIL_LIMIT


def test_binary_noise_cannot_inflate_the_archived_log(tmp_path):
    """Replacing malformed bytes would triple them; the archive bound must hold."""
    log = tmp_path / "server.log"
    log.write_bytes(b"\xff" * (_SERVER_LOG_TAIL_LIMIT * 2) + b"\nSIGSEGV in fused_moe\n")
    snapshot_round(tmp_path, _res(bench_result={"server_log_path": str(log)}))
    dest = tmp_path / "reports" / "enablement" / "abc123" / "server.log"
    note = _LOG_TRUNCATION_NOTE.format(dropped=log.stat().st_size - _SERVER_LOG_TAIL_LIMIT)
    assert dest.stat().st_size <= _SERVER_LOG_TAIL_LIMIT + len(note)
    assert "SIGSEGV in fused_moe" in dest.read_text(encoding="utf-8")


def test_round_before_a_bench_has_no_server_log(tmp_path):
    """A patch rejected or a build broken never started a server to log."""
    archive = snapshot_round(tmp_path, _res())
    assert archive.path_for(ROLE_SERVER_LOG) == ""
    assert not (tmp_path / "reports" / "enablement" / "abc123" / "server.log").exists()


def test_missing_server_log_is_skipped(tmp_path):
    archive = snapshot_round(tmp_path, _res(bench_result={"server_log_path": str(tmp_path / "gone.log")}))
    assert archive.path_for(ROLE_SERVER_LOG) == ""


def test_snapshot_round_archives_artifacts_and_preimages(tmp_path):
    """Applied artifacts and their pre-images are archived by snapshot_round."""
    source = tmp_path / "src" / "mod.py"
    source.parent.mkdir(parents=True)
    source.write_text("# patched\n", encoding="utf-8")
    backup = tmp_path / "bak" / "mod.py.bak"
    backup.parent.mkdir(parents=True)
    backup.write_text("# original\n", encoding="utf-8")

    res = _res(
        artifacts_applied=[
            {
                "target": "/sgl-workspace/sglang/mod.py",
                "source": str(source),
                "backup": str(backup),
                "rel_target": "mod.py",
                "kind": "python_source",
                "existed": True,
            }
        ]
    )
    archive = snapshot_round(tmp_path, res)

    art_dir = tmp_path / "reports" / "enablement" / "abc123" / "artifacts"
    assert (art_dir / "000_mod.py").read_text() == "# patched\n"
    assert (art_dir / "000_mod.py.orig").read_text() == "# original\n"
    assert archive.path_for(ROLE_ARTIFACT_SOURCE) == "reports/enablement/abc123/artifacts/000_mod.py"
    assert archive.path_for(ROLE_ARTIFACT_PREIMAGE) == "reports/enablement/abc123/artifacts/000_mod.py.orig"


def test_oversized_artifact_is_skipped(tmp_path):
    src = tmp_path / "runs" / "specialist" / "abc123" / "patches"
    src.mkdir(parents=True)
    big = src / "003_big.patch"
    big.write_bytes(b"x" * (_FILE_SIZE_LIMIT + 1))
    archive = snapshot_round(tmp_path, _res(patches_applied=[str(big)]))
    assert not (tmp_path / "reports" / "enablement" / "abc123" / "patches" / "003_big.patch").exists()
    assert archive.to_list() == []


def test_unsafe_task_id_is_refused(tmp_path):
    with pytest.raises(ValueError):
        snapshot_round(tmp_path, _res(specialist_task_id="../evil"))


def test_round_without_a_specialist_is_skipped(tmp_path):
    """Phase-synthesised rounds carry no task id and would all collide."""
    archive = snapshot_round(tmp_path, {"enablement": True, "status": "reverted", "reason": "artifact_unreadable"})
    assert archive.to_list() == []
    assert not (tmp_path / "reports" / "enablement").exists()


def _patch(tmp_path, rel, body="diff --git a/f b/f\n"):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return str(p)


def _round(task_id, *patches, artifacts=()):
    """One accepted enablement round for ``EnablementRound.kept_rounds``.

    Mirrors the structure EnablementRound.push_kept_round writes.
    """
    return {"task_id": task_id, "patches": list(patches), "artifacts": list(artifacts)}


def _collected_and_vetted_round(tmp_path, target, before, after):
    workspace = tmp_path / "runs" / "specialist" / "abc123"
    worktree = workspace / "worktree"
    worktree.mkdir(parents=True)
    _git("init", "-q", str(worktree))
    (worktree / "runtime.py").write_text("# Original comment\nENABLED = True\n", encoding="utf-8")
    if before:
        original = worktree / target
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_text(before, encoding="utf-8")
    _git("-C", str(worktree), "add", ".")
    _git("-C", str(worktree), "-c", "user.email=a@b", "-c", "user.name=x", "commit", "-qm", "init")
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{target}" if before else "/dev/null",
            tofile=f"b/{target}",
        )
    )
    patch = Path(_patch(worktree, "patches/manual.patch", diff))
    collected, _roots = SpecialistSubprocessDispatcher._collect_patches(worktree, workspace, worktree)
    assert collected == [str(patch)]
    kept, dropped, _grounding, _spans_roots = vet_patches(collected, base_checkout=worktree)
    payload = {"patches_written": kept, "setup_commands": ["pip install local-runtime==1.0"]}
    (workspace / "specialist_done.json").write_text(json.dumps(payload), encoding="utf-8")
    return worktree, patch, kept, dropped


@pytest.mark.parametrize(
    ("target", "before", "after", "verdict"),
    [
        ("scratch/rebench/probe.py", "", "print('one-off probe')\n", "work_artifact"),
        (
            "runtime.py",
            "# Original comment\nENABLED = True\n",
            "# Revised comment\nENABLED = True\n",
            "annotation_only",
        ),
    ],
)
def test_snapshot_does_not_readmit_vet_rejected_patch(tmp_path, target, before, after, verdict):
    worktree, patch, kept, dropped = _collected_and_vetted_round(tmp_path, target, before, after)
    assert kept == []
    assert [(d["path"], d["verdict"]) for d in dropped] == [(str(patch), verdict)]
    evidence = patch.read_bytes()

    archive = snapshot_round(tmp_path, _res(patches_applied=kept, framework_root=str(worktree)))

    assert patch.read_bytes() == evidence
    assert archive.path_for(ROLE_SPECIALIST_RESULT)
    assert archive.paths_for(ROLE_PATCH) == ()
    archived_evidence = archive.path_for(ROLE_PATCH_EVIDENCE)
    assert archived_evidence == "reports/enablement/abc123/attempted_patches/manual.patch"
    assert (tmp_path / archived_evidence).read_bytes() == evidence


@pytest.mark.parametrize(
    ("target", "before", "after", "verdict"),
    [
        ("scratch/rebench/probe.py", "", "print('one-off probe')\n", "work_artifact"),
        (
            "runtime.py",
            "# Original comment\nENABLED = True\n",
            "# Revised comment\nENABLED = True\n",
            "annotation_only",
        ),
    ],
)
def test_setup_only_round_does_not_install_rejected_scan_patch(tmp_path, target, before, after, verdict):
    worktree, patch, kept, dropped = _collected_and_vetted_round(tmp_path, target, before, after)
    assert kept == []
    assert dropped[0]["verdict"] == verdict
    setup = ["pip install local-runtime==1.0"]
    result = _res(patches_applied=kept, setup_commands_applied=setup, framework_root=str(worktree))
    snapshot_round(tmp_path, result)
    en = EnablementRound()
    en.framework_root = result["framework_root"]
    en.setup_commands = result["setup_commands_applied"]
    en.kept_rounds = [_round(result["specialist_task_id"], *result["patches_applied"])]
    closure = {"interpreter": "/usr/bin/python3", "distributions": {"local-runtime": "1.0"}}
    en.environment_closure = closure
    en.environment_closure_baseline = {"interpreter": "/usr/bin/python3", "distributions": {"local-runtime": "0.9"}}

    write_setting_script(tmp_path, en, "sglang", model="/models/M")

    text = (tmp_path / "reports" / "enablement" / "enablement_setup.sh").read_text(encoding="utf-8")
    assert setup[0] in text
    assert "place_file" not in text
    assert en.kept_rounds == [{"task_id": result["specialist_task_id"], "patches": [], "artifacts": []}]
    assert not (tmp_path / "reports" / "enablement" / "setup_files").exists()
    assert patch.is_file()


@pytest.mark.parametrize(
    ("target", "before", "after"),
    [
        ("runtime.py", "ENABLED = False\n", "ENABLED = True\n"),
        ("kernels/new_kernel.py", "", "def block_size():\n    return 128\n"),
        ("configs/runtime.yaml", "block_size: 64\n", "block_size: 128\n"),
        ("configs/runtime.json", '{"block_size": 64}\n', '{"block_size": 128}\n'),
    ],
)
def test_vetted_source_and_config_patches_are_archived(tmp_path, target, before, after):
    worktree, patch, kept, dropped = _collected_and_vetted_round(tmp_path, target, before, after)
    assert kept == [str(patch)]
    assert dropped == []

    archive = snapshot_round(tmp_path, _res(patches_applied=kept, framework_root=str(worktree)))
    archived_patch = archive.path_for(ROLE_PATCH)
    assert (tmp_path / archived_patch).read_bytes() == patch.read_bytes()


def test_whole_file_artifact_is_archived_without_claiming_a_patch(tmp_path):
    source = Path(_patch(tmp_path, "runs/runtime.yaml", "block_size: 128\n"))
    content = source.read_bytes()
    target = tmp_path / "framework" / "configs" / "runtime.yaml"
    archive = snapshot_round(tmp_path, _res(artifacts_applied=[{"source": str(source), "target": str(target)}]))
    assert (tmp_path / archive.path_for(ROLE_ARTIFACT_SOURCE)).read_bytes() == content
    assert archive.paths_for(ROLE_PATCH) == ()


def _git(*args):
    subprocess.run(["git", *args], check=True, capture_output=True)
