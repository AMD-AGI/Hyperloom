# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tool-build provenance, and the phase timeline it lands beside.

Which build of tracelens, GEAK or forge produced a session's numbers is only
in scope while the run that used it is running. Nothing downstream can
recover it, so these pin the recording at each point it is decided, and the
``versions`` map it composes into.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hyperloom.inference_optimizer.breakdown.recorder import (
    assemble_parts,
    instrument,
    tool_versions,
)


def _init_git_repo(path: Path) -> str:
    for argv in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "commit", "--allow-empty", "-q", "-m", "init"],
    ):
        subprocess.run(argv, cwd=path, check=True, capture_output=True)
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _distribution_version() -> str:
    """The version forge's provenance entry must now report.

    Resolved independently of the recorder's own probe so the assertion is an
    oracle rather than a tautology. A source checkout with no installed
    distribution resolves to the empty string, which is still the answer the
    probe owes: an unresolvable build reports nothing rather than borrowing the
    git SHA the caller's ``root`` happens to yield.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("hyperloom-inference_optimizer")
    except PackageNotFoundError:
        return ""


def test_versions_map_composed_at_top_level(tmp_path: Path) -> None:
    # Every producer that resolves a build writes to one map, keyed by tool.
    tool_versions.record_tool_version(tmp_path, tool="tracelens", root=str(tmp_path))
    instrument.record_backend_versions_and_timeline(
        tmp_path,
        {
            "kernel_id": "k1",
            "run_id": "r1",
            "attempts": [
                {"attempt_id": "a1", "backend": "geak", "status": "completed"},
            ],
        },
    )
    out = assemble_parts(tmp_path)
    versions = out["metadata"]["versions"]["tools"]
    assert isinstance(versions, dict)
    assert set(versions) >= {"tracelens", "geak"}
    assert versions["geak"]["tool"] == "geak"


def test_forge_backend_mints_versions_entry(tmp_path: Path) -> None:
    # A forge attempt mints a provenance entry distinct from geak's.
    sha = _init_git_repo(tmp_path)
    instrument.record_backend_versions_and_timeline(
        tmp_path,
        {
            "kernel_id": "k1",
            "run_id": "r1",
            "attempts": [
                {
                    "attempt_id": "a1",
                    "backend": "forge",
                    "status": "completed",
                    "metadata": {"root_dir": str(tmp_path)},
                },
            ],
        },
    )
    out = assemble_parts(tmp_path)
    versions = out["metadata"]["versions"]["tools"]
    assert versions["forge"]["tool"] == "forge"
    # KernelForge ships inside this distribution, so its version IS Hyperloom's;
    # there is no separate checkout left to ``git rev-parse``. The producer-supplied
    # root_dir still yields a commit, so provenance keeps both halves.
    assert versions["forge"]["version"] == _distribution_version()
    assert versions["forge"]["version"] != sha
    assert versions["forge"]["commit"] == sha


def test_a_result_with_no_attempts_still_credits_the_backend_it_names(tmp_path: Path) -> None:
    # A run that failed before any backend launched still knows which build was
    # in play, and that is the case where provenance matters most.
    instrument.record_backend_versions_and_timeline(
        tmp_path,
        {"kernel_id": "k1", "backend": "geak", "status": "failed", "attempts": []},
    )
    out = assemble_parts(tmp_path)
    assert out["metadata"]["versions"]["tools"]["geak"]["tool"] == "geak"


def test_a_pre_dispatch_gating_failure_invents_no_build(tmp_path: Path) -> None:
    # No backend was resolved, so there is no build to report. Minting an
    # all-empty entry would read as "this tool ran and reported nothing".
    instrument.record_backend_versions_and_timeline(
        tmp_path,
        {"kernel_id": "k1", "status": "failed", "attempts": []},
    )
    out = assemble_parts(tmp_path)
    assert not (out.get("metadata") or {}).get("versions")


def test_geak_provenance_resolves_geak_root_env_without_explicit_root(tmp_path: Path, monkeypatch) -> None:
    # With no producer-supplied root, ``geak`` falls back to $GEAK_ROOT so
    # versions["geak"] records that repo's SHA.
    geak_root = tmp_path / "GEAK"
    geak_root.mkdir()
    geak_sha = _init_git_repo(geak_root)
    monkeypatch.setenv("GEAK_ROOT", str(geak_root))

    meta = tool_versions._tool_metadata("geak")

    assert meta["tool"] == "geak"
    assert meta["root_dir"] == str(geak_root)
    assert meta["commit"] == geak_sha
    assert meta["version"] == geak_sha


def test_the_bypass_reader_reports_this_distribution(tmp_path: Path) -> None:
    # The bypass trace reader ships inside Hyperloom, so its entry must carry
    # this distribution's version rather than the all-empty shell it wrote
    # while "bypass" was missing from the provenance registry.
    tool_versions.record_tool_version(tmp_path, tool="bypass")
    out = assemble_parts(tmp_path)
    entry = out["metadata"]["versions"]["tools"]["bypass"]
    assert entry["tool"] == "bypass"
    assert entry["version"] == _distribution_version()


def test_tool_version_probe_git_strategies(tmp_path: Path) -> None:
    sha = _init_git_repo(tmp_path)
    # geak -> git short SHA; commit == version.
    meta = tool_versions._tool_metadata("geak", root=str(tmp_path))
    assert meta["commit"] == sha
    assert meta["version"] == sha
    # tracelens -> git describe (--always falls back to the short sha here).
    meta_tl = tool_versions._tool_metadata("tracelens", root=str(tmp_path))
    assert meta_tl["version"]  # non-empty describe output
    # forge -> the distribution version: it is vendored into Hyperloom, not a
    # checkout, so the git strategy geak uses does not apply to it. The commit
    # still comes from the root the caller passed.
    meta_forge = tool_versions._tool_metadata("forge", root=str(tmp_path))
    assert meta_forge["commit"] == sha
    assert meta_forge["version"] == _distribution_version()
    assert meta_forge["version"] != sha
    # A caller-supplied version always wins over the probe.
    meta_explicit = tool_versions._tool_metadata(
        "geak",
        root=str(tmp_path),
        version="v9.9",
    )
    assert meta_explicit["version"] == "v9.9"


def test_tool_version_probe_cmd_and_dist() -> None:
    # CLI strategy: python3 --version.
    assert (
        tool_versions._probe_tool_version(
            ("cmd", ("python3", "--version")),
            "",
        )
        .lower()
        .startswith("python")
    )
    # dist strategy resolves an installed package and rejects a bogus name.
    assert tool_versions._dist_version(("pytest",))
    assert tool_versions._dist_version(("definitely-not-a-real-dist-xyz",)) == ""
