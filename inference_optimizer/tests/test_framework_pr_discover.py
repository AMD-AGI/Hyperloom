"""Tests for ``inference_optimizer.orchestrator.framework_pr_discover``.

Hermetic - patches ``subprocess.run`` and ``shutil.which`` so no real
``fa`` invocation, no real git/pip, no GPU. Each test exercises one
slice of the c-light-auto contract.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator import framework_pr_discover as fpd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build a stand-in argparse-style namespace with sane defaults."""
    base = dict(
        framework_pr_discover=False,
        framework_pr="",
        framework_gap="",
        framework_repo_url="",
        framework_primus_url="",
        framework_keywords="",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _completed(rc=0, stdout="", stderr=""):
    """Construct a ``subprocess.CompletedProcess`` for monkeypatch."""
    import subprocess
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# _resolve_fa_binary
# ---------------------------------------------------------------------------


def test_resolve_fa_binary_via_which(monkeypatch):
    """When `fa` is on PATH, return that path."""
    monkeypatch.setattr(fpd.shutil, "which", lambda _name: "/usr/local/bin/fa")
    assert fpd._resolve_fa_binary() == "/usr/local/bin/fa"


def test_resolve_fa_binary_fallback_to_opt_venv(monkeypatch, tmp_path):
    """When PATH lookup fails but /opt/venv/bin/fa exists, fall back to it."""
    monkeypatch.setattr(fpd.shutil, "which", lambda _name: None)
    fake_venv = tmp_path / "fa"
    fake_venv.write_text("#!/bin/sh\n")
    fake_venv.chmod(0o755)
    monkeypatch.setattr(fpd, "_DEFAULT_SGLANG_PATH", fake_venv)  # unrelated; just to silence linter
    monkeypatch.setattr(fpd.os.path, "isfile", lambda p: p == "/opt/venv/bin/fa")
    monkeypatch.setattr(fpd.os, "access", lambda p, _mode: p == "/opt/venv/bin/fa")
    assert fpd._resolve_fa_binary() == "/opt/venv/bin/fa"


def test_resolve_fa_binary_missing_raises(monkeypatch):
    """When neither PATH nor /opt/venv has fa, raise FrameworkPRError."""
    monkeypatch.setattr(fpd.shutil, "which", lambda _name: None)
    monkeypatch.setattr(fpd.os.path, "isfile", lambda _p: False)
    with pytest.raises(fpd.FrameworkPRError, match="fa CLI not found"):
        fpd._resolve_fa_binary()


# ---------------------------------------------------------------------------
# _parse_pr_number
# ---------------------------------------------------------------------------


def test_parse_pr_number_ok():
    """PR:N -> N as int."""
    assert fpd._parse_pr_number("PR:123") == 123


def test_parse_pr_number_non_pr_raises():
    """A non-PR ref is rejected."""
    with pytest.raises(fpd.FrameworkPRError, match="only handles PR:N"):
        fpd._parse_pr_number("main")


def test_parse_pr_number_bad_int_raises():
    """`PR:abc` does not parse."""
    with pytest.raises(fpd.FrameworkPRError, match="could not parse PR number"):
        fpd._parse_pr_number("PR:abc")


# ---------------------------------------------------------------------------
# _build_explore_request
# ---------------------------------------------------------------------------


def test_build_explore_request_shape(tmp_path):
    """The synthesized payload has the keys fa requires + sane defaults."""
    req = fpd._build_explore_request(
        gap_description="improve sglang fp8 MoE",
        repo_url="https://github.com/sgl-project/sglang.git",
        primus_cortex_url="http://primus",
        work_dir=tmp_path / "work",
    )
    assert req["framework"] == "sglang"
    assert req["repo_url"] == "https://github.com/sgl-project/sglang.git"
    assert req["search_perf_prs"] is True
    assert req["max_search_candidates"] == 1
    assert req["search_modes"] == ["primus_cortex"]
    assert req["primus_cortex"]["base_url"] == "http://primus"
    assert req["gap_description"] == "improve sglang fp8 MoE"
    assert req["prepare_candidate_env"] is False
    assert req["commands"] == {}
    assert isinstance(req["baseline"]["throughput"], float)


# ---------------------------------------------------------------------------
# discover_pr
# ---------------------------------------------------------------------------


def _stub_fa_output(work_dir: Path, *, candidates: list[dict]) -> None:
    """Pre-populate ``plan_summary.json`` so discover_pr can parse it."""
    out = work_dir / "plan_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")


def test_discover_pr_happy_path(monkeypatch, tmp_path):
    """A single candidate is returned with the expected hand-off shape."""
    monkeypatch.setattr(fpd, "_resolve_fa_binary", lambda: "/fake/fa")

    def fake_run(_argv, **_kw):
        # fa would have written plan_summary.json; the stub does it for us.
        _stub_fa_output(
            tmp_path,
            candidates=[
                {
                    "candidate": {
                        "ref": "PR:24984",
                        "head_sha": "033ba939f9d96744f0171521e9beb2f229b2a53e",
                        "source": "primus_cortex",
                    },
                    "candidate_dir": str(tmp_path / "candidates" / "01_pr-24984"),
                    "patches_path": str(tmp_path / "candidates" / "01_pr-24984" / "pr.patches"),
                    "files_json_path": str(
                        tmp_path / "candidates" / "01_pr-24984" / "pr_files.json"
                    ),
                }
            ],
        )
        return _completed()

    monkeypatch.setattr(fpd.subprocess, "run", fake_run)
    handoff = fpd.discover_pr(
        gap_description="improve fp8 MoE",
        repo_url="https://github.com/sgl-project/sglang.git",
        primus_cortex_url="http://primus",
        work_dir=tmp_path,
    )
    assert handoff["winner_ref"] == "PR:24984"
    assert handoff["head_sha"].startswith("033ba93")
    assert handoff["winner_dir"].endswith("01_pr-24984")
    assert handoff["patch_path"].endswith("pr.patches")


def test_discover_pr_no_candidates_raises(monkeypatch, tmp_path):
    """When fa returns 0 candidates, discover_pr raises a clear error."""
    monkeypatch.setattr(fpd, "_resolve_fa_binary", lambda: "/fake/fa")

    def fake_run(_argv, **_kw):
        _stub_fa_output(tmp_path, candidates=[])
        return _completed()

    monkeypatch.setattr(fpd.subprocess, "run", fake_run)
    with pytest.raises(fpd.FrameworkPRError, match="no candidates"):
        fpd.discover_pr(
            gap_description="x",
            repo_url="https://github.com/sgl-project/sglang.git",
            primus_cortex_url="http://primus",
            work_dir=tmp_path,
        )


def test_discover_pr_non_pr_ref_raises(monkeypatch, tmp_path):
    """A `main` / branch ref is rejected (we only handle PR:N)."""
    monkeypatch.setattr(fpd, "_resolve_fa_binary", lambda: "/fake/fa")

    def fake_run(_argv, **_kw):
        _stub_fa_output(
            tmp_path,
            candidates=[
                {
                    "candidate": {"ref": "main", "source": "explicit"},
                    "candidate_dir": str(tmp_path / "candidates" / "01_main"),
                }
            ],
        )
        return _completed()

    monkeypatch.setattr(fpd.subprocess, "run", fake_run)
    with pytest.raises(fpd.FrameworkPRError, match="not a PR ref"):
        fpd.discover_pr(
            gap_description="x",
            repo_url="https://github.com/sgl-project/sglang.git",
            primus_cortex_url="http://primus",
            work_dir=tmp_path,
        )


def test_discover_pr_fa_rc_nonzero_raises(monkeypatch, tmp_path):
    """A non-zero fa exit code surfaces as FrameworkPRError."""
    monkeypatch.setattr(fpd, "_resolve_fa_binary", lambda: "/fake/fa")
    monkeypatch.setattr(
        fpd.subprocess, "run",
        lambda *_a, **_kw: _completed(rc=2, stderr="primus_cortex unreachable"),
    )
    with pytest.raises(fpd.FrameworkPRError, match="primus_cortex unreachable"):
        fpd.discover_pr(
            gap_description="x",
            repo_url="https://github.com/sgl-project/sglang.git",
            primus_cortex_url="http://nope",
            work_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# apply_to_sglang
# ---------------------------------------------------------------------------


def test_apply_to_sglang_requires_git(tmp_path):
    """If sglang_path is not a git checkout, raise rather than blunder on."""
    not_git = tmp_path / "no-git"
    not_git.mkdir()
    with pytest.raises(fpd.FrameworkPRError, match="not a git checkout"):
        fpd.apply_to_sglang(
            head_sha="abc1234",
            pr_number=1,
            sglang_path=not_git,
        )


def test_apply_to_sglang_no_python_subdir_raises(tmp_path, monkeypatch):
    """When sglang/python/ is missing, reinstall step refuses to proceed."""
    sglang = tmp_path / "sglang"
    (sglang / ".git").mkdir(parents=True)
    monkeypatch.setattr(fpd.subprocess, "run", lambda *_a, **_kw: _completed())
    with pytest.raises(fpd.FrameworkPRError, match="python does not exist"):
        fpd.apply_to_sglang(
            head_sha="abc1234567890",
            pr_number=1,
            sglang_path=sglang,
            pip_reinstall=True,
        )


def test_apply_to_sglang_skips_reinstall(tmp_path, monkeypatch):
    """pip_reinstall=False (default now) short-circuits the reinstall."""
    sglang = tmp_path / "sglang"
    (sglang / ".git").mkdir(parents=True)
    calls = []

    def fake_run(argv, **_kw):
        calls.append(list(argv))
        return _completed()

    monkeypatch.setattr(fpd.subprocess, "run", fake_run)
    fpd.apply_to_sglang(
        head_sha="abc1234567890",
        pr_number=1,
        sglang_path=sglang,
        pip_reinstall=False,
        auto_stash=False,
    )
    # fetch + checkout, but NO pip install
    cmds = [argv[0:3] for argv in calls if argv]
    assert any(c[:2] == ["git", "fetch"] for c in cmds)
    assert any(c[:2] == ["git", "checkout"] for c in cmds)
    assert not any("pip" in str(argv) for argv in calls)


def test_apply_to_sglang_default_no_reinstall(tmp_path, monkeypatch):
    """Default `pip_reinstall` value is now False; no pip call invoked."""
    sglang = tmp_path / "sglang"
    (sglang / ".git").mkdir(parents=True)
    calls = []

    def fake_run(argv, **_kw):
        calls.append(list(argv))
        return _completed()

    monkeypatch.setattr(fpd.subprocess, "run", fake_run)
    fpd.apply_to_sglang(
        head_sha="abc1234567890",
        pr_number=1,
        sglang_path=sglang,
        auto_stash=False,
    )
    assert not any("pip" in str(argv) for argv in calls)


def test_apply_to_sglang_auto_stash_when_dirty(tmp_path, monkeypatch):
    """`auto_stash=True` (default) triggers `git stash push -u` on dirty tree."""
    sglang = tmp_path / "sglang"
    (sglang / ".git").mkdir(parents=True)
    calls = []

    def fake_run(argv, **_kw):
        calls.append(list(argv))
        # Simulate dirty `git status --porcelain --untracked-files=no` -> non-empty
        if argv[:3] == ["git", "status", "--porcelain"]:
            return _completed(stdout=" M python/pyproject.toml\n")
        return _completed()

    monkeypatch.setattr(fpd.subprocess, "run", fake_run)
    fpd.apply_to_sglang(
        head_sha="abc1234567890",
        pr_number=1,
        sglang_path=sglang,
        auto_stash=True,
    )
    # Verify the stash push happened before the checkout
    stash_idx = next(i for i, c in enumerate(calls) if c[:3] == ["git", "stash", "push"])
    checkout_idx = next(i for i, c in enumerate(calls) if c[:2] == ["git", "checkout"])
    assert stash_idx < checkout_idx, "stash must precede checkout"


def test_apply_to_sglang_no_stash_when_clean(tmp_path, monkeypatch):
    """When the worktree is clean, auto_stash does not call `git stash`."""
    sglang = tmp_path / "sglang"
    (sglang / ".git").mkdir(parents=True)
    calls = []

    def fake_run(argv, **_kw):
        calls.append(list(argv))
        # Clean tree -> empty status
        if argv[:3] == ["git", "status", "--porcelain"]:
            return _completed(stdout="")
        return _completed()

    monkeypatch.setattr(fpd.subprocess, "run", fake_run)
    fpd.apply_to_sglang(
        head_sha="abc1234567890",
        pr_number=1,
        sglang_path=sglang,
        auto_stash=True,
    )
    assert not any(c[:3] == ["git", "stash", "push"] for c in calls)


def test_apply_to_sglang_no_auto_stash_flag(tmp_path, monkeypatch):
    """`auto_stash=False` skips the stash even on a dirty tree."""
    sglang = tmp_path / "sglang"
    (sglang / ".git").mkdir(parents=True)
    calls = []

    def fake_run(argv, **_kw):
        calls.append(list(argv))
        if argv[:3] == ["git", "status", "--porcelain"]:
            return _completed(stdout=" M python/pyproject.toml\n")
        return _completed()

    monkeypatch.setattr(fpd.subprocess, "run", fake_run)
    fpd.apply_to_sglang(
        head_sha="abc1234567890",
        pr_number=1,
        sglang_path=sglang,
        auto_stash=False,
    )
    assert not any(c[:3] == ["git", "stash", "push"] for c in calls)


# ---------------------------------------------------------------------------
# run() top-level
# ---------------------------------------------------------------------------


def test_run_mutually_exclusive_args_raises():
    """--framework-pr and --framework-pr-discover cannot coexist."""
    args = _make_args(framework_pr="PR:1", framework_pr_discover=True)
    with pytest.raises(fpd.FrameworkPRError, match="mutually exclusive"):
        fpd.run(args)


def test_run_neither_arg_raises():
    """run() must not be invoked when no framework-pr flag is set."""
    args = _make_args()
    with pytest.raises(fpd.FrameworkPRError, match="should not be called"):
        fpd.run(args)


def test_run_discover_requires_gap_or_keywords():
    """--framework-pr-discover with neither --framework-gap NOR --framework-keywords raises."""
    args = _make_args(framework_pr_discover=True)
    with pytest.raises(fpd.FrameworkPRError, match="--framework-gap or --framework-keywords"):
        fpd.run(args)


# ---------------------------------------------------------------------------
# C: --framework-keywords parsing + propagation
# ---------------------------------------------------------------------------


def test_build_explore_request_includes_keywords(tmp_path):
    """When keywords is non-empty, the req dict carries them as a list."""
    req = fpd._build_explore_request(
        gap_description="improve sglang",
        repo_url="https://github.com/sgl-project/sglang.git",
        primus_cortex_url="http://primus",
        work_dir=tmp_path / "w",
        keywords=["mi300x", "throughput"],
    )
    assert req["keywords"] == ["mi300x", "throughput"]


def test_build_explore_request_omits_keywords_when_empty(tmp_path):
    """No keywords -> 'keywords' key not present in req (back-compat with fa schema)."""
    req = fpd._build_explore_request(
        gap_description="improve sglang fp8 MoE",
        repo_url="https://github.com/sgl-project/sglang.git",
        primus_cortex_url="http://primus",
        work_dir=tmp_path / "w",
        keywords=None,
    )
    assert "keywords" not in req


def test_run_parses_framework_keywords_comma_separated(monkeypatch):
    """`--framework-keywords 'fp8,moe,sglang'` -> ['fp8','moe','sglang']."""
    captured = {}

    def fake_discover(**kw):
        captured.update(kw)
        return {
            "winner_ref": "PR:1", "head_sha": "abc1234567890",
            "winner_dir": "", "patch_path": "", "files_json_path": "",
            "candidate": {"ref": "PR:1", "head_sha": "abc1234567890"},
        }

    monkeypatch.setattr(fpd, "discover_pr", fake_discover)
    monkeypatch.setattr(fpd, "apply_to_sglang", lambda *_a, **_kw: None)
    monkeypatch.setattr(fpd, "_resolve_head_sha", lambda *_a, **_kw: "abc1234567890")

    args = _make_args(
        framework_pr_discover=True,
        framework_gap="improve sglang fp8 MoE",
        framework_keywords="fp8,moe,sglang",
    )
    fpd.run(args)
    assert captured["keywords"] == ["fp8", "moe", "sglang"]


def test_run_parses_framework_keywords_space_separated(monkeypatch):
    """`--framework-keywords 'fp8 moe sglang'` also works."""
    captured = {}

    def fake_discover(**kw):
        captured.update(kw)
        return {
            "winner_ref": "PR:1", "head_sha": "abc1234567890",
            "winner_dir": "", "patch_path": "", "files_json_path": "",
            "candidate": {"ref": "PR:1"},
        }

    monkeypatch.setattr(fpd, "discover_pr", fake_discover)
    monkeypatch.setattr(fpd, "apply_to_sglang", lambda *_a, **_kw: None)
    monkeypatch.setattr(fpd, "_resolve_head_sha", lambda *_a, **_kw: "abc1234567890")

    args = _make_args(
        framework_pr_discover=True,
        framework_gap="anything",
        framework_keywords="fp8 moe sglang",
    )
    fpd.run(args)
    assert captured["keywords"] == ["fp8", "moe", "sglang"]


def test_run_keywords_without_gap_is_ok(monkeypatch):
    """`--framework-keywords` alone (no --framework-gap) is sufficient."""
    captured = {}

    def fake_discover(**kw):
        captured.update(kw)
        return {
            "winner_ref": "PR:1", "head_sha": "abc1234567890",
            "winner_dir": "", "patch_path": "", "files_json_path": "",
            "candidate": {"ref": "PR:1"},
        }

    monkeypatch.setattr(fpd, "discover_pr", fake_discover)
    monkeypatch.setattr(fpd, "apply_to_sglang", lambda *_a, **_kw: None)
    monkeypatch.setattr(fpd, "_resolve_head_sha", lambda *_a, **_kw: "abc1234567890")

    args = _make_args(
        framework_pr_discover=True,
        framework_gap="",                # explicitly empty
        framework_keywords="moe",
    )
    fpd.run(args)
    assert captured["keywords"] == ["moe"]
    assert captured["gap_description"] == ""
