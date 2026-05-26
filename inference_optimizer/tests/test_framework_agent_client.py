"""framework_agent_client.fetch_pr_candidates tests.

Coverage:

* Happy path: a fake ``fa`` binary returns a JSON payload mimicking the
  real ``fa candidates`` schema; the helper normalises each candidate
  into the prompt-friendly dict shape.
* Graceful degrade on missing binary / non-zero exit / timeout / parse
  error — every failure path returns ``[]``.
* ``_resolve_fa_binary`` respects ``$FA_BIN`` override.
* ``repo_url_for_framework`` lookup table.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import framework_agent_client as fac


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_fake_fa(
    tmp_path: Path,
    *,
    name: str = "fa",
    body: str,
) -> Path:
    fa_path = tmp_path / name
    fa_path.write_text(body, encoding="utf-8")
    fa_path.chmod(fa_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return fa_path


def _happy_fa_body() -> str:
    """Stub bash script that emits a realistic fa-candidates JSON
    response to stdout regardless of args. Built without textwrap so
    the shebang stays at column 0 even when the embedded JSON spans
    multiple lines."""
    payload = {
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "search_modes": ["primus_cortex", "github"],
        "search_perf_prs": False,
        "max_search_candidates": 5,
        "count": 2,
        "candidates": [
            {
                "ref": "PR:1234",
                "repo": "sgl-project/sglang",
                "source": "github",
                "head_sha": "deadbeef",
                "title": "MoE expert dispatch speedup",
                "labels": ["moe", "perf"],
                "author": "alice",
                "changed_files": [],
                "updated_at": "2026-04-01T00:00:00Z",
                "html_url": "https://github.com/sgl-project/sglang/pull/1234",
                "score": 0.94,
            },
            {
                "ref": "PR:5678",
                "repo": "sgl-project/sglang",
                "source": "primus_cortex",
                "head_sha": "cafebabe",
                "title": "Cuda graph fixes",
                "labels": [],
                "author": "bob",
                "changed_files": [],
                "updated_at": "",
                "html_url": "https://github.com/sgl-project/sglang/pull/5678",
                "score": 0.41,
            },
        ],
    }
    return (
        "#!/usr/bin/env bash\n"
        "cat <<'JSON_EOF'\n"
        + json.dumps(payload, indent=2)
        + "\nJSON_EOF\n"
    )


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_pr_candidates_happy_path(tmp_path, monkeypatch):
    fa_path = _write_fake_fa(tmp_path, body=_happy_fa_body())
    monkeypatch.setenv("FA_BIN", str(fa_path))

    out = await fac.fetch_pr_candidates(
        gap_description="MoE routing slow",
        framework="sglang",
        session_dir=tmp_path,
    )

    assert len(out) == 2
    first = out[0]
    assert first["repo"] == "sgl-project/sglang"
    assert first["pr_number"] == 1234
    assert first["ref"] == "PR:1234"
    assert first["title"] == "MoE expert dispatch speedup"
    assert first["labels"] == ["moe", "perf"]
    # diff_url constructed from html_url
    assert first["diff_url"] == (
        "https://github.com/sgl-project/sglang/pull/1234.diff"
    )
    assert first["source_url"] == (
        "https://github.com/sgl-project/sglang/pull/1234"
    )
    assert first["score"] == pytest.approx(0.94)
    assert first["author"] == "alice"
    # summary is joined labels (fallback since Candidate has no summary field).
    assert first["summary"] == "moe, perf"

    # No diff body inlined anywhere.
    for entry in out:
        assert "diff" not in entry
        assert "patches" not in entry


@pytest.mark.asyncio
async def test_fetch_pr_candidates_respects_max(tmp_path, monkeypatch):
    fa_path = _write_fake_fa(tmp_path, body=_happy_fa_body())
    monkeypatch.setenv("FA_BIN", str(fa_path))

    out = await fac.fetch_pr_candidates(
        gap_description="anything",
        framework="sglang",
        session_dir=tmp_path,
        max_candidates=1,
    )
    assert len(out) == 1


# ---------------------------------------------------------------------------
# 2. Graceful degrade
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_pr_candidates_returns_empty_when_binary_missing(
    tmp_path, monkeypatch,
):
    # Point FA_BIN at something that doesn't exist.
    monkeypatch.setenv("FA_BIN", str(tmp_path / "no-such-fa"))
    monkeypatch.setenv("PATH", "")
    out = await fac.fetch_pr_candidates(
        gap_description="x", framework="sglang", session_dir=tmp_path,
    )
    assert out == []


@pytest.mark.asyncio
async def test_fetch_pr_candidates_returns_empty_on_nonzero_exit(
    tmp_path, monkeypatch,
):
    fa_path = _write_fake_fa(
        tmp_path,
        body='#!/usr/bin/env bash\necho "boom" >&2\nexit 17\n',
    )
    monkeypatch.setenv("FA_BIN", str(fa_path))
    out = await fac.fetch_pr_candidates(
        gap_description="x", framework="sglang", session_dir=tmp_path,
    )
    assert out == []


@pytest.mark.asyncio
async def test_fetch_pr_candidates_returns_empty_on_json_parse_error(
    tmp_path, monkeypatch,
):
    fa_path = _write_fake_fa(
        tmp_path,
        body='#!/usr/bin/env bash\necho "this is not json"\n',
    )
    monkeypatch.setenv("FA_BIN", str(fa_path))
    out = await fac.fetch_pr_candidates(
        gap_description="x", framework="sglang", session_dir=tmp_path,
    )
    assert out == []


@pytest.mark.asyncio
async def test_fetch_pr_candidates_returns_empty_on_unknown_framework(
    tmp_path, monkeypatch,
):
    fa_path = _write_fake_fa(tmp_path, body=_happy_fa_body())
    monkeypatch.setenv("FA_BIN", str(fa_path))
    out = await fac.fetch_pr_candidates(
        gap_description="x",
        framework="unknown-framework",
        session_dir=tmp_path,
    )
    assert out == []


@pytest.mark.asyncio
async def test_fetch_pr_candidates_cleans_up_tmp(tmp_path, monkeypatch):
    """The temporary ExploreRequest JSON must be removed after the
    subprocess call so ``.fa-tmp/`` doesn't grow unboundedly."""
    fa_path = _write_fake_fa(tmp_path, body=_happy_fa_body())
    monkeypatch.setenv("FA_BIN", str(fa_path))

    await fac.fetch_pr_candidates(
        gap_description="x", framework="sglang", session_dir=tmp_path,
    )
    fa_tmp = tmp_path / ".fa-tmp"
    # Directory exists but should be empty (request file deleted).
    assert fa_tmp.is_dir()
    leftover = list(fa_tmp.iterdir())
    assert leftover == []


# ---------------------------------------------------------------------------
# 3. Repo URL helpers
# ---------------------------------------------------------------------------
def test_repo_url_for_framework_known():
    assert fac.repo_url_for_framework("sglang") == (
        "https://github.com/sgl-project/sglang.git"
    )
    assert fac.repo_url_for_framework("vllm") == (
        "https://github.com/ROCm/vllm.git"
    )


def test_repo_url_for_framework_unknown_returns_empty():
    assert fac.repo_url_for_framework("rust-burn") == ""
    assert fac.repo_url_for_framework("") == ""


def test_resolve_fa_binary_prefers_env_var(tmp_path, monkeypatch):
    fa_path = _write_fake_fa(tmp_path, body='#!/usr/bin/env bash\nexit 0\n')
    monkeypatch.setenv("FA_BIN", str(fa_path))
    # Even if PATH is empty, the env var wins.
    monkeypatch.setenv("PATH", "")
    assert fac._resolve_fa_binary() == str(fa_path)


def test_resolve_fa_binary_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("FA_BIN", raising=False)
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    monkeypatch.setenv("PATH", "")
    assert fac._resolve_fa_binary() is None
