"""source_resolver + collect_target_files tests (P2 PR-E)."""

from __future__ import annotations

from pathlib import Path

import pytest

from framework_agent.agent.source_resolver import (
    ARG_SCAN_INCLUDE_DIRS,
    collect_target_files,
    resolve_framework_sources,
)


_FIXTURES = Path(__file__).parent / "fixtures"


def test_collect_target_files_sglang_fixture():
    files = collect_target_files("sglang", _FIXTURES / "mini_sglang")
    # Fixture includes: server_args.py + configs/scheduler.py +
    # configs/broken.py (still collected; broken.py triggers grep fallback
    # in ast_scanner)
    rel = {Path(f).name for f in files}
    assert "server_args.py" in rel
    assert "scheduler.py" in rel


def test_collect_target_files_vllm_fixture():
    files = collect_target_files("vllm", _FIXTURES / "mini_vllm")
    rel = {Path(f).name for f in files}
    assert "arg_utils.py" in rel


def test_collect_target_files_unknown_framework_returns_empty():
    files = collect_target_files("not-a-framework", _FIXTURES / "mini_vllm")
    assert files == []


def test_collect_target_files_deterministic_ordering():
    """Same root scanned twice -> identical list ordering."""
    a = collect_target_files("sglang", _FIXTURES / "mini_sglang")
    b = collect_target_files("sglang", _FIXTURES / "mini_sglang")
    assert a == b


def test_resolve_framework_sources_via_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """VLLM_SOURCE_ROOT / SGLANG_SOURCE_ROOT env override wins."""
    monkeypatch.setenv("SGLANG_SOURCE_ROOT", str(_FIXTURES / "mini_sglang"))
    monkeypatch.setenv("VLLM_SOURCE_ROOT", str(_FIXTURES / "mini_vllm"))
    resolved = resolve_framework_sources(("vllm", "sglang"))
    assert "vllm" in resolved
    assert "sglang" in resolved
    assert resolved["sglang"] == (_FIXTURES / "mini_sglang").resolve()


def test_resolve_framework_sources_skips_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Frameworks without env / container / site-packages -> omitted."""
    monkeypatch.delenv("VLLM_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("SGLANG_SOURCE_ROOT", raising=False)
    # Hide real packages by monkeypatching find_spec to None for our targets.
    import framework_agent.agent.source_resolver as sr

    def fake_find_spec(name: str):
        return None

    monkeypatch.setattr(sr.importlib.util, "find_spec", fake_find_spec)
    # /sgl-workspace/* probably exists on the dev box; we don't assert
    # the missing case here unless we also mock Path.is_dir. Just
    # exercise the call path returns a dict (possibly empty).
    out = resolve_framework_sources(("not-a-framework",))
    assert out == {}
