# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_server_patcher`` (Hyperloom issue #194 §4 / §5).

Per-framework patchers that are fail-soft, idempotent, concurrency-safe, and
atomic. Fixtures synthesize fake vLLM/SGLang installs + a fake TraceLens patch
tree inside ``tmp_path`` so no real ``$TRACELENS_ROOT`` or site-packages is touched.
"""

from __future__ import annotations

import shutil
import sys
import textwrap
import threading
import types
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors import _server_patcher
from inference_optimizer.orchestrator.action_executors._server_patcher import (
    ensure_sglang_patched_for_tracelens,
    ensure_vllm_patched_for_tracelens,
)


# Fixtures: fake TraceLens patch tree + fake vLLM / SGLang installs
_FAKE_VLLM_VERSION = "0.99.0-fake"
_FAKE_SGLANG_VERSION = "0.5.9"  # must match one of _SGLANG_SUPPORTED_VERSIONS


def _make_fake_tracelens(tmp_path: Path) -> Path:
    """Build the TraceLens patch-tree skeleton (v0.3.1 per-version subdir layout)."""
    root = tmp_path / "TraceLens-internal"
    base = root / "examples" / "custom_workflows" / "inference_analysis"
    (base / "vllm_patches").mkdir(parents=True)
    (
        base / "sglang_roofline_patches"
        / _versioned_subdir_for_fake_sglang()
    ).mkdir(parents=True)
    return root


def _versioned_subdir_for_fake_sglang() -> str:
    """Subdir name TraceLens v0.3.1 ships for ``_FAKE_SGLANG_VERSION``."""
    return _server_patcher._versioned_patches_subdir_name(_FAKE_SGLANG_VERSION) or ""


def _make_fake_vllm_install(tmp_path: Path) -> Path:
    """Build a fake ``site-packages/vllm/...`` tree with a patchable ``vllm/config/profiler.py``."""
    site_packages = tmp_path / "site_packages"
    vllm_pkg = site_packages / "vllm"
    (vllm_pkg / "config").mkdir(parents=True)
    (vllm_pkg / "__init__.py").write_text(
        f'__version__ = "{_FAKE_VLLM_VERSION}"\n',
        encoding="utf-8",
    )
    (vllm_pkg / "config" / "profiler.py").write_text(
        textwrap.dedent(
            """\
            # Synthetic vLLM profiler config — used only by tests.
            class ProfilerConfig:
                profiler: str = ""
                ignore_frontend: bool = False
            """
        ),
        encoding="utf-8",
    )
    return site_packages


def _write_fake_vllm_patch(
    tracelens_root: Path, version: str,
    sentinels: tuple[str, ...] = (
        "capture_torch_profiler_dir", "detailed_trace_annotation",
    ),
) -> Path:
    """Generate a minimal unified diff adding both sentinel markers (PR-D §4) to the fake profiler.py."""
    patch_path = (
        tracelens_root
        / "examples" / "custom_workflows" / "inference_analysis"
        / "vllm_patches" / f"config_vllm_v{version}.patch"
    )
    new_lines = "\n".join(f"+    {s}: str = \"\"" for s in sentinels)
    added_count = len(sentinels)
    patch_path.write_text(
        textwrap.dedent(
            f"""\
            diff --git a/vllm/config/profiler.py b/vllm/config/profiler.py
            index 0000001..0000002 100644
            --- a/vllm/config/profiler.py
            +++ b/vllm/config/profiler.py
            @@ -1,4 +1,{4 + added_count} @@
             # Synthetic vLLM profiler config — used only by tests.
             class ProfilerConfig:
                 profiler: str = ""
            {new_lines}
                 ignore_frontend: bool = False
            """
        ),
        encoding="utf-8",
    )
    return patch_path


@pytest.fixture
def fake_vllm_world(tmp_path: Path, monkeypatch):
    """Fake TRACELENS_ROOT + fake vllm module + matching patch; returns (tracelens_root, install_root, patch_file)."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    install_root = _make_fake_vllm_install(tmp_path)
    patch_file = _write_fake_vllm_patch(tracelens_root, _FAKE_VLLM_VERSION)

    fake_mod = types.ModuleType("vllm")
    fake_mod.__version__ = _FAKE_VLLM_VERSION  # type: ignore[attr-defined]
    fake_mod.__file__ = str(install_root / "vllm" / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))
    return tracelens_root, install_root, patch_file


def _make_fake_sglang_install(tmp_path: Path) -> Path:
    """Build the editable ``python/sglang/...`` layout; returns the apply root (parent of ``python/``).

    Includes stub files for the extra_sentinels annotation markers so that
    post-apply sentinel verification passes after fake patches are applied.
    """
    apply_root = tmp_path / "sgl_repo"
    pkg = apply_root / "python" / "sglang" / "srt" / "utils"
    pkg.mkdir(parents=True)
    (apply_root / "python" / "sglang" / "__init__.py").write_text(
        f'__version__ = "{_FAKE_SGLANG_VERSION}"\n',
        encoding="utf-8",
    )
    (apply_root / "python" / "sglang" / "srt" / "__init__.py").write_text("")
    (apply_root / "python" / "sglang" / "srt" / "utils" / "__init__.py").write_text("")
    # Pre-populate extra_sentinels targets with annotation marker text so
    # post-apply sentinel checks pass in test fixtures.
    managers = apply_root / "python" / "sglang" / "srt" / "managers"
    managers.mkdir(parents=True, exist_ok=True)
    (managers / "scheduler.py").write_text(
        "# stub\ndef _build_profile_annotation(): pass\ndef profile_annotation(): pass\n",
    )
    (managers / "scheduler_profiler_mixin.py").write_text(
        "# stub roofline_annotations execute_ torch.profiler.record_function\n",
    )
    (managers / "io_struct.py").write_text(
        "# stub shape_discovery roofline_annotations\n",
    )
    entrypoints = apply_root / "python" / "sglang" / "srt" / "entrypoints"
    entrypoints.mkdir(parents=True, exist_ok=True)
    (entrypoints / "http_server.py").write_text(
        "# stub shape_discovery roofline_annotations\n",
    )
    return apply_root


def _write_fake_sglang_patches(
    tracelens_root: Path, *, count: int = 1, include_new_file: bool = True,
) -> list[Path]:
    """Write ``count`` minimal patches into the v0.3.1 per-version subdir (first creates the sentinel file)."""
    base = (
        tracelens_root / "examples" / "custom_workflows"
        / "inference_analysis" / "sglang_roofline_patches"
        / _versioned_subdir_for_fake_sglang()
    )
    patches: list[Path] = []
    if include_new_file:
        p1 = base / "kernel_shape_profiler.patch"
        p1.write_text(
            textwrap.dedent(
                """\
                diff --git a/python/sglang/srt/utils/kernel_shape_profiler.py b/python/sglang/srt/utils/kernel_shape_profiler.py
                new file mode 100644
                index 000000000..1111111
                --- /dev/null
                +++ b/python/sglang/srt/utils/kernel_shape_profiler.py
                @@ -0,0 +1,3 @@
                +# kernel_shape_profiler stub — TraceLens patch fixture.
                +def hello():
                +    return "kernel_shape_profiler"
                """
            ),
            encoding="utf-8",
        )
        patches.append(p1)
    for i in range(len(patches), count):
        p = base / f"misc_{i}.patch"
        target = f"python/sglang/srt/utils/extra_{i}.py"
        p.write_text(
            textwrap.dedent(
                f"""\
                diff --git a/{target} b/{target}
                new file mode 100644
                index 000000000..2222222
                --- /dev/null
                +++ b/{target}
                @@ -0,0 +1 @@
                +# extra_{i} stub
                """
            ),
            encoding="utf-8",
        )
        patches.append(p)
    return patches


@pytest.fixture
def fake_sglang_world(tmp_path: Path, monkeypatch):
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    patches = _write_fake_sglang_patches(tracelens_root, count=3)

    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = _FAKE_SGLANG_VERSION  # type: ignore[attr-defined]
    fake_mod.__file__ = str(apply_root / "python" / "sglang" / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))

    return tracelens_root, apply_root, patches


def _git_available() -> bool:
    return shutil.which("git") is not None


_REQUIRES_GIT = pytest.mark.skipif(
    not _git_available(), reason="git not available in test environment",
)


# vLLM happy path + idempotency
@_REQUIRES_GIT
def test_vllm_first_call_applies_patch(fake_vllm_world):
    _, install_root, _ = fake_vllm_world
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is True
    text = (install_root / "vllm" / "config" / "profiler.py").read_text()
    assert "capture_torch_profiler_dir" in text


@_REQUIRES_GIT
def test_vllm_second_call_is_noop(fake_vllm_world):
    """Idempotency: re-applying short-circuits on the sentinel and does not mutate the file."""
    _, install_root, _ = fake_vllm_world
    sentinel_path = install_root / "vllm" / "config" / "profiler.py"
    ensure_vllm_patched_for_tracelens()
    after_first = sentinel_path.read_text()
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is True
    assert sentinel_path.read_text() == after_first


# vLLM fail-soft paths
def test_vllm_returns_false_without_tracelens_root(monkeypatch):
    """No TRACELENS_ROOT, no explicit arg → fail-soft False."""
    monkeypatch.delenv("TRACELENS_ROOT", raising=False)
    assert ensure_vllm_patched_for_tracelens(None) is False


def test_vllm_returns_false_when_vllm_not_importable(
    fake_vllm_world, monkeypatch,
):
    """An environment without vllm must not crash — discover returns None → False."""
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    def _blocked_import(name, *args, **kwargs):  # noqa: ANN001 - hook
        if name == "vllm":
            raise ImportError("vllm not installed (test simulation)")
        return _real_import(name, *args, **kwargs)

    import builtins
    _real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    assert ensure_vllm_patched_for_tracelens() is False


def test_vllm_returns_false_when_no_patch_for_version(
    tmp_path, monkeypatch,
):
    """vLLM version with no matching TraceLens patch → fail-soft."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    install_root = _make_fake_vllm_install(tmp_path)
    fake_mod = types.ModuleType("vllm")
    fake_mod.__version__ = "9.9.9-no-patch"  # type: ignore[attr-defined]
    fake_mod.__file__ = str(install_root / "vllm" / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))
    assert ensure_vllm_patched_for_tracelens() is False


def test_vllm_returns_false_when_install_layout_unexpected(
    tmp_path, monkeypatch,
):
    """A broken vLLM install layout fails soft."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    _write_fake_vllm_patch(tracelens_root, _FAKE_VLLM_VERSION)
    bogus = tmp_path / "isolated_no_config" / "vllm" / "__init__.py"
    bogus.parent.mkdir(parents=True)
    bogus.write_text("")
    fake_mod = types.ModuleType("vllm")
    fake_mod.__version__ = _FAKE_VLLM_VERSION  # type: ignore[attr-defined]
    fake_mod.__file__ = str(bogus)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))
    assert ensure_vllm_patched_for_tracelens() is False


# SGLang happy path + version gating + atomic application
@_REQUIRES_GIT
def test_sglang_first_call_applies_all_patches(fake_sglang_world):
    _, apply_root, _ = fake_sglang_world
    rc = ensure_sglang_patched_for_tracelens()
    assert rc is True
    sentinel = (
        apply_root / "python" / "sglang" / "srt" / "utils"
        / "kernel_shape_profiler.py"
    )
    assert sentinel.exists()
    assert "kernel_shape_profiler" in sentinel.read_text()


@_REQUIRES_GIT
def test_sglang_second_call_is_noop(fake_sglang_world):
    _, apply_root, _ = fake_sglang_world
    ensure_sglang_patched_for_tracelens()
    sentinel = (
        apply_root / "python" / "sglang" / "srt" / "utils"
        / "kernel_shape_profiler.py"
    )
    before = sentinel.read_text()
    rc = ensure_sglang_patched_for_tracelens()
    assert rc is True
    assert sentinel.read_text() == before


def test_sglang_rejects_unsupported_version(tmp_path, monkeypatch):
    """SGLang versions outside the ``0.5.x`` allowlist fail-soft and must NOT touch the install tree."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    _write_fake_sglang_patches(tracelens_root)
    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = "0.6.0"  # type: ignore[attr-defined]
    fake_mod.__file__ = str(apply_root / "python" / "sglang" / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))
    assert ensure_sglang_patched_for_tracelens() is False
    sentinel = (
        apply_root / "python" / "sglang" / "srt" / "utils"
        / "kernel_shape_profiler.py"
    )
    assert not sentinel.exists(), (
        "patcher must NOT have written to disk for unsupported version"
    )


def test_sglang_rejects_unknown_install_layout(tmp_path, monkeypatch):
    """An unrecognised SGLang install layout fails soft."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    _write_fake_sglang_patches(tracelens_root)
    install = tmp_path / "weird_namespace" / "not_sglang"
    install.mkdir(parents=True)
    (install / "__init__.py").write_text(
        f'__version__ = "{_FAKE_SGLANG_VERSION}"\n',
        encoding="utf-8",
    )
    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = _FAKE_SGLANG_VERSION  # type: ignore[attr-defined]
    fake_mod.__file__ = str(install / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))
    assert ensure_sglang_patched_for_tracelens() is False


@_REQUIRES_GIT
def test_sglang_precheck_failure_skips_all(tmp_path, monkeypatch):
    """If ANY patch fails `git apply --check`, NONE are applied."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    _write_fake_sglang_patches(tracelens_root, count=1)
    bad = (
        tracelens_root / "examples" / "custom_workflows"
        / "inference_analysis" / "sglang_roofline_patches"
        / _versioned_subdir_for_fake_sglang() / "zzz_bad.patch"
    )
    bad.write_text(
        textwrap.dedent(
            """\
            diff --git a/python/sglang/does_not_exist.py b/python/sglang/does_not_exist.py
            index 0000001..0000002 100644
            --- a/python/sglang/does_not_exist.py
            +++ b/python/sglang/does_not_exist.py
            @@ -1,1 +1,1 @@
            -original
            +modified
            """
        ),
        encoding="utf-8",
    )
    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = _FAKE_SGLANG_VERSION  # type: ignore[attr-defined]
    fake_mod.__file__ = str(apply_root / "python" / "sglang" / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))

    assert ensure_sglang_patched_for_tracelens() is False
    sentinel = (
        apply_root / "python" / "sglang" / "srt" / "utils"
        / "kernel_shape_profiler.py"
    )
    assert not sentinel.exists(), (
        "patcher partial-applied even though --check predicted failure"
    )


# Concurrency: threads racing the same fake install converge.
@_REQUIRES_GIT
def test_vllm_concurrent_patchers_converge(fake_vllm_world):
    _, install_root, _ = fake_vllm_world
    results: list[bool] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(ensure_vllm_patched_for_tracelens())
        except BaseException as exc:  # noqa: BLE001 - test-only
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors
    assert all(results), results
    text = (install_root / "vllm" / "config" / "profiler.py").read_text()
    assert text.count("capture_torch_profiler_dir") == 1


# Misc: missing git binary
def test_returns_false_when_git_missing(fake_vllm_world, monkeypatch):
    """No `git` on PATH → fail-soft so git-less containers still run benchmarks."""
    monkeypatch.setattr(_server_patcher.shutil, "which", lambda _name: None)
    _, install_root, _ = fake_vllm_world
    assert ensure_vllm_patched_for_tracelens() is False


# PR-C §1 (tightened by PR-D §6): patch -p1 --fuzz=2 fallback when git apply
# --check rejects. fuzz=2 tolerates single-line drift but rejects multi-line drift.
_REQUIRES_PATCH = pytest.mark.skipif(
    shutil.which("patch") is None,
    reason="`patch` binary not available in test environment",
)


@_REQUIRES_GIT
@_REQUIRES_PATCH
def test_apply_atomic_fuzzy_fallback_when_git_strict_check_fails(fake_vllm_world):
    """When strict ``git apply --check`` rejects but ``patch --fuzz=2`` accepts, the patcher uses the fuzzy fallback."""
    _, install_root, _ = fake_vllm_world
    # One drift line is within fuzz=2 tolerance, so the fuzzy fallback applies.
    target = install_root / "vllm" / "config" / "profiler.py"
    target.write_text(
        textwrap.dedent(
            """\
            # Synthetic vLLM profiler config — used only by tests.
            # PR-C drift comment that breaks strict git apply --check.
            class ProfilerConfig:
                profiler: str = ""
                ignore_frontend: bool = False
            """
        ),
        encoding="utf-8",
    )
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is True
    assert "capture_torch_profiler_dir" in target.read_text(encoding="utf-8")


@_REQUIRES_GIT
def test_apply_atomic_returns_false_when_strict_and_fuzzy_both_fail(
    fake_vllm_world, monkeypatch,
):
    """When strict AND fuzzy both reject, the patcher returns False without touching the install."""
    _, install_root, _ = fake_vllm_world
    target = install_root / "vllm" / "config" / "profiler.py"
    target.write_text(
        "# Completely unrelated content — every patch hunk must miss.\n",
        encoding="utf-8",
    )
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is False
    assert "capture_torch_profiler_dir" not in target.read_text(encoding="utf-8")


@_REQUIRES_GIT
def test_apply_atomic_routes_clean_apply_through_strict_path(fake_vllm_world):
    """Regression guard: a clean-context install goes through strict ``git apply`` and never invokes the fuzzy fallback."""
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is True
    _, install_root, _ = fake_vllm_world
    text = (install_root / "vllm" / "config" / "profiler.py").read_text()
    assert "capture_torch_profiler_dir" in text
    assert ensure_vllm_patched_for_tracelens() is True


def test_patch_dry_run_returns_false_when_patch_binary_missing(tmp_path):
    """``_patch_dry_run`` fails soft when no ``patch`` binary is on PATH."""
    fake_diff = tmp_path / "fake.patch"
    fake_diff.write_text("--- a/x\n+++ b/x\n", encoding="utf-8")
    rc = _server_patcher._patch_dry_run(
        "/nonexistent/patch", fake_diff, tmp_path,
    )
    assert rc is False


# PR-D §6 safety guarantee: the ``_FUZZ`` constant must stay at GNU patch's
# default (2); bumping it back to 10 re-opens the silent mis-apply risk.
def test_fuzz_value_is_default_two_not_maximum_ten():
    """PR-D §6: ``_FUZZ`` MUST be 2; ``--fuzz=10`` would silently mis-apply on multi-line drift."""
    assert _server_patcher._FUZZ == 2, (
        f"_FUZZ must be 2 (GNU patch default, PR-D §6 safety floor); "
        f"found {_server_patcher._FUZZ}. Bumping it back up to 10 or "
        f"higher re-opens the silent multi-line mis-apply risk this "
        f"constant was introduced to close."
    )


@_REQUIRES_GIT
@_REQUIRES_PATCH
def test_fuzz_fallback_rejects_multi_line_context_mismatch(fake_vllm_world):
    """PR-D §6: mutating more than ``_FUZZ`` (=2) context lines makes the fuzzy fallback reject (fuzz=10 would accept)."""
    _, install_root, _ = fake_vllm_world
    target = install_root / "vllm" / "config" / "profiler.py"
    # All 3 before-context lines mutated; trailing context kept identical.
    target.write_text(
        textwrap.dedent(
            """\
            # COMPLETELY different header — upstream renamed the file's purpose.
            class ProfilerConfigRenamed:
                profile_dir: str = "renamed_field"
                ignore_frontend: bool = False
            """
        ),
        encoding="utf-8",
    )
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is False, (
        "fuzz=2 must reject when 3 of 4 context lines mismatch; "
        "fuzz=10 would have silently applied here and inserted the "
        "TraceLens fields into the renamed class — a semantically "
        "wrong patched install"
    )
    text = target.read_text(encoding="utf-8")
    assert "capture_torch_profiler_dir" not in text, (
        "fuzz=2 must not have mutated the install when context "
        "mismatch exceeded _FUZZ tolerance"
    )
    assert "detailed_trace_annotation" not in text


@_REQUIRES_GIT
@_REQUIRES_PATCH
def test_fuzz_fallback_tolerates_offset_slippage(fake_vllm_world):
    """Pure OFFSET slippage (context preserved verbatim, just shifted) still applies under fuzz=2."""
    _, install_root, _ = fake_vllm_world
    target = install_root / "vllm" / "config" / "profiler.py"
    target.write_text(
        textwrap.dedent(
            """\
            # Synthetic vLLM profiler config — used only by tests.
            # Drift line 1: upstream comment from a profiler refactor.
            # Drift line 2: docstring fragment.
            # Drift line 3: license header chunk.
            # Drift line 4: deprecation note.
            # Drift line 5: TODO marker.
            class ProfilerConfig:
                profiler: str = ""
                ignore_frontend: bool = False
            """
        ),
        encoding="utf-8",
    )
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is True, (
        "fuzz=2 must accept offset slippage when every context line "
        "is preserved verbatim — that's the legitimate point-release "
        "drift case the fuzzy fallback was designed for"
    )
    text = target.read_text(encoding="utf-8")
    assert "capture_torch_profiler_dir" in text
    assert "detailed_trace_annotation" in text


# PR-C §2: SGLang minor-version allowlist (was: exact-version pin)
@pytest.mark.parametrize(
    "env, version, expected",
    [
        pytest.param({}, "0.5.9", True, id="default_minor_covers_059"),
        pytest.param({}, "0.5.10", True, id="default_minor_covers_0510"),
        pytest.param({}, "0.5.11", True, id="default_minor_covers_0511"),
        pytest.param({}, "0.6.0", False, id="default_rejects_06x"),
        pytest.param({}, "0.4.9", False, id="default_rejects_04x"),
        pytest.param({}, "0.50.0", False, id="default_rejects_naive_prefix"),
        pytest.param(
            {"HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS": "0.5.9"},
            "0.5.9",
            True,
            id="exact_pin_accepts_pinned",
        ),
        pytest.param(
            {"HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS": "0.5.9"},
            "0.5.10",
            False,
            id="exact_pin_rejects_default_minor",
        ),
        pytest.param(
            {"HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS": "0.5,0.6"},
            "0.5.9",
            True,
            id="minor_env_keeps_default",
        ),
        pytest.param(
            {"HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS": "0.5,0.6"},
            "0.6.0",
            True,
            id="minor_env_adds_06",
        ),
        pytest.param(
            {"HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS": "0.5,0.6"},
            "0.7.0",
            False,
            id="minor_env_still_rejects_07",
        ),
        pytest.param({}, "", False, id="empty_version_rejected"),
        pytest.param({}, "   ", False, id="whitespace_version_rejected"),
    ],
)
def test_sglang_version_accepted(monkeypatch, env, version, expected):
    """Minor-version allowlist (PR-C §2): default 0.5.x band, narrowable/extendable via env overrides."""
    for key in (
        "HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS",
        "HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS",
    ):
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert _server_patcher._sglang_version_accepted(version) is expected


# PR-D §1: wheel-install SGLang patching via -p3 strip
def _make_fake_wheel_sglang_install(tmp_path: Path) -> Path:
    """Synthesise a pip-wheel SGLang layout (``site-packages/sglang/...``, no ``python/`` parent)."""
    site_packages = tmp_path / "site-packages"
    pkg = site_packages / "sglang" / "srt" / "utils"
    pkg.mkdir(parents=True)
    (site_packages / "sglang" / "__init__.py").write_text(
        f'__version__ = "{_FAKE_SGLANG_VERSION}"\n',
        encoding="utf-8",
    )
    (site_packages / "sglang" / "srt" / "__init__.py").write_text("")
    (site_packages / "sglang" / "srt" / "utils" / "__init__.py").write_text("")
    managers = site_packages / "sglang" / "srt" / "managers"
    managers.mkdir(parents=True, exist_ok=True)
    (managers / "scheduler.py").write_text(
        "# stub\ndef _build_profile_annotation(): pass\ndef profile_annotation(): pass\n",
    )
    (managers / "scheduler_profiler_mixin.py").write_text(
        "# stub roofline_annotations execute_ torch.profiler.record_function\n",
    )
    (managers / "io_struct.py").write_text(
        "# stub shape_discovery roofline_annotations\n",
    )
    entrypoints = site_packages / "sglang" / "srt" / "entrypoints"
    entrypoints.mkdir(parents=True, exist_ok=True)
    (entrypoints / "http_server.py").write_text(
        "# stub shape_discovery roofline_annotations\n",
    )
    return site_packages


@_REQUIRES_GIT
def test_sglang_wheel_install_patches_via_p3_strip(tmp_path, monkeypatch):
    """A wheel-layout SGLang install applies patches with ``-p3`` in place, modifying real wheel files."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    tracelens_root = _make_fake_tracelens(tmp_path)
    site_packages = _make_fake_wheel_sglang_install(tmp_path)
    _write_fake_sglang_patches(tracelens_root, count=1)

    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = _FAKE_SGLANG_VERSION  # type: ignore[attr-defined]
    fake_mod.__file__ = str(site_packages / "sglang" / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))

    rc = ensure_sglang_patched_for_tracelens()
    assert rc is True
    sentinel_real = (
        site_packages / "sglang" / "srt" / "utils" / "kernel_shape_profiler.py"
    )
    assert sentinel_real.exists(), (
        "wheel install's sglang/srt/utils/kernel_shape_profiler.py must exist "
        "after patching — the -p3 path should have modified the wheel directly"
    )
    assert "kernel_shape_profiler" in sentinel_real.read_text(encoding="utf-8")


@_REQUIRES_GIT
def test_sglang_wheel_install_is_idempotent(tmp_path, monkeypatch):
    """Second wheel-install invocation short-circuits via the sentinel check."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    tracelens_root = _make_fake_tracelens(tmp_path)
    site_packages = _make_fake_wheel_sglang_install(tmp_path)
    _write_fake_sglang_patches(tracelens_root, count=1)

    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = _FAKE_SGLANG_VERSION  # type: ignore[attr-defined]
    fake_mod.__file__ = str(site_packages / "sglang" / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))

    assert ensure_sglang_patched_for_tracelens() is True
    sentinel_real = (
        site_packages / "sglang" / "srt" / "utils" / "kernel_shape_profiler.py"
    )
    snapshot = sentinel_real.read_text(encoding="utf-8")
    assert ensure_sglang_patched_for_tracelens() is True
    assert sentinel_real.read_text(encoding="utf-8") == snapshot


def test_resolve_sglang_apply_root_editable_returns_p1(tmp_path):
    """Editable install keeps the ``-p1`` strip + repo-root apply path."""
    repo = tmp_path / "sgl_repo"
    pkg = repo / "python" / "sglang"
    pkg.mkdir(parents=True)
    sglang_module = pkg / "__init__.py"
    sglang_module.write_text("__version__ = '0.5.9'", encoding="utf-8")
    result = _server_patcher._resolve_sglang_apply_root(sglang_module)
    assert result is not None
    apply_root, strip = result
    assert apply_root == repo
    assert strip == 1


def test_resolve_sglang_apply_root_wheel_returns_p3(tmp_path):
    """Wheel install returns ``(<sglang_dir>, 3)`` to strip the patch prefix to the wheel layout."""
    site_packages = _make_fake_wheel_sglang_install(tmp_path)
    sglang_module = site_packages / "sglang" / "__init__.py"
    result = _server_patcher._resolve_sglang_apply_root(sglang_module)
    assert result is not None
    apply_root, strip = result
    assert apply_root == site_packages / "sglang"
    assert strip == 3


def test_resolve_sglang_apply_root_rejects_unexpected_layout(tmp_path):
    """An sglang module under an unrecognised layout fails soft."""
    weird_root = tmp_path / "weird_namespace" / "not_sglang"
    weird_root.mkdir(parents=True)
    sglang_module = weird_root / "__init__.py"
    sglang_module.write_text("__version__ = '0.5.9'", encoding="utf-8")
    assert _server_patcher._resolve_sglang_apply_root(sglang_module) is None


# PR-D §4: tuple-of-substrings sentinel for vLLM (false-positive guard)
def test_is_patched_requires_all_substrings_in_tuple(tmp_path):
    """``_is_patched`` requires EVERY substring in ``plan.sentinel_text``; one marker alone is rejected."""
    sentinel = tmp_path / "fake_sentinel.py"
    sentinel.write_text(
        "class ProfilerConfig:\n    capture_torch_profiler_dir: str = ''\n",
        encoding="utf-8",
    )
    plan = _server_patcher._PatchPlan(
        framework="vllm",
        version="0.20.0",
        apply_root=tmp_path,
        patches=(),
        sentinel_file=sentinel,
        sentinel_text=("capture_torch_profiler_dir", "detailed_trace_annotation"),
    )
    assert _server_patcher._is_patched(plan) is False, (
        "_is_patched must require BOTH markers; one alone is not enough"
    )
    sentinel.write_text(
        "class ProfilerConfig:\n"
        "    capture_torch_profiler_dir: str = ''\n"
        "    detailed_trace_annotation: bool = False\n",
        encoding="utf-8",
    )
    assert _server_patcher._is_patched(plan) is True


def test_is_patched_handles_single_element_tuple(tmp_path):
    """Single-element tuple sentinel (SGLang): presence of the lone marker counts as patched."""
    sentinel = tmp_path / "fake_kernel_shape_profiler.py"
    sentinel.write_text(
        "# kernel_shape_profiler module stub\n", encoding="utf-8",
    )
    plan = _server_patcher._PatchPlan(
        framework="sglang",
        version="0.5.9",
        apply_root=tmp_path,
        patches=(),
        sentinel_file=sentinel,
        sentinel_text=("kernel_shape_profiler",),
    )
    assert _server_patcher._is_patched(plan) is True
    sentinel.write_text("# unrelated file\n", encoding="utf-8")
    assert _server_patcher._is_patched(plan) is False


def test_vllm_plan_uses_two_marker_sentinel(tmp_path, monkeypatch):
    """PR-D §4: the vLLM plan declares a 2-tuple sentinel (both markers)."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    install_root = _make_fake_vllm_install(tmp_path)
    _write_fake_vllm_patch(tracelens_root, _FAKE_VLLM_VERSION)
    fake_mod = types.ModuleType("vllm")
    fake_mod.__version__ = _FAKE_VLLM_VERSION  # type: ignore[attr-defined]
    fake_mod.__file__ = str(install_root / "vllm" / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))

    plan = _server_patcher._discover_vllm_plan(None)
    assert plan is not None
    assert isinstance(plan.sentinel_text, tuple)
    assert set(plan.sentinel_text) == {
        "capture_torch_profiler_dir", "detailed_trace_annotation",
    }, plan.sentinel_text


def test_sglang_plan_keeps_single_marker_sentinel(fake_sglang_world):
    """SGLang keeps a single-substring sentinel in a 1-tuple for type uniformity (PR-D §4)."""
    tracelens_root, _, _ = fake_sglang_world
    plan = _server_patcher._discover_sglang_plan(tracelens_root)
    assert plan is not None
    assert isinstance(plan.sentinel_text, tuple)
    assert plan.sentinel_text == ("kernel_shape_profiler",), plan.sentinel_text


# PR-D §5: a TraceLens-shipped SUPPORTED_VERSIONS manifest takes precedence
# over the hardcoded minor allowlist (auto-adapts without a code change).
def _write_sglang_versions_manifest(
    tracelens_root: Path, body: str, *,
    version: str = _FAKE_SGLANG_VERSION,
    filename: str = "SUPPORTED_VERSIONS.txt",
) -> Path:
    """Write a TraceLens-style version manifest into the per-version SGLang patches subdir."""
    subdir = _server_patcher._versioned_patches_subdir_name(version) or ""
    patches_dir = (
        tracelens_root / "examples" / "custom_workflows"
        / "inference_analysis" / "sglang_roofline_patches" / subdir
    )
    patches_dir.mkdir(parents=True, exist_ok=True)
    manifest = patches_dir / filename
    manifest.write_text(body, encoding="utf-8")
    return manifest


def test_load_sglang_manifest_returns_none_when_absent(tmp_path):
    """No manifest → the loader returns None so the caller falls back to the PR-C.2 minor allowlist."""
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    assert _server_patcher._load_sglang_supported_versions_from_manifest(
        patches_dir,
    ) is None


def test_load_sglang_manifest_parses_versions_skipping_comments(tmp_path):
    """Format: one version per line, ``#`` comments and blank lines ignored, whitespace stripped."""
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    (patches_dir / "SUPPORTED_VERSIONS.txt").write_text(
        "# TraceLens-supported SGLang versions\n"
        "0.5.9\n"
        "   0.5.10   \n"
        "\n"
        "0.6.0  # post-bump\n"
        "# 0.7.0  (planned, not yet shipped)\n",
        encoding="utf-8",
    )
    versions = _server_patcher._load_sglang_supported_versions_from_manifest(
        patches_dir,
    )
    assert versions == frozenset({"0.5.9", "0.5.10", "0.6.0"}), versions


def test_load_sglang_manifest_empty_returns_empty_frozenset(tmp_path):
    """An explicit empty manifest returns an empty frozenset (NOT None) → caller rejects every version."""
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    (patches_dir / "SUPPORTED_VERSIONS.txt").write_text(
        "# Intentionally empty — no versions supported.\n"
        "\n",
        encoding="utf-8",
    )
    assert _server_patcher._load_sglang_supported_versions_from_manifest(
        patches_dir,
    ) == frozenset()


def test_load_sglang_manifest_prefers_dot_txt_over_no_extension(tmp_path):
    """Both manifest filenames are valid; ``.txt`` wins when both are present."""
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    (patches_dir / "SUPPORTED_VERSIONS.txt").write_text(
        "from_dot_txt\n", encoding="utf-8",
    )
    (patches_dir / "SUPPORTED_VERSIONS").write_text(
        "from_no_extension\n", encoding="utf-8",
    )
    assert _server_patcher._load_sglang_supported_versions_from_manifest(
        patches_dir,
    ) == frozenset({"from_dot_txt"})


def test_load_sglang_manifest_falls_back_to_no_extension(tmp_path):
    """The no-extension manifest variant is picked up when it's the only one present."""
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    (patches_dir / "SUPPORTED_VERSIONS").write_text(
        "0.5.9\n", encoding="utf-8",
    )
    assert _server_patcher._load_sglang_supported_versions_from_manifest(
        patches_dir,
    ) == frozenset({"0.5.9"})


def test_sglang_version_accepted_consults_manifest_when_present(
    tmp_path, monkeypatch,
):
    """A shipped manifest is the source of truth, overriding the hardcoded ``0.5.x`` default both ways."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    (patches_dir / "SUPPORTED_VERSIONS.txt").write_text(
        "0.6.0\n0.7.0\n", encoding="utf-8",
    )
    assert _server_patcher._sglang_version_accepted(
        "0.5.9", patches_dir=patches_dir,
    ) is False
    assert _server_patcher._sglang_version_accepted(
        "0.7.0", patches_dir=patches_dir,
    ) is True


def test_sglang_version_accepted_empty_manifest_rejects_all(
    tmp_path, monkeypatch,
):
    """An explicit empty manifest beats the hardcoded default → all versions rejected."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    (patches_dir / "SUPPORTED_VERSIONS.txt").write_text(
        "# all entries commented out → empty manifest\n", encoding="utf-8",
    )
    assert _server_patcher._sglang_version_accepted(
        "0.5.9", patches_dir=patches_dir,
    ) is False, (
        "empty manifest must reject every version — not silently fall through "
        "to the hardcoded default"
    )


def test_sglang_version_accepted_no_manifest_falls_back_to_default(
    tmp_path, monkeypatch,
):
    """With no manifest the helper falls back to ``_SGLANG_DEFAULT_ALLOWED_MINORS = ("0.5",)``."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()  # No manifest written.
    assert _server_patcher._sglang_version_accepted(
        "0.5.9", patches_dir=patches_dir,
    ) is True
    assert _server_patcher._sglang_version_accepted(
        "0.5.99", patches_dir=patches_dir,
    ) is True
    assert _server_patcher._sglang_version_accepted(
        "0.6.0", patches_dir=patches_dir,
    ) is False


def test_sglang_version_accepted_operator_env_vars_beat_manifest(
    tmp_path, monkeypatch,
):
    """An operator's exact pin takes precedence over a shipped manifest."""
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    (patches_dir / "SUPPORTED_VERSIONS.txt").write_text(
        "0.5.9\n", encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", "0.7.0")
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    assert _server_patcher._sglang_version_accepted(
        "0.7.0", patches_dir=patches_dir,
    ) is True
    assert _server_patcher._sglang_version_accepted(
        "0.5.9", patches_dir=patches_dir,
    ) is False


@_REQUIRES_GIT
def test_sglang_e2e_manifest_admits_version_outside_default_allowlist(
    tmp_path, monkeypatch,
):
    """Full integration: a shipped manifest admits SGLang 0.7.0 even though the default 0.5.x allowlist would reject."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    sgl_init = apply_root / "python" / "sglang" / "__init__.py"
    sgl_init.write_text('__version__ = "0.7.0"\n', encoding="utf-8")
    _write_versioned_sglang_patches(tracelens_root, "sglang_0_7_0", count=1)
    _write_sglang_versions_manifest(tracelens_root, "0.7.0\n", version="0.7.0")

    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = "0.7.0"  # type: ignore[attr-defined]
    fake_mod.__file__ = str(sgl_init)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))

    assert ensure_sglang_patched_for_tracelens() is True
    sentinel = (
        apply_root / "python" / "sglang" / "srt" / "utils"
        / "kernel_shape_profiler.py"
    )
    assert sentinel.exists(), (
        "0.7.0 should have been patched because the manifest admits it — "
        "the default 0.5.x allowlist alone would have rejected"
    )


# TraceLens v0.3.1: per-version patch subdirs (sglang_0_5_9/, ...) are the only
# supported layout; the flat v0.3 layout has been retired.
def _write_versioned_sglang_patches(
    tracelens_root: Path, subdir: str, *, count: int = 1,
) -> list[Path]:
    """Write minimal v0.3.1-style patches into a per-version subdir."""
    base = (
        tracelens_root / "examples" / "custom_workflows"
        / "inference_analysis" / "sglang_roofline_patches" / subdir
    )
    base.mkdir(parents=True, exist_ok=True)
    patches: list[Path] = []
    p1 = base / "kernel_shape_profiler.patch"
    p1.write_text(
        textwrap.dedent(
            """\
            diff --git a/python/sglang/srt/utils/kernel_shape_profiler.py b/python/sglang/srt/utils/kernel_shape_profiler.py
            new file mode 100644
            index 000000000..1111111
            --- /dev/null
            +++ b/python/sglang/srt/utils/kernel_shape_profiler.py
            @@ -0,0 +1,3 @@
            +# kernel_shape_profiler stub — TraceLens patch fixture.
            +def hello():
            +    return "kernel_shape_profiler"
            """
        ),
        encoding="utf-8",
    )
    patches.append(p1)
    for i in range(len(patches), count):
        p = base / f"misc_{i}.patch"
        target = f"python/sglang/srt/utils/extra_{i}.py"
        p.write_text(
            textwrap.dedent(
                f"""\
                diff --git a/{target} b/{target}
                new file mode 100644
                index 000000000..2222222
                --- /dev/null
                +++ b/{target}
                @@ -0,0 +1 @@
                +# extra_{i} stub
                """
            ),
            encoding="utf-8",
        )
        patches.append(p)
    return patches


def test_versioned_patches_subdir_name_helper():
    """Dotted versions map to ``sglang_<underscored>`` subdir names; bad input returns None."""
    assert _server_patcher._versioned_patches_subdir_name("0.5.11") == "sglang_0_5_11"
    assert _server_patcher._versioned_patches_subdir_name("0.5.9") == "sglang_0_5_9"
    assert _server_patcher._versioned_patches_subdir_name("1.2.3.4") == "sglang_1_2_3_4"
    assert _server_patcher._versioned_patches_subdir_name("0.5.11-rc1") == "sglang_0_5_11"
    assert _server_patcher._versioned_patches_subdir_name("0.5.11+local") == "sglang_0_5_11"
    assert _server_patcher._versioned_patches_subdir_name("") is None
    assert _server_patcher._versioned_patches_subdir_name("not-a-version") is None
    assert _server_patcher._versioned_patches_subdir_name(None) is None  # type: ignore[arg-type]


def test_resolve_sglang_patches_dir_returns_versioned_subdir(tmp_path):
    """The resolver returns the per-version subdir when it exists and has a ``*.patch`` file."""
    root = tmp_path / "sglang_roofline_patches"
    root.mkdir()
    subdir = root / "sglang_0_5_11"
    subdir.mkdir()
    (subdir / "kernel_shape_profiler.patch").write_text("p\n", encoding="utf-8")
    assert (
        _server_patcher._resolve_sglang_patches_dir(root, "0.5.11") == subdir
    )


def test_resolve_sglang_patches_dir_returns_none_when_subdir_missing(tmp_path):
    """No per-version subdir → ``None`` (flat layout is no longer supported)."""
    root = tmp_path / "sglang_roofline_patches"
    root.mkdir()
    (root / "decoy.patch").write_text("placeholder\n", encoding="utf-8")
    assert _server_patcher._resolve_sglang_patches_dir(root, "0.5.11") is None


def test_resolve_sglang_patches_dir_returns_none_when_subdir_empty(tmp_path):
    """An empty ``sglang_0_5_11/`` is not a valid patch set."""
    root = tmp_path / "sglang_roofline_patches"
    root.mkdir()
    (root / "sglang_0_5_11").mkdir()
    assert _server_patcher._resolve_sglang_patches_dir(root, "0.5.11") is None


@_REQUIRES_GIT
def test_sglang_e2e_versioned_layout_applies_from_subdir(tmp_path, monkeypatch):
    """End-to-end: sglang 0.5.11 resolves the ``sglang_0_5_11/`` subdir and applies every patch."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    sgl_init = apply_root / "python" / "sglang" / "__init__.py"
    sgl_init.write_text('__version__ = "0.5.11"\n', encoding="utf-8")
    _write_versioned_sglang_patches(tracelens_root, "sglang_0_5_11", count=1)

    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = "0.5.11"  # type: ignore[attr-defined]
    fake_mod.__file__ = str(sgl_init)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))

    assert ensure_sglang_patched_for_tracelens() is True
    sentinel = (
        apply_root / "python" / "sglang" / "srt" / "utils"
        / "kernel_shape_profiler.py"
    )
    assert sentinel.exists(), (
        "0.5.11 should pick sglang_0_5_11/ subdir and apply its patches"
    )


def test_discover_sglang_plan_marks_versioned_layout(tmp_path, monkeypatch):
    """The discovered ``_PatchPlan``'s patches point inside the per-version subdir."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    sgl_init = apply_root / "python" / "sglang" / "__init__.py"
    sgl_init.write_text('__version__ = "0.5.11"\n', encoding="utf-8")
    _write_versioned_sglang_patches(tracelens_root, "sglang_0_5_11", count=1)

    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = "0.5.11"  # type: ignore[attr-defined]
    fake_mod.__file__ = str(sgl_init)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))

    plan = _server_patcher._discover_sglang_plan(None)
    assert plan is not None
    for p in plan.patches:
        assert "sglang_0_5_11" in p.parts, (
            f"versioned layout should be selected, got patch path {p}"
        )


# Issue #505: a "new file" member patch whose target is ALREADY pre-baked into
# the sglang 0.5.11 image (byte-identical post-image) must be reverse-check
# detected and SKIPPED from the atomic set — so the remaining annotation
# patches still apply — instead of the whole set fail-soft skipping (which
# silently disabled per-step kernel-shape annotations -> empty kernel shape ->
# kernel-opt/GEAK never dispatched for the whole run).
@_REQUIRES_GIT
def test_sglang_0511_already_applied_member_is_skipped_not_failsoft(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    sgl_init = apply_root / "python" / "sglang" / "__init__.py"
    sgl_init.write_text('__version__ = "0.5.11"\n', encoding="utf-8")
    # 3 patches: kernel_shape_profiler.patch (the sentinel new-file), plus
    # misc_1 (extra_1.py) and misc_2 (extra_2.py) annotation new-files.
    _write_versioned_sglang_patches(tracelens_root, "sglang_0_5_11", count=3)

    # Force the fuzzy `patch` fallback OFF (git stays on PATH) so an
    # already-applied new-file member deterministically routes through the
    # reverse `git apply -R --check` skip branch rather than `patch --dry-run`.
    real_which = shutil.which
    monkeypatch.setattr(
        _server_patcher.shutil,
        "which",
        lambda name: real_which("git") if name == "git" else None,
    )

    utils_dir = apply_root / "python" / "sglang" / "srt" / "utils"
    # Pre-bake ONE non-sentinel member byte-identically (== already applied in
    # the image). We deliberately do NOT pre-bake the sentinel
    # kernel_shape_profiler.py, so the idempotency sentinel check does not
    # short-circuit before _apply_atomic runs.
    prebaked = utils_dir / "extra_1.py"
    prebaked.write_text("# extra_1 stub\n", encoding="utf-8")

    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = "0.5.11"  # type: ignore[attr-defined]
    fake_mod.__file__ = str(sgl_init)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))

    rc = ensure_sglang_patched_for_tracelens()
    assert rc is True, (
        "an already-applied member must be skipped so the remaining patches "
        "still apply; got fail-soft False (the #505 regression that left "
        "kernel-shape annotations disabled and GEAK never dispatched)"
    )
    # The sentinel + the other annotation patch landed despite the skip.
    assert (utils_dir / "kernel_shape_profiler.py").exists()
    assert (utils_dir / "extra_2.py").exists()
    # The pre-baked member is untouched (skipped — not rewritten or rolled back).
    assert prebaked.read_text(encoding="utf-8") == "# extra_1 stub\n"


@_REQUIRES_GIT
def test_apply_atomic_skips_already_applied_member(tmp_path, monkeypatch):
    """Unit-level: ``_apply_atomic`` returns True and applies the pending
    member while skipping the already-applied one (no ``patch`` binary needed).
    """
    real_which = shutil.which
    monkeypatch.setattr(
        _server_patcher.shutil,
        "which",
        lambda name: real_which("git") if name == "git" else None,
    )
    apply_root = tmp_path / "tree"
    apply_root.mkdir()
    # already.py is pre-baked identical to its patch's post-image.
    (apply_root / "already.py").write_text("# already applied\n", encoding="utf-8")

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    already_patch = patches_dir / "01_already.patch"
    already_patch.write_text(
        textwrap.dedent(
            """\
            diff --git a/already.py b/already.py
            new file mode 100644
            index 000000000..1111111
            --- /dev/null
            +++ b/already.py
            @@ -0,0 +1 @@
            +# already applied
            """
        ),
        encoding="utf-8",
    )
    pending_patch = patches_dir / "02_pending.patch"
    pending_patch.write_text(
        textwrap.dedent(
            """\
            diff --git a/pending.py b/pending.py
            new file mode 100644
            index 000000000..2222222
            --- /dev/null
            +++ b/pending.py
            @@ -0,0 +1 @@
            +# freshly applied
            """
        ),
        encoding="utf-8",
    )
    plan = _server_patcher._PatchPlan(
        framework="sglang",
        version="0.5.11",
        apply_root=apply_root,
        patches=(already_patch, pending_patch),
        sentinel_file=apply_root / "pending.py",
        sentinel_text=("freshly applied",),
        apply_strip=1,
    )
    assert _server_patcher._apply_atomic(plan) is True
    assert (apply_root / "pending.py").exists()
    assert (apply_root / "already.py").read_text(encoding="utf-8") == (
        "# already applied\n"
    )


def test_apply_atomic_rolls_back_when_post_apply_sentinel_fails(
    tmp_path, monkeypatch,
):
    """Transaction integrity: when patches apply cleanly to disk but the
    post-apply sentinel check fails, ``_apply_atomic`` must roll back the
    already-written patches (not leave the framework tree modified while
    reporting failure)."""
    real_which = shutil.which
    monkeypatch.setattr(
        _server_patcher.shutil,
        "which",
        lambda name: real_which("git") if name == "git" else None,
    )
    apply_root = tmp_path / "tree"
    apply_root.mkdir()

    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    new_patch = patches_dir / "01_new.patch"
    new_patch.write_text(
        textwrap.dedent(
            """\
            diff --git a/created.py b/created.py
            new file mode 100644
            index 000000000..3333333
            --- /dev/null
            +++ b/created.py
            @@ -0,0 +1 @@
            +# freshly applied
            """
        ),
        encoding="utf-8",
    )
    # Sentinel text the applied content does NOT contain -> _is_patched False
    # after a clean apply, exercising the post-apply gate.
    plan = _server_patcher._PatchPlan(
        framework="sglang",
        version="0.5.11",
        apply_root=apply_root,
        patches=(new_patch,),
        sentinel_file=apply_root / "created.py",
        sentinel_text=("MARKER_THAT_IS_NEVER_PRESENT",),
        apply_strip=1,
    )

    assert _server_patcher._apply_atomic(plan) is False
    # The patch was applied to disk then must be reverted; the file must not
    # linger (disk state must match the reported failure).
    assert not (apply_root / "created.py").exists(), (
        "post-apply sentinel failure must roll back already-applied patches"
    )
