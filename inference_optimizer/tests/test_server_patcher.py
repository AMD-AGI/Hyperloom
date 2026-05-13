"""Tests for ``_server_patcher`` (Hyperloom issue #194 §4 / §5).

Contract recap (see ``_server_patcher.py`` docstring for the long
version):

* Per-framework patchers (vLLM, SGLang) — caller invokes only the one
  matching the YAML's framework.
* Fail-soft on every error path — caller treats ``False`` as
  "do not inject TraceLens-only flags".
* Idempotent — sentinel check short-circuits after the first apply.
* Concurrency-safe — flock serializes the write window.
* Atomic for multi-patch sets — ``--check`` all first, rollback on
  mid-apply failure.

The tests synthesize a fake vLLM / SGLang install tree + a fake
TraceLens patch directory inside ``tmp_path`` so we never touch the
real ``$TRACELENS_ROOT`` or any real ``site-packages``.
"""

from __future__ import annotations

import shutil
import subprocess
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


# ===========================================================================
# Fixtures: fake TraceLens patch tree + fake vLLM / SGLang installs
# ===========================================================================
_FAKE_VLLM_VERSION = "0.99.0-fake"
_FAKE_SGLANG_VERSION = "0.5.9"  # must match one of _SGLANG_SUPPORTED_VERSIONS


def _make_fake_tracelens(tmp_path: Path) -> Path:
    """Build the ``<root>/examples/custom_workflows/inference_analysis/``
    skeleton the patcher expects to find patches under."""
    root = tmp_path / "TraceLens-internal"
    base = root / "examples" / "custom_workflows" / "inference_analysis"
    (base / "vllm_patches").mkdir(parents=True)
    (base / "sglang_roofline_patches").mkdir(parents=True)
    return root


def _make_fake_vllm_install(tmp_path: Path) -> Path:
    """Build a fake ``site-packages/vllm/...`` tree with a sentinel
    file (``vllm/config/profiler.py``) ready to be patched."""
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
    """Generate a minimal unified diff that adds the sentinel strings
    to the fake profiler.py — `git apply` accepts it without needing
    a real .git/ directory.

    PR-D §4: real TraceLens vLLM patches add BOTH
    ``capture_torch_profiler_dir`` and ``detailed_trace_annotation`` as
    new fields on ``ProfilerConfig``; the _server_patcher sentinel
    requires BOTH to be present before declaring the install patched.
    The fixture must follow suit so the synthetic patch leaves both
    markers in the file, matching the real-world contract.
    """
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
    """Set up: fake TRACELENS_ROOT + fake vllm in sys.modules + a
    matching patch file. Returns (tracelens_root, install_root, patch_file)."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    install_root = _make_fake_vllm_install(tmp_path)
    patch_file = _write_fake_vllm_patch(tracelens_root, _FAKE_VLLM_VERSION)

    # Inject a fake `vllm` module so `import vllm` inside the discover
    # function sees our staged install rather than the real one (which
    # may or may not be present in CI).
    fake_mod = types.ModuleType("vllm")
    fake_mod.__version__ = _FAKE_VLLM_VERSION  # type: ignore[attr-defined]
    fake_mod.__file__ = str(install_root / "vllm" / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))

    return tracelens_root, install_root, patch_file


def _make_fake_sglang_install(tmp_path: Path) -> Path:
    """SGLang patches use `a/python/sglang/...` prefix so the install
    must have a `python/sglang/...` layout (the editable repo form).
    Returns the apply root (parent of `python/`)."""
    apply_root = tmp_path / "sgl_repo"
    pkg = apply_root / "python" / "sglang" / "srt" / "utils"
    pkg.mkdir(parents=True)
    (apply_root / "python" / "sglang" / "__init__.py").write_text(
        f'__version__ = "{_FAKE_SGLANG_VERSION}"\n',
        encoding="utf-8",
    )
    (apply_root / "python" / "sglang" / "srt" / "__init__.py").write_text("")
    (apply_root / "python" / "sglang" / "srt" / "utils" / "__init__.py").write_text("")
    return apply_root


def _write_fake_sglang_patches(
    tracelens_root: Path, *, count: int = 1, include_new_file: bool = True,
) -> list[Path]:
    """Write ``count`` minimal patches into the sglang patches dir.

    The first patch always creates ``kernel_shape_profiler.py``
    (matching the real patch set's sentinel file). Additional patches
    are no-op header-comment additions to make the multi-patch atomic
    apply path testable.
    """
    base = (
        tracelens_root / "examples" / "custom_workflows"
        / "inference_analysis" / "sglang_roofline_patches"
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
        # Touch a fresh file each time so patches don't conflict.
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


# ===========================================================================
# vLLM happy path + idempotency
# ===========================================================================
@_REQUIRES_GIT
def test_vllm_first_call_applies_patch(fake_vllm_world):
    _, install_root, _ = fake_vllm_world
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is True
    text = (install_root / "vllm" / "config" / "profiler.py").read_text()
    assert "capture_torch_profiler_dir" in text


@_REQUIRES_GIT
def test_vllm_second_call_is_noop(fake_vllm_world):
    """Idempotency: sentinel-based short-circuit — re-applying must not
    mutate the file."""
    _, install_root, _ = fake_vllm_world
    sentinel_path = install_root / "vllm" / "config" / "profiler.py"
    ensure_vllm_patched_for_tracelens()
    after_first = sentinel_path.read_text()
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is True
    assert sentinel_path.read_text() == after_first


# ===========================================================================
# vLLM fail-soft paths
# ===========================================================================
def test_vllm_returns_false_without_tracelens_root(monkeypatch):
    """No TRACELENS_ROOT, no explicit arg → fail-soft False (callers
    treat as "don't inject TraceLens flags")."""
    monkeypatch.delenv("TRACELENS_ROOT", raising=False)
    assert ensure_vllm_patched_for_tracelens(None) is False


def test_vllm_returns_false_when_vllm_not_importable(
    fake_vllm_world, monkeypatch,
):
    """Even with TRACELENS_ROOT set, an environment without vllm
    must not crash — discover returns None → patcher returns False."""
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    # Block re-import to simulate vllm-less environment.

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
    """vLLM version exists but TraceLens hasn't shipped a matching
    patch file → fail-soft (this is the common case for brand-new
    vLLM releases)."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    install_root = _make_fake_vllm_install(tmp_path)
    # NB: we deliberately do not write a patch file.
    fake_mod = types.ModuleType("vllm")
    fake_mod.__version__ = "9.9.9-no-patch"  # type: ignore[attr-defined]
    fake_mod.__file__ = str(install_root / "vllm" / "__init__.py")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_mod)
    monkeypatch.setenv("TRACELENS_ROOT", str(tracelens_root))
    assert ensure_vllm_patched_for_tracelens() is False


def test_vllm_returns_false_when_install_layout_unexpected(
    tmp_path, monkeypatch,
):
    """`vllm.__file__` points at a path whose parent.parent is NOT a
    site-packages-like dir — install layout broken, fail-soft."""
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


# ===========================================================================
# SGLang happy path + version gating + atomic application
# ===========================================================================
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
    """SGLang versions outside the configured minor allowlist (default
    ``0.5.x``) must fail-soft — and crucially must NOT touch the
    install tree on disk. PR-C widened the gate from an exact 0.5.9
    pin to a 0.5.x prefix so freshly bumped point releases reach the
    fuzzy patch fallback; bigger minor bumps (0.6.x, 0.4.x) still
    fail-soft here so we don't risk applying a stale patch set against
    an incompatible install."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    _write_fake_sglang_patches(tracelens_root)
    fake_mod = types.ModuleType("sglang")
    # 0.6.0 — outside the 0.5.x allowlist; would need a real minor
    # bump in TraceLens's patch set, not just fuzzy context drift.
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
    """SGLang installed at neither ``python/sglang/`` (editable) nor
    ``site-packages/sglang/`` (wheel) — e.g. someone renamed the
    package or placed it under a custom dir — must fail-soft. Wheel
    installs now go through the shim path (covered by the
    ``test_sglang_wheel_install_uses_symlink_shim`` test); only
    layouts the resolver doesn't recognise return None here."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    _write_fake_sglang_patches(tracelens_root)
    # Place the module under a parent dir named something other than
    # ``sglang`` so the wheel-install check rejects it. (A real-world
    # cause would be a namespace-package install or a fork rename.)
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
    """If ANY patch in the set fails `git apply --check`, NONE are
    applied — the install stays in its original state."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    # First patch is fine.
    _write_fake_sglang_patches(tracelens_root, count=1)
    # Add a corrupt patch that won't apply (refers to a file that
    # doesn't exist in our fake install).
    bad = (
        tracelens_root / "examples" / "custom_workflows"
        / "inference_analysis" / "sglang_roofline_patches" / "zzz_bad.patch"
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


# ===========================================================================
# Concurrency: threads racing the same fake install converge.
# ===========================================================================
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
    # Sentinel substring appears exactly once — no double-patching.
    assert text.count("capture_torch_profiler_dir") == 1


# ===========================================================================
# Misc: missing git binary
# ===========================================================================
def test_returns_false_when_git_missing(fake_vllm_world, monkeypatch):
    """If `git` isn't on PATH we cannot apply patches; fail-soft so
    containers without git still run benchmarks (just without
    TraceLens flags)."""
    monkeypatch.setattr(_server_patcher.shutil, "which", lambda _name: None)
    # Reset to pre-patch state by removing the sentinel from profiler.py.
    _, install_root, _ = fake_vllm_world
    # The sentinel file is freshly written by the fixture — no
    # cleanup needed; the patcher will see "not patched" → try git →
    # find no git → return False.
    assert ensure_vllm_patched_for_tracelens() is False


# ===========================================================================
# PR-C §1: patch -p1 --fuzz=10 fallback when git apply --check rejects
# ===========================================================================
_REQUIRES_PATCH = pytest.mark.skipif(
    shutil.which("patch") is None,
    reason="`patch` binary not available in test environment",
)


@_REQUIRES_GIT
@_REQUIRES_PATCH
def test_apply_atomic_fuzzy_fallback_when_git_strict_check_fails(fake_vllm_world):
    """When ``git apply --check`` rejects a patch because the install
    has drifted slightly from the patch's recorded context, but
    ``patch -p1 --fuzz=10 --dry-run`` still accepts it, the patcher
    falls back to the fuzzy path and applies successfully.

    Simulates the common case where TraceLens hasn't shipped a
    patch revision yet for a freshly bumped SGLang / vLLM point
    release. Without this fallback the run silently loses the
    TraceLens-only profiler flags."""
    _, install_root, _ = fake_vllm_world
    # Add a drift comment between the patched lines so the strict
    # ``git apply --check`` context check fails but ``patch --fuzz=10``
    # still locates the hunk.
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
    # Patch must have landed via the fuzzy fallback path.
    assert "capture_torch_profiler_dir" in target.read_text(encoding="utf-8")


@_REQUIRES_GIT
def test_apply_atomic_returns_false_when_strict_and_fuzzy_both_fail(
    fake_vllm_world, monkeypatch,
):
    """When ``git apply --check`` rejects the patch AND
    ``patch -p1 --fuzz=10`` is unavailable (or itself rejects), the
    patcher returns False without touching the install — the fuzzy
    fallback never opens a new error mode."""
    _, install_root, _ = fake_vllm_world
    # Drift the target enough that strict + fuzzy both refuse the patch.
    target = install_root / "vllm" / "config" / "profiler.py"
    target.write_text(
        "# Completely unrelated content — every patch hunk must miss.\n",
        encoding="utf-8",
    )
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is False
    # Install must be untouched (no half-applied patch).
    assert "capture_torch_profiler_dir" not in target.read_text(encoding="utf-8")


@_REQUIRES_GIT
def test_apply_atomic_routes_clean_apply_through_strict_path(fake_vllm_world):
    """Regression guard: when the install matches the patch context
    exactly, the strict ``git apply --check`` accepts it and the
    fuzzy fallback is never invoked. Without this guard PR-C could
    accidentally route every patch through the slower fuzzy path."""
    rc = ensure_vllm_patched_for_tracelens()
    assert rc is True
    _, install_root, _ = fake_vllm_world
    # Sanity: patched content present; we can't easily mock the
    # subprocess invocations without making the test brittle, so we
    # rely on the next call being a no-op (the sentinel grep).
    text = (install_root / "vllm" / "config" / "profiler.py").read_text()
    assert "capture_torch_profiler_dir" in text
    # Second call short-circuits without trying any apply at all.
    assert ensure_vllm_patched_for_tracelens() is True


def test_patch_dry_run_returns_false_when_patch_binary_missing(tmp_path):
    """``_patch_dry_run`` must fail-soft when no ``patch`` binary is
    on PATH so callers can decide to skip the fuzzy fallback rather
    than crash."""
    # Pass a non-existent ``patch_bin`` path; subprocess raises
    # FileNotFoundError which the helper catches.
    fake_diff = tmp_path / "fake.patch"
    fake_diff.write_text("--- a/x\n+++ b/x\n", encoding="utf-8")
    rc = _server_patcher._patch_dry_run(
        "/nonexistent/patch", fake_diff, tmp_path,
    )
    assert rc is False


# ===========================================================================
# PR-C §2: SGLang minor-version allowlist (was: exact-version pin)
# ===========================================================================
def test_sglang_version_accepted_default_minor_covers_059(monkeypatch):
    """Default allowlist must still accept the original exact pin so
    pre-PR-C behaviour is preserved on the current deployment."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    assert _server_patcher._sglang_version_accepted("0.5.9") is True


def test_sglang_version_accepted_default_minor_covers_0510(monkeypatch):
    """Default allowlist accepts a freshly bumped point release —
    the whole point of PR-C is to let 0.5.10 / 0.5.11 reach the
    fuzzy fallback layer (which can still reject if the patch
    fundamentally conflicts)."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    assert _server_patcher._sglang_version_accepted("0.5.10") is True
    assert _server_patcher._sglang_version_accepted("0.5.11") is True


def test_sglang_version_accepted_default_minor_rejects_different_minor(monkeypatch):
    """0.5.x allowlist must NOT accept 0.6.x or 0.4.x — those are
    minor-version bumps with bigger surface change than fuzzy patch
    contextual drift can tolerate."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    assert _server_patcher._sglang_version_accepted("0.6.0") is False
    assert _server_patcher._sglang_version_accepted("0.4.9") is False
    # Edge case: 0.50.0 must not match the 0.5 prefix — guard against
    # naive ``startswith``.
    assert _server_patcher._sglang_version_accepted("0.50.0") is False


def test_sglang_version_accepted_exact_pin_env_overrides_minor(monkeypatch):
    """Operators who want to lock down to a known-good list can set
    ``HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS`` — exact pins win over
    the minor allowlist for the duration of the run."""
    monkeypatch.setenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", "0.5.9")
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    assert _server_patcher._sglang_version_accepted("0.5.9") is True
    # 0.5.10 IS in the default minor allowlist, but the exact-pin env
    # narrows it back to just 0.5.9.
    assert _server_patcher._sglang_version_accepted("0.5.10") is False


def test_sglang_version_accepted_minor_env_extends_default(monkeypatch):
    """``HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS=0.5,0.6`` lets operators
    extend the default allowlist when TraceLens promises a coming
    patch set for the next minor."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    monkeypatch.setenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", "0.5,0.6")
    assert _server_patcher._sglang_version_accepted("0.5.9") is True
    assert _server_patcher._sglang_version_accepted("0.6.0") is True
    assert _server_patcher._sglang_version_accepted("0.7.0") is False


def test_sglang_version_accepted_empty_version_rejected(monkeypatch):
    """Empty / whitespace ``sglang.__version__`` must not silently
    pass — the patcher needs a real version to pick the right patch
    set."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    assert _server_patcher._sglang_version_accepted("") is False
    assert _server_patcher._sglang_version_accepted("   ") is False


# ===========================================================================
# PR-D §1: wheel-install SGLang patching via -p3 strip
# ===========================================================================
def _make_fake_wheel_sglang_install(tmp_path: Path) -> Path:
    """Synthesise a pip-wheel SGLang layout: ``site-packages/sglang/...``
    with no ``python/`` parent. Mirrors what
    ``pip install sglang --no-build-isolation`` produces from a release
    wheel, distinct from the editable repo layout the fixture above
    builds."""
    site_packages = tmp_path / "site-packages"
    pkg = site_packages / "sglang" / "srt" / "utils"
    pkg.mkdir(parents=True)
    (site_packages / "sglang" / "__init__.py").write_text(
        f'__version__ = "{_FAKE_SGLANG_VERSION}"\n',
        encoding="utf-8",
    )
    (site_packages / "sglang" / "srt" / "__init__.py").write_text("")
    (site_packages / "sglang" / "srt" / "utils" / "__init__.py").write_text("")
    return site_packages


@_REQUIRES_GIT
def test_sglang_wheel_install_patches_via_p3_strip(tmp_path, monkeypatch):
    """A wheel-layout SGLang install (``site-packages/sglang/...`` with
    no ``python/`` parent) must apply patches with ``-p3`` from inside
    the wheel sglang/ dir itself — no symlinks, no tmpdirs, real
    wheel files modified in place. ``git apply``'s symlink-safety
    check would refuse a symlink-shim approach (verified empirically
    on git 2.40+); ``-p3`` sidesteps the issue entirely by stripping
    the patch's ``a/python/sglang/`` prefix down to the wheel-relative
    ``srt/...`` path."""
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
    # Patch landed on the real wheel install file.
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
    """Second invocation short-circuits via the sentinel check without
    re-running git apply. Guards against the wheel-install path
    racing on every call."""
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
    """Editable install (``<repo>/python/sglang/__init__.py``) keeps
    the historical ``-p1`` strip + repo-root apply path."""
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
    """Wheel install (``site-packages/sglang/__init__.py``) returns
    ``(<sglang_dir>, 3)`` so the patch's ``a/python/sglang/`` prefix
    is stripped down to the wheel layout's ``srt/...`` path."""
    site_packages = _make_fake_wheel_sglang_install(tmp_path)
    sglang_module = site_packages / "sglang" / "__init__.py"
    result = _server_patcher._resolve_sglang_apply_root(sglang_module)
    assert result is not None
    apply_root, strip = result
    assert apply_root == site_packages / "sglang"
    assert strip == 3


def test_resolve_sglang_apply_root_rejects_unexpected_layout(tmp_path):
    """An sglang module that lives under neither ``python/sglang/`` nor
    ``site-packages/sglang/`` (e.g. the user moved it under a custom
    dir, or a fork renamed the package) must fail-soft."""
    weird_root = tmp_path / "weird_namespace" / "not_sglang"
    weird_root.mkdir(parents=True)
    sglang_module = weird_root / "__init__.py"
    sglang_module.write_text("__version__ = '0.5.9'", encoding="utf-8")
    assert _server_patcher._resolve_sglang_apply_root(sglang_module) is None


# ===========================================================================
# PR-D §4: tuple-of-substrings sentinel for vLLM (false-positive guard)
# ===========================================================================
def test_is_patched_requires_all_substrings_in_tuple(tmp_path):
    """``_is_patched`` must require EVERY substring in
    ``plan.sentinel_text`` to be present. Drop one of the two vLLM
    markers and the check must reject — that's the false-positive
    guard if upstream ever merges one of the marker identifiers
    without adopting the rest of the TraceLens patch."""
    sentinel = tmp_path / "fake_sentinel.py"
    # Has only the first marker, not the second.
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
    # Add the second marker → now it counts as patched.
    sentinel.write_text(
        "class ProfilerConfig:\n"
        "    capture_torch_profiler_dir: str = ''\n"
        "    detailed_trace_annotation: bool = False\n",
        encoding="utf-8",
    )
    assert _server_patcher._is_patched(plan) is True


def test_is_patched_handles_single_element_tuple(tmp_path):
    """Single-element tuple sentinel (the SGLang case) behaves
    identically to the historical single-string sentinel: presence
    of the lone marker counts as patched."""
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
    # Wipe the marker → rejected.
    sentinel.write_text("# unrelated file\n", encoding="utf-8")
    assert _server_patcher._is_patched(plan) is False


def test_vllm_plan_uses_two_marker_sentinel(tmp_path, monkeypatch):
    """The vLLM plan must declare a 2-tuple sentinel
    (``capture_torch_profiler_dir`` AND ``detailed_trace_annotation``).
    This is the structural guarantee PR-D §4 makes: changing the plan
    to a 1-tuple would silently revert to the false-positive-prone
    historical behaviour, so we pin it explicitly."""
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
    """SGLang's sentinel file is *created* by the patch — its presence
    + a unique internal identifier is already false-positive-proof, so
    we keep the historical single-substring sentinel wrapped in a
    1-tuple for type uniformity with the vLLM plan (PR-D §4)."""
    tracelens_root, _, _ = fake_sglang_world
    plan = _server_patcher._discover_sglang_plan(tracelens_root)
    assert plan is not None
    assert isinstance(plan.sentinel_text, tuple)
    assert plan.sentinel_text == ("kernel_shape_profiler",), plan.sentinel_text


# ===========================================================================
# PR-D §5: TraceLens-shipped SUPPORTED_VERSIONS manifest takes precedence
# over the hardcoded minor allowlist. The day TraceLens starts shipping
# this file, Hyperloom's version gate auto-adapts without a code change —
# the decoupling intent of the #194 §5 follow-up recommendation.
# ===========================================================================
def _write_sglang_versions_manifest(
    tracelens_root: Path, body: str, *, filename: str = "SUPPORTED_VERSIONS.txt",
) -> Path:
    """Write a TraceLens-style version manifest into the SGLang patches
    dir. Tests use this fixture to simulate TraceLens shipping the
    decoupling manifest before it actually does."""
    patches_dir = (
        tracelens_root / "examples" / "custom_workflows"
        / "inference_analysis" / "sglang_roofline_patches"
    )
    manifest = patches_dir / filename
    manifest.write_text(body, encoding="utf-8")
    return manifest


def test_load_sglang_manifest_returns_none_when_absent(tmp_path):
    """Today's TraceLens doesn't ship the manifest. The loader must
    return None (not an empty frozenset) so the caller falls back to
    the PR-C.2 minor allowlist — preserving today's behaviour
    byte-for-byte until TraceLens decides to ship the file."""
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    assert _server_patcher._load_sglang_supported_versions_from_manifest(
        patches_dir,
    ) is None


def test_load_sglang_manifest_parses_versions_skipping_comments(tmp_path):
    """Format contract: one version per line, ``#`` starts a comment,
    blank lines ignored. Whitespace around versions is stripped."""
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
    """If TraceLens explicitly ships an empty manifest (all entries
    commented out, no versions listed), the loader returns an empty
    frozenset — NOT None. Caller then rejects every version, which is
    the operator's signal that TraceLens has declared no versions
    are supported on this patch revision."""
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
    """Both ``SUPPORTED_VERSIONS.txt`` and ``SUPPORTED_VERSIONS`` are
    valid filenames; precedence is ``.txt`` first (more discoverable
    in IDEs / file-listing tooling). Both files present → ``.txt``
    wins."""
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
    """If only the no-extension variant exists, the loader picks it
    up. Lets TraceLens use the LICENSE/README naming convention if
    they prefer it."""
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
    """End-to-end gate semantics: when a manifest is shipped, IT is
    the source of truth — bypasses the hardcoded ``0.5.x`` default.
    A version that the default would accept but the manifest doesn't
    list must be rejected; a version the manifest lists but the
    default wouldn't accept must be accepted."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    (patches_dir / "SUPPORTED_VERSIONS.txt").write_text(
        "0.6.0\n0.7.0\n", encoding="utf-8",
    )
    # 0.5.9 — accepted by default allowlist, NOT listed in manifest
    # → manifest must override and reject.
    assert _server_patcher._sglang_version_accepted(
        "0.5.9", patches_dir=patches_dir,
    ) is False
    # 0.7.0 — rejected by default allowlist (0.7.x not in 0.5.x), IS
    # listed in manifest → manifest must override and accept.
    assert _server_patcher._sglang_version_accepted(
        "0.7.0", patches_dir=patches_dir,
    ) is True


def test_sglang_version_accepted_empty_manifest_rejects_all(
    tmp_path, monkeypatch,
):
    """An explicit empty manifest means "TraceLens declares no
    supported versions" — and that declaration beats the hardcoded
    default. Operators see all versions rejected, prompting them to
    investigate whether the patch set has been deprecated or split."""
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
    """Today's behaviour: TraceLens doesn't ship the manifest, so the
    helper falls back to ``_SGLANG_DEFAULT_ALLOWED_MINORS = ("0.5",)``.
    This preserves PR-C.2 byte-for-byte and is the safety net that
    prevents D.5 from regressing anyone who hasn't pulled the new
    TraceLens release."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()  # No manifest written.
    assert _server_patcher._sglang_version_accepted(
        "0.5.9", patches_dir=patches_dir,
    ) is True
    assert _server_patcher._sglang_version_accepted(
        "0.5.99", patches_dir=patches_dir,
    ) is True  # 0.5.x covers all point releases.
    assert _server_patcher._sglang_version_accepted(
        "0.6.0", patches_dir=patches_dir,
    ) is False  # 0.6 not in default allowlist; no manifest to override.


def test_sglang_version_accepted_operator_env_vars_beat_manifest(
    tmp_path, monkeypatch,
):
    """The operator escape hatch must still work even when TraceLens
    ships a manifest — an operator pinning a specific version (e.g.
    to roll back after a bad release) takes precedence over the
    vendor's declared support list."""
    patches_dir = tmp_path / "sglang_roofline_patches"
    patches_dir.mkdir()
    (patches_dir / "SUPPORTED_VERSIONS.txt").write_text(
        "0.5.9\n", encoding="utf-8",
    )
    # Operator pins 0.7.0 exactly, even though manifest doesn't list it.
    monkeypatch.setenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", "0.7.0")
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    assert _server_patcher._sglang_version_accepted(
        "0.7.0", patches_dir=patches_dir,
    ) is True
    # And the manifest-listed 0.5.9 is now REJECTED because the
    # operator's exact pinset takes over completely.
    assert _server_patcher._sglang_version_accepted(
        "0.5.9", patches_dir=patches_dir,
    ) is False


@_REQUIRES_GIT
def test_sglang_e2e_manifest_admits_version_outside_default_allowlist(
    tmp_path, monkeypatch,
):
    """Full integration: SGLang 0.7.0 (well outside the hardcoded
    ``0.5.x`` default) gets patched because TraceLens has shipped a
    manifest that explicitly lists it. This is the exact "TraceLens
    bumps support without requiring a Hyperloom code change" workflow
    the reviewer asked for."""
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS", raising=False)
    monkeypatch.delenv("HYPERLOOM_SGLANG_PATCH_ALLOWED_MINORS", raising=False)
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    # Replace the fake sglang's __version__ with 0.7.0 in both the
    # injected module and the on-disk __init__.py.
    sgl_init = apply_root / "python" / "sglang" / "__init__.py"
    sgl_init.write_text('__version__ = "0.7.0"\n', encoding="utf-8")
    _write_fake_sglang_patches(tracelens_root, count=1)
    _write_sglang_versions_manifest(tracelens_root, "0.7.0\n")

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
