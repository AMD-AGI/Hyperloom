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
real ``/wekafs/InferenceX`` or any real ``site-packages``.
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
    tracelens_root: Path, version: str, sentinel: str = "capture_torch_profiler_dir",
) -> Path:
    """Generate a minimal unified diff that adds the sentinel string
    to the fake profiler.py — `git apply` accepts it without needing
    a real .git/ directory."""
    patch_path = (
        tracelens_root
        / "examples" / "custom_workflows" / "inference_analysis"
        / "vllm_patches" / f"config_vllm_v{version}.patch"
    )
    patch_path.write_text(
        textwrap.dedent(
            f"""\
            diff --git a/vllm/config/profiler.py b/vllm/config/profiler.py
            index 0000001..0000002 100644
            --- a/vllm/config/profiler.py
            +++ b/vllm/config/profiler.py
            @@ -1,4 +1,5 @@
             # Synthetic vLLM profiler config — used only by tests.
             class ProfilerConfig:
                 profiler: str = ""
            +    {sentinel}: str = ""
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
    """SGLang versions outside `_SGLANG_SUPPORTED_VERSIONS` (today only
    "0.5.9") must fail-soft — and crucially must NOT touch the
    install tree on disk."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    apply_root = _make_fake_sglang_install(tmp_path)
    _write_fake_sglang_patches(tracelens_root)
    fake_mod = types.ModuleType("sglang")
    fake_mod.__version__ = "0.5.10"  # current Hyperloom default — unsupported
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


def test_sglang_rejects_non_editable_install_layout(tmp_path, monkeypatch):
    """A pip-wheel install where `sglang/` sits directly in site-packages
    (no `python/` parent) cannot use the `a/python/sglang/...` patches —
    fail-soft so wheel users keep working."""
    tracelens_root = _make_fake_tracelens(tmp_path)
    _write_fake_sglang_patches(tracelens_root)
    # Layout: site-packages/sglang/__init__.py (no `python/` dir).
    install = tmp_path / "site_packages" / "sglang"
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
