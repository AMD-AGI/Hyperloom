###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for kernel-source discovery (``source_env``).

Covers version fallback, native ``csrc`` detection, ``*_meta`` canonicalization,
and auto-enumeration of arbitrary kernel libraries via a temporary
``site-packages`` on ``sys.path``. No installed framework required; runnable
directly via ``python3 test_source_env.py`` (no pytest).
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from hyperloom.agents.kernel.tools import source_env


@contextmanager
def _env(**overrides: str | None):
    """Temporarily set/clear env vars, restoring prior state on exit."""
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@contextmanager
def _on_syspath(path: Path):
    """Temporarily prepend ``path`` to ``sys.path``."""
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            # Idempotent teardown: the entry may already be gone (nested contexts
            # or other cleanup removed it); a missing entry is not an error here.
            pass


def _kernel_pkg(site: Path, pkg: str, subdir: str = "csrc") -> Path:
    """Create ``site/pkg/subdir/k.cu`` with a native kernel; return the pkg dir."""
    d = site / pkg / subdir
    d.mkdir(parents=True)
    (d / "k.cu").write_text("__global__ void k(){}", encoding="utf-8")
    return site / pkg


# --- version ----------------------------------------------------------------
def test_version_reads_version_py_fallback() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "atom"
        root.mkdir()
        (root / "_version.py").write_text("__version__ = '1.2.3'\n", encoding="utf-8")
        assert source_env._version("definitely-not-installed-pkg", root) == "1.2.3"


def test_version_missing_is_empty() -> None:
    with tempfile.TemporaryDirectory() as raw:
        assert source_env._version("definitely-not-installed-pkg", Path(raw)) == ""
        assert source_env._version("definitely-not-installed-pkg", None) == ""


# --- native csrc detection --------------------------------------------------
def test_find_csrc_detects_native_dir() -> None:
    with tempfile.TemporaryDirectory() as raw:
        pkg = _kernel_pkg(Path(raw), "atom")
        roots = source_env._find_csrc(pkg)
        assert any(str(p).endswith("atom/csrc") for p in roots), roots


def test_find_csrc_ignores_python_only_dir() -> None:
    """A ``kernels`` dir with no native files is not treated as source."""
    with tempfile.TemporaryDirectory() as raw:
        pkg = Path(raw) / "pylib"
        (pkg / "kernels").mkdir(parents=True)
        (pkg / "kernels" / "impl.py").write_text("x = 1\n", encoding="utf-8")
        assert source_env._find_csrc(pkg) == ()


# --- canonicalization -------------------------------------------------------
def test_canonical_strips_meta_suffix() -> None:
    assert source_env._canonical("aiter_meta") == "aiter"
    assert source_env._canonical("atom") == "atom"


# --- auto-enumeration -------------------------------------------------------
def test_discover_auto_enumerates_new_library() -> None:
    """A brand-new kernel library on sys.path is found with no code change."""
    with tempfile.TemporaryDirectory() as raw:
        site = Path(raw) / "site-packages"
        site.mkdir()
        _kernel_pkg(site, "atom")
        with _on_syspath(site), _env(HYPERLOOM_DISCOVER_ONLY="atom", HYPERLOOM_FRAMEWORK_SOURCE_ROOTS=None):
            fw = source_env.discover_frameworks()
        assert "atom" in fw, fw
        assert any(str(p).endswith("atom/csrc") for p in fw["atom"].csrc_roots)


def test_discover_merges_meta_sibling_into_base() -> None:
    """``aiter_meta/csrc`` is attributed to ``aiter`` (keeping aiter's version)."""
    with tempfile.TemporaryDirectory() as raw:
        site = Path(raw) / "site-packages"
        site.mkdir()
        (site / "aiter").mkdir()
        (site / "aiter" / "_version.py").write_text("__version__='0.1.99'\n", encoding="utf-8")
        _kernel_pkg(site, "aiter_meta")
        with (
            _on_syspath(site),
            _env(
                HYPERLOOM_DISCOVER_ONLY="aiter",
                HYPERLOOM_FRAMEWORK_SOURCE_ROOTS=f"aiter={site / 'aiter'}",
            ),
        ):
            fw = source_env.discover_frameworks()
        assert "aiter" in fw and "aiter_meta" not in fw, fw
        assert fw["aiter"].version == "0.1.99", fw["aiter"].version
        assert any(str(p).endswith("aiter_meta/csrc") for p in fw["aiter"].csrc_roots)


def test_discover_only_filters_out_others() -> None:
    """``DISCOVER_ONLY`` excludes kernel libraries not named."""
    with tempfile.TemporaryDirectory() as raw:
        site = Path(raw) / "site-packages"
        site.mkdir()
        _kernel_pkg(site, "atom")
        _kernel_pkg(site, "flydsl")
        with _on_syspath(site), _env(HYPERLOOM_DISCOVER_ONLY="atom", HYPERLOOM_FRAMEWORK_SOURCE_ROOTS=None):
            fw = source_env.discover_frameworks()
        assert "atom" in fw and "flydsl" not in fw, fw


_TESTS = [
    test_version_reads_version_py_fallback,
    test_version_missing_is_empty,
    test_find_csrc_detects_native_dir,
    test_find_csrc_ignores_python_only_dir,
    test_canonical_strips_meta_suffix,
    test_discover_auto_enumerates_new_library,
    test_discover_merges_meta_sibling_into_base,
    test_discover_only_filters_out_others,
]


def _run_all() -> int:
    """Run every check, print PASS/FAIL per test, return a process exit code."""
    failures = 0
    for test in _TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report any unexpected error.
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    total = len(_TESTS)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
