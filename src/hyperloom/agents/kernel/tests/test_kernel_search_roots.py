###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Guards on the framework roots the grep tier searches for kernel source."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import tracelens_analysis as tl  # noqa: E402


class TestInstalledPackageDir:
    def test_locates_a_real_package(self):
        located = tl._installed_package_dir("json")
        assert located and Path(located).is_dir()

    def test_absent_package_returns_empty(self):
        assert tl._installed_package_dir("no_such_package_xyz") == ""

    def test_rejects_non_identifier(self):
        """Guards a path fragment being passed where a package name belongs."""
        assert tl._installed_package_dir("/usr/local/lib") == ""
        assert tl._installed_package_dir("") == ""


class TestDiscoverKernelSearchRoots:
    def test_drops_roots_that_do_not_exist(self, monkeypatch, tmp_path):
        present = tmp_path / "vllm"
        present.mkdir()
        monkeypatch.setattr(
            tl,
            "_resolve_kernel_search_roots",
            lambda: (f"{present}/", "/gone/aiter/"),
        )
        tl.kernel_search_roots.cache_clear()
        assert tl.kernel_search_roots() == (str(present),)

    def test_strips_trailing_separator(self, monkeypatch, tmp_path):
        """Callers match these as prefixes without a separator of their own."""
        root = tmp_path / "aiter"
        root.mkdir()
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", lambda: (f"{root}/",))
        tl.kernel_search_roots.cache_clear()
        assert tl.kernel_search_roots() == (str(root),)

    def test_deduplicates_while_preserving_order(self, monkeypatch, tmp_path):
        first = tmp_path / "vllm"
        second = tmp_path / "aiter"
        first.mkdir()
        second.mkdir()
        monkeypatch.setattr(
            tl,
            "_resolve_kernel_search_roots",
            lambda: (f"{first}/", f"{second}/", str(first), f"{first}//"),
        )
        tl.kernel_search_roots.cache_clear()
        assert tl.kernel_search_roots() == (str(first), str(second))

    def test_falls_back_to_local_discovery_without_the_orchestrator(self, monkeypatch, tmp_path):
        """Standalone CLI use must still find the installed frameworks."""
        located = tmp_path / "aiter"
        located.mkdir()
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", None)
        monkeypatch.setattr(tl, "_KERNEL_SOURCE_PACKAGES", ("aiter",))
        monkeypatch.setattr(tl, "_FALLBACK_SEARCH_ROOTS", ())
        monkeypatch.setattr(tl, "_installed_package_dir", lambda package: str(located) if package == "aiter" else "")
        tl.kernel_search_roots.cache_clear()
        assert tl.kernel_search_roots() == (str(located),)

    def test_pinned_layouts_are_a_last_resort_not_a_requirement(self, monkeypatch, tmp_path):
        """A pinned root is used only when it exists, never assumed."""
        checkout = tmp_path / "sgl-workspace" / "vllm"
        checkout.mkdir(parents=True)
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", None)
        monkeypatch.setattr(tl, "_KERNEL_SOURCE_PACKAGES", ())
        monkeypatch.setattr(tl, "_FALLBACK_SEARCH_ROOTS", (str(checkout), "/sgl-workspace/gone"))
        tl.kernel_search_roots.cache_clear()
        assert tl.kernel_search_roots() == (str(checkout),)

    def test_an_empty_central_answer_still_reaches_local_discovery(self, monkeypatch, tmp_path):
        """\"The resolver imported\" is not \"the resolver found something\"."""
        located = tmp_path / "sgl_kernel"
        located.mkdir()
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", lambda: ())
        monkeypatch.setattr(tl, "_KERNEL_SOURCE_PACKAGES", ("sgl_kernel",))
        monkeypatch.setattr(tl, "_FALLBACK_SEARCH_ROOTS", ())
        monkeypatch.setattr(
            tl,
            "_installed_package_dir",
            lambda package: str(located) if package == "sgl_kernel" else "",
        )
        tl.kernel_search_roots.cache_clear()
        assert tl.kernel_search_roots() == (str(located),)

    def test_a_central_answer_is_not_widened_by_the_fallback(self, monkeypatch, tmp_path):
        """The resolver stays authoritative whenever it has an answer."""
        central = tmp_path / "vllm"
        central.mkdir()
        local = tmp_path / "aiter"
        local.mkdir()
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", lambda: (str(central),))
        monkeypatch.setattr(tl, "_KERNEL_SOURCE_PACKAGES", ("aiter",))
        monkeypatch.setattr(tl, "_FALLBACK_SEARCH_ROOTS", (str(local),))
        monkeypatch.setattr(tl, "_installed_package_dir", lambda _package: str(local))
        tl.kernel_search_roots.cache_clear()
        assert tl.kernel_search_roots() == (str(central),)

    def test_no_searchable_root_is_reported_loudly(self, monkeypatch, caplog):
        """An unsearchable host must not look like a healthy one."""
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", lambda: ("/gone/vllm/",))
        tl.kernel_search_roots.cache_clear()
        with caplog.at_level(logging.WARNING, logger=tl.log.name):
            assert tl.kernel_search_roots() == ()
        assert "no framework source root" in caplog.text

    def teardown_method(self):
        """Drop the cached roots so the next test resolves them afresh."""
        tl.kernel_search_roots.cache_clear()


class TestRefreshKernelSearchRoots:
    """A framework installed after import must still become searchable."""

    def test_a_root_that_appears_later_is_picked_up(self, monkeypatch, tmp_path):
        appears = tmp_path / "aiter"
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", lambda: (str(appears),))
        tl.kernel_search_roots.cache_clear()
        assert tl.kernel_search_roots() == ()

        appears.mkdir()

        # Still the cached answer until the run boundary drops it.
        assert tl.kernel_search_roots() == ()
        assert tl.refresh_kernel_search_roots() == (str(appears),)
        assert tl.kernel_search_roots() == (str(appears),)

    def test_everything_derived_from_the_roots_is_dropped_with_them(self, monkeypatch, tmp_path):
        """A stale derived cache reproduces the bug one step further on."""
        checkout = tmp_path / "sgl-workspace" / "aiter"
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", lambda: (str(checkout),))
        tl.kernel_search_roots.cache_clear()
        tl._harness_search_bases.cache_clear()
        assert tl._harness_search_bases() == ()

        checkout.mkdir(parents=True)
        tl.refresh_kernel_search_roots()

        assert tl._harness_search_bases() == (str(checkout.parent), str(checkout))

    def teardown_method(self):
        tl.kernel_search_roots.cache_clear()
        tl._harness_search_bases.cache_clear()


class TestThePackageListHasOneOwner:
    """Two lists of kernel-source packages is one list that goes stale."""

    def test_the_tool_defers_to_the_orchestrator_list(self):
        from hyperloom.orchestrator.framework.paths import FRAMEWORK_SOURCE_PACKAGES

        assert tl._KERNEL_SOURCE_PACKAGES == FRAMEWORK_SOURCE_PACKAGES

    def test_the_standalone_default_does_not_drift_from_it(self):
        """The literal is the standalone fallback, not a competing answer."""
        from hyperloom.orchestrator.framework.paths import FRAMEWORK_SOURCE_PACKAGES

        assert set(tl._STANDALONE_KERNEL_SOURCE_PACKAGES) == set(FRAMEWORK_SOURCE_PACKAGES)
