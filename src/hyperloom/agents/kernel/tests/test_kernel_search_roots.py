###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Guards on the framework roots the grep tier searches for kernel source.

``KNOWN_SEARCH_ROOTS`` used to be a literal naming one container's layout
(``/sgl-workspace/...`` and a python3.10 venv). On a host that installs the
frameworks anywhere else -- a wheel under ``dist-packages``, a different Python
minor -- every root was absent, so the grep tier searched nothing, returned no
hit for any kernel, and the LLM tiers got an empty shortlist and a validation
gate that rejected every path outside those absent roots. The run still
succeeded: it reported zero routable kernels, which reads exactly like a trace
with nothing worth optimizing, and kernel-opt sat idle with no work to dispatch.

These tests pin the properties that keep that silent failure from returning:
roots are discovered at runtime, non-existent ones never survive, and a host
with nothing installed says so instead of looking healthy.
"""

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
        tl._discover_kernel_search_roots.cache_clear()
        assert tl._discover_kernel_search_roots() == (str(present),)

    def test_strips_trailing_separator(self, monkeypatch, tmp_path):
        """Callers match these as prefixes without a separator of their own."""
        root = tmp_path / "aiter"
        root.mkdir()
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", lambda: (f"{root}/",))
        tl._discover_kernel_search_roots.cache_clear()
        assert tl._discover_kernel_search_roots() == (str(root),)

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
        tl._discover_kernel_search_roots.cache_clear()
        assert tl._discover_kernel_search_roots() == (str(first), str(second))

    def test_falls_back_to_local_discovery_without_the_orchestrator(
        self, monkeypatch, tmp_path
    ):
        """Standalone CLI use must still find the installed frameworks."""
        located = tmp_path / "aiter"
        located.mkdir()
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", None)
        monkeypatch.setattr(tl, "_KERNEL_SOURCE_PACKAGES", ("aiter",))
        monkeypatch.setattr(tl, "_FALLBACK_SEARCH_ROOTS", ())
        monkeypatch.setattr(
            tl, "_installed_package_dir", lambda package: str(located) if package == "aiter" else ""
        )
        tl._discover_kernel_search_roots.cache_clear()
        assert tl._discover_kernel_search_roots() == (str(located),)

    def test_pinned_layouts_are_a_last_resort_not_a_requirement(
        self, monkeypatch, tmp_path
    ):
        """A pinned root is used only when it exists, never assumed."""
        checkout = tmp_path / "sgl-workspace" / "vllm"
        checkout.mkdir(parents=True)
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", None)
        monkeypatch.setattr(tl, "_KERNEL_SOURCE_PACKAGES", ())
        monkeypatch.setattr(
            tl, "_FALLBACK_SEARCH_ROOTS", (str(checkout), "/sgl-workspace/gone")
        )
        tl._discover_kernel_search_roots.cache_clear()
        assert tl._discover_kernel_search_roots() == (str(checkout),)

    def test_no_searchable_root_is_reported_loudly(self, monkeypatch, caplog):
        """An unsearchable host must not look like a healthy one."""
        monkeypatch.setattr(tl, "_resolve_kernel_search_roots", lambda: ("/gone/vllm/",))
        tl._discover_kernel_search_roots.cache_clear()
        with caplog.at_level(logging.WARNING, logger=tl.log.name):
            assert tl._discover_kernel_search_roots() == ()
        assert "no framework source root" in caplog.text

    def teardown_method(self):
        """Drop the cached roots so the next test resolves them afresh."""
        tl._discover_kernel_search_roots.cache_clear()
