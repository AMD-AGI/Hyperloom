# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for framework source-scope path-resolution helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.framework import paths as fp
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.orchestrator.framework.paths import (
    probe_framework_source_roots_for_env,
    resolve_source_file_allowlist,
)
from hyperloom.orchestrator.prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    build_orchestration_prompt,
)
from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir


@pytest.fixture(autouse=True)
def _clean_framework_env(monkeypatch):
    """Reset every env var the helpers read."""
    for key in (
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        "INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS",
        "INFERENCE_OPTIMIZER_VLLM_ARG_UTILS",
        "INFERENCE_OPTIMIZER_ATOM_ARG_UTILS",
        "VIRTUAL_ENV",
        "VLLM_VENV_ROOT",
        "DSL2_ROOT",
        "FLYDSL_ROOT",
        "FLYDSL_EXTRA_SOURCE_DIRS",
    ):
        monkeypatch.delenv(key, raising=False)


class TestNormalizeRoot:
    def test_appends_trailing_slash(self):
        assert fp._normalize_root("/sgl-workspace/aiter") == "/sgl-workspace/aiter/"

    def test_preserves_existing_trailing_slash(self):
        assert fp._normalize_root("/foo/") == "/foo/"

    def test_empty_input_returns_empty(self):
        assert fp._normalize_root("") == ""
        assert fp._normalize_root("   ") == ""


class TestResolveSourceFileAllowlist:
    def test_default_when_env_empty(self, monkeypatch):
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: ())
        monkeypatch.setattr(fp, "_discover_installed_package_roots", lambda: ())
        # The enablement ROCm/HIP roots are always merged (default-on capability).
        assert fp.resolve_source_file_allowlist() == (fp._DEFAULT_SOURCE_ROOTS + fp._ROCM_HIP_SOURCE_ROOTS)

    def test_merges_discovered_roots(self, monkeypatch):
        monkeypatch.setattr(fp, "_discover_installed_package_roots", lambda: ("/venv/site-packages/",))
        monkeypatch.setattr(
            fp, "_discover_installed_framework_roots", lambda: ("/usr/local/lib/python3.12/dist-packages/vllm/",)
        )
        roots = fp.resolve_source_file_allowlist()
        assert fp._DEFAULT_SOURCE_ROOTS[0] in roots
        assert "/venv/site-packages/" in roots
        assert "/usr/local/lib/python3.12/dist-packages/vllm/" in roots

    def test_appends_extra_roots_unique_in_order(self, monkeypatch):
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: ())
        monkeypatch.setattr(fp, "_discover_installed_package_roots", lambda: ())
        monkeypatch.setenv(
            "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
            "/opt/custom/sglang:/sgl-workspace/aiter:/opt/other/vllm",
        )
        out = fp.resolve_source_file_allowlist()
        n = len(fp._DEFAULT_SOURCE_ROOTS)
        assert out[:n] == fp._DEFAULT_SOURCE_ROOTS
        assert "/opt/custom/sglang/" in out
        assert "/opt/other/vllm/" in out
        assert out.count("/sgl-workspace/aiter/") == 1

    def test_discovers_active_site_packages_root(self, tmp_path, monkeypatch):
        root = tmp_path / "lib" / "python3.12" / "site-packages"
        root.mkdir(parents=True)
        monkeypatch.setattr(fp.site, "getsitepackages", lambda: [str(root)])
        monkeypatch.setattr(fp.site, "getusersitepackages", lambda: "")
        monkeypatch.setattr(fp.sysconfig, "get_path", lambda _key: None)
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: ())
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())
        monkeypatch.setattr(fp, "_ROCM_HIP_SOURCE_ROOTS", ())
        assert f"{root}/" in fp.resolve_source_file_allowlist()


class TestFindSpecOrigin:
    def test_returns_none_when_spec_missing(self, monkeypatch):
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: None,
        )
        assert fp._find_spec_origin("does_not_matter") is None

    def test_returns_none_when_origin_missing(self, monkeypatch):
        spec = SimpleNamespace(origin=None)
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        assert fp._find_spec_origin("pkg") is None

    def test_init_origin_returns_parent_dir(self, monkeypatch, tmp_path):
        init = tmp_path / "pkg" / "__init__.py"
        init.parent.mkdir(parents=True)
        init.write_text("# stub")
        spec = SimpleNamespace(origin=str(init))
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        assert fp._find_spec_origin("pkg") == init.parent

    def test_handles_find_spec_raising(self, monkeypatch):
        def boom(_):
            raise ValueError("malformed")

        monkeypatch.setattr(importlib.util, "find_spec", boom)
        assert fp._find_spec_origin("pkg") is None


class TestGlobInstallPackageRoots:
    def test_finds_dist_packages_under_usr_local(self, tmp_path, monkeypatch):
        base = tmp_path / "usr_local_lib" / "python3.12" / "dist-packages"
        (base / "vllm").mkdir(parents=True)
        (base / "aiter_meta").mkdir()
        monkeypatch.setattr(
            fp,
            "_INSTALL_GLOB_PARENTS",
            (tmp_path / "usr_local_lib",),
        )
        roots = fp._glob_install_package_roots()
        assert any("dist-packages/vllm/" in r for r in roots)
        assert any("dist-packages/aiter_meta/" in r for r in roots)


class TestResolvePatchTargetRoots:
    def test_includes_static_fallback_when_discovery_empty(self, monkeypatch):
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: ())
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", raising=False)
        roots = fp.resolve_patch_target_roots()
        assert "/usr/local/lib/python3.12/dist-packages/vllm/" in roots
        assert "/aiter_meta/csrc/" in roots

    def test_includes_flydsl_roots(self):
        assert "/opt/flydsl/" in fp.resolve_patch_target_roots()

    @pytest.mark.parametrize("env_key", ["DSL2_ROOT", "FLYDSL_ROOT"])
    def test_honours_flydsl_root_env(self, monkeypatch, env_key):
        monkeypatch.setenv(env_key, "/checkouts/FlyDSL")
        roots = fp.resolve_patch_target_roots()
        # Both variants: the apply gate matches a lower-cased path verbatim,
        # while a path-resolving consumer needs the real case.
        assert "/checkouts/FlyDSL/" in roots
        assert "/checkouts/flydsl/" in roots

    def test_source_file_allowlist_excludes_flydsl(self, monkeypatch):
        monkeypatch.setenv("FLYDSL_ROOT", "/checkouts/flydsl")
        allowlist = fp.resolve_source_file_allowlist()
        assert not any("flydsl" in root.lower() for root in allowlist)


class TestResolveKernelSearchRoots:
    def test_drops_roots_that_do_not_exist(self, monkeypatch, tmp_path):
        """A pinned root that no longer exists must not reach the caller.

        Grepping an absent directory yields no hits, which is indistinguishable
        from a kernel whose source is genuinely absent -- the exact failure that
        silently emptied kernel-opt's candidate list.
        """
        present = tmp_path / "vllm"
        present.mkdir()
        monkeypatch.setattr(
            fp,
            "_discover_installed_framework_roots",
            lambda: (f"{present}/", "/gone/aiter/"),
        )
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())
        monkeypatch.setattr(fp, "resolve_flydsl_source_roots", lambda: ())
        assert fp.resolve_kernel_search_roots() == (f"{present}/",)

    def test_excludes_bare_site_packages_parents(self, monkeypatch, tmp_path):
        """Only package dirs, never the whole site-packages tree.

        The allowlist reports the parent so an editability check can contain any
        installed file; grepping it would scan every wheel on the host.
        """
        parent = tmp_path / "dist-packages"
        (parent / "vllm").mkdir(parents=True)
        monkeypatch.setattr(fp, "_discover_installed_package_roots", lambda: (f"{parent}/",))
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: (f"{parent}/vllm/",))
        roots = fp.resolve_kernel_search_roots()
        assert f"{parent}/vllm/" in roots
        assert f"{parent}/" not in roots

    def test_empty_when_nothing_is_installed(self, monkeypatch):
        """No searchable root is reported as such, not as a silent success."""
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: ())
        monkeypatch.setattr(fp, "_discover_scriptable_repo_roots", lambda: ())
        monkeypatch.setattr(fp, "_discover_explicit_framework_root", lambda: ())
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ("/gone/vllm/",))
        monkeypatch.setattr(fp, "resolve_flydsl_source_roots", lambda: ("/gone/flydsl/",))
        assert fp.resolve_kernel_search_roots() == ()

    def test_includes_explicit_framework_checkout(self, monkeypatch, tmp_path):
        """An editable checkout is invisible to importlib; the env var finds it."""
        checkout = tmp_path / "my-vllm"
        checkout.mkdir()
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: ())
        monkeypatch.setenv(fp.GENERIC_FRAMEWORK_ROOT_ENV, str(checkout))
        assert f"{checkout}/" in fp.resolve_kernel_search_roots()


class TestEveryKernelSourcePackageIsDiscoverable:
    """One package list, reached by all three discovery mechanisms.

    ``sgl_kernel`` holds SGLang's kernel sources and was named by the tool that
    greps for them but by none of the discovery paths here. Because this
    resolver imports successfully in every non-standalone run, the tool's own
    list was never consulted -- so a host with a standalone ``sgl_kernel`` wheel
    reported it as searched and never searched it. A package present in only
    some of the three mechanisms is the shape of that bug, so the tests below
    assert all three derive from the same tuple.
    """

    def test_sgl_kernel_is_a_framework_source_package(self):
        assert "sgl_kernel" in fp.FRAMEWORK_SOURCE_PACKAGES
        assert "aiter_meta" in fp.FRAMEWORK_SOURCE_PACKAGES

    def test_importlib_discovery_covers_every_package(self, monkeypatch, tmp_path):
        """A standalone wheel is found by spec origin alone."""
        located = tmp_path / "sgl_kernel"
        located.mkdir()
        monkeypatch.setattr(
            fp,
            "_find_spec_origin",
            lambda name: str(located) if name == "sgl_kernel" else None,
        )
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr(fp, "_glob_install_package_roots", lambda: ())
        assert f"{located}/" in fp._discover_installed_framework_roots()

    def test_the_venv_glob_covers_every_package(self, monkeypatch, tmp_path):
        """A wheel under ``$VIRTUAL_ENV`` that importlib cannot see."""
        venv = tmp_path / "venv"
        installed = venv / "lib" / "python3.12" / "site-packages" / "sgl_kernel"
        installed.mkdir(parents=True)
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: None)
        monkeypatch.setattr(fp, "_glob_install_package_roots", lambda: ())
        assert f"{installed}/" in fp._discover_installed_framework_roots()

    def test_the_install_glob_covers_every_package(self, monkeypatch, tmp_path):
        """Both ``site-`` and ``dist-`` spellings, under a bare install parent."""
        for flavour in ("site", "dist"):
            installed = tmp_path / flavour / "lib" / "python3.12" / f"{flavour}-packages" / "sgl_kernel"
            installed.mkdir(parents=True)
            monkeypatch.setattr(fp, "_INSTALL_GLOB_PARENTS", (tmp_path / flavour / "lib",))
            assert f"{installed}/" in fp._glob_install_package_roots(), flavour

    def test_a_standalone_sgl_kernel_wheel_becomes_a_search_root(self, monkeypatch, tmp_path):
        """End to end: the only framework on the host is ``sgl_kernel``."""
        installed = tmp_path / "lib" / "python3.12" / "dist-packages" / "sgl_kernel"
        installed.mkdir(parents=True)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: None)
        monkeypatch.setattr(fp, "_INSTALL_GLOB_PARENTS", (tmp_path / "lib",))
        monkeypatch.setattr(fp, "_discover_scriptable_repo_roots", lambda: ())
        monkeypatch.setattr(fp, "_discover_explicit_framework_root", lambda: ())
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())
        monkeypatch.setattr(fp, "resolve_flydsl_source_roots", lambda: ())
        assert fp.resolve_kernel_search_roots() == (f"{installed}/",)


class TestFlydslExtraSourceDirs:
    def test_lists_only_roots_that_exist(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FLYDSL_ROOT", str(tmp_path / "missing"))
        assert fp.flydsl_extra_source_dirs() == ""

        real = tmp_path / "flydsl"
        real.mkdir()
        monkeypatch.setenv("FLYDSL_ROOT", str(real))
        assert fp.flydsl_extra_source_dirs() == str(real)

    def test_preserves_an_operator_supplied_value(self, monkeypatch, tmp_path):
        real = tmp_path / "flydsl"
        real.mkdir()
        monkeypatch.setenv("FLYDSL_ROOT", str(real))
        monkeypatch.setenv("FLYDSL_EXTRA_SOURCE_DIRS", "/custom/dir")
        assert fp.flydsl_extra_source_dirs() == f"/custom/dir:{real}"


class TestProbeFrameworkSourceRootsForEnv:
    def test_returns_existing_dirs_only(self, tmp_path, monkeypatch):
        present = tmp_path / "fake_root"
        present.mkdir()
        monkeypatch.setattr(
            fp,
            "_discover_installed_framework_roots",
            lambda: (
                f"{present}/",
                f"{tmp_path / 'missing'}/",
            ),
        )
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())
        monkeypatch.setattr(fp, "_discover_installed_package_roots", lambda: ())
        # Isolate from the always-on enablement ROCm/HIP root (may exist on disk).
        monkeypatch.setattr(fp, "_ROCM_HIP_SOURCE_ROOTS", ())
        result = fp.probe_framework_source_roots_for_env()
        assert result == f"{present}/"

    def test_includes_site_packages_when_virtual_env_set(
        self,
        tmp_path,
        monkeypatch,
    ):
        venv = tmp_path / "venv"
        site = venv / "lib" / "python3.12" / "site-packages"
        for name in ("vllm", "sglang", "aiter", "aiter_meta"):
            (site / name).mkdir(parents=True)
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: None)
        monkeypatch.setattr(fp, "_glob_install_package_roots", lambda: ())
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        result = fp.probe_framework_source_roots_for_env()
        for name in ("vllm", "sglang", "aiter", "aiter_meta"):
            assert f"{name}/" in result

    def test_isolated_vllm_venv_root_fallback(self, tmp_path, monkeypatch):
        # Isolated vLLM: main VIRTUAL_ENV has no vllm; VLLM_VENV_ROOT points at
        # the isolated venv holding vllm + split AITER, which must be discovered.
        main_venv = tmp_path / "opt-venv"
        (main_venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
        iso_venv = tmp_path / "vllm-venv"
        iso_site = iso_venv / "lib" / "python3.12" / "site-packages"
        for name in ("vllm", "aiter", "aiter_meta"):
            (iso_site / name).mkdir(parents=True)
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: None)
        monkeypatch.setattr(fp, "_glob_install_package_roots", lambda: ())
        monkeypatch.setenv("VIRTUAL_ENV", str(main_venv))
        monkeypatch.setenv("VLLM_VENV_ROOT", str(iso_venv))
        result = fp._discover_installed_framework_roots()
        assert any(r.endswith("/vllm/") for r in result)
        assert any(r.endswith("/aiter/") for r in result)
        assert any(r.endswith("/aiter_meta/") for r in result)

    def test_dedupes_origins_against_defaults(self, tmp_path, monkeypatch):
        shared = tmp_path / "shared"
        shared.mkdir()
        monkeypatch.setattr(
            fp,
            "_DEFAULT_SOURCE_ROOTS",
            (f"{shared}/",),
        )
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: shared)
        monkeypatch.setattr(fp, "_discover_installed_package_roots", lambda: ())
        monkeypatch.setattr(fp, "_glob_install_package_roots", lambda: ())
        monkeypatch.setattr(fp, "_ROCM_HIP_SOURCE_ROOTS", ())
        result = fp.probe_framework_source_roots_for_env()
        assert result == f"{shared}/"


# xdit enablement


class TestDefaultSourceRootsIncludesXdit:
    def test_xdit_root_present_in_defaults(self):
        """/app/xDiT/ must be in the PolicyGate source-file allowlist."""
        assert any("/app/xDiT" in r for r in fp._DEFAULT_SOURCE_ROOTS), (
            f"_DEFAULT_SOURCE_ROOTS missing xDiT entry: {fp._DEFAULT_SOURCE_ROOTS!r}"
        )

    def test_xdit_root_visible_in_resolve_allowlist(self):
        """The public resolver must also surface the xDiT root."""
        out = fp.resolve_source_file_allowlist()
        assert any("/app/xDiT" in r for r in out)

    def test_xfuser_in_framework_packages(self):
        """xfuser must be in _FRAMEWORK_PACKAGES for importlib discovery."""
        assert "xfuser" in fp._FRAMEWORK_PACKAGES

    def test_xdit_in_framework_buckets(self):
        """xdit must be in _FRAMEWORK_BUCKETS for summarise_framework_root_discovery."""
        assert "xdit" in fp._FRAMEWORK_BUCKETS

    def test_custom_in_framework_buckets(self):
        """custom must be in _FRAMEWORK_BUCKETS for root discovery summaries."""
        assert "custom" in fp._FRAMEWORK_BUCKETS

    def test_xdit_in_static_patch_fallback_roots(self):
        """/app/xDiT/ must be in the static patch fallback roots."""
        assert any("/app/xDiT" in r for r in fp._STATIC_PATCH_FALLBACK_ROOTS)


class TestScriptableRepoRootDiscovery:
    """A scriptable framework runs from a checkout, not an installed package.

    A live session probed the framework as ``missing`` with the checkout
    checkout on disk, so PolicyGate would have rejected any patch against
    ``hyvideo/`` and framework-agent had no source to work on.
    """

    def test_repo_path_env_lands_in_allowlist(self, tmp_path, monkeypatch):
        checkout = tmp_path / "my-framework"
        (checkout / "hyvideo").mkdir(parents=True)
        monkeypatch.setenv("CUSTOM_REPO_PATH", str(checkout))

        assert f"{checkout}/" in fp.resolve_source_file_allowlist()

    def test_dir_alias_also_discovered(self, tmp_path, monkeypatch):
        checkout = tmp_path / "my-framework"
        checkout.mkdir()
        monkeypatch.delenv("CUSTOM_REPO_PATH", raising=False)
        monkeypatch.setenv("CUSTOM_DIR", str(checkout))

        assert f"{checkout}/" in fp.resolve_source_file_allowlist()

    def test_missing_checkout_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CUSTOM_REPO_PATH", str(tmp_path / "absent"))

        assert not any("absent" in r for r in fp.resolve_source_file_allowlist())


class TestGenericFrameworkRepoPath:
    """A session is single-framework, so the operator should not need the prefix.

    ``<FRAMEWORK>_REPO_PATH`` requires knowing the framework name before the right
    variable can be set, and switching frameworks means switching variable names —
    for a value that cannot collide, since the CLI locks ``$FRAMEWORK`` for the run.
    The generic form is also the only way to point at a framework that is neither
    pip-installed nor registered as scriptable, such as an editable vllm checkout.
    """

    def test_generic_env_lands_in_allowlist(self, tmp_path, monkeypatch):
        checkout = tmp_path / "some-framework"
        (checkout / "pkg").mkdir(parents=True)
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(checkout))

        assert f"{checkout}/" in fp.resolve_source_file_allowlist()

    def test_generic_env_works_for_a_non_scriptable_framework(self, tmp_path, monkeypatch):
        """An editable vllm tree is not discoverable by importlib or site-packages."""
        checkout = tmp_path / "vllm-src"
        (checkout / "vllm").mkdir(parents=True)
        monkeypatch.delenv("CUSTOM_REPO_PATH", raising=False)
        monkeypatch.setenv("FRAMEWORK", "vllm")
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(checkout))

        assert f"{checkout}/" in fp.resolve_source_file_allowlist()

    def test_prefixed_value_still_wins_so_nothing_existing_changes(self, tmp_path, monkeypatch):
        """Both are accepted, and the prefixed one keeps its precedence."""
        prefixed = tmp_path / "prefixed"
        (prefixed / "hyvideo").mkdir(parents=True)
        generic = tmp_path / "generic"
        generic.mkdir()
        monkeypatch.setenv("CUSTOM_REPO_PATH", str(prefixed))
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(generic))

        roots = fp.resolve_source_file_allowlist()
        assert f"{prefixed}/" in roots
        assert roots.index(f"{prefixed}/") < roots.index(f"{generic}/")

    def test_missing_generic_checkout_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(tmp_path / "absent"))

        assert not any("absent" in r for r in fp.resolve_source_file_allowlist())

    def test_summary_accepts_repo_dirname(self, tmp_path, monkeypatch):
        """The checkout dir is xDiT, not xdit — summary must still say ok."""
        checkout = tmp_path / "xDiT"
        checkout.mkdir()
        monkeypatch.setenv("XDIT_REPO_PATH", str(checkout))

        summary = fp.summarise_framework_root_discovery(fp.probe_framework_source_roots_for_env())

        assert "xdit=ok" in summary


class TestProbeIncludesXditWhenInstalled:
    def test_xfuser_picked_up_via_find_spec(self, tmp_path, monkeypatch):
        """A real ``find_spec('xfuser')`` origin is included."""
        origin = tmp_path / "xfuser_pkg"
        origin.mkdir()
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())

        spec_map = {"xfuser": origin}
        monkeypatch.setattr(
            fp,
            "_find_spec_origin",
            lambda name: spec_map.get(name),
        )
        result = fp.probe_framework_source_roots_for_env()
        assert f"{origin}/" in result

    def test_xfuser_picked_up_via_venv_site_packages(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A wheel-installed xfuser is picked up via the VIRTUAL_ENV glob."""
        venv = tmp_path / "venv"
        site = venv / "lib" / "python3.12" / "site-packages"
        (site / "xfuser").mkdir(parents=True)
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: None)
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        result = fp.probe_framework_source_roots_for_env()
        assert "xfuser/" in result


# atom enablement


class TestDefaultSourceRootsIncludesAtom:
    def test_atom_root_present_in_defaults(self):
        """/app/ATOM/atom/ must be in the PolicyGate source-file allowlist."""
        assert any("/app/ATOM/atom" in r for r in fp._DEFAULT_SOURCE_ROOTS), (
            f"_DEFAULT_SOURCE_ROOTS missing atom entry: {fp._DEFAULT_SOURCE_ROOTS!r}"
        )

    def test_atom_root_visible_in_resolve_allowlist(self):
        """The public resolver must also surface the atom root."""
        out = fp.resolve_source_file_allowlist()
        assert any("/app/ATOM/atom" in r for r in out)


class TestProbeIncludesAtomWhenInstalled:
    def test_atom_picked_up_via_find_spec(self, tmp_path, monkeypatch):
        """A real ``find_spec('atom')`` origin is included even without a /app/ATOM/atom/ default root."""
        origin = tmp_path / "atom_pkg"
        origin.mkdir()
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())

        spec_map = {"atom": origin}
        monkeypatch.setattr(
            fp,
            "_find_spec_origin",
            lambda name: spec_map.get(name),
        )
        result = fp.probe_framework_source_roots_for_env()
        assert f"{origin}/" in result

    def test_atom_picked_up_via_venv_site_packages(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A wheel-installed atom is picked up via the VIRTUAL_ENV ``python*/site-packages/atom`` glob."""
        venv = tmp_path / "venv"
        site = venv / "lib" / "python3.12" / "site-packages"
        (site / "atom").mkdir(parents=True)
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: None)
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        result = fp.probe_framework_source_roots_for_env()
        assert "atom/" in result


class TestSummariseFrameworkRootDiscovery:
    def test_buckets_atom_ok(self):
        """The install.sh log helper reports atom=ok when an atom root appears in the discovery string."""
        out = fp.summarise_framework_root_discovery(
            "/sgl-workspace/aiter/:/sgl-workspace/sglang/:/sgl-workspace/vllm/:/app/ATOM/atom/"
        )
        assert "atom=ok" in out
        assert "sglang=ok" in out
        assert "vllm=ok" in out
        assert "aiter=ok" in out

    def test_buckets_xdit_ok(self):
        """Reports xdit=ok when /app/xDiT/ appears in the discovery string."""
        out = fp.summarise_framework_root_discovery("/sgl-workspace/aiter/:/app/xDiT/")
        assert "xdit=ok" in out
        assert "aiter=ok" in out

    def test_buckets_xdit_missing_when_absent(self):
        out = fp.summarise_framework_root_discovery("/sgl-workspace/aiter/:/sgl-workspace/sglang/")
        assert "xdit=missing" in out

    def test_buckets_atom_missing_on_non_atom_box(self):
        out = fp.summarise_framework_root_discovery("/sgl-workspace/aiter/:/sgl-workspace/sglang/:/sgl-workspace/vllm/")
        assert "atom=missing" in out
        assert "sglang=ok" in out

    def test_handles_empty_input(self):
        out = fp.summarise_framework_root_discovery("")
        assert "atom=missing" in out
        assert "sglang=missing" in out
        assert "vllm=missing" in out
        assert "aiter=missing" in out
        assert "xdit=missing" in out

    def test_does_not_substring_match_unrelated_paths(self):
        """Only paths whose last directory IS ``atom`` count; a substring like ``atomic_kernel`` must not."""
        out = fp.summarise_framework_root_discovery("/sgl-workspace/atomic_kernel/")
        assert "atom=missing" in out

    def test_does_not_substring_match_xdit_unrelated(self):
        """A path like ``/xdit_tools/`` must not match the ``xdit`` bucket."""
        out = fp.summarise_framework_root_discovery("/sgl-workspace/xdit_tools/")
        assert "xdit=missing" in out


class TestAtomPathPresentInAllThreeLocations:
    """Pin atom-source-path entries across the three sister lists so a cleanup can't drop one."""

    def test_atom_present_in_default_source_roots(self):
        assert any("/app/atom/atom" in r.lower() for r in fp._DEFAULT_SOURCE_ROOTS)

    def test_atom_present_in_reusable_source_roots(self):
        from hyperloom.orchestrator.kernel import (
            request_handlers as krh,
        )

        assert any("/app/atom/atom" in r.lower() for r in krh._reusable_source_roots())

    def test_atom_present_in_tracelens_reusable_roots(self):
        """The kernel-agent's tracelens_analysis ``_REUSABLE_SOURCE_ROOTS`` must track the orchestrator-side list."""
        ka_path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "hyperloom"
            / "agents"
            / "kernel"
            / "tools"
            / "tracelens_analysis.py"
        )
        if not ka_path.is_file():
            pytest.skip(f"kernel-agent tracelens_analysis not on disk at {ka_path}")
        text = ka_path.read_text(encoding="utf-8")
        assert "/app/atom/atom/" in text.lower(), (
            "src/hyperloom/agents/kernel/tools/tracelens_analysis.py _REUSABLE_SOURCE_ROOTS "
            "is out of sync with src/hyperloom/orchestrator/kernel/"
            "request_handlers._REUSABLE_SOURCE_ROOTS (atom missing)"
        )

    def test_kernel_request_handlers_and_tracelens_analysis_atom_paths_in_sync(self):
        """The orchestrator gate and kernel-agent classifier derive reusable roots from the same source, so their atom subsets must match."""
        ka_path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "hyperloom"
            / "agents"
            / "kernel"
            / "tools"
            / "tracelens_analysis.py"
        )
        if not ka_path.is_file():
            pytest.skip(f"kernel-agent tracelens_analysis not on disk at {ka_path}")
        from hyperloom.orchestrator.kernel import (
            request_handlers as krh,
        )

        orch_atom = frozenset(r.lower() for r in krh._reusable_source_roots() if "/atom/" in r.lower())
        # Put the tools dir on sys.path: the sister tool imports sibling kernel-agent tools.
        import importlib.util as _ilu
        import sys as _sys

        tools_dir = str(ka_path.parent)
        added = tools_dir not in _sys.path
        if added:
            _sys.path.insert(0, tools_dir)
        try:
            spec = _ilu.spec_from_file_location(
                "_tracelens_atom_sync_probe",
                ka_path,
            )
            assert spec is not None and spec.loader is not None
            mod = _ilu.module_from_spec(spec)
            # Register before exec so self-referential dataclass annotations resolve.
            _sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            ka_atom = frozenset(r.lower() for r in mod._reusable_roots() if "/atom/" in r.lower())
        finally:
            if added and tools_dir in _sys.path:
                _sys.path.remove(tools_dir)
        assert orch_atom, "orchestrator reusable roots carry no atom entry"
        assert ka_atom, "tracelens reusable roots carry no atom entry"
        assert orch_atom == ka_atom, f"atom subsets diverged — orch={sorted(orch_atom)!r} ka={sorted(ka_atom)!r}"


# ROCm/HIP enablement source roots (default-on)


class TestRocmHipSourceRoots:
    """The ROCm/HIP source-edit capability is always on (enablement path)."""

    def test_always_returns_rocm_root(self):
        """No opt-in gate: the ROCm/HIP roots are always surfaced."""
        assert fp.resolve_rocm_hip_source_roots() == ("/opt/rocm/",)

    def test_merged_into_allowlist_by_default(self, monkeypatch):
        """/opt/rocm/ is always present in the resolved allowlist."""
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: ())
        assert "/opt/rocm/" in fp.resolve_source_file_allowlist()

    def test_aiter_allowed(self):
        """aiter stays in the default allowlist too."""
        assert any("/aiter/" in r for r in fp._DEFAULT_SOURCE_ROOTS)


# Source-root resolution + prompt injection
def test_resolve_source_file_allowlist_unions_env_override(monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        "/custom/vllm/:/extra/pkg/",
    )
    roots = resolve_source_file_allowlist()
    assert "/sgl-workspace/vllm/" in roots
    assert "/custom/vllm/" in roots
    assert "/extra/pkg/" in roots


def test_prompt_renders_framework_source_roots(registry=None):
    registry = registry or ACTION_CATALOGUE
    custom = ("/custom/sglang/", "/opt/venv/lib/python3.12/site-packages/vllm/")
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        max_minutes=60,
        rules_fragment_path=asset_system_prompts_dir() / "orchestration.md",
        framework_source_roots=custom,
    )
    assert "framework_source_roots:" in text
    assert "/custom/sglang/" in text
    assert "site-packages/vllm/" in text


def test_probe_framework_source_roots_includes_defaults(tmp_path, monkeypatch):
    ws = tmp_path / "sgl-workspace" / "sglang"
    ws.mkdir(parents=True)
    monkeypatch.setattr(
        "hyperloom.orchestrator.framework.paths._DEFAULT_SOURCE_ROOTS",
        (str(ws) + "/",),
    )
    out = probe_framework_source_roots_for_env()
    assert str(ws) in out or (str(ws) + "/") in out


# apply_kernel_patch known-target roots
_APPLY_TOOL_PATH = (
    Path(__file__).resolve().parents[4] / "src" / "hyperloom" / "agents" / "kernel" / "tools" / "apply_kernel_patch.py"
)


@pytest.fixture(scope="module")
def apply_tool() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_apply_kernel_patch_roots_test",
        _APPLY_TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_known_target_roots_includes_dist_packages_vllm(
    apply_tool,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fp, "_discover_installed_framework_roots", lambda: ("/usr/local/lib/python3.12/dist-packages/vllm/",)
    )
    monkeypatch.delenv("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", raising=False)
    apply_tool._CACHED_KNOWN_TARGET_ROOTS = None
    roots = apply_tool.known_target_roots()
    assert "/usr/local/lib/python3.12/dist-packages/vllm/" in roots


def test_detect_strategy_accepts_dist_packages_vllm_py(
    apply_tool,
    monkeypatch,
) -> None:
    target = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/parameter.py",
    )
    monkeypatch.setattr(
        apply_tool,
        "known_target_roots",
        lambda: ("/usr/local/lib/python3.12/dist-packages/vllm/",),
    )
    strat = apply_tool._detect_strategy(target, allow_unknown_target=False)
    assert strat["compiled"] is False


# --- aiter_meta split-wheel rebuild recognition (regression) ---
# aiter device sources ship in the sibling ``aiter_meta`` package, so hot
# kernels land under ``.../dist-packages/aiter_meta/csrc/...``. The JIT/cpp_itfs
# rebuild gates keyed only ``/aiter/csrc/``, so a KEPT aiter_meta .cu deployed
# but never re-JIT'd -> integrate saw a stale binary and REVERT'd
# (fault_attempts_exhausted; observed 07.25-07.30 on Qwen3-8B/Llama/Mixtral).

_AITER_META_CU = Path("/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/quant_kernels.cu")
_AITER_META_CPP_ITFS_CU = Path("/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/cpp_itfs/mha_fwd.cu")


def test_target_is_in_aiter_csrc_matches_aiter_meta(apply_tool) -> None:
    # split-wheel layout must be recognised as an aiter csrc source
    assert apply_tool._target_is_in_aiter_csrc(_AITER_META_CU) is True
    # classic layout still recognised
    assert apply_tool._target_is_in_aiter_csrc(Path("/sgl-workspace/aiter/csrc/kernels/quant_kernels.cu")) is True
    # unrelated source stays out
    assert (
        apply_tool._target_is_in_aiter_csrc(
            Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/parameter.py")
        )
        is False
    )


def test_target_is_in_aiter_cpp_itfs_matches_aiter_meta(apply_tool) -> None:
    assert apply_tool._target_is_in_aiter_cpp_itfs(_AITER_META_CPP_ITFS_CU) is True
    # a non-cpp_itfs aiter_meta source is csrc but NOT cpp_itfs
    assert apply_tool._target_is_in_aiter_cpp_itfs(_AITER_META_CU) is False


def test_invalidate_aiter_jit_build_runs_for_aiter_meta_target(apply_tool, tmp_path) -> None:
    jit_build = tmp_path / "aiter" / "jit" / "build"
    jit_build.mkdir(parents=True)
    (jit_build / "module_aiter_core.so").write_bytes(b"stale")
    backup_dir = tmp_path / "backup"

    res = apply_tool._invalidate_aiter_jit_build(
        _AITER_META_CU,
        backup_dir,
        jit_build_dir=jit_build,
    )

    assert res["status"] == "ok", res
    assert not jit_build.exists()  # moved aside so the next import re-JITs
    backups = list(backup_dir.glob("jit_build_*/module_aiter_core.so"))
    assert len(backups) == 1


def test_invalidate_aiter_jit_build_ignores_orphaned_prior_backup(
    apply_tool,
    tmp_path,
) -> None:
    jit_build = tmp_path / "aiter" / "jit" / "build"
    jit_build.mkdir(parents=True)
    (jit_build / "first.so").write_bytes(b"first")
    backup_dir = tmp_path / "backup"

    first = apply_tool._invalidate_aiter_jit_build(
        _AITER_META_CU,
        backup_dir,
        jit_build_dir=jit_build,
    )
    jit_build.mkdir(parents=True)
    (jit_build / "second.so").write_bytes(b"second")
    second = apply_tool._invalidate_aiter_jit_build(
        _AITER_META_CU,
        backup_dir,
        jit_build_dir=jit_build,
    )

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert first["backup_path"] != second["backup_path"]
    assert Path(first["backup_path"]).is_dir()
    assert Path(second["backup_path"]).is_dir()


class TestResolveFrameworkTree:
    def test_prefixed_env_wins(self, monkeypatch, tmp_path):
        tree = tmp_path / "sglang"
        tree.mkdir()
        monkeypatch.setenv("SGLANG_REPO_PATH", str(tree))
        assert fp.resolve_framework_tree("sglang") == f"{tree}/"

    def test_generic_env_is_used_when_no_prefixed_one(self, monkeypatch, tmp_path):
        tree = tmp_path / "generic"
        tree.mkdir()
        monkeypatch.delenv("SGLANG_REPO_PATH", raising=False)
        monkeypatch.delenv("SGLANG_DIR", raising=False)
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(tree))
        assert fp.resolve_framework_tree("sglang") == f"{tree}/"

    def test_absent_env_falls_to_package_origin(self, monkeypatch, tmp_path):
        pkg_parent = tmp_path / "site-packages"
        (pkg_parent / "myfw").mkdir(parents=True)
        monkeypatch.delenv("FRAMEWORK_REPO_PATH", raising=False)
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: pkg_parent if name == "myfw" else None)
        assert fp.resolve_framework_tree("myfw") == f"{pkg_parent}/"

    def test_unknown_framework_resolves_to_nothing(self, monkeypatch):
        monkeypatch.delenv("FRAMEWORK_REPO_PATH", raising=False)
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: None)
        assert fp.resolve_framework_tree("not-a-framework") == ""

    def test_empty_name_resolves_to_nothing(self):
        assert fp.resolve_framework_tree("") == ""
