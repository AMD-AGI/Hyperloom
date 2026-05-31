"""Unit tests for ``orchestrator.framework_paths``.

Covers the path-resolution helpers that decide where to look for sglang
``server_args.py`` / vLLM ``arg_utils.py``, plus the env-driven roots
allowlist used by PolicyGate. Each test isolates the module's reliance
on /sgl-workspace + importlib.util.find_spec so we can verify the
control-flow without the real container layout.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator import framework_paths as fp


@pytest.fixture(autouse=True)
def _clean_framework_env(monkeypatch):
    """Reset every env var the helpers read so each test gets a fresh slate."""
    for key in (
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        "INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS",
        "INFERENCE_OPTIMIZER_VLLM_ARG_UTILS",
        "VIRTUAL_ENV",
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
        assert fp.resolve_source_file_allowlist() == fp._DEFAULT_SOURCE_ROOTS

    def test_merges_discovered_roots(self, monkeypatch):
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: (
            "/usr/local/lib/python3.12/dist-packages/vllm/",
        ))
        roots = fp.resolve_source_file_allowlist()
        assert fp._DEFAULT_SOURCE_ROOTS[0] in roots
        assert "/usr/local/lib/python3.12/dist-packages/vllm/" in roots

    def test_appends_extra_roots_unique_in_order(self, monkeypatch):
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: ())
        monkeypatch.setenv(
            "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
            "/opt/custom/sglang:/sgl-workspace/aiter:/opt/other/vllm",
        )
        out = fp.resolve_source_file_allowlist()
        # Defaults preserved, extras appended, duplicates of defaults dropped.
        assert out[:3] == fp._DEFAULT_SOURCE_ROOTS
        assert "/opt/custom/sglang/" in out
        assert "/opt/other/vllm/" in out
        # No duplicate of the existing default.
        assert out.count("/sgl-workspace/aiter/") == 1


class TestFindSpecOrigin:
    def test_returns_none_when_spec_missing(self, monkeypatch):
        monkeypatch.setattr(
            importlib.util, "find_spec",
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
        # _find_spec_origin returns origin.parent in both __init__ and
        # non-__init__ branches; the contract is "package directory".
        assert fp._find_spec_origin("pkg") == init.parent

    def test_handles_find_spec_raising(self, monkeypatch):
        def boom(_):
            raise ValueError("malformed")

        monkeypatch.setattr(importlib.util, "find_spec", boom)
        assert fp._find_spec_origin("pkg") is None


class TestResolveSglangServerArgs:
    def test_explicit_env_pointing_to_file(self, tmp_path, monkeypatch):
        target = tmp_path / "server_args.py"
        target.write_text("# fake")
        monkeypatch.setenv(
            "INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS", str(target),
        )
        path, source = fp.resolve_sglang_server_args_path()
        assert path == target
        assert source == str(target)

    def test_explicit_env_pointing_to_missing_file(self, tmp_path, monkeypatch):
        target = tmp_path / "missing.py"
        monkeypatch.setenv(
            "INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS", str(target),
        )
        path, source = fp.resolve_sglang_server_args_path()
        assert path == target
        assert "not found" in source

    def test_fallback_via_find_spec(self, tmp_path, monkeypatch):
        # Pretend /sgl-workspace/.../server_args.py does not exist.
        monkeypatch.setattr(
            fp, "_DEFAULT_SGLANG_SERVER_ARGS", tmp_path / "absent.py",
        )
        # Build a fake sglang origin that contains srt/server_args.py.
        origin = tmp_path / "sglang_pkg"
        (origin / "srt").mkdir(parents=True)
        sa = origin / "srt" / "server_args.py"
        sa.write_text("# fake")
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: origin)
        path, source = fp.resolve_sglang_server_args_path()
        assert path == sa
        assert source == str(sa)

    def test_alt_layout_when_primary_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            fp, "_DEFAULT_SGLANG_SERVER_ARGS", tmp_path / "absent.py",
        )
        origin = tmp_path / "sglang_pkg"
        alt = origin / "python" / "sglang" / "srt" / "server_args.py"
        alt.parent.mkdir(parents=True)
        alt.write_text("# fake")
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: origin)
        path, source = fp.resolve_sglang_server_args_path()
        assert path == alt

    def test_default_path_returned_when_nothing_resolves(
        self, tmp_path, monkeypatch,
    ):
        sentinel = tmp_path / "still_absent.py"
        monkeypatch.setattr(fp, "_DEFAULT_SGLANG_SERVER_ARGS", sentinel)
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: None)
        path, source = fp.resolve_sglang_server_args_path()
        assert path == sentinel
        assert "not found" in source


class TestResolveVllmArgUtils:
    def test_default_path_when_present(self, tmp_path, monkeypatch):
        default = tmp_path / "arg_utils.py"
        default.write_text("# fake")
        monkeypatch.setattr(fp, "_DEFAULT_VLLM_ARG_UTILS", default)
        path, source = fp.resolve_vllm_arg_utils_path()
        assert path == default

    def test_alt_layout_via_find_spec(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fp, "_DEFAULT_VLLM_ARG_UTILS", tmp_path / "absent.py")
        origin = tmp_path / "vllm_pkg"
        alt = origin / "vllm" / "engine" / "arg_utils.py"
        alt.parent.mkdir(parents=True)
        alt.write_text("# fake")
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: origin)
        path, source = fp.resolve_vllm_arg_utils_path()
        assert path == alt

    def test_explicit_env_missing_file_returns_not_found_source(
        self, tmp_path, monkeypatch,
    ):
        target = tmp_path / "no.py"
        monkeypatch.setenv("INFERENCE_OPTIMIZER_VLLM_ARG_UTILS", str(target))
        path, source = fp.resolve_vllm_arg_utils_path()
        assert path == target
        assert "not found" in source


class TestGlobInstallPackageRoots:
    def test_finds_dist_packages_under_usr_local(self, tmp_path, monkeypatch):
        base = tmp_path / "usr_local_lib" / "python3.12" / "dist-packages"
        (base / "vllm").mkdir(parents=True)
        monkeypatch.setattr(
            fp, "_INSTALL_GLOB_PARENTS", (tmp_path / "usr_local_lib",),
        )
        roots = fp._glob_install_package_roots()
        assert any("dist-packages/vllm/" in r for r in roots)


class TestResolvePatchTargetRoots:
    def test_includes_static_fallback_when_discovery_empty(self, monkeypatch):
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: ())
        monkeypatch.delenv("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", raising=False)
        roots = fp.resolve_patch_target_roots()
        assert "/usr/local/lib/python3.12/dist-packages/vllm/" in roots
        assert "/aiter_meta/csrc/" in roots


class TestProbeFrameworkSourceRootsForEnv:
    def test_returns_existing_dirs_only(self, tmp_path, monkeypatch):
        present = tmp_path / "fake_root"
        present.mkdir()
        monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: (
            f"{present}/",
            f"{tmp_path / 'missing'}/",
        ))
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())
        result = fp.probe_framework_source_roots_for_env()
        assert result == f"{present}/"

    def test_includes_site_packages_when_virtual_env_set(
        self, tmp_path, monkeypatch,
    ):
        venv = tmp_path / "venv"
        site = venv / "lib" / "python3.12" / "site-packages"
        for name in ("vllm", "sglang", "aiter"):
            (site / name).mkdir(parents=True)
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: None)
        monkeypatch.setattr(fp, "_glob_install_package_roots", lambda: ())
        monkeypatch.setenv("VIRTUAL_ENV", str(venv))
        result = fp.probe_framework_source_roots_for_env()
        for name in ("vllm", "sglang", "aiter"):
            assert f"{name}/" in result
