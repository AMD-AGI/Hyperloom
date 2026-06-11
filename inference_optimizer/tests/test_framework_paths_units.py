# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``orchestrator.framework_paths`` path-resolution helpers and the PolicyGate roots allowlist."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator import framework_paths as fp
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.framework_paths import (
    probe_framework_source_roots_for_env,
    resolve_sglang_server_args_path,
    resolve_source_file_allowlist,
    resolve_vllm_arg_utils_path,
)
from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    build_orchestration_prompt,
)
from inference_optimizer.paths import asset_system_prompts_dir


@pytest.fixture(autouse=True)
def _clean_framework_env(monkeypatch):
    """Reset every env var the helpers read."""
    for key in (
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        "INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS",
        "INFERENCE_OPTIMIZER_VLLM_ARG_UTILS",
        "INFERENCE_OPTIMIZER_ATOM_ARG_UTILS",
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
        n = len(fp._DEFAULT_SOURCE_ROOTS)
        assert out[:n] == fp._DEFAULT_SOURCE_ROOTS
        assert "/opt/custom/sglang/" in out
        assert "/opt/other/vllm/" in out
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
        monkeypatch.setattr(
            fp, "_DEFAULT_SGLANG_SERVER_ARGS", tmp_path / "absent.py",
        )
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

    def test_dedupes_origins_against_defaults(self, tmp_path, monkeypatch):
        shared = tmp_path / "shared"
        shared.mkdir()
        monkeypatch.setattr(
            fp, "_DEFAULT_SOURCE_ROOTS", (f"{shared}/",),
        )
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: shared)
        monkeypatch.setattr(fp, "_glob_install_package_roots", lambda: ())
        result = fp.probe_framework_source_roots_for_env()
        assert result == f"{shared}/"


# atom enablement

class TestDefaultSourceRootsIncludesAtom:
    def test_atom_root_present_in_defaults(self):
        """/app/ATOM/atom/ must be in the PolicyGate source-file allowlist."""
        assert any(
            "/app/ATOM/atom" in r for r in fp._DEFAULT_SOURCE_ROOTS
        ), (
            f"_DEFAULT_SOURCE_ROOTS missing atom entry: "
            f"{fp._DEFAULT_SOURCE_ROOTS!r}"
        )

    def test_atom_root_visible_in_resolve_allowlist(self):
        """The public resolver must also surface the atom root."""
        out = fp.resolve_source_file_allowlist()
        assert any("/app/ATOM/atom" in r for r in out)


class TestResolveAtomArgUtils:
    def test_env_override_pointing_to_file(self, tmp_path, monkeypatch):
        target = tmp_path / "arg_utils.py"
        target.write_text("# fake")
        monkeypatch.setenv(
            "INFERENCE_OPTIMIZER_ATOM_ARG_UTILS", str(target),
        )
        path, source = fp.resolve_atom_arg_utils_path()
        assert path == target
        assert source == str(target)

    def test_env_override_pointing_to_missing_file(self, tmp_path, monkeypatch):
        target = tmp_path / "missing.py"
        monkeypatch.setenv(
            "INFERENCE_OPTIMIZER_ATOM_ARG_UTILS", str(target),
        )
        path, source = fp.resolve_atom_arg_utils_path()
        assert path == target
        assert "not found" in source

    def test_default_location_when_present(self, tmp_path, monkeypatch):
        default = tmp_path / "arg_utils.py"
        default.write_text("# fake")
        monkeypatch.setattr(fp, "_DEFAULT_ATOM_ARG_UTILS", default)
        path, source = fp.resolve_atom_arg_utils_path()
        assert path == default

    def test_spec_fallback_resolves_model_engine_arg_utils(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(
            fp, "_DEFAULT_ATOM_ARG_UTILS", tmp_path / "absent.py",
        )
        origin = tmp_path / "atom_pkg"
        au = origin / "model_engine" / "arg_utils.py"
        au.parent.mkdir(parents=True)
        au.write_text("# fake")
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: origin)
        path, source = fp.resolve_atom_arg_utils_path()
        assert path == au
        assert source == str(au)

    def test_default_path_returned_when_nothing_resolves(
        self, tmp_path, monkeypatch,
    ):
        sentinel = tmp_path / "still_absent.py"
        monkeypatch.setattr(fp, "_DEFAULT_ATOM_ARG_UTILS", sentinel)
        monkeypatch.setattr(fp, "_find_spec_origin", lambda name: None)
        path, source = fp.resolve_atom_arg_utils_path()
        assert path == sentinel
        assert "not found" in source


class TestProbeIncludesAtomWhenInstalled:
    def test_atom_picked_up_via_find_spec(self, tmp_path, monkeypatch):
        """A real ``find_spec('atom')`` origin is included even without a /app/ATOM/atom/ default root."""
        origin = tmp_path / "atom_pkg"
        origin.mkdir()
        monkeypatch.setattr(fp, "_DEFAULT_SOURCE_ROOTS", ())

        spec_map = {"atom": origin}
        monkeypatch.setattr(
            fp, "_find_spec_origin",
            lambda name: spec_map.get(name),
        )
        result = fp.probe_framework_source_roots_for_env()
        assert f"{origin}/" in result

    def test_atom_picked_up_via_venv_site_packages(
        self, tmp_path, monkeypatch,
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


class TestFrameworkPathsThreeFrameworksSymmetric:
    """Static guard: all three frameworks' arg-utils resolvers return the ``(Path, str)`` contract on env-override."""

    @pytest.mark.parametrize(
        "resolver_name",
        [
            "resolve_sglang_server_args_path",
            "resolve_vllm_arg_utils_path",
            "resolve_atom_arg_utils_path",
        ],
    )
    def test_resolver_returns_path_str_pair(
        self, resolver_name, tmp_path, monkeypatch,
    ):
        env_map = {
            "resolve_sglang_server_args_path":
                "INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS",
            "resolve_vllm_arg_utils_path":
                "INFERENCE_OPTIMIZER_VLLM_ARG_UTILS",
            "resolve_atom_arg_utils_path":
                "INFERENCE_OPTIMIZER_ATOM_ARG_UTILS",
        }
        target = tmp_path / "fake_args.py"
        target.write_text("# stub")
        monkeypatch.setenv(env_map[resolver_name], str(target))
        path, source = getattr(fp, resolver_name)()
        assert isinstance(path, Path)
        assert isinstance(source, str)
        assert path == target


class TestSummariseFrameworkRootDiscovery:
    def test_buckets_atom_ok(self):
        """The install.sh log helper reports atom=ok when an atom root appears in the discovery string."""
        out = fp.summarise_framework_root_discovery(
            "/sgl-workspace/aiter/:/sgl-workspace/sglang/:/sgl-workspace/vllm/"
            ":/app/ATOM/atom/"
        )
        assert "atom=ok" in out
        assert "sglang=ok" in out
        assert "vllm=ok" in out
        assert "aiter=ok" in out

    def test_buckets_atom_missing_on_non_atom_box(self):
        out = fp.summarise_framework_root_discovery(
            "/sgl-workspace/aiter/:/sgl-workspace/sglang/:/sgl-workspace/vllm/"
        )
        assert "atom=missing" in out
        assert "sglang=ok" in out

    def test_handles_empty_input(self):
        out = fp.summarise_framework_root_discovery("")
        assert "atom=missing" in out
        assert "sglang=missing" in out
        assert "vllm=missing" in out
        assert "aiter=missing" in out

    def test_does_not_substring_match_unrelated_paths(self):
        """Only paths whose last directory IS ``atom`` count; a substring like ``atomic_kernel`` must not."""
        out = fp.summarise_framework_root_discovery(
            "/sgl-workspace/atomic_kernel/"
        )
        assert "atom=missing" in out


class TestAtomPathPresentInAllThreeLocations:
    """Pin atom-source-path entries across the three sister lists so a cleanup can't drop one."""

    def test_atom_present_in_default_source_roots(self):
        assert any(
            "/app/atom/atom" in r.lower() for r in fp._DEFAULT_SOURCE_ROOTS
        )

    def test_atom_present_in_reusable_source_roots(self):
        from inference_optimizer.orchestrator import (
            kernel_request_handlers as krh,
        )
        assert any(
            "/app/atom/atom" in r.lower() for r in krh._reusable_source_roots()
        )

    def test_atom_present_in_tracelens_reusable_roots(self):
        """The kernel-agent's tracelens_analysis ``_REUSABLE_SOURCE_ROOTS`` must track the orchestrator-side list."""
        ka_path = (
            Path(__file__).resolve().parents[2]
            / "kernel-agent" / "tools" / "tracelens_analysis.py"
        )
        if not ka_path.is_file():
            pytest.skip(
                f"kernel-agent tracelens_analysis not on disk at {ka_path}"
            )
        text = ka_path.read_text(encoding="utf-8")
        assert "/app/atom/atom/" in text.lower(), (
            "kernel-agent/tools/tracelens_analysis.py _REUSABLE_SOURCE_ROOTS "
            "is out of sync with inference_optimizer/orchestrator/"
            "kernel_request_handlers._REUSABLE_SOURCE_ROOTS (atom missing)"
        )

    def test_kernel_request_handlers_and_tracelens_analysis_atom_paths_in_sync(self):
        """The orchestrator gate and kernel-agent classifier derive reusable roots from the same source, so their atom subsets must match."""
        ka_path = (
            Path(__file__).resolve().parents[2]
            / "kernel-agent" / "tools" / "tracelens_analysis.py"
        )
        if not ka_path.is_file():
            pytest.skip(
                f"kernel-agent tracelens_analysis not on disk at {ka_path}"
            )
        from inference_optimizer.orchestrator import (
            kernel_request_handlers as krh,
        )
        orch_atom = frozenset(
            r.lower() for r in krh._reusable_source_roots()
            if "/atom/" in r.lower()
        )
        # Put the tools dir on sys.path: the sister tool imports sibling kernel-agent tools.
        import importlib.util as _ilu
        import sys as _sys
        tools_dir = str(ka_path.parent)
        added = tools_dir not in _sys.path
        if added:
            _sys.path.insert(0, tools_dir)
        try:
            spec = _ilu.spec_from_file_location(
                "_tracelens_atom_sync_probe", ka_path,
            )
            assert spec is not None and spec.loader is not None
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            ka_atom = frozenset(
                r.lower() for r in mod._reusable_roots()
                if "/atom/" in r.lower()
            )
        finally:
            if added and tools_dir in _sys.path:
                _sys.path.remove(tools_dir)
        assert orch_atom, "orchestrator reusable roots carry no atom entry"
        assert ka_atom, "tracelens reusable roots carry no atom entry"
        assert orch_atom == ka_atom, (
            f"atom subsets diverged — orch={sorted(orch_atom)!r} "
            f"ka={sorted(ka_atom)!r}"
        )


# Source-root resolution + prompt injection (was test_framework_source_roots.py)
def test_resolve_source_file_allowlist_unions_env_override(monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS",
        "/custom/vllm/:/extra/pkg/",
    )
    roots = resolve_source_file_allowlist()
    assert "/sgl-workspace/vllm/" in roots
    assert "/custom/vllm/" in roots
    assert "/extra/pkg/" in roots


def test_find_spec_fallback_returns_note_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS", "")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_VLLM_ARG_UTILS", "")
    path, note = resolve_sglang_server_args_path()
    assert path.name == "server_args.py"
    assert "not found" in note.lower() or str(path) in note
    vpath, vnote = resolve_vllm_arg_utils_path()
    assert vpath.name == "arg_utils.py"
    assert vnote


def test_prompt_renders_framework_source_roots(registry=None):
    registry = registry or ActionRegistry().load()
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
        "inference_optimizer.orchestrator.framework_paths._DEFAULT_SOURCE_ROOTS",
        (str(ws) + "/",),
    )
    out = probe_framework_source_roots_for_env()
    assert str(ws) in out or (str(ws) + "/") in out


# apply_kernel_patch known-target roots (was test_apply_kernel_patch_roots.py)
_APPLY_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "kernel-agent"
    / "tools"
    / "apply_kernel_patch.py"
)


@pytest.fixture(scope="module")
def apply_tool() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_apply_kernel_patch_roots_test", _APPLY_TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_known_target_roots_includes_dist_packages_vllm(
    apply_tool, monkeypatch,
) -> None:
    monkeypatch.setattr(fp, "_discover_installed_framework_roots", lambda: (
        "/usr/local/lib/python3.12/dist-packages/vllm/",
    ))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS", raising=False)
    apply_tool._CACHED_KNOWN_TARGET_ROOTS = None
    roots = apply_tool.known_target_roots()
    assert "/usr/local/lib/python3.12/dist-packages/vllm/" in roots


def test_detect_strategy_accepts_dist_packages_vllm_py(
    apply_tool, monkeypatch,
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
