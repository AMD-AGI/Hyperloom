# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-framework runtime dependency installation.

Covers the manifest contract, the load-bearing torch guard, and -- most
importantly -- that the pass actually runs in the documented flow: install.sh
executes before ``--framework`` is known, so if only that side installed the
deps a scriptable framework would reach baseline with nothing installed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pytest

from hyperloom.inference_optimizer import framework_deps as fd

REPO_ROOT = Path(__file__).resolve().parents[4]
ASSETS = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets"
INSTALL_SH = ASSETS / "install.sh"
PREFLIGHT_PY = (
    REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "cli" / "preflight.py"
)


# --------------------------------------------------------------------------
# manifest parsing
# --------------------------------------------------------------------------


def test_blank_lines_and_comments_are_ignored():
    reqs, refused, invalid = fd.parse_manifest("# header\n\n  einops  # why\n\n#end\n")
    assert [r.spec for r in reqs] == ["einops"]
    assert (refused, invalid) == ([], [])


def test_hash_inside_a_vcs_spec_is_not_a_comment():
    """A '#' only starts a comment at line start or after whitespace."""
    reqs, _, _ = fd.parse_manifest("git+https://host/repo.git#egg=thing:thing\n")
    assert [r.spec for r in reqs] == ["git+https://host/repo.git#egg=thing"]


def test_import_name_override_and_default():
    reqs, _, _ = fd.parse_manifest("opencv-python:cv2\nsome-dist\n")
    assert (reqs[0].spec, reqs[0].import_name) == ("opencv-python", "cv2")
    assert (reqs[1].spec, reqs[1].import_name) == ("some-dist", "some_dist")


def test_version_specifier_stripped_from_default_import_name():
    reqs, _, _ = fd.parse_manifest("einops>=0.8.0\n")
    assert (reqs[0].spec, reqs[0].import_name) == ("einops>=0.8.0", "einops")


def test_url_spec_without_an_import_name_is_rejected():
    """No import name can be derived from a URL, and guessing would mean the
    probe never resolves and the package reinstalls on every single run."""
    reqs, _, invalid = fd.parse_manifest("git+https://host/repo.git\n")
    assert reqs == []
    assert invalid == ["git+https://host/repo.git"]


def test_url_spec_with_an_explicit_import_name_is_kept():
    reqs, _, invalid = fd.parse_manifest("git+https://host/repo.git:thing\n")
    assert (reqs[0].spec, reqs[0].import_name) == ("git+https://host/repo.git", "thing")
    assert invalid == []


@pytest.mark.parametrize("core", sorted(fd.CORE_PACKAGES))
def test_load_bearing_packages_are_refused(core):
    """Honouring an upstream torch==2.7.1 pin would brick the shared venv."""
    reqs, refused, _ = fd.parse_manifest(f"{core}==1.2.3\neinops\n")
    assert [r.spec for r in reqs] == ["einops"]
    assert refused == [core]


# --------------------------------------------------------------------------
# install behaviour
# --------------------------------------------------------------------------


class FakePython:
    """Stands in for subprocess.run against a probed/installed interpreter."""

    def __init__(self, installed=(), uninstallable=(), clobbers_torch=False):
        self.installed = set(installed)
        self.uninstallable = set(uninstallable)
        self.clobbers_torch = clobbers_torch
        self.pip_calls: list[str] = []
        self.hip = "7.2.53211"

    def __call__(self, cmd, **kwargs):
        if "-m" in cmd and "pip" in cmd:
            spec = cmd[-1]
            self.pip_calls.append(spec)
            if spec in self.uninstallable:
                return subprocess.CompletedProcess(cmd, 1, "", "boom")
            if self.clobbers_torch:
                self.hip = ""
            self.installed.add(re.split(r"[<>=!~;\[]", spec)[0].replace("-", "_"))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        script = cmd[2] if len(cmd) > 2 else ""
        if "find_spec" in script:
            absent = [n for n in cmd[3:] if n not in self.installed]
            return subprocess.CompletedProcess(cmd, 0, "\n".join(absent), "")
        if "importlib.metadata" in script:
            return subprocess.CompletedProcess(cmd, 0, "torch==2.9.1\n", "")
        if "torch.version.hip" in script:
            return subprocess.CompletedProcess(cmd, 0, self.hip, "")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def _manifest(tmp_path: Path, framework: str, body: str) -> Path:
    root = tmp_path / "framework_deps"
    root.mkdir(exist_ok=True)
    (root / f"{framework}.txt").write_text(body, encoding="utf-8")
    return root


def test_missing_manifest_is_a_noop(tmp_path, monkeypatch):
    fake = FakePython()
    monkeypatch.setattr(fd.subprocess, "run", fake)
    out = fd.ensure("sglang", python_exe="py", root=tmp_path)
    assert out.skipped_reason
    assert fake.pip_calls == []


def test_unset_framework_is_a_noop(tmp_path, monkeypatch):
    fake = FakePython()
    monkeypatch.setattr(fd.subprocess, "run", fake)
    out = fd.ensure("", python_exe="py", root=tmp_path)
    assert out.skipped_reason
    assert fake.pip_calls == []


def test_installs_only_what_is_absent(tmp_path, monkeypatch):
    root = _manifest(tmp_path, "demo", "einops\nplyfile\n")
    fake = FakePython(installed={"plyfile"})
    monkeypatch.setattr(fd.subprocess, "run", fake)
    out = fd.ensure("demo", python_exe="py", root=root)
    assert fake.pip_calls == ["einops"]
    assert out.installed == ["einops"]
    assert out.already_present == ["plyfile"]


def test_import_name_override_avoids_a_redundant_install(tmp_path, monkeypatch):
    root = _manifest(tmp_path, "demo", "opencv-python:cv2\n")
    fake = FakePython(installed={"cv2"})
    monkeypatch.setattr(fd.subprocess, "run", fake)
    out = fd.ensure("demo", python_exe="py", root=root)
    assert fake.pip_calls == []
    assert out.ok


def test_one_failure_does_not_strand_the_rest(tmp_path, monkeypatch):
    """gsplat builds from sdist; its failure must not block the cheap wheels."""
    root = _manifest(tmp_path, "demo", "gsplat\neinops\n")
    fake = FakePython(uninstallable={"gsplat"})
    monkeypatch.setattr(fd.subprocess, "run", fake)
    out = fd.ensure("demo", python_exe="py", root=root)
    assert fake.pip_calls == ["gsplat", "einops"]
    assert out.installed == ["einops"]
    assert out.failed == ["gsplat"]
    assert not out.ok


def test_every_install_is_constrained(tmp_path, monkeypatch):
    """Without -c the resolver can pull a CUDA torch over the ROCm one."""
    root = _manifest(tmp_path, "demo", "einops\n")
    seen = []

    class Recorder(FakePython):
        def __call__(self, cmd, **kwargs):
            if "-m" in cmd and "pip" in cmd:
                seen.append(cmd)
            return super().__call__(cmd, **kwargs)

    monkeypatch.setattr(fd.subprocess, "run", Recorder())
    fd.ensure("demo", python_exe="py", root=root)
    assert seen and "-c" in seen[0]


def test_clobbered_rocm_torch_raises(tmp_path, monkeypatch):
    root = _manifest(tmp_path, "demo", "einops\n")
    monkeypatch.setattr(fd.subprocess, "run", FakePython(clobbers_torch=True))
    with pytest.raises(fd.TorchClobberedError, match="ROCm torch"):
        fd.ensure("demo", python_exe="py", root=root)


def test_check_only_reports_without_installing(tmp_path, monkeypatch):
    root = _manifest(tmp_path, "demo", "einops\n")
    fake = FakePython()
    monkeypatch.setattr(fd.subprocess, "run", fake)
    out = fd.ensure("demo", python_exe="py", root=root, check_only=True)
    assert out.would_install == ["einops"]
    assert fake.pip_calls == []


# --------------------------------------------------------------------------
# wiring: both entry points must actually run this
# --------------------------------------------------------------------------


def test_preflight_covers_a_framework_install_sh_could_not_see(monkeypatch):
    """The regression that motivated the preflight pass.

    install.sh runs before --framework exists, so its own attempt no-ops. If
    preflight did not repeat the pass, a worldmirror baseline would start with
    none of HY-World-2.0's imports installed.
    """
    from hyperloom.inference_optimizer.cli import preflight

    monkeypatch.delenv("FRAMEWORK", raising=False)  # as at install time
    seen = {}

    def fake_ensure(framework, **kwargs):
        seen["framework"] = framework
        seen["python_exe"] = kwargs.get("python_exe")
        return fd.Outcome(framework=framework)

    monkeypatch.setattr(fd, "ensure", fake_ensure)
    args = argparse.Namespace(framework="worldmirror")
    preflight._ensure_framework_deps(args, "/venv/bin/python3", [])

    assert seen == {"framework": "worldmirror", "python_exe": "/venv/bin/python3"}


def test_preflight_falls_back_to_env_then_default(monkeypatch):
    from hyperloom.inference_optimizer.cli import preflight

    seen = []
    monkeypatch.setattr(fd, "ensure", lambda framework, **kw: (
        seen.append(framework), fd.Outcome(framework=framework))[1])

    monkeypatch.setenv("FRAMEWORK", "worldmirror")
    preflight._ensure_framework_deps(argparse.Namespace(framework=None), "py", [])
    monkeypatch.delenv("FRAMEWORK")
    preflight._ensure_framework_deps(argparse.Namespace(framework=None), "py", [])

    assert seen == ["worldmirror", "sglang"]


def test_preflight_invokes_the_pass():
    text = PREFLIGHT_PY.read_text(encoding="utf-8")
    assert "_ensure_framework_deps(args, benchmark_python, pip_extra)" in text


def test_install_sh_delegates_to_this_module():
    """install.sh must shell out here rather than reimplement the contract."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    assert "hyperloom.inference_optimizer.framework_deps" in text
    assert re.search(r"^ensure_framework_deps$", text, re.M), (
        "ensure_framework_deps is defined but never invoked"
    )


# --------------------------------------------------------------------------
# the shipped manifest
# --------------------------------------------------------------------------


def test_shipped_worldmirror_manifest_is_well_formed():
    """Properties that must hold however the package list evolves."""
    reqs, refused, invalid = fd.parse_manifest(
        (ASSETS / "framework_deps" / "worldmirror.txt").read_text(encoding="utf-8")
    )
    assert reqs, "manifest parsed to nothing"
    assert refused == [], f"manifest names load-bearing packages: {refused}"
    assert invalid == [], f"manifest entries with underivable import names: {invalid}"
    names = {fd._package_base(r.spec) for r in reqs}
    # flash-attn has no ROCm wheel and is served by the SDPA shim; listing it
    # here would make every install attempt fail.
    assert "flash-attn" not in names and "flash_attn" not in names
    assert not names & fd.CORE_PACKAGES
    # gsplat is called on the model forward path, so it cannot be dropped.
    assert "gsplat" in names
