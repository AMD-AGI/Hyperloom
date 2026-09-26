# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prebuilt-image vLLM source activation tests."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from hyperloom.orchestrator.actions.executors import _workload_envs as we

_INSTALL = Path(__file__).resolve().parents[1] / "assets" / "install.sh"
_COMMIT = "f46a9dfe2c5f57bebbd29556cbbb25eabd874226"


def _functions(*names: str) -> str:
    text = _INSTALL.read_text(encoding="utf-8")
    chunks = []
    for name in names:
        match = re.search(rf"^{name}\(\) \{{.*?^\}}$", text, re.MULTILINE | re.DOTALL)
        assert match, f"{name}() not found"
        chunks.append(match.group())
    return "\n\n".join(chunks)


def _bash(names: tuple[str, ...], body: str) -> subprocess.CompletedProcess[str]:
    script = f"set -euo pipefail\n{_functions(*names)}\n{body}\n"
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=False)


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()


def _image_tree(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git("init", "-q", cwd=upstream)
    _git("config", "user.email", "test@example.com", cwd=upstream)
    _git("config", "user.name", "Test", cwd=upstream)
    files = {
        "vllm/__init__.py": "SOURCE = True\n",
        "vllm/tracked.py": "tracked = 1\n",
        "docker/Dockerfile.rocm": "upstream rocm\n",
        "docker/Dockerfile.rocm_base": "upstream base\n",
        "tools/vllm-rocm/pin_rocm_dependencies.py": "upstream pin\n",
    }
    for relative, content in files.items():
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git("add", ".", cwd=upstream)
    _git("commit", "-qm", "source", cwd=upstream)
    commit = _git("rev-parse", "HEAD", cwd=upstream)

    source = tmp_path / "app-vllm"
    shutil.copytree(upstream, source, ignore=shutil.ignore_patterns(".git"))
    (source / "docker" / "Dockerfile.rocm").write_text("image rocm\n", encoding="utf-8")
    (source / "docker" / "Dockerfile.rocm_base").write_text("image base\n", encoding="utf-8")
    (source / "tools" / "vllm-rocm" / "pin_rocm_dependencies.py").unlink()
    (source / "vllm" / "_C.fake.so").write_bytes(b"existing native")
    wheel = tmp_path / "site-packages" / "vllm"
    wheel.mkdir(parents=True)
    (wheel / "tracked.py").write_text("wheel must not overwrite\n", encoding="utf-8")
    (wheel / "_version.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    (wheel / "_C.fake.so").write_bytes(b"wheel native")
    (wheel / "_rocm_C.fake.so").write_bytes(b"native")
    return upstream, source, wheel, commit


def test_image_deltas_form_clean_synthetic_baseline(tmp_path: Path) -> None:
    upstream, source, wheel, commit = _image_tree(tmp_path)
    body = f"""
die() {{ echo "$*" >&2; return 1; }}
VLLM_IMAGE_REPO="{upstream}"; VLLM_IMAGE_SOURCE_COMMIT="{commit}"; PYTHON="{os.sys.executable}"
prepare_vllm_image_git_tree "{source}"
copy_missing_vllm_wheel_artifacts "{source}" "{wheel}"; prepare_vllm_image_git_tree "{source}"
"""
    result = _bash(("prepare_vllm_image_git_tree", "copy_missing_vllm_wheel_artifacts"), body)

    assert result.returncode == 0, result.stderr
    head = _git("rev-parse", "HEAD", cwd=source)
    assert _git("rev-parse", "HEAD^", cwd=source) == commit
    assert _git("rev-parse", "refs/hyperloom/upstream", cwd=source) == commit
    assert _git("rev-parse", "refs/hyperloom/image-baseline", cwd=source) == head
    assert _git("show", "-s", "--format=%s", cwd=source) == "Hyperloom prebuilt vLLM image baseline"
    identity = _git("show", "-s", "--format=%an <%ae>%n%cn <%ce>", cwd=source)
    assert identity.splitlines() == ["Hyperloom <hyperloom@amd.com>", "Hyperloom <hyperloom@amd.com>"]
    assert _git("status", "--porcelain", "--untracked-files=no", cwd=source) == ""
    assert _git("diff", "--name-only", commit, "HEAD", cwd=source).splitlines() == [
        "docker/Dockerfile.rocm",
        "docker/Dockerfile.rocm_base",
        "tools/vllm-rocm/pin_rocm_dependencies.py",
    ]
    assert (source / "docker" / "Dockerfile.rocm").read_text() == "image rocm\n"
    assert not (source / "tools" / "vllm-rocm" / "pin_rocm_dependencies.py").exists()
    assert (source / "vllm" / "tracked.py").read_text() == "tracked = 1\n"
    assert (source / "vllm" / "_C.fake.so").read_bytes() == b"existing native"
    assert (source / "vllm" / "_version.py").read_text() == "__version__ = 'test'\n"
    assert (source / "vllm" / "_rocm_C.fake.so").read_bytes() == b"native"
    assert _git(
        "check-ignore", "vllm/_C.fake.so", "vllm/_version.py", "vllm/_rocm_C.fake.so", cwd=source
    ).splitlines() == [
        "vllm/_C.fake.so",
        "vllm/_version.py",
        "vllm/_rocm_C.fake.so",
    ]
    specialist = tmp_path / "specialist"
    _git("worktree", "add", "--detach", str(specialist), cwd=source)
    assert (specialist / "vllm" / "tracked.py").read_text() == "tracked = 1\n"


def test_dirty_synthetic_baseline_fails_without_changing_source(tmp_path: Path) -> None:
    upstream, source, _wheel, commit = _image_tree(tmp_path)
    tracked = source / "vllm" / "tracked.py"
    body = f"""
die() {{ echo "$*" >&2; return 1; }}
VLLM_IMAGE_REPO="{upstream}"; VLLM_IMAGE_SOURCE_COMMIT="{commit}"
prepare_vllm_image_git_tree "{source}"
printf 'user change\\n' > "{tracked}"
prepare_vllm_image_git_tree "{source}"
"""
    result = _bash(("prepare_vllm_image_git_tree",), body)

    assert result.returncode != 0
    assert tracked.read_text() == "user change\n"
    assert "user changes" in result.stderr


def test_commit_mismatch_leaves_other_images_untouched() -> None:
    body = f"""
log() {{ :; }}; die() {{ echo "$*" >&2; return 1; }}
probe_vllm_image_wheel() {{ printf '0.27.0+gdeadbee\\tdeadbee\\t/opt/python/site-packages/vllm\\n'; }}
prepare_vllm_image_git_tree() {{ echo MUTATION; }}
copy_missing_vllm_wheel_artifacts() {{ echo MUTATION; }}
verify_vllm_image_source_import() {{ echo MUTATION; }}
VLLM_IMAGE_SOURCE_ROOT=/app/vllm; VLLM_IMAGE_SOURCE_COMMIT={_COMMIT}
activate_vllm_image_source
"""
    result = _bash(("activate_vllm_image_source",), body)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_matching_runtime_persists_verified_source_aliases(tmp_path: Path) -> None:
    source = tmp_path / "app-vllm"
    (source / "vllm").mkdir(parents=True)
    env_file = tmp_path / "kernel.env"
    body = f"""
log() {{ :; }}; die() {{ echo "$*" >&2; return 1; }}
probe_vllm_image_wheel() {{ printf '0.27.0+gf46a9dfe\\tf46a9dfe\\t/opt/python/site-packages/vllm\\n'; }}
prepare_vllm_image_git_tree() {{
  export VLLM_IMAGE_UPSTREAM_SHA="$VLLM_IMAGE_SOURCE_COMMIT"
  export VLLM_IMAGE_BASELINE_SHA=0123456789abcdef
}}
copy_missing_vllm_wheel_artifacts() {{ :; }}
verify_vllm_image_source_import() {{ :; }}
VLLM_IMAGE_SOURCE_ROOT="{source}"; VLLM_IMAGE_SOURCE_COMMIT={_COMMIT}
VLLM_IMAGE_SOURCE_ACTIVE=0; KERNEL_AGENT_ENV="{env_file}"; DRY_RUN=0; CHECK_ONLY=0
activate_vllm_image_source
persist_vllm_image_source_env
"""
    result = _bash(("activate_vllm_image_source", "persist_vllm_image_source_env"), body)

    assert result.returncode == 0, result.stderr
    assert env_file.read_text().splitlines() == [
        f"export FRAMEWORK_REPO_PATH='{source}'",
        f"export VLLM_REPO_PATH='{source}'",
        f"export VLLM_DIR='{source}'",
        "export HYPERLOOM_VLLM_IMAGE_SOURCE='1'",
        f"export VLLM_IMAGE_UPSTREAM_SHA='{_COMMIT}'",
        "export VLLM_IMAGE_BASELINE_SHA='0123456789abcdef'",
    ]


def test_failed_source_import_never_activates(tmp_path: Path) -> None:
    source = tmp_path / "app-vllm"
    (source / "vllm").mkdir(parents=True)
    body = f"""
log() {{ :; }}; die() {{ echo "$*" >&2; return 1; }}
probe_vllm_image_wheel() {{ printf '0.27.0+gf46a9dfe\\tf46a9dfe\\t/opt/python/site-packages/vllm\\n'; }}
prepare_vllm_image_git_tree() {{ :; }}
copy_missing_vllm_wheel_artifacts() {{ :; }}
verify_vllm_image_source_import() {{ return 1; }}
VLLM_IMAGE_SOURCE_ROOT="{source}"; VLLM_IMAGE_SOURCE_COMMIT={_COMMIT}
VLLM_IMAGE_SOURCE_ACTIVE=0; DRY_RUN=0; CHECK_ONLY=0
activate_vllm_image_source
"""
    result = _bash(("activate_vllm_image_source",), body)
    assert result.returncode != 0
    assert "import verification failed" in result.stderr


def test_vllm_launch_uses_one_source_root(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_VLLM_IMAGE_SOURCE", "1")
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", "/app/vllm")
    monkeypatch.setenv("PYTHONPATH", "/workspace:/opt/python")
    bench = {"framework": "vllm"}
    envs: dict[str, str] = {}

    we._apply_vllm_source_runtime(bench, envs)

    assert {key: envs[key] for key in ("FRAMEWORK_REPO_PATH", "VLLM_REPO_PATH", "VLLM_DIR")} == {
        "FRAMEWORK_REPO_PATH": "/app/vllm",
        "VLLM_REPO_PATH": "/app/vllm",
        "VLLM_DIR": "/app/vllm",
    }
    assert envs["PYTHONPATH"] == "/app/vllm:/workspace:/opt/python"


def test_non_vllm_launch_is_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_VLLM_IMAGE_SOURCE", "1")
    monkeypatch.setenv("FRAMEWORK_REPO_PATH", "/app/vllm")
    envs = {"PYTHONPATH": "/original"}

    we._apply_vllm_source_runtime({"framework": "sglang"}, envs)

    assert envs == {"PYTHONPATH": "/original"}
