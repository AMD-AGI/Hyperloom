# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Setup checks use one Python owner without installing or touching the workspace."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.inference_optimizer import setup

_INSTALLER = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
_CHECK_ARGS = ("--check-only", "--install-framework", "none", "--frameworks", "atom", "--require-frameworks")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _shell_path(path: Path) -> str:
    value = path.as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


def _python_owner(tmp_path: Path, *, venv: bool = True) -> tuple[Path, Path]:
    """Execute real Python probes/imports with isolated package and interpreter metadata."""
    root = tmp_path / "selected env"
    python = root / "bin" / "python"
    packages = root / "packages"
    _write(
        packages / "torch.py",
        "import os\nfrom types import SimpleNamespace\n"
        "__version__ = '2.10.0'\n"
        "version = SimpleNamespace(hip=None if os.environ.get('TEST_HIP_FAILURE') else '7.2.0')\n",
    )
    _write(
        packages / "atom" / "__init__.py",
        "import os\n"
        "if os.environ.get('TEST_ATOM_IMPORT_FAILURE'):\n"
        "    raise ImportError('test ATOM import failure')\n"
        "print('ATOM_IMPORTED')\n",
    )
    _write(packages / "atom" / "entrypoints" / "__init__.py", "")
    _write(
        packages / "atom" / "entrypoints" / "openai_server.py",
        "import os, sys\n"
        "assert sys.argv[1:] == ['--help'], sys.argv\n"
        "print('ATOM_HELP_PYTHON=' + sys.executable, file=sys.stderr)\n"
        "if os.environ.get('TEST_ATOM_HELP_FAILURE'):\n"
        "    raise SystemExit('test ATOM help failure')\n",
    )
    bootstrap = _write(
        root / "bootstrap.py",
        "import os, runpy, sys\n"
        "if sys.argv[1:2] == ['-B']:\n"
        "    sys.dont_write_bytecode = True\n"
        "    sys.argv.pop(1)\n"
        f"sys.executable = {_shell_path(python)!r}\n"
        f"sys.prefix = {_shell_path(root)!r}\n"
        f"sys.base_prefix = {('base-python' if venv else _shell_path(root))!r}\n"
        f"sys.path.insert(0, {packages.as_posix()!r})\n"
        "args = sys.argv[1:]\n"
        "if args[0] == '--version':\n"
        "    print('Python test owner')\n"
        "elif args[0] == '-c':\n"
        "    sys.argv = ['-c', *args[2:]]\n"
        "    exec(args[1])\n"
        "elif args[0] == '-':\n"
        "    sys.argv = args\n"
        "    exec(sys.stdin.read())\n"
        "elif args[0] == '-m':\n"
        "    sys.argv = [args[1], *args[2:]]\n"
        "    runpy.run_module(args[1], run_name='__main__')\n"
        "else:\n"
        "    raise SystemExit('Unexpected Python invocation: ' + repr(args))\n",
    )
    launcher = f'#!/usr/bin/env bash\nexec {shlex.quote(Path(sys.executable).as_posix())} {shlex.quote(bootstrap.as_posix())} "$@"\n'
    for name in ("python", "python3"):
        executable = _write(python.parent / name, launcher)
        executable.chmod(0o755)
    return python, root


def _run_setup(
    tmp_path: Path,
    python: Path,
    *,
    before: str = "",
    body: str = "main",
    extra_env: dict[str, str] | None = None,
    args: tuple[str, ...] = _CHECK_ARGS,
) -> subprocess.CompletedProcess[str]:
    text = _INSTALLER.read_text(encoding="utf-8")
    marker = '\nmain "$@"\n'
    assert marker in text
    library = _write(tmp_path / "installer-lib.sh", text.replace(marker, "\n"))
    runner = _write(
        tmp_path / "runner.sh",
        f"source {shlex.quote(library.as_posix())}\n"
        "rocm-smi() { printf 'test MI300X\\n'; }\n"
        "rocminfo() { printf 'gfx942\\n'; }\n"
        "check_torch_rocm_shared_libs() { :; }\n"
        "check_rocm_toolchain_alignment() { :; }\n"
        "check_torch_triton_alignment() { :; }\n"
        "export_rocm_sdk_toolchain_root() { :; }\n"
        "upsert_dotenv_var() { die 'MUTATION: dotenv'; }\n"
        "remove_dotenv_var() { die 'MUTATION: dotenv'; }\n"
        "download_rocm_profiler_hotfix_libs() { die 'MUTATION: download'; }\n"
        f"{before}\n{body}\n"
        'printf \'SELECTED_PYTHON=%s\\nSELECTED_VENV=%s\\n\' "${PYTHON:-}" "${VIRTUAL_ENV:-}"\n'
        "printf 'PATH_PYTHON3=%s\\n' \"$(command -v python3)\"\n",
    )
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("ANTHROPIC_", "OPENAI_", "DEEPSEEK_"))
    }
    for key in (
        "VIRTUAL_ENV",
        "FRAMEWORK_ENV",
        "FRAMEWORKS",
        "USER_DATA_PATH",
        "HYPERLOOM_SETUP_ENV_AUTHORITATIVE",
        "KERNEL_OPT_BACKEND_ORDER",
        "HYPERLOOM_RUN_MODE",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "LLM_GATEWAY_KEY",
    ):
        env.pop(key, None)
    env.update(
        REPO_ROOT=tmp_path.as_posix(),
        PYTHON=python.as_posix(),
        INFERENCE_OPTIMIZER_FORCE_PYTHON="1",
        MSYS2_ENV_CONV_EXCL="VIRTUAL_ENV",
        ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR=(tmp_path / "absent-rocm-libs").as_posix(),
    )
    env.update(extra_env or {})
    return subprocess.run(["bash", runner.as_posix(), *args], cwd=tmp_path, env=env, text=True, capture_output=True)


@pytest.mark.parametrize("pin", ["missing", "empty", "not-python"])
def test_invalid_forced_python_never_falls_back_to_valid_path(tmp_path: Path, pin: str) -> None:
    python, _ = _python_owner(tmp_path)
    bad = _write(tmp_path / "not-python", "#!/usr/bin/env bash\nexit 7\n")
    bad.chmod(0o755)
    value = {"missing": (tmp_path / "missing").as_posix(), "empty": "", "not-python": bad.as_posix()}[pin]
    result = _run_setup(
        tmp_path,
        python,
        before=f'export PATH={shlex.quote(_shell_path(python.parent))}:"$PATH"\npython3 --version',
        extra_env={"PYTHON": value},
    )
    assert "Python test owner" in result.stdout
    assert result.returncode != 0
    assert "PYTHON" in result.stderr
    assert "base preflight OK" not in result.stdout


@pytest.mark.parametrize("active_venv", ["unset", "same-prefix", "mismatch", "non-venv"])
def test_selected_python_owns_virtualenv_and_path(tmp_path: Path, active_venv: str) -> None:
    python, root = _python_owner(tmp_path, venv=active_venv != "non-venv")
    wrapper = _write(tmp_path / "python-wrapper", f'#!/usr/bin/env bash\nexec {shlex.quote(python.as_posix())} "$@"\n')
    wrapper.chmod(0o755)
    unrelated = tmp_path / "other env"
    unrelated.mkdir()
    active = _shell_path(root) + "/." if active_venv == "same-prefix" else _shell_path(unrelated)
    before = f'export PATH="$PATH":{shlex.quote(_shell_path(python.parent))}'
    result = _run_setup(
        tmp_path,
        wrapper,
        before=before,
        extra_env={} if active_venv == "unset" else {"VIRTUAL_ENV": active},
    )
    if active_venv in {"mismatch", "non-venv"}:
        assert result.returncode != 0
        assert "VIRTUAL_ENV" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert f"SELECTED_PYTHON={_shell_path(python)}" in result.stdout
        assert f"SELECTED_VENV={_shell_path(root)}" in result.stdout
        assert f"PATH_PYTHON3={_shell_path(python.parent)}/python3" in result.stdout


@pytest.mark.parametrize(
    ("failure", "diagnostic"),
    [
        ("TEST_ATOM_IMPORT_FAILURE", "test ATOM import failure"),
        ("TEST_ATOM_HELP_FAILURE", "test ATOM help failure"),
        ("TEST_HIP_FAILURE", "NOT a ROCm build"),
    ],
)
def test_required_atom_checks_real_import_help_and_rocm_torch(tmp_path: Path, failure: str, diagnostic: str) -> None:
    python, _ = _python_owner(tmp_path)
    result = _run_setup(tmp_path, python, extra_env={failure: "1"})
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert "base preflight OK" not in result.stdout


def test_check_only_atom_uses_selected_help_and_preserves_workspace(tmp_path: Path) -> None:
    python, root = _python_owner(tmp_path)
    dotenv = _write(tmp_path / ".env", "KEEP_ME=unchanged\n")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = _run_setup(tmp_path, python)
    assert result.returncode == 0, result.stderr
    assert f"ATOM_HELP_PYTHON={_shell_path(python)}" in result.stderr
    assert "verification pass complete" in result.stdout
    assert "MUTATION" not in result.stdout + result.stderr
    assert dotenv.read_text() == "KEEP_ME=unchanged\n"
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize("conflict", [False, True])
def test_readonly_workspace_root_must_match_selected_root(tmp_path: Path, conflict: bool) -> None:
    python, _ = _python_owner(tmp_path)
    root = (tmp_path / "data").as_posix()
    selected = (tmp_path / "other-data").as_posix() if conflict else root
    _write(tmp_path / ".env", f"USER_DATA_PATH={selected}\n")
    result = _run_setup(tmp_path, python, before=f"readonly USER_DATA_PATH={shlex.quote(root)}")
    if conflict:
        assert result.returncode != 0
        assert "USER_DATA_PATH" in result.stderr
        assert "conflict" in result.stderr.lower()
    else:
        assert result.returncode == 0, result.stderr
        assert "readonly variable" not in result.stderr


def test_readonly_workspace_root_rejects_conflicting_cli_override(tmp_path: Path) -> None:
    python, _ = _python_owner(tmp_path)
    result = _run_setup(
        tmp_path,
        python,
        before=f"readonly USER_DATA_PATH={shlex.quote((tmp_path / 'data').as_posix())}",
        args=(*_CHECK_ARGS, "--user-data-path", (tmp_path / "other-data").as_posix()),
    )
    assert result.returncode != 0
    assert "USER_DATA_PATH" in result.stderr
    assert "conflict" in result.stderr.lower()


def test_docker_check_ignores_dotenv_host_python_owner(tmp_path: Path) -> None:
    python, root = _python_owner(tmp_path)
    _write(
        tmp_path / ".env",
        "HYPERLOOM_RUN_MODE=docker\nPYTHON=/host/missing/python\n"
        "VIRTUAL_ENV=/host/venv\nINFERENCE_OPTIMIZER_FORCE_PYTHON=0\n",
    )
    result = _run_setup(tmp_path, python, extra_env={"HYPERLOOM_RUN_MODE": "docker"})
    assert result.returncode == 0, result.stderr
    assert f"SELECTED_PYTHON={_shell_path(python)}" in result.stdout
    assert f"SELECTED_VENV={_shell_path(root)}" in result.stdout
    assert "/host/" not in result.stdout + result.stderr


def test_setup_entrypoint_preserves_explicit_python_pin_without_mutating_caller(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON", "/caller/python")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_FORCE_PYTHON", "1")
    monkeypatch.setenv("PYTHONPATH", "/caller/pip-target")
    _write(tmp_path / ".env", "HYPERLOOM_RUN_MODE=docker\nPYTHON=/host/missing/python\n")
    seen = {}

    def run(cmd, *, env):
        seen.update(env)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(setup.subprocess, "run", run)
    assert setup.main(["--check-only", "--", "--install-framework", "none"]) == 0
    assert seen["PYTHON"] == "/caller/python"
    assert seen["INFERENCE_OPTIMIZER_FORCE_PYTHON"] == "1"
    assert os.environ["PYTHONPATH"] == "/caller/pip-target"
