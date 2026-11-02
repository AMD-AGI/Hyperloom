# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two sessions installed from one checkout each get their own runtime dir.

Every session run from a checkout shares its ``.env``. install.sh used to persist the
runtime paths it derives from ``USER_DATA_PATH`` there, so a second session that set
only ``USER_DATA_PATH`` read the first one's ``HYPERLOOM_RUNTIME_DIR`` / ``KERNEL_AGENT_ENV``
back and rewrote the first session's ``kernel-agent.env.sh`` (board #166).

The runs below execute install.sh's own path setup (from the top of the script through
the runtime-path derivation, with the shipped ``runtime_env.sh`` loader) followed by the
shipped ``write_env_file`` step that writes ``kernel-agent.env.sh`` and ``.env``. Only the
package/clone steps between the two are left out: they need a GPU image and a network.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from hyperloom.inference_optimizer.cli import preflight_env

from ._kernel_tools_harness import ASSETS, INSTALL_SH, sourceable_kernel_tools

_DERIVATION_END = "\nruntime_env_derive_runtime_paths\n"
_DERIVED_KEYS = ("HYPERLOOM_RUNTIME_DIR", "KERNEL_AGENT_ENV", "HYPERLOOM_ROOT")
_PINNED_GEAK_REF = "0" * 40


def _installer(tmp_path: Path, *, print_env_file: bool = False) -> Path:
    """install.sh's path setup, then the kernel-tools env step or (``print_env_file``) a print of the file it selected."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    head = text[: text.index(_DERIVATION_END) + len(_DERIVATION_END)]
    assets = tmp_path / "assets"
    if not assets.is_dir():
        assets.mkdir()
        (assets / "runtime_env.sh").write_bytes((ASSETS / "runtime_env.sh").read_bytes())
        kernel_tools = sourceable_kernel_tools(assets)
        (assets / "install.sh").write_text(
            f"{head}\nCHECK_ONLY=0\nDRY_RUN=0\n. {shlex.quote(kernel_tools.as_posix())}\nwrite_env_file\n",
            encoding="utf-8",
            newline="\n",
        )
        (assets / "selected_env_file.sh").write_text(
            f'{head}\nprintf "%s\\n" "$KERNEL_AGENT_ENV"\n', encoding="utf-8", newline="\n"
        )
    return assets / ("selected_env_file.sh" if print_env_file else "install.sh")


def _run_installer(script: Path, checkout: Path, shell: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
    env.update(REPO_ROOT=str(checkout), GEAK_REF=_PINNED_GEAK_REF, ANTHROPIC_BASE_URL="https://gateway.invalid")
    env.update(shell)
    proc = subprocess.run(
        ["bash", script.as_posix()], cwd=checkout, env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    return proc


def _install(
    tmp_path: Path, checkout: Path, user_data_path: Path | None, **shell: str
) -> subprocess.CompletedProcess[str]:
    if user_data_path is not None:
        shell["USER_DATA_PATH"] = str(user_data_path)
    return _run_installer(_installer(tmp_path), checkout, shell)


def _kernel_env(workspace: Path) -> Path:
    return workspace / "runtime" / "kernel-agent.env.sh"


def _dotenv_keys(checkout: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (checkout / ".env").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out.setdefault(key.strip(), value)
    return out


@pytest.fixture
def sessions(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    return checkout, tmp_path / "c16" / "workspace", tmp_path / "c4" / "workspace"


def _assert_own_runtime(workspace: Path) -> None:
    text = _kernel_env(workspace).read_text(encoding="utf-8")
    assert f"export USER_DATA_PATH='{workspace}'" in text
    assert f"export HYPERLOOM_RUNTIME_DIR='{workspace}/runtime'" in text
    assert f"export KERNEL_AGENT_ENV='{_kernel_env(workspace)}'" in text


def test_second_session_writes_its_own_runtime_dir_and_leaves_the_first_alone(tmp_path, sessions):
    checkout, c16, c4 = sessions
    _install(tmp_path, checkout, c16)
    first = _kernel_env(c16).read_bytes()
    first_mtime = _kernel_env(c16).stat().st_mtime_ns

    _install(tmp_path, checkout, c4)

    _assert_own_runtime(c4)
    assert _kernel_env(c16).read_bytes() == first
    assert _kernel_env(c16).stat().st_mtime_ns == first_mtime
    _assert_own_runtime(c16)
    assert not any(key in _dotenv_keys(checkout) for key in _DERIVED_KEYS)


def test_runtime_paths_an_older_install_left_in_dotenv_are_ignored_and_purged(tmp_path, sessions):
    """The #166 repro: .env as the previous install.sh wrote it for C16, then C4 sets only USER_DATA_PATH."""
    checkout, c16, c4 = sessions
    _install(tmp_path, checkout, c16)
    first = _kernel_env(c16).read_bytes()
    with (checkout / ".env").open("a", encoding="utf-8") as fh:
        fh.write(
            f"HYPERLOOM_RUNTIME_DIR={c16}/runtime\n"
            f"KERNEL_AGENT_ENV={_kernel_env(c16)}\n"
            f"HYPERLOOM_ROOT={c16}/runtime/source-mirrors\n"
        )

    _install(tmp_path, checkout, c4)

    _assert_own_runtime(c4)
    assert _kernel_env(c16).read_bytes() == first
    keys = _dotenv_keys(checkout)
    assert keys["USER_DATA_PATH"] == str(c4)
    assert not any(key in keys for key in _DERIVED_KEYS)


def test_single_session_rerun_without_user_data_path_keeps_its_workspace(tmp_path, sessions):
    checkout, c16, _ = sessions
    _install(tmp_path, checkout, c16)
    _kernel_env(c16).unlink()

    _install(tmp_path, checkout, None)

    _assert_own_runtime(c16)


def test_operator_runtime_path_in_dotenv_is_kept(tmp_path, sessions):
    checkout, c16, _ = sessions
    custom = tmp_path / "private" / "ka.env.sh"
    (checkout / ".env").write_text(f"USER_DATA_PATH={c16}\nKERNEL_AGENT_ENV={custom}\n", encoding="utf-8")

    _install(tmp_path, checkout, None)

    assert custom.is_file()
    assert _dotenv_keys(checkout)["KERNEL_AGENT_ENV"] == str(custom)


def test_preflight_takes_runtime_paths_from_the_callers_workspace(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".env").write_text(
        "USER_DATA_PATH=/c16/workspace\n"
        "HYPERLOOM_RUNTIME_DIR=/c16/workspace/runtime\n"
        "KERNEL_AGENT_ENV=/c16/workspace/runtime/kernel-agent.env.sh\n"
        "HYPERLOOM_ROOT=/c16/workspace/runtime/source-mirrors\n",
        encoding="utf-8",
    )
    base = {key: value for key, value in os.environ.items() if key not in (*_DERIVED_KEYS, "USER_DATA_PATH")}
    base["REPO_ROOT"] = str(checkout)

    with mock.patch.dict(os.environ, {**base, "USER_DATA_PATH": "/c4/workspace"}, clear=True):
        preflight_env._load_dotenv_fallback()
        assert os.environ["USER_DATA_PATH"] == "/c4/workspace"
        assert not any(key in os.environ for key in _DERIVED_KEYS)

    with mock.patch.dict(os.environ, base, clear=True):
        preflight_env._load_dotenv_fallback()
        assert os.environ["KERNEL_AGENT_ENV"] == "/c16/workspace/runtime/kernel-agent.env.sh"


def _caller_env(checkout: Path, shell: dict[str, str]) -> dict[str, str]:
    """This process's environment minus the workspace variables, plus what one session's shell sets."""
    ours = (*_DERIVED_KEYS, "USER_DATA_PATH", "HYPERLOOM_SETUP_ENV_AUTHORITATIVE")
    base = {key: value for key, value in os.environ.items() if key not in ours}
    return {**base, "REPO_ROOT": str(checkout), **shell}


def _preflight_env_file(checkout: Path, shell: dict[str, str]) -> str:
    """Run preflight's two env fallbacks as `hyperloom optimize` does and return the env file it loaded."""
    with mock.patch.dict(os.environ, _caller_env(checkout, shell), clear=True):
        preflight_env._load_dotenv_fallback()
        outcome = preflight_env._load_kernel_agent_env_fallback()
    return outcome["detail"]["env_file"]


def test_preflight_finds_the_env_file_under_an_exported_runtime_dir(tmp_path, sessions):
    """troubleshooting.md: export USER_DATA_PATH and a node-local HYPERLOOM_RUNTIME_DIR, install, then optimize."""
    checkout, workspace, _ = sessions
    runtime_dir = tmp_path / "var" / "lib" / "hyperloom" / "runtime"
    shell = {"USER_DATA_PATH": str(workspace), "HYPERLOOM_RUNTIME_DIR": str(runtime_dir)}

    _install(tmp_path, checkout, workspace, HYPERLOOM_RUNTIME_DIR=str(runtime_dir))

    written = runtime_dir / "kernel-agent.env.sh"
    assert written.is_file()
    assert not _kernel_env(workspace).exists()
    assert _preflight_env_file(checkout, shell) == str(written)


def test_preflight_finds_the_env_file_under_a_runtime_dir_kept_in_dotenv(tmp_path, sessions):
    """A single session whose operator-set HYPERLOOM_RUNTIME_DIR lives only in .env."""
    checkout, workspace, _ = sessions
    runtime_dir = tmp_path / "private-runtime"
    (checkout / ".env").write_text(
        f"USER_DATA_PATH={workspace}\nHYPERLOOM_RUNTIME_DIR={runtime_dir}\n", encoding="utf-8"
    )

    _install(tmp_path, checkout, None)

    written = runtime_dir / "kernel-agent.env.sh"
    assert written.is_file()
    assert _dotenv_keys(checkout)["HYPERLOOM_RUNTIME_DIR"] == str(runtime_dir)
    assert _preflight_env_file(checkout, {}) == str(written)


def test_setup_written_dotenv_selects_the_runtime_dir_for_install_and_preflight(tmp_path, sessions):
    """The reviewer's probe: `hyperloom setup` wrote .env (it holds HYPERLOOM_RUN_MODE), the shell names another root."""
    checkout, a, b = sessions
    (checkout / ".env").write_text(f"HYPERLOOM_RUN_MODE=baremetal\nUSER_DATA_PATH={a}\n", encoding="utf-8")

    _install(tmp_path, checkout, b)

    assert _kernel_env(a).is_file()
    assert not _kernel_env(b).exists()
    assert _preflight_env_file(checkout, {"USER_DATA_PATH": str(b)}) == str(_kernel_env(a))


def test_setup_written_dotenv_outranks_a_runtime_dir_the_shell_exports(tmp_path, sessions):
    checkout, a, b = sessions
    stale = tmp_path / "stale-runtime"
    (checkout / ".env").write_text(f"HYPERLOOM_RUN_MODE=baremetal\nUSER_DATA_PATH={a}\n", encoding="utf-8")
    shell = {"USER_DATA_PATH": str(b), "HYPERLOOM_RUNTIME_DIR": str(stale)}

    _install(tmp_path, checkout, b, HYPERLOOM_RUNTIME_DIR=str(stale))

    assert _kernel_env(a).is_file()
    assert not stale.exists()
    assert _preflight_env_file(checkout, shell) == str(_kernel_env(a))


# (.env as written, what the session's shell exports, the env file both must select); {a} and {b} are two workspaces.
_SELECTION_CASES = {
    "caller-root-over-a-plain-dotenv": ("USER_DATA_PATH={a}", {"USER_DATA_PATH": "{b}"}, "{b}/runtime"),
    "caller-root-over-derived-copies-in-dotenv": (
        "USER_DATA_PATH={a}\nHYPERLOOM_RUNTIME_DIR={a}/runtime\nKERNEL_AGENT_ENV={a}/runtime/kernel-agent.env.sh",
        {"USER_DATA_PATH": "{b}"},
        "{b}/runtime",
    ),
    "dotenv-root-when-the-shell-is-silent": ("USER_DATA_PATH={a}", {}, "{a}/runtime"),
    "caller-env-file-over-a-plain-dotenv": (
        "USER_DATA_PATH={a}",
        {"USER_DATA_PATH": "{b}", "KERNEL_AGENT_ENV": "{b}-ka.sh"},
        None,
    ),
    "setup-dotenv-over-caller-root": (
        "HYPERLOOM_RUN_MODE=baremetal\nUSER_DATA_PATH={a}",
        {"USER_DATA_PATH": "{b}"},
        "{a}/runtime",
    ),
    "setup-dotenv-over-caller-runtime-paths": (
        "HYPERLOOM_RUN_MODE=baremetal\nUSER_DATA_PATH={a}",
        {"USER_DATA_PATH": "{b}", "HYPERLOOM_RUNTIME_DIR": "{b}-rt", "KERNEL_AGENT_ENV": "{b}-ka.sh"},
        "{a}/runtime",
    ),
    "setup-dotenv-keeps-its-own-runtime-dir": (
        "HYPERLOOM_RUN_MODE=baremetal\nUSER_DATA_PATH={a}\nHYPERLOOM_RUNTIME_DIR={a}-rt",
        {"USER_DATA_PATH": "{b}"},
        "{a}-rt",
    ),
    "setup-dotenv-with-a-quoted-root": (
        'HYPERLOOM_RUN_MODE=baremetal\nUSER_DATA_PATH="{a} with space"',
        {"USER_DATA_PATH": "{b}"},
        "{a} with space/runtime",
    ),
    "setup-flag-in-the-shell-makes-any-dotenv-authoritative": (
        "USER_DATA_PATH={a}",
        {"USER_DATA_PATH": "{b}", "HYPERLOOM_SETUP_ENV_AUTHORITATIVE": "1"},
        "{a}/runtime",
    ),
}


@pytest.mark.parametrize(("dotenv", "shell", "runtime_dir"), _SELECTION_CASES.values(), ids=_SELECTION_CASES)
def test_install_preflight_and_the_documented_source_line_select_the_same_env_file(
    tmp_path, sessions, dotenv, shell, runtime_dir
):
    """One rule, three readers: runtime_env.sh decides for install.sh and the documented `source` lines, and
    preflight's Python mirror of it must give the same file for the same shell and .env."""
    checkout, a, b = sessions
    names = {"a": str(a), "b": str(b)}
    (checkout / ".env").write_text(dotenv.format(**names) + "\n", encoding="utf-8")
    shell = {key: value.format(**names) for key, value in shell.items()}
    expected = f"{runtime_dir.format(**names)}/kernel-agent.env.sh" if runtime_dir else shell["KERNEL_AGENT_ENV"]

    installed = _run_installer(_installer(tmp_path, print_env_file=True), checkout, shell).stdout.splitlines()[-1]
    documented = subprocess.run(
        ["bash", "-c", '. "$1" && runtime_env_kernel_agent_env_path', "bash", str(ASSETS / "runtime_env.sh")],
        cwd=checkout,
        env=_caller_env(checkout, shell),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    with mock.patch.dict(os.environ, _caller_env(checkout, shell), clear=True):
        preflight_env._load_dotenv_fallback()
        looked_for = preflight_env._kernel_agent_env_candidate()

    assert installed == expected
    assert documented == expected
    assert looked_for == expected
