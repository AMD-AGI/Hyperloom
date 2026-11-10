# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two sessions installed from one checkout each get their own runtime dir and kernel-agent env file.

Every session run from a checkout shares its ``.env``. install.sh used to persist the
runtime paths it derives from ``USER_DATA_PATH`` there, so a second session that set
only ``USER_DATA_PATH`` read the first one's ``HYPERLOOM_RUNTIME_DIR`` / ``KERNEL_AGENT_ENV``
back and rewrote the first session's ``kernel-agent.env.sh`` (board #166).

``kernel-agent.env.sh`` carries credentials and is sourced, so it is kept off ``USER_DATA_PATH``,
which every pod of the user shares and can write: it is written into a directory of this
account's own under the container's ``TMPDIR``, one per runtime dir keyed absolute, and a copy an
older install left in the runtime dir is removed. That name is predictable, so every reader
refuses a directory or file there that another account could have put or changed.

The runs below execute install.sh's own path setup (from the top of the script through
the runtime-path derivation, with the shipped ``runtime_env.sh`` loader) followed by the
shipped ``write_env_file`` step that writes ``kernel-agent.env.sh`` and ``.env``. Only the
package/clone steps between the two are left out: they need a GPU image and a network.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli import preflight_env

from ._kernel_tools_harness import ASSETS, INSTALL_SH, RUNTIME_ENV_SH, sourceable_kernel_tools

_DERIVATION_END = "\nruntime_env_derive_runtime_paths\n"
_DERIVED_KEYS = ("HYPERLOOM_RUNTIME_DIR", "KERNEL_AGENT_ENV", "HYPERLOOM_ROOT")
_PINNED_GEAK_REF = "0" * 40
_GATEWAY = "https://gateway.invalid"


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


def _installer_run(
    script: Path, checkout: Path, shell: dict[str, str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ}
    env.update(REPO_ROOT=str(checkout), GEAK_REF=_PINNED_GEAK_REF, ANTHROPIC_BASE_URL=_GATEWAY)
    env.update(shell)
    return subprocess.run(
        ["bash", script.as_posix()], cwd=cwd or checkout, env=env, capture_output=True, text=True, timeout=60
    )


def _run_installer(script: Path, checkout: Path, shell: dict[str, str]) -> subprocess.CompletedProcess[str]:
    proc = _installer_run(script, checkout, shell)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    return proc


def _install(
    tmp_path: Path, checkout: Path, user_data_path: Path | None, **shell: str
) -> subprocess.CompletedProcess[str]:
    if user_data_path is not None:
        shell["USER_DATA_PATH"] = str(user_data_path)
    return _run_installer(_installer(tmp_path), checkout, shell)


def _private_env_file(runtime_dir: str) -> Path:
    """The documented default: ``${TMPDIR:-/tmp}/hyperloom-<uid>-<first 16 hex of sha256(runtime dir)>/...``."""
    key = hashlib.sha256(runtime_dir.encode()).hexdigest()[:16]
    return Path(os.environ["TMPDIR"]) / f"hyperloom-{os.getuid()}-{key}" / "kernel-agent.env.sh"


def _kernel_env(workspace: Path) -> Path:
    return _private_env_file(f"{workspace}/runtime")


def _stand_in_installed_file(env_file: Path) -> None:
    """Leave a private file where install.sh writes one: the documented line prints only a file that is there."""
    env_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    env_file.write_text(f"export ANTHROPIC_BASE_URL='{_GATEWAY}'\n", encoding="utf-8")
    env_file.chmod(0o600)


def _legacy_env(workspace: Path) -> Path:
    return workspace / "runtime" / "kernel-agent.env.sh"


def _dotenv_keys(checkout: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (checkout / ".env").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out.setdefault(key.strip(), value)
    return out


@pytest.fixture
def sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """A checkout and two workspaces; this test's own TMPDIR stands for the container's disk in every reader."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (tmp_path / "tmp").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
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
            f"KERNEL_AGENT_ENV={_legacy_env(c16)}\n"
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


def test_preflight_finds_the_env_file_of_an_exported_runtime_dir(tmp_path, sessions):
    """troubleshooting.md: export USER_DATA_PATH and a node-local HYPERLOOM_RUNTIME_DIR, install, then optimize."""
    checkout, workspace, _ = sessions
    runtime_dir = tmp_path / "var" / "lib" / "hyperloom" / "runtime"
    shell = {"USER_DATA_PATH": str(workspace), "HYPERLOOM_RUNTIME_DIR": str(runtime_dir)}

    _install(tmp_path, checkout, workspace, HYPERLOOM_RUNTIME_DIR=str(runtime_dir))

    written = _private_env_file(str(runtime_dir))
    assert written.is_file()
    assert not (runtime_dir / "kernel-agent.env.sh").exists()
    assert not _legacy_env(workspace).exists()
    assert _preflight_env_file(checkout, shell) == str(written)


def test_preflight_finds_the_env_file_under_a_runtime_dir_kept_in_dotenv(tmp_path, sessions):
    """A single session whose operator-set HYPERLOOM_RUNTIME_DIR lives only in .env."""
    checkout, workspace, _ = sessions
    runtime_dir = tmp_path / "private-runtime"
    (checkout / ".env").write_text(
        f"USER_DATA_PATH={workspace}\nHYPERLOOM_RUNTIME_DIR={runtime_dir}\n", encoding="utf-8"
    )

    _install(tmp_path, checkout, None)

    written = _private_env_file(str(runtime_dir))
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
    # Where an older install put the file, kept by its .env or by a shell that sourced the file: derived again.
    "legacy-env-file-in-dotenv": (
        "USER_DATA_PATH={a}\nKERNEL_AGENT_ENV={a}/runtime/kernel-agent.env.sh",
        {},
        "{a}/runtime",
    ),
    "legacy-env-file-in-the-shell": (
        "USER_DATA_PATH={a}",
        {"USER_DATA_PATH": "{b}", "KERNEL_AGENT_ENV": "{b}/runtime/kernel-agent.env.sh"},
        "{b}/runtime",
    ),
    "legacy-env-file-of-an-exported-runtime-dir": (
        "USER_DATA_PATH={a}",
        {"USER_DATA_PATH": "{b}", "HYPERLOOM_RUNTIME_DIR": "{b}-rt", "KERNEL_AGENT_ENV": "{b}-rt/kernel-agent.env.sh"},
        "{b}-rt",
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
    expected = str(_private_env_file(runtime_dir.format(**names))) if runtime_dir else shell["KERNEL_AGENT_ENV"]

    installed = _run_installer(_installer(tmp_path, print_env_file=True), checkout, shell).stdout.splitlines()[-1]
    _stand_in_installed_file(Path(expected))
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


def _preflight_loads(checkout: Path, shell: dict[str, str]) -> tuple[str, str | None]:
    """The env file preflight loaded for this shell, and the ``ANTHROPIC_BASE_URL`` it left in the environment."""
    env = {key: value for key, value in _caller_env(checkout, shell).items() if not key.startswith("ANTHROPIC_")}
    with mock.patch.dict(os.environ, env, clear=True):
        preflight_env._load_dotenv_fallback()
        outcome = preflight_env._load_kernel_agent_env_fallback()
        return outcome["detail"]["env_file"], os.environ.get("ANTHROPIC_BASE_URL")


def test_the_env_file_is_written_0600_into_a_0700_directory_of_its_own_off_the_workspace(tmp_path, sessions):
    checkout, workspace, _ = sessions

    _install(tmp_path, checkout, workspace)

    written = _kernel_env(workspace)
    assert stat.S_IMODE(written.stat().st_mode) == 0o600
    assert stat.S_IMODE(written.parent.stat().st_mode) == 0o700
    assert written.stat().st_uid == written.parent.stat().st_uid == os.geteuid()
    assert [entry.name for entry in written.parent.iterdir()] == [written.name]
    assert workspace not in written.parents
    assert not _legacy_env(workspace).exists()


def test_preflight_and_the_documented_source_lines_read_the_file_install_wrote(tmp_path, sessions):
    checkout, workspace, _ = sessions
    shell = {"USER_DATA_PATH": str(workspace)}
    _install(tmp_path, checkout, workspace)
    _link_assets(checkout)

    sourced = [_source_documented(checkout, shell, where, line) for where, line in _documented_lines()]

    assert all(run[:3] == (0, _GATEWAY, str(_kernel_env(workspace))) for run in sourced), sourced
    assert _preflight_loads(checkout, shell) == (str(_kernel_env(workspace)), _GATEWAY)


def test_a_copy_an_older_install_left_in_the_runtime_dir_is_removed(tmp_path, sessions):
    checkout, workspace, _ = sessions
    legacy = _legacy_env(workspace)
    legacy.parent.mkdir(parents=True)
    legacy.write_text("export ANTHROPIC_API_KEY='left-by-an-older-install'\n", encoding="utf-8")

    _install(tmp_path, checkout, workspace)

    assert not legacy.exists()
    assert _kernel_env(workspace).is_file()


@pytest.mark.parametrize("blocker", ["read-only-runtime-dir", "directory-in-its-place"])
def test_a_copy_that_cannot_be_removed_is_named_and_never_read(tmp_path, sessions, blocker):
    checkout, workspace, _ = sessions
    legacy = _legacy_env(workspace)
    legacy.parent.mkdir(parents=True)
    if blocker == "directory-in-its-place":
        legacy.mkdir()
    elif os.geteuid() == 0:
        pytest.skip("root removes an entry of a read-only directory")
    else:
        legacy.write_text("export ANTHROPIC_BASE_URL='https://planted.invalid'\n", encoding="utf-8")
        legacy.parent.chmod(0o555)
    try:
        proc = _install(tmp_path, checkout, workspace)
    finally:
        legacy.parent.chmod(0o755)

    assert f"cannot remove the old shared copy of the kernel-agent env file {legacy}; it is not read" in proc.stderr
    assert "planted" not in proc.stdout + proc.stderr
    assert _preflight_loads(checkout, {"USER_DATA_PATH": str(workspace)}) == (str(_kernel_env(workspace)), _GATEWAY)


def test_an_earlier_file_another_account_owns_is_replaced_not_rewritten(tmp_path, sessions):
    """The new file is renamed over an earlier one: truncating or chmod-ing it in place fails on a file of another
    account's, and as root would leave the credentials in a file that account owns."""
    checkout, workspace, _ = sessions
    own = tmp_path / "own"
    own.mkdir(mode=0o700)
    earlier = own / "kernel-agent.env.sh"
    earlier.write_text("export KERNEL_AGENT_LOG_LEVEL=stale\n", encoding="utf-8")
    if os.geteuid() == 0:
        os.chown(earlier, 65534, 65534)
        earlier.chmod(0o600)
    else:
        earlier.chmod(0o400)
    before = earlier.stat()

    _install(tmp_path, checkout, workspace, KERNEL_AGENT_ENV=str(earlier))

    after = earlier.stat()
    assert after.st_ino != before.st_ino
    assert (after.st_uid, stat.S_IMODE(after.st_mode)) == (os.geteuid(), 0o600)
    assert f"export KERNEL_AGENT_ENV='{earlier}'" in earlier.read_text(encoding="utf-8")


def test_a_directory_other_accounts_can_write_is_refused(tmp_path, sessions):
    """A user directory every pod of the user can write, as on shared storage, never holds the file."""
    checkout, workspace, _ = sessions
    workspace.mkdir(parents=True)
    workspace.chmod(0o2770)
    target = workspace / "kernel-agent.env.sh"

    proc = _installer_run(
        _installer(tmp_path), checkout, {"USER_DATA_PATH": str(workspace), "KERNEL_AGENT_ENV": str(target)}
    )

    assert proc.returncode != 0
    assert f"{workspace} can be written by its group or others (mode 2770)" in proc.stderr
    assert not target.exists()


_PLANTED_VALUES = ("https://planted.invalid", "/planted/runner")
_PLANTED = f"export ANTHROPIC_BASE_URL='{_PLANTED_VALUES[0]}'\nexport GEAK_E2E_RUNNER='{_PLANTED_VALUES[1]}'\n"
_REFUSAL_ADVICE = "Set KERNEL_AGENT_ENV to a file in such a directory on this container's own disk, or TMPDIR to such"
_REPO = ASSETS.parents[3]
_DOC_SUFFIXES = frozenset({".md", ".rst", ".sh", ".in", ".example", ".sbatch", ".template"})
# Where the line is documented today; the scan below must at least find these, so it cannot pass by finding nothing.
_DOCUMENTED_IN = frozenset(
    {
        "src/hyperloom/inference_optimizer/assets/setup_env.sh.example",
        "src/hyperloom/inference_optimizer/assets/slurm/_incontainer.sh.in",
        "docs/reference/upgrade.md",
    }
)


def _documented_lines() -> list[tuple[str, str]]:
    """Every line the docs and the shipped assets give for loading the file, as (file, the line a shell runs).

    Scanned rather than listed, so a line documented later is held to the same tests; a diff's leading ``+`` is dropped.
    """
    found: list[tuple[str, str]] = []
    candidates = {*(_REPO / "docs").rglob("*"), *(_REPO / "src" / "hyperloom").rglob("*"), *_REPO.glob("*")}
    for path in sorted(candidates):
        where = path.relative_to(_REPO)
        if path.suffix not in _DOC_SUFFIXES or path == RUNTIME_ENV_SH or "tests" in where.parts or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip().removeprefix("+ ").strip()
            if "runtime_env_kernel_agent_env_path" in text and not text.startswith("#"):
                found.append((where.as_posix(), text))
    assert {where for where, _ in found} >= _DOCUMENTED_IN, found
    return found


def _link_assets(checkout: Path, assets: Path = ASSETS) -> None:
    """Put ``assets`` where the documented lines look from the checkout: the src layout, and the docs' older one."""
    for package in (checkout / "src" / "hyperloom", checkout / "hyperloom"):
        (package / "inference_optimizer").mkdir(parents=True)
        (package / "inference_optimizer" / "assets").symlink_to(assets)


def _source_documented(
    checkout: Path, shell: dict[str, str], where: str, line: str
) -> tuple[int, str | None, str | None, str]:
    """Run one documented line from the checkout as its file runs it, the slurm job under ``set -euo pipefail``.

    Returns the line's status, the ``ANTHROPIC_BASE_URL`` and ``KERNEL_AGENT_ENV`` it left (None when the job stopped
    at the line), and stderr.
    """
    prelude = "set -euo pipefail\n" if where.endswith(".sh.in") else ""
    report = 'printf "status %s loaded %s from %s\\n" "$?" "${ANTHROPIC_BASE_URL-}" "${KERNEL_AGENT_ENV-}"'
    proc = _shell_reader(checkout, shell, f"{prelude}{line}\n{report}\n")
    ran_on = re.fullmatch(r"status (\d+) loaded (.*) from (.*)\n", proc.stdout)
    if ran_on is None:
        assert proc.returncode != 0 and not proc.stdout, (where, proc)
        return proc.returncode, None, None, proc.stderr
    return int(ran_on[1]), ran_on[2], ran_on[3], proc.stderr


def _assert_every_documented_line_stops(checkout: Path, shell: dict[str, str], *said: str) -> None:
    """Every documented line fails with the helper's message as its only output and loads nothing; the job stops."""
    for where, line in _documented_lines():
        status, loaded, _, err = _source_documented(checkout, shell, where, line)
        assert not loaded, (where, loaded)
        assert status != 0, where
        assert len(err.splitlines()) == 1, (where, err)
        assert all(part in err for part in said), (where, err)
        assert not any(value in err for value in _PLANTED_VALUES), where


def _give_another_group(entry: Path) -> None:
    """Give ``entry`` a group other than this account's own where this account may: as root, or as a member of one."""
    others = [65534] if os.geteuid() == 0 else [gid for gid in os.getgroups() if gid != os.getegid()]
    if others:
        os.chown(entry, -1, others[0])


def _plant(env_file: Path, plant: str) -> str:
    """Put what another account could leave at the derived place; return the refusal every reader must give."""
    directory = env_file.parent
    if plant.endswith("other-owner") and os.geteuid() != 0:
        pytest.skip("only root can give an entry to another account")
    if plant == "directory-is-a-link":
        elsewhere = env_file.parents[1] / "elsewhere"
        elsewhere.mkdir(mode=0o700)
        (elsewhere / env_file.name).write_text(_PLANTED, encoding="utf-8")
        (elsewhere / env_file.name).chmod(0o600)
        directory.symlink_to(elsewhere)
        return f"{directory} is a symbolic link"
    directory.mkdir(mode=0o700)
    if plant == "file-is-a-link":
        elsewhere = env_file.parents[1] / "elsewhere.sh"
        elsewhere.write_text(_PLANTED, encoding="utf-8")
        elsewhere.chmod(0o600)
        env_file.symlink_to(elsewhere)
        return f"{env_file} is a symbolic link"
    env_file.write_text(_PLANTED, encoding="utf-8")
    env_file.chmod(0o600)
    uid = os.geteuid()
    if plant == "directory-of-another-owner":
        directory.chmod(0o755)
        os.chown(directory, 65534, 65534)
        return f"{directory} belongs to uid 65534, not to this account (uid {uid})"
    if plant == "world-writable-directory":
        directory.chmod(0o777)
        return f"{directory} can be written by its group or others (mode 777)"
    if plant == "group-writable-directory":
        directory.chmod(0o775)
        _give_another_group(directory)
        return f"{directory} can be written by its group or others (mode 775)"
    if plant == "file-of-another-owner":
        os.chown(env_file, 65534, 65534)
        return f"{env_file} belongs to uid 65534, not to this account (uid {uid})"
    if plant == "group-writable-file":
        env_file.chmod(0o620)
        _give_another_group(env_file)
        return f"{env_file} can be written by its group or others (mode 620)"
    assert plant == "world-writable-file"
    env_file.chmod(0o666)
    return f"{env_file} can be written by its group or others (mode 666)"


def _shell_reader(checkout: Path, shell: dict[str, str], script: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` from the checkout, with the caller's environment minus any gateway it already holds."""
    env = {key: value for key, value in _caller_env(checkout, shell).items() if not key.startswith("ANTHROPIC_")}
    return subprocess.run(
        ["bash", "-c", script, "bash", str(ASSETS / "runtime_env.sh")],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "plant",
    [
        "directory-of-another-owner",
        "world-writable-directory",
        "group-writable-directory",
        "directory-is-a-link",
        "file-of-another-owner",
        "world-writable-file",
        "group-writable-file",
        "file-is-a-link",
    ],
)
def test_a_planted_env_file_is_refused_by_every_reader(tmp_path, sessions, capsys, plant):
    """Another account that works out the predictable path cannot get its file sourced: the documented source lines,
    the slurm job and preflight all refuse it, name the entry and why, and never show what it holds; install.sh
    refuses to write into a planted directory. The group-writable cases hold the check to the group as well as to
    others: a 0775 directory or 0620 file passes a check narrowed to others' write bit."""
    checkout, workspace, _ = sessions
    shell = {"USER_DATA_PATH": str(workspace)}
    refusal = _plant(_kernel_env(workspace), plant)
    _link_assets(checkout)

    _assert_every_documented_line_stops(
        checkout, shell, f"refusing the kernel-agent env file {_kernel_env(workspace)}: {refusal}; ", _REFUSAL_ADVICE
    )
    caller = {key: value for key, value in _caller_env(checkout, shell).items() if not key.startswith("ANTHROPIC_")}
    with mock.patch.dict(os.environ, caller, clear=True):
        preflight_env._load_dotenv_fallback()
        with pytest.raises(SystemExit) as preflight:
            preflight_env._load_kernel_agent_env_fallback()
        preflight_loaded = os.environ.get("ANTHROPIC_BASE_URL"), os.environ.get("GEAK_E2E_RUNNER")
    preflight_err = capsys.readouterr().err

    assert preflight.value.code == 2
    assert f"refusing the kernel-agent env file {_kernel_env(workspace)}: {refusal}; " in preflight_err
    assert _REFUSAL_ADVICE in preflight_err
    assert not any(value in preflight_err for value in _PLANTED_VALUES)
    assert preflight_loaded == (None, None)
    if plant.startswith("directory") or plant.endswith("directory"):
        installed = _installer_run(_installer(tmp_path), checkout, shell)
        assert installed.returncode != 0
        assert f"refusing the kernel-agent env file {_kernel_env(workspace)}: {refusal}; " in installed.stderr


@pytest.mark.parametrize("missing", ["workspace", "directory", "file"])
def test_a_reader_refuses_an_env_file_install_has_not_written(sessions, missing):
    """A path printed for a name that does not exist yet would be sourced after another account created it there, so
    the helper prints nothing and fails, saying to run install.sh in this container first; every documented line then
    stops with that message alone rather than sourcing an empty name."""
    checkout, workspace, _ = sessions
    env_file = _kernel_env(workspace)
    shell = {} if missing == "workspace" else {"USER_DATA_PATH": str(workspace)}
    if missing == "file":
        env_file.parent.mkdir(mode=0o700)
    _link_assets(checkout)
    if missing == "workspace":
        said = (
            "no kernel-agent env file to load: none of KERNEL_AGENT_ENV, HYPERLOOM_RUNTIME_DIR, USER_DATA_PATH is set"
        )
    else:
        said = f"no kernel-agent env file at {env_file}: {env_file.parent if missing == 'directory' else env_file} does"

    helper = _shell_reader(checkout, shell, '. "$1" && runtime_env_kernel_agent_env_path')

    assert (helper.returncode, helper.stdout) == (1, "")
    assert said in helper.stderr
    assert "run install.sh in this container first" in helper.stderr
    _assert_every_documented_line_stops(checkout, shell, said, "run install.sh in this container first")
    assert not env_file.exists()


# runtime_env.sh as shipped, with another account creating the file's directory and the file the moment the helper has
# answered, before the caller's `.` runs: the window a path printed for a missing name leaves open.
_RACE_RUNTIME_ENV = """. {shipped}
eval "_answered_$(declare -f runtime_env_kernel_agent_env_path)"
runtime_env_kernel_agent_env_path() {{
  local status=0
  _answered_runtime_env_kernel_agent_env_path "$@" || status=$?
  mkdir -p -m 700 -- "${{RACE_FILE%/*}}" && printf '%s' "$RACE_PLANT" > "$RACE_FILE" && chmod 600 -- "$RACE_FILE"
  [ "$(id -u)" != 0 ] || chown -R 65534:65534 -- "${{RACE_FILE%/*}}"
  return "$status"
}}
"""


def test_a_file_planted_after_the_helper_answered_is_never_sourced(tmp_path, sessions):
    """The race: the helper runs while the directory does not exist, then another account (uid 65534 when the test
    runs as root) creates it and the file before the caller sources what the helper printed. The helper has already
    failed and printed nothing, so no documented line loads what was planted."""
    checkout, workspace, _ = sessions
    env_file = _kernel_env(workspace)
    shell = {"USER_DATA_PATH": str(workspace), "RACE_FILE": str(env_file), "RACE_PLANT": _PLANTED}
    racing = tmp_path / "racing-assets"
    racing.mkdir()
    (racing / "runtime_env.sh").write_text(
        _RACE_RUNTIME_ENV.format(shipped=shlex.quote(RUNTIME_ENV_SH.as_posix())), encoding="utf-8"
    )
    _link_assets(checkout, racing)

    for where, line in _documented_lines():
        shutil.rmtree(env_file.parent, ignore_errors=True)
        status, loaded, _, err = _source_documented(checkout, shell, where, line)
        assert env_file.is_file(), f"{where}: the plant never landed, so the window was not exercised"
        assert not loaded, (where, loaded)
        assert status != 0, where
        assert f"no kernel-agent env file at {env_file}: {env_file.parent} does not exist" in err, (where, err)
        assert not any(value in err for value in _PLANTED_VALUES), where


class _Parsed(Exception):
    pass


def _hyperloom_candidate(cwd: Path, env: dict[str, str]) -> tuple[str | None, str | None]:
    """Preflight's env file for this shell as is, and after ``hyperloom``'s own entry point made it absolute."""
    with mock.patch.dict(os.environ, env, clear=True):
        os.chdir(cwd)
        preflight_env._load_dotenv_fallback()
        as_exported = preflight_env._kernel_agent_env_candidate()
        with mock.patch.object(cli, "_build_parser", side_effect=_Parsed), pytest.raises(_Parsed):
            cli.main([])
        return as_exported, preflight_env._kernel_agent_env_candidate()


# (USER_DATA_PATH, HYPERLOOM_RUNTIME_DIR or None, run from the link to the base dir rather than the base dir itself,
# HOME a link, the runtime dir the file is keyed on); {base} is the run's directory, {home} is HOME.
_ABSOLUTE_CASES = {
    "relative": ("ud", None, False, False, "{base}/ud/runtime"),
    "relative-with-dot-and-trailing-slashes": ("./ud//", None, False, False, "{base}/ud/runtime"),
    "tilde": ("~", None, False, False, "{home}/runtime"),
    "tilde-slash": ("~/ud/", None, False, False, "{home}/ud/runtime"),
    "relative-from-a-working-directory-entered-through-a-link": ("ud", None, True, False, "{base}/ud/runtime"),
    "tilde-with-home-a-link": ("~/ud", None, False, True, "{home}/ud/runtime"),
    "relative-runtime-dir": ("{base}/ud", "rt/", False, False, "{base}/rt"),
}


@pytest.mark.parametrize(
    ("user_data", "runtime_dir", "via_link", "home_link", "keyed_on"), _ABSOLUTE_CASES.values(), ids=_ABSOLUTE_CASES
)
def test_install_preflight_and_the_documented_source_line_key_the_same_absolute_runtime_dir(
    tmp_path, sessions, monkeypatch, user_data, runtime_dir, via_link, home_link, keyed_on
):
    """A relative or ``~`` workspace is keyed absolute on every side, as ``hyperloom`` makes USER_DATA_PATH absolute
    before preflight: install.sh, the documented source line, and preflight before and after that step agree."""
    checkout, _, _ = sessions
    base = tmp_path / "base"
    base.mkdir()
    (tmp_path / "base-link").symlink_to(base)
    home = tmp_path / "home"
    home.mkdir()
    if home_link:
        (tmp_path / "home-link").symlink_to(home)
        home = tmp_path / "home-link"
    cwd = tmp_path / "base-link" if via_link else base
    names = {"base": str(base), "home": str(home)}
    shell = {"USER_DATA_PATH": user_data.format(**names), "HOME": str(home), "PWD": str(cwd)}
    if runtime_dir is not None:
        shell["HYPERLOOM_RUNTIME_DIR"] = runtime_dir
    expected = str(_private_env_file(keyed_on.format(**names)))
    monkeypatch.chdir(tmp_path)

    installed = _installer_run(_installer(tmp_path, print_env_file=True), checkout, shell, cwd=cwd)
    _stand_in_installed_file(Path(expected))
    documented = subprocess.run(
        ["bash", "-c", '. "$1" && runtime_env_kernel_agent_env_path', "bash", str(ASSETS / "runtime_env.sh")],
        cwd=cwd,
        env=_caller_env(checkout, shell),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert installed.returncode == 0, installed.stderr
    assert documented.returncode == 0, documented.stderr
    assert installed.stdout.splitlines()[-1] == expected
    assert documented.stdout.strip() == expected
    assert _hyperloom_candidate(cwd, _caller_env(checkout, shell)) == (expected, expected)


def test_preflight_loads_the_file_install_wrote_for_a_relative_workspace(tmp_path, sessions, monkeypatch):
    """The review's probe: USER_DATA_PATH=ud exported, install.sh, then `hyperloom optimize` from the same directory."""
    checkout, _, _ = sessions
    base = tmp_path / "base"
    base.mkdir()
    shell = {"USER_DATA_PATH": "ud"}
    monkeypatch.chdir(tmp_path)

    proc = _installer_run(_installer(tmp_path), checkout, shell, cwd=base)
    assert proc.returncode == 0, proc.stderr
    env = {key: value for key, value in _caller_env(checkout, shell).items() if not key.startswith("ANTHROPIC_")}
    with mock.patch.dict(os.environ, env, clear=True):
        os.chdir(base)
        with mock.patch.object(cli, "_build_parser", side_effect=_Parsed), pytest.raises(_Parsed):
            cli.main([])
        outcome = preflight_env._load_kernel_agent_env_fallback()
        loaded = os.environ.get("ANTHROPIC_BASE_URL")

    assert outcome["detail"]["env_file"] == str(_private_env_file(f"{base}/ud/runtime"))
    assert loaded == _GATEWAY
