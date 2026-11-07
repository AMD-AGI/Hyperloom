# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A checkout another account can write keeps no credentials in its ``.env``.

install_kernel_tools.sh ``write_env_file`` upserts the Anthropic key, the gateway URL and the subscription token into
``$REPO_ROOT/.env``. A checkout bind-mounted from shared storage (the slurm job's ``HYPERLOOM_SOURCE_DIR``) keeps that
file where other accounts read and rewrite it. On a checkout another account can write -- its directory or ``.env``
group- or other-writable, another account's, or a link -- the credentials go to the container-local kernel-agent env
file alone, the credential lines already in ``.env`` are removed (atomically, never through a link, every other byte
kept), and every reader that took them from ``.env`` takes them from that file. A private checkout is unchanged.

The installs below run install.sh's own path setup and credential preflight, then the shipped ``write_env_file``; the
package and clone steps between them need a GPU image and a network and are left out.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from hyperloom.inference_optimizer.cli import preflight as cli_preflight
from hyperloom.inference_optimizer.cli import preflight_env

from ._kernel_tools_harness import INSTALL_SH, RUNTIME_ENV_SH, install_sh_function, sourceable_kernel_tools
from .test_install_session_paths import (
    _DERIVATION_END,
    _GATEWAY,
    _PINNED_GEAK_REF,
    _documented_lines,
    _give_another_group,
    _kernel_env,
    _link_assets,
)

# Test values only; nothing here is a real credential.
_CREDENTIALS = {
    "ANTHROPIC_BASE_URL": _GATEWAY,
    "ANTHROPIC_API_KEY": "sk-test-shell-key",
    "ANTHROPIC_CUSTOM_HEADERS": "X-Tenant: shell-tenant\nX-Team: shell-team",
    "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test-shell-token",
}
# What an older install, or the operator, left in .env.
_OLD_CREDENTIALS = {
    "ANTHROPIC_BASE_URL": "https://old-gateway.invalid",
    "ANTHROPIC_API_KEY": "sk-test-old-key",
    "ANTHROPIC_CUSTOM_HEADERS": 'X-Tenant: old-tenant\nX-Team: "old-team"',
    "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test-old-token",
}
# The credential lines of _EXISTING, each of the shapes the loaders read: a plain one, an `export` one with single
# quotes, a double-quoted value over two lines with escaped quotes, an indented one, and a retired spelling.
_CREDENTIAL_LINES = (
    b"ANTHROPIC_BASE_URL=https://old-gateway.invalid\n",
    b"export ANTHROPIC_API_KEY='sk-test-old-key'\n",
    b'ANTHROPIC_CUSTOM_HEADERS="X-Tenant: old-tenant\n',
    b'X-Team: \\"old-team\\""\n',
    b"  CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-test-old-token\n",
    b"ANTHROPIC_AUTH_TOKEN=sk-test-retired\n",
)
# The other lines, which stay byte for byte: a comment that names a key, a CRLF line, the operator's OpenAI side, and a
# last line with no newline.
_OTHER_LINES = (
    b"# shared checkout settings\n",
    b"HF_HOME=/models/hf\n",
    b"KERNEL_OPT_BACKEND_ORDER=geak\r\n",
    b"# ANTHROPIC_API_KEY=only-a-comment\n",
    b"OPENAI_BASE_URL=https://gateway.invalid/v1\n",
    b"OPENAI_API_KEY=ak-test-openai-side",
)
_EXISTING = b"".join(
    (
        _OTHER_LINES[0],
        _OTHER_LINES[1],
        *_CREDENTIAL_LINES[:4],
        _OTHER_LINES[2],
        *_CREDENTIAL_LINES[4:],
        *_OTHER_LINES[3:],
    )
)
_SECRETS = (b"sk-test-", b"sk-ant-oat01-test", b"-tenant", b"-team", b"old-gateway")
_CREDENTIAL_KEYS = tuple(_CREDENTIALS)
# install.sh's credential preflight, run before the kernel-agent steps are sourced, as install.sh runs it.
_CREDENTIAL_PREFLIGHT = (
    "log",
    "warn",
    "die",
    "preflight_load_dotenv",
    "normalize_legacy_deepseek_env",
    "preflight_reject_cross_provider",
    "preflight_validate_credentials",
)
_OTHER_UID = 65534
_MARKER = "HYPERLOOM_DOTENV_CREDENTIALS"
_PLANTED = ("https://planted.invalid", "sk-test-planted")


def _write_installer(where: Path, before_write: str = "") -> Path:
    """install.sh's path setup, its credential preflight, then ``write_env_file`` (after ``before_write``), in files any
    account can read."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    head = text[: text.index(_DERIVATION_END) + len(_DERIVATION_END)]
    where.mkdir(parents=True)
    shutil.copyfile(RUNTIME_ENV_SH, where / "runtime_env.sh")
    kernel_tools = sourceable_kernel_tools(where)
    preflight = "\n".join(install_sh_function(name) for name in _CREDENTIAL_PREFLIGHT)
    script = where / "install.sh"
    script.write_text(
        f"{head}\nCHECK_ONLY=0\nDRY_RUN=0\n{preflight}\npreflight_validate_credentials\n"
        f". {shlex.quote(kernel_tools.as_posix())}\n{before_write}\nwrite_env_file\n",
        encoding="utf-8",
        newline="\n",
    )
    for entry in (where, script, where / "runtime_env.sh", kernel_tools):
        entry.chmod(0o755)
    return script


def _run(script: Path, checkout: Path, shell: dict[str, str], uid: int | None = None) -> subprocess.CompletedProcess:
    """Run ``script`` from the checkout with only what this shell exports, as ``uid`` when given."""
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ}
    env.update(REPO_ROOT=str(checkout), GEAK_REF=_PINNED_GEAK_REF, **shell)
    argv = ["bash", script.as_posix()]
    if uid is not None:
        argv = ["setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--", *argv]
    return subprocess.run(argv, cwd=checkout, env=env, capture_output=True, text=True, timeout=120)


def _install(script: Path, checkout: Path, workspace: Path, shell: dict[str, str]) -> subprocess.CompletedProcess:
    proc = _run(script, checkout, {"USER_DATA_PATH": str(workspace), **shell})
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    return proc


def _no_credentials() -> dict[str, str]:
    """A shell that exports none (empty is unset to every loader; _run always passes the gateway otherwise)."""
    return {"ANTHROPIC_BASE_URL": ""}


def _launch_env(checkout: Path, workspace: Path, **exported: str) -> dict[str, str]:
    """This process's environment without any credential, workspace or provider variable, plus a launch shell's."""
    dropped = ("ANTHROPIC_", "OPENAI_", "CLAUDE_", "DEEPSEEK_", "GEAK_", "HYPERLOOM_", "KERNEL_AGENT_", "LLM_")
    env = {key: value for key, value in os.environ.items() if not key.startswith(dropped)}
    env.pop("USER_DATA_PATH", None)
    return {**env, "REPO_ROOT": str(checkout), "USER_DATA_PATH": str(workspace), **exported}


def _assignments(path: Path) -> dict[str, str]:
    return preflight_env._parse_env_assignments(path.read_text(encoding="utf-8"))


def _credentials_in(path: Path) -> dict[str, str | None]:
    assignments = _assignments(path)
    return {key: assignments.get(key) for key in _CREDENTIAL_KEYS}


def _preflight_credentials(checkout: Path, workspace: Path, **exported: str) -> dict[str, str | None]:
    """The credentials preflight leaves for ``hyperloom optimize`` launched from a shell that exports ``exported``.

    It runs preflight's whole env normalisation, so the single-provider rule that strips the side ``.env`` did not
    supply is applied too: with the operator's OpenAI side in ``.env``, a credential not counted as ``.env``'s would be
    stripped here.
    """
    with (
        mock.patch.dict(os.environ, _launch_env(checkout, workspace, **exported), clear=True),
        mock.patch.object(cli_preflight, "_validate_explicit_server_limits", lambda args: None),
    ):
        cli_preflight._normalize_preflight_env(argparse.Namespace(), {"ext": {"steps": []}})
        return {key: os.environ.get(key) for key in _CREDENTIAL_KEYS}


def _sourced_credentials(checkout: Path, workspace: Path, where: str, line: str) -> dict[str, str]:
    """The credentials a documented ``source`` line (the slurm job's under ``set -euo pipefail``) leaves in its shell."""
    prelude = "set -euo pipefail\n" if where.endswith(".sh.in") else ""
    report = "".join(f'printf "%s\\0" "${{{key}-}}"\n' for key in _CREDENTIAL_KEYS)
    proc = subprocess.run(
        ["bash", "-c", f"{prelude}{line}\n{report}", "bash", RUNTIME_ENV_SH.as_posix()],
        cwd=checkout,
        env=_launch_env(checkout, workspace),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (where, proc.stderr)
    return dict(zip(_CREDENTIAL_KEYS, proc.stdout.split("\0")[:-1], strict=True))


def _assert_every_reader_gets(checkout: Path, workspace: Path, expected: dict[str, str]) -> None:
    assert _preflight_credentials(checkout, workspace) == expected
    _link_assets(checkout)
    for where, line in _documented_lines():
        assert _sourced_credentials(checkout, workspace, where, line) == expected, where


def _assert_no_credentials_and_the_rest_kept(dotenv: Path) -> None:
    """No credential assignment and no credential value in ``.env``; its other lines are there, in order, unchanged."""
    text = dotenv.read_bytes()
    assert not set(_assignments(dotenv)) & {*_CREDENTIAL_KEYS, "ANTHROPIC_AUTH_TOKEN"}
    assert not [secret for secret in _SECRETS if secret in text]
    lines = iter(text.splitlines(keepends=True))
    assert all(any(line.rstrip(b"\n") == other.rstrip(b"\n") for line in lines) for other in _OTHER_LINES)


@pytest.fixture
def container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A private checkout holding ``_EXISTING`` as its .env, and a workspace; this test's TMPDIR is the container's
    own disk."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    checkout.chmod(0o755)
    (checkout / ".env").write_bytes(_EXISTING)
    (checkout / ".env").chmod(0o600)
    (tmp_path / "tmp").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    return checkout, tmp_path / "workspace"


def _make_shared(checkout: Path, kind: str) -> str:
    """Make ``checkout`` one another account can write, the way ``kind`` names; return why the check must say so."""
    dotenv = checkout / ".env"
    uid = os.geteuid()
    if kind.endswith("another-account") and uid != 0:
        pytest.skip("only root can give an entry to another account")
    if kind == "group-writable-checkout":
        checkout.chmod(0o2775)
        _give_another_group(checkout)
        return f"{checkout} can be written by its group or others (mode 2775)"
    if kind == "world-writable-checkout":
        checkout.chmod(0o777)
        return f"{checkout} can be written by its group or others (mode 777)"
    if kind == "checkout-of-another-account":
        os.chown(checkout, _OTHER_UID, _OTHER_UID)
        return f"{checkout} belongs to uid {_OTHER_UID}, not to this account (uid {uid})"
    if kind == "other-writable-dotenv":
        dotenv.chmod(0o606)
        return f"{dotenv} can be written by its group or others (mode 606)"
    if kind == "group-writable-dotenv":
        dotenv.chmod(0o620)
        _give_another_group(dotenv)
        return f"{dotenv} can be written by its group or others (mode 620)"
    if kind == "dotenv-of-another-account":
        os.chown(dotenv, _OTHER_UID, _OTHER_UID)
        return f"{dotenv} belongs to uid {_OTHER_UID}, not to this account (uid {uid})"
    assert kind == "dotenv-is-a-link"
    elsewhere = checkout.parent / "dotenv-elsewhere"
    dotenv.rename(elsewhere)
    dotenv.symlink_to(elsewhere)
    return f"{dotenv} is a symbolic link"


_SHARED_KINDS = (
    "group-writable-checkout",
    "world-writable-checkout",
    "checkout-of-another-account",
    "other-writable-dotenv",
    "group-writable-dotenv",
    "dotenv-of-another-account",
    "dotenv-is-a-link",
)


@pytest.mark.parametrize("from_shell", [True, False], ids=["exported", "left-in-dotenv"])
@pytest.mark.parametrize("kind", _SHARED_KINDS)
def test_a_shared_checkouts_dotenv_keeps_no_credentials_and_every_reader_still_gets_them(
    tmp_path, container, kind, from_shell
):
    """Exported in the install shell, or only in .env as an older install left them: either way they end up in the
    container-local file alone, .env keeps every other line, and preflight, the documented source lines and the slurm
    job all get them, from a launch shell that exports none."""
    checkout, workspace = container
    said = _make_shared(checkout, kind)
    elsewhere = checkout.parent / "dotenv-elsewhere"
    expected = _CREDENTIALS if from_shell else _OLD_CREDENTIALS

    proc = _install(
        _write_installer(tmp_path / "assets"), checkout, workspace, _CREDENTIALS if from_shell else _no_credentials()
    )

    assert f"credentials kept out of {checkout}/.env, which another account can write ({said})" in proc.stdout
    _assert_no_credentials_and_the_rest_kept(checkout / ".env")
    assert _assignments(checkout / ".env")[_MARKER] == "kernel-agent-env"
    assert _credentials_in(_kernel_env(workspace)) == expected
    if kind == "dotenv-is-a-link":
        assert not (checkout / ".env").is_symlink()
        assert elsewhere.read_bytes() == _EXISTING
    _assert_every_reader_gets(checkout, workspace, expected)


def test_a_private_checkout_keeps_its_credentials_in_dotenv_as_before(tmp_path, container):
    """Its .env takes the URL, key and token as it did; the other lines and the readers are as before."""
    checkout, workspace = container
    without_header = [line for line in _CREDENTIAL_LINES if b"X-T" not in line]
    (checkout / ".env").write_bytes(b"".join((*_OTHER_LINES[:-1], *without_header, _OTHER_LINES[-1])))

    proc = _install(_write_installer(tmp_path / "assets"), checkout, workspace, _CREDENTIALS)

    assert "credentials kept out of" not in proc.stdout
    persisted = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")
    assert {key: _assignments(checkout / ".env").get(key) for key in persisted} == {
        key: _CREDENTIALS[key] for key in persisted
    }
    assert _MARKER not in _assignments(checkout / ".env")
    assert stat.S_IMODE((checkout / ".env").stat().st_mode) == 0o600
    assert _credentials_in(_kernel_env(workspace)) == _CREDENTIALS
    with mock.patch.dict(os.environ, _launch_env(checkout, workspace), clear=True):
        assert "credentials" not in preflight_env._load_dotenv_fallback()["detail"]
    _assert_every_reader_gets(checkout, workspace, _CREDENTIALS)


def test_a_reinstall_in_the_same_container_takes_the_credentials_from_its_file(tmp_path, container):
    """install.sh took them from .env when it ran again; on a shared checkout it takes them from the file the previous
    install wrote in this container."""
    checkout, workspace = container
    _make_shared(checkout, "group-writable-checkout")
    installer = _write_installer(tmp_path / "assets")
    _install(installer, checkout, workspace, _CREDENTIALS)

    again = _install(installer, checkout, workspace, _no_credentials())

    env_file = _kernel_env(workspace)
    assert f"loaded 4 missing credential(s) from {env_file} (env wins)" in again.stdout
    assert _credentials_in(env_file) == _CREDENTIALS
    _assert_no_credentials_and_the_rest_kept(checkout / ".env")
    # A key exported for a rotation wins over the file's, in install.sh and in preflight alike.
    rotated = {"ANTHROPIC_API_KEY": "sk-test-rotated-key"}
    _install(installer, checkout, workspace, {**_no_credentials(), **rotated})
    assert _credentials_in(env_file) == {**_CREDENTIALS, **rotated}
    _install(installer, checkout, workspace, _CREDENTIALS)
    assert _preflight_credentials(checkout, workspace, **rotated) == {**_CREDENTIALS, **rotated}


# The shipped .env writers, each recording .env as it finds it before it rewrites it.
_RECORD_EVERY_DOTENV_WRITE = """
eval "_shipped_$(declare -f upsert_dotenv_var)"
eval "_shipped_$(declare -f remove_dotenv_var)"
upsert_dotenv_var() { cat -- "$DOTENV" >> "$RECORD_FILE"; _shipped_upsert_dotenv_var "$@"; }
remove_dotenv_var() { cat -- "$DOTENV" >> "$RECORD_FILE"; _shipped_remove_dotenv_var "$@"; }
"""


@pytest.mark.parametrize("kind", ["group-writable-checkout", "other-writable-dotenv"])
def test_no_later_write_copies_a_credential_line_of_a_shared_dotenv(tmp_path, container, kind):
    """The credential lines, the retired spellings with them, are gone before write_env_file's other writes copy .env
    into a new file: none of those writes ever finds one."""
    checkout, workspace = container
    _make_shared(checkout, kind)
    record = tmp_path / "dotenv-before-each-write"

    _install(
        _write_installer(tmp_path / "assets", _RECORD_EVERY_DOTENV_WRITE),
        checkout,
        workspace,
        {**_no_credentials(), "RECORD_FILE": str(record)},
    )

    assert record.read_bytes().count(b"HF_HOME=/models/hf") > 5
    assert not [secret for secret in _SECRETS if secret in record.read_bytes()]


def test_a_dotenv_the_install_made_this_accounts_takes_them_from_the_next_install(tmp_path, container):
    """An install over a .env another account could write, in a private directory, replaces it with one of this
    account's and leaves the marker, so the readers in between take the credentials from the container-local file;
    the next install finds a private checkout and keeps them in .env as for any other."""
    checkout, workspace = container
    _make_shared(checkout, "other-writable-dotenv")
    installer = _write_installer(tmp_path / "assets")
    _install(installer, checkout, workspace, _CREDENTIALS)
    assert preflight_env._dotenv_not_private(checkout / ".env") is None
    _assert_every_reader_gets(checkout, workspace, _CREDENTIALS)

    again = _install(installer, checkout, workspace, _no_credentials())

    assert "credentials kept out of" not in again.stdout
    assignments = _assignments(checkout / ".env")
    assert _MARKER not in assignments
    assert assignments["ANTHROPIC_API_KEY"] == _CREDENTIALS["ANTHROPIC_API_KEY"]
    assert _credentials_in(_kernel_env(workspace)) == _CREDENTIALS


def test_a_shared_checkout_without_credentials_says_where_they_go(tmp_path, container):
    checkout, workspace = container
    (checkout / ".env").write_bytes(b"".join(_OTHER_LINES[:4]))
    said = _make_shared(checkout, "group-writable-checkout")

    proc = _run(
        _write_installer(tmp_path / "assets"), checkout, {"USER_DATA_PATH": str(workspace), **_no_credentials()}
    )

    assert proc.returncode == 2, proc.stderr
    assert "Missing required credential group(s)" in proc.stderr
    assert f"$KERNEL_AGENT_ENV  ({_kernel_env(workspace)}): .env keeps no credentials, as\n" in proc.stderr
    assert f"another account can write it ({said}); export them in the shell that\n" in proc.stderr


def test_a_planted_env_file_lends_no_credentials_to_a_shared_checkouts_readers(tmp_path, container, capsys):
    """Taking the credentials from the container-local file is held to that file's rule: one another account could
    have written is refused, unread, by install.sh's preflight and by preflight's .env loader alike."""
    checkout, workspace = container
    (checkout / ".env").write_bytes(b"".join(_OTHER_LINES[:4]))
    _make_shared(checkout, "group-writable-checkout")
    env_file = _kernel_env(workspace)
    env_file.parent.mkdir(mode=0o700, parents=True)
    env_file.write_text(
        "export ANTHROPIC_BASE_URL='https://planted.invalid'\nexport ANTHROPIC_API_KEY='sk-test-planted'\n",
        encoding="utf-8",
    )
    env_file.chmod(0o620)
    refusal = (
        f"refusing the kernel-agent env file {env_file}: {env_file} can be written by its group or others (mode 620)"
    )

    proc = _run(
        _write_installer(tmp_path / "assets"), checkout, {"USER_DATA_PATH": str(workspace), **_no_credentials()}
    )
    with mock.patch.dict(os.environ, _launch_env(checkout, workspace), clear=True):
        with pytest.raises(SystemExit) as preflight:
            preflight_env._load_dotenv_fallback()
        preflight_loaded = os.environ.get("ANTHROPIC_API_KEY")
    preflight_err = capsys.readouterr().err

    assert proc.returncode == 2
    assert refusal in proc.stderr
    assert f"credentials not taken from {env_file}; export them in this shell" in proc.stderr
    assert not [value for value in _PLANTED if value in proc.stdout + proc.stderr]
    assert preflight.value.code == 2
    assert refusal in preflight_err
    assert preflight_loaded is None
    assert not [value for value in _PLANTED if value in preflight_err]


@pytest.mark.parametrize(
    "kind",
    ["private", "private-with-the-marker", "checkout-is-a-link", "dotenv-is-a-directory", *_SHARED_KINDS],
)
def test_shell_and_preflight_hold_a_checkout_to_the_same_rule(tmp_path, container, kind):
    checkout, _ = container
    dotenv = checkout / ".env"
    env = {key: value for key, value in os.environ.items() if key != _MARKER}
    if kind == "private":
        said = None
    elif kind == "private-with-the-marker":
        env[_MARKER] = "kernel-agent-env"
        said = f"it carries {_MARKER}=kernel-agent-env"
    elif kind == "checkout-is-a-link":
        (tmp_path / "via-link").symlink_to(checkout)
        dotenv = tmp_path / "via-link" / ".env"
        said = f"another account can write it ({tmp_path / 'via-link'} is a symbolic link)"
    elif kind == "dotenv-is-a-directory":
        dotenv.unlink()
        dotenv.mkdir()
        said = f"another account can write it ({dotenv} is not a regular file)"
    else:
        said = f"another account can write it ({_make_shared(checkout, kind)})"

    shell = subprocess.run(
        ["bash", "-c", '. "$1"; runtime_env_dotenv_credentials_elsewhere "$2"', "bash", RUNTIME_ENV_SH.as_posix()]
        + [str(dotenv)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with mock.patch.dict(os.environ, env, clear=True):
        preflight = preflight_env._dotenv_credentials_elsewhere(dotenv)

    assert (shell or None) == said
    assert preflight == said


def test_shell_and_preflight_name_the_same_credentials():
    listed = subprocess.run(
        ["bash", "-c", '. "$1"; printf "%s" "$RUNTIME_ENV_DOTENV_CREDENTIAL_KEYS"', "bash", RUNTIME_ENV_SH.as_posix()],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert tuple(listed.split()) == preflight_env._DOTENV_CREDENTIAL_KEYS == _CREDENTIAL_KEYS


def _remove(dotenv: Path, *extra: str, uid: int | None = None) -> subprocess.CompletedProcess:
    argv = ["bash", "-c", '. "$1"; shift; runtime_env_remove_dotenv_credentials "$@"', "bash"]
    argv += [RUNTIME_ENV_SH.as_posix(), str(dotenv), *extra]
    if uid is not None:
        argv = ["setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--", *argv]
    return subprocess.run(argv, capture_output=True, text=True, timeout=60)


def test_removal_keeps_every_other_byte_and_the_mode_and_replaces_the_file(container):
    checkout, _ = container
    dotenv = checkout / ".env"
    # A continuation line that reads like a credential belongs to the value above it and stays.
    other_value = b'GEAK_NOTES="first\nANTHROPIC_API_KEY=part-of-the-note"\n'
    dotenv.write_bytes(other_value + _EXISTING)
    dotenv.chmod(0o664)
    before = dotenv.stat().st_ino

    proc = _remove(dotenv, "ANTHROPIC_AUTH_TOKEN")

    assert proc.returncode == 0, proc.stderr
    assert dotenv.read_bytes() == other_value + b"".join(_OTHER_LINES)
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o664
    assert dotenv.stat().st_ino != before
    assert sorted(entry.name for entry in checkout.iterdir()) == [".env"]


def test_removal_writes_nothing_when_there_is_nothing_to_remove(container):
    checkout, _ = container
    dotenv = checkout / ".env"
    dotenv.write_bytes(b"".join(_OTHER_LINES))
    before = dotenv.stat()

    assert _remove(dotenv).returncode == 0

    after = dotenv.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert sorted(entry.name for entry in checkout.iterdir()) == [".env"]


def test_removal_replaces_a_link_and_never_writes_through_it(tmp_path, container):
    checkout, _ = container
    dotenv = checkout / ".env"
    elsewhere = tmp_path / "elsewhere.env"
    dotenv.rename(elsewhere)
    dotenv.symlink_to(elsewhere)

    assert _remove(dotenv, "ANTHROPIC_AUTH_TOKEN").returncode == 0

    assert not dotenv.is_symlink()
    assert dotenv.read_bytes() == b"".join(_OTHER_LINES)
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o600
    assert elsewhere.read_bytes() == _EXISTING


def test_removal_leaves_a_file_it_cannot_read_apart_untouched(container):
    """A quoted value that never ends hides where the credential lines stop; nothing is written."""
    checkout, _ = container
    dotenv = checkout / ".env"
    dotenv.write_bytes(_EXISTING + b'\nHF_TOKEN="never closed\n')

    proc = _remove(dotenv)

    assert proc.returncode == 1
    assert f"cannot remove the credentials from {dotenv}: a quoted value in it never ends" in proc.stderr
    assert dotenv.read_bytes() == _EXISTING + b'\nHF_TOKEN="never closed\n'
    assert sorted(entry.name for entry in checkout.iterdir()) == [".env"]


def _shared_root() -> Path:
    """A directory every account can enter, outside this test's own tmp_path, which only its owner can."""
    if os.geteuid() != 0 or shutil.which("setpriv") is None:
        pytest.skip("acting as a second account needs root and setpriv")
    root = Path(tempfile.mkdtemp(prefix="hl-shared-checkout-"))
    root.chmod(0o755)
    return root


def _private_dir(path: Path, uid: int) -> Path:
    path.mkdir(mode=0o700)
    os.chown(path, uid, uid)
    return path


def test_a_bind_mounted_checkout_two_accounts_install_from(monkeypatch):
    """The slurm case: one checkout on shared storage, group-writable, that a second account (uid 65534) installs from
    first, with its own credentials, over a .env its older install left them in; then this account installs from it.
    Each account's credentials reach only its own container-local file, .env ends up with neither's, and this
    account's preflight gets its own."""
    root = _shared_root()
    try:
        checkout = root / "hyperloom"
        checkout.mkdir()
        os.chown(checkout, 0, _OTHER_UID)
        checkout.chmod(0o2775)
        dotenv = checkout / ".env"
        dotenv.write_bytes(_EXISTING)
        os.chown(dotenv, _OTHER_UID, _OTHER_UID)
        dotenv.chmod(0o600)
        installer = _write_installer(root / "assets")
        theirs = {key: value.replace("shell", "second-account") for key, value in _CREDENTIALS.items()}
        theirs["ANTHROPIC_BASE_URL"] = "https://second-account-gateway.invalid"
        their_tmp = _private_dir(root / "their-tmp", _OTHER_UID)
        their_workspace = _private_dir(root / "their-workspace", _OTHER_UID)
        their_home = _private_dir(root / "their-home", _OTHER_UID)
        shell = {"USER_DATA_PATH": str(their_workspace), "TMPDIR": str(their_tmp), "HOME": str(their_home), **theirs}

        second = _run(installer, checkout, shell, uid=_OTHER_UID)

        assert second.returncode == 0, second.stderr
        (their_file,) = their_tmp.glob("hyperloom-*/kernel-agent.env.sh")
        assert their_file.stat().st_uid == _OTHER_UID
        assert _credentials_in(their_file) == theirs
        _assert_no_credentials_and_the_rest_kept(dotenv)
        assert dotenv.stat().st_uid == _OTHER_UID
        their_bytes = their_file.read_bytes()

        our_tmp = _private_dir(root / "our-tmp", 0)
        our_workspace = root / "our-workspace"
        monkeypatch.setenv("TMPDIR", str(our_tmp))
        _install(installer, checkout, our_workspace, _CREDENTIALS)

        assert _credentials_in(_kernel_env(our_workspace)) == _CREDENTIALS
        _assert_no_credentials_and_the_rest_kept(dotenv)
        assert not [value for value in theirs.values() if value.encode() in dotenv.read_bytes()]
        assert their_file.read_bytes() == their_bytes
        assert _preflight_credentials(checkout, our_workspace) == _CREDENTIALS
    finally:
        shutil.rmtree(root)


def test_an_install_that_cannot_remove_the_credentials_stops(monkeypatch):
    """A second account whose checkout directory it cannot write (another account's, 0755) cannot replace .env, so
    the install stops before anything is written there and says what is left."""
    root = _shared_root()
    try:
        checkout = root / "hyperloom"
        checkout.mkdir()
        checkout.chmod(0o755)
        dotenv = checkout / ".env"
        dotenv.write_bytes(_EXISTING)
        dotenv.chmod(0o666)
        installer = _write_installer(root / "assets")
        their_tmp = _private_dir(root / "their-tmp", _OTHER_UID)
        their_workspace = _private_dir(root / "their-workspace", _OTHER_UID)
        shell = {
            "USER_DATA_PATH": str(their_workspace),
            "TMPDIR": str(their_tmp),
            "HOME": str(their_tmp),
            **_CREDENTIALS,
        }

        proc = _run(installer, checkout, shell, uid=_OTHER_UID)

        assert proc.returncode != 0
        said = f"{checkout} belongs to uid 0, not to this account (uid {_OTHER_UID})"
        assert f"the credentials in {dotenv} could not be removed, and another account can write it ({said})" in (
            proc.stderr
        )
        assert dotenv.read_bytes() == _EXISTING
        assert not [secret for secret in (b"sk-test-shell", b"shell-tenant") if secret in dotenv.read_bytes()]
    finally:
        shutil.rmtree(root)


def test_removal_fails_on_a_dotenv_it_cannot_read():
    """Another account's 0600 .env cannot be told apart: the removal says so and fails, never reporting that nothing
    was there to remove."""
    root = _shared_root()
    try:
        checkout = root / "hyperloom"
        checkout.mkdir()
        checkout.chmod(0o777)
        dotenv = checkout / ".env"
        dotenv.write_bytes(_EXISTING)
        dotenv.chmod(0o600)

        proc = _remove(dotenv, uid=_OTHER_UID)

        assert proc.returncode == 1
        assert f"cannot remove the credentials from {dotenv}: it could not be read, or what it keeps written" in (
            proc.stderr
        )
        assert dotenv.read_bytes() == _EXISTING
        assert sorted(entry.name for entry in checkout.iterdir()) == [".env"]
    finally:
        shutil.rmtree(root)
