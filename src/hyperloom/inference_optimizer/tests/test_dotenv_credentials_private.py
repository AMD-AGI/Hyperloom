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

from hyperloom.common.env_safety import is_lent_not_private_dotenv_assignment as is_lent
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
_THIRD_UID = 65533
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
    # The steps source runtime_env.sh from this tree, which a second account may not be able to enter: the copy here.
    kernel_text = kernel_tools.read_text(encoding="utf-8")
    sourced = f". {shlex.quote(RUNTIME_ENV_SH.as_posix())}\n"
    assert sourced in kernel_text
    kernel_tools.write_text(
        kernel_text.replace(sourced, f". {shlex.quote((where / 'runtime_env.sh').as_posix())}\n", 1), encoding="utf-8"
    )
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


def _openai_side_only() -> dict[str, str]:
    """A shell that exports the OpenAI side alone, which a .env another account can write no longer lends either."""
    return {
        **_no_credentials(),
        "OPENAI_BASE_URL": "https://gateway.invalid/v1",
        "OPENAI_API_KEY": "ak-test-openai-side",
    }


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


# Of _OTHER_LINES, what a .env another account can write still holds once an install rewrote it: the comments. Every
# assignment it would not lend (HF_HOME, the backend order, the OpenAI side) is gone with the credentials.
_COMMENT_LINES = tuple(line for line in _OTHER_LINES if line.startswith(b"#"))


def _assert_only_what_it_lends_is_kept(dotenv: Path) -> None:
    """A shared ``.env`` an install rewrote: no credential, nothing it would not lend, its comments in order."""
    text = dotenv.read_bytes()
    assignments = _assignments(dotenv)
    assert not set(assignments) & {*_CREDENTIAL_KEYS, "ANTHROPIC_AUTH_TOKEN"}
    assert not [secret for secret in _SECRETS if secret in text]
    assert {key: value for key, value in assignments.items() if not is_lent(key, value)} == {}
    assert [line for line in text.splitlines(keepends=True) if line.startswith(b"#")] == list(_COMMENT_LINES)
    assert not [line for line in _OTHER_LINES if line not in _COMMENT_LINES and line.rstrip(b"\r\n") in text]


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
        checkout.chmod(0o775)
        _give_another_group(checkout)
        return f"{checkout} can be written by its group or others (mode 775)"
    if kind == "world-writable-checkout":
        checkout.chmod(0o777)
        return f"{checkout} can be written by its group or others (mode 777)"
    if kind == "checkout-of-another-account":
        # Root takes the checkout's owner for its own (root on the user's bind-mounted checkout), so the .env of a
        # third account is what it must refuse there.
        os.chown(checkout, _OTHER_UID, _OTHER_UID)
        os.chown(dotenv, _THIRD_UID, _THIRD_UID)
        return f"{dotenv} belongs to uid {_THIRD_UID}, not to this account (uid {uid})"
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


@pytest.mark.parametrize("kind", _SHARED_KINDS)
def test_a_shared_checkouts_dotenv_keeps_no_credentials_and_every_reader_still_gets_them(tmp_path, container, kind):
    """Exported in the install shell over a .env an older install left its own in: they end up in the container-local
    file alone, .env keeps every other line and none of the old ones, and preflight, the documented source lines and
    the slurm job all get them, from a launch shell that exports none."""
    checkout, workspace = container
    said = _make_shared(checkout, kind)
    elsewhere = checkout.parent / "dotenv-elsewhere"

    proc = _install(_write_installer(tmp_path / "assets"), checkout, workspace, _CREDENTIALS)

    assert f"credentials kept out of {checkout}/.env, which another account can write ({said})" in proc.stdout
    assert f"ignored 5 credential line(s) in {checkout}/.env, which keeps none" in proc.stderr
    _assert_only_what_it_lends_is_kept(checkout / ".env")
    assert _assignments(checkout / ".env")[_MARKER] == "kernel-agent-env"
    assert _credentials_in(_kernel_env(workspace)) == _CREDENTIALS
    if kind == "dotenv-is-a-link":
        assert not (checkout / ".env").is_symlink()
        assert elsewhere.read_bytes() == _EXISTING
    _assert_every_reader_gets(checkout, workspace, _CREDENTIALS)


@pytest.mark.parametrize("kind", ["group-writable-checkout", "dotenv-of-another-account"])
def test_credentials_left_in_a_shared_dotenv_are_neither_used_nor_copied(tmp_path, container, kind):
    """Credentials only in a .env another account can write may be that account's: the install, with only the OpenAI
    side exported, takes none of them, copies none into the container-local file and removes them from .env;
    preflight takes none either."""
    checkout, workspace = container
    said = _make_shared(checkout, kind)

    proc = _install(_write_installer(tmp_path / "assets"), checkout, workspace, _openai_side_only())

    assert f"ignored 5 credential line(s) in {checkout}/.env, which keeps none, as another account can write it" in (
        proc.stderr
    )
    assert f"({said})" in proc.stderr
    assert _credentials_in(_kernel_env(workspace)) == dict.fromkeys(_CREDENTIAL_KEYS)
    assert not [secret for secret in _SECRETS if secret in _kernel_env(workspace).read_bytes()]
    _assert_only_what_it_lends_is_kept(checkout / ".env")
    assert not [secret for secret in _SECRETS if secret.decode() in proc.stdout + proc.stderr]
    with mock.patch.dict(os.environ, _launch_env(checkout, workspace), clear=True):
        detail = preflight_env._load_dotenv_fallback()["detail"]
        loaded = {key: os.environ.get(key) for key in (*_CREDENTIAL_KEYS, "ANTHROPIC_AUTH_TOKEN")}
    assert loaded == dict.fromkeys(loaded)
    assert detail["credentials"]["vars_loaded"] == 0


def test_a_private_checkouts_dotenv_still_lends_its_credentials(tmp_path, container):
    """A private .env holding the credentials, with nothing exported, still gives them to the install, which copies
    them into the container-local file, and to preflight, as before."""
    checkout, workspace = container

    proc = _install(_write_installer(tmp_path / "assets"), checkout, workspace, _no_credentials())

    assert "ignored" not in proc.stderr
    assert _credentials_in(_kernel_env(workspace)) == _OLD_CREDENTIALS
    assert _preflight_credentials(checkout, workspace) == _OLD_CREDENTIALS


def test_a_private_dotenv_with_the_marker_lends_no_credentials_to_either_loader(container):
    """The marker, read from the file itself before anything is loaded, says the credentials are in the kernel-agent
    env file: install.sh's loader and preflight's both skip a credential line next to it."""
    checkout, workspace = container
    (checkout / ".env").write_bytes(
        b"".join(_OTHER_LINES[:2])
        + b"ANTHROPIC_BASE_URL=https://planted.invalid\n"
        + f"{_MARKER}=kernel-agent-env\n".encode()
    )
    shell = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; load_dotenv_no_clobber; printf "%s|%s" "${ANTHROPIC_BASE_URL-}" "$' + _MARKER + '"',
            "bash",
            RUNTIME_ENV_SH.as_posix(),
        ],
        env=_launch_env(checkout, workspace),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with mock.patch.dict(os.environ, _launch_env(checkout, workspace), clear=True):
        preflight_env._load_dotenv_fallback()
        preflight = (os.environ.get("ANTHROPIC_BASE_URL"), os.environ.get(_MARKER))

    assert shell == "|kernel-agent-env"
    assert preflight == (None, "kernel-agent-env")


# One of each: what another account plants in a shared .env after this account's install, to be ignored by every
# reader. A URL or headers of its own would carry this account's key to it; a key or token of its own would run this
# account's jobs on its account; the retired spellings normalize into them.
_PLANTS = {
    "ANTHROPIC_BASE_URL": "https://planted.invalid",
    "ANTHROPIC_CUSTOM_HEADERS": "X-Planted: planted-tenant",
    "ANTHROPIC_API_KEY": "sk-test-planted",
    "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test-planted",
    "ANTHROPIC_AUTH_TOKEN": "sk-test-planted-auth",
    "DEEPSEEK_BASE_URL": "https://planted-deepseek.invalid",
    "DEEPSEEK_API_KEY": "sk-test-planted-deepseek",
}


@pytest.mark.parametrize("kind", ["world-writable-checkout", "group-writable-checkout", "other-writable-dotenv"])
@pytest.mark.parametrize("key", list(_PLANTS))
def test_a_credential_planted_in_a_shared_dotenv_after_the_install_reaches_no_reader(tmp_path, container, kind, key):
    checkout, workspace = container
    _make_shared(checkout, kind)
    installer = _write_installer(tmp_path / "assets")
    _install(installer, checkout, workspace, _CREDENTIALS)
    dotenv = checkout / ".env"
    if kind == "other-writable-dotenv":
        # The install replaced it with a private file of this account's; the other account's lever is the directory.
        checkout.chmod(0o777)
    with dotenv.open("a", encoding="utf-8") as handle:
        handle.write(f"{key}={_PLANTS[key]}\n")
    planted = _PLANTS[key]

    with (
        mock.patch.dict(os.environ, _launch_env(checkout, workspace), clear=True),
        mock.patch.object(cli_preflight, "_validate_explicit_server_limits", lambda args: None),
    ):
        cli_preflight._normalize_preflight_env(argparse.Namespace(), {"ext": {"steps": []}})
        preflight = dict(os.environ)
    assert {name: preflight.get(name) for name in _CREDENTIAL_KEYS} == _CREDENTIALS
    assert not [name for name, value in preflight.items() if planted in value]

    again = _install(installer, checkout, workspace, _no_credentials())

    assert planted not in _kernel_env(workspace).read_text(encoding="utf-8")
    assert _credentials_in(_kernel_env(workspace)) == _CREDENTIALS
    assert key not in _assignments(dotenv)
    assert planted not in again.stdout + again.stderr
    _assert_every_reader_gets(checkout, workspace, _CREDENTIALS)


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
    _assert_only_what_it_lends_is_kept(checkout / ".env")
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
        {**_openai_side_only(), "RECORD_FILE": str(record)},
    )

    # Each of those writes found .env (its first comment), and none found a credential or a line it does not lend.
    assert record.read_bytes().count(_COMMENT_LINES[0]) > 5
    assert not [secret for secret in _SECRETS if secret in record.read_bytes()]
    assert b"HF_HOME=" not in record.read_bytes()


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
    [
        "private",
        "private-with-the-marker",
        "checkout-is-a-link",
        "link-to-a-shared-checkout",
        "root-on-the-users-checkout",
        "dotenv-is-a-directory",
        *_SHARED_KINDS,
    ],
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
        # This account's link to its own private checkout: judged where it leads, by both.
        (tmp_path / "via-link").symlink_to(checkout)
        dotenv = tmp_path / "via-link" / ".env"
        said = None
    elif kind == "link-to-a-shared-checkout":
        (tmp_path / "via-link").symlink_to(checkout)
        dotenv = tmp_path / "via-link" / ".env"
        said = f"another account can write it ({_make_shared(checkout, 'world-writable-checkout')})"
    elif kind == "root-on-the-users-checkout":
        # A container's root on the checkout the user bind-mounted: the user's checkout and .env, no one else's write.
        if os.geteuid() != 0:
            pytest.skip("only root can give an entry to another account")
        os.chown(checkout, _OTHER_UID, _OTHER_UID)
        os.chown(dotenv, _OTHER_UID, _OTHER_UID)
        said = None
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


def _judged_as(uid: int, dotenv: Path) -> tuple[str | None, str | None]:
    """What runtime_env_dotenv_not_private and preflight's _dotenv_not_private say of ``dotenv`` to the account ``uid``
    (not root): the shell through setpriv, preflight in a child process that has become that account."""
    shell = subprocess.run(
        ["setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "--", "bash", "--noprofile", "--norc", "-c"]
        + ['eval "$RUNTIME_ENV_TEXT"; runtime_env_dotenv_not_private "$1"', "bash", str(dotenv)],
        env={"PATH": os.environ["PATH"], "RUNTIME_ENV_TEXT": RUNTIME_ENV_SH.read_text(encoding="utf-8")},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    read_end, write_end = os.pipe()
    child = os.fork()
    if child == 0:
        status = 1
        try:
            os.close(read_end)
            os.setgroups([])
            os.setresgid(uid, uid, uid)
            os.setresuid(uid, uid, uid)
            said = preflight_env._dotenv_not_private(dotenv)
            os.write(write_end, ("-" if said is None else "+" + said).encode())
            status = 0
        finally:
            os._exit(status)
    os.close(write_end)
    with os.fdopen(read_end, "rb") as reader:
        answer = reader.read().decode()
    assert os.waitpid(child, 0)[1] == 0
    return shell or None, None if answer == "-" else answer[1:]


@pytest.mark.parametrize("owner", [_THIRD_UID, _OTHER_UID], ids=["another-accounts-checkout", "its-own-checkout"])
def test_an_account_that_is_not_root_takes_no_other_owner_for_its_own(owner):
    """Only root takes the checkout's owner for its own: an account that is not root refuses another account's 0755
    checkout and 0644 .env in the shell and in preflight alike, and accepts its own."""
    root = _shared_root()
    try:
        checkout = root / "checkout"
        checkout.mkdir(mode=0o755)
        dotenv = checkout / ".env"
        dotenv.write_text("FRAMEWORK=vllm\n", encoding="utf-8")
        dotenv.chmod(0o644)
        for entry in (checkout, dotenv):
            os.chown(entry, owner, owner)

        shell, preflight = _judged_as(_OTHER_UID, dotenv)

        said = None
        if owner != _OTHER_UID:
            said = f"{checkout} belongs to uid {owner}, not to this account (uid {_OTHER_UID})"
        assert shell == said
        assert preflight == said
    finally:
        shutil.rmtree(root)


def test_shell_and_preflight_name_the_same_credentials():
    listed = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; printf "%s\\n" "$RUNTIME_ENV_DOTENV_CREDENTIAL_KEYS" "$RUNTIME_ENV_DOTENV_RETIRED_CREDENTIAL_KEYS"',
            "bash",
            RUNTIME_ENV_SH.as_posix(),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert tuple(listed[0].split()) == preflight_env._DOTENV_CREDENTIAL_KEYS == _CREDENTIAL_KEYS
    assert tuple(listed[1].split()) == preflight_env._DOTENV_RETIRED_CREDENTIAL_KEYS
    assert {"ANTHROPIC_AUTH_TOKEN", *preflight_env.LEGACY_DEEPSEEK_ENV_KEYS} <= set(listed[1].split())


def _remove(dotenv: Path, *extra: str, uid: int | None = None) -> subprocess.CompletedProcess:
    runtime_env = RUNTIME_ENV_SH
    if uid is not None:
        # A copy beside the checkout's directory, which the second account can enter; it may not enter this tree.
        runtime_env = dotenv.parent.parent / "runtime_env.sh"
        shutil.copyfile(RUNTIME_ENV_SH, runtime_env)
        runtime_env.chmod(0o644)
    argv = ["bash", "-c", '. "$1"; shift; runtime_env_remove_dotenv_credentials "$@"', "bash"]
    argv += [runtime_env.as_posix(), str(dotenv), *extra]
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


def test_removal_replaces_a_link_to_a_file_with_nothing_to_remove(tmp_path, container):
    checkout, _ = container
    dotenv = checkout / ".env"
    elsewhere = tmp_path / "elsewhere.env"
    elsewhere.write_bytes(b"".join(_OTHER_LINES))
    dotenv.unlink()
    dotenv.symlink_to(elsewhere)

    assert _remove(dotenv).returncode == 0

    assert not dotenv.is_symlink()
    assert dotenv.read_bytes() == b"".join(_OTHER_LINES)
    assert elsewhere.read_bytes() == b"".join(_OTHER_LINES)


def test_removal_removes_a_link_to_a_directory(tmp_path, container):
    checkout, _ = container
    dotenv = checkout / ".env"
    theirs = tmp_path / "their-dir"
    theirs.mkdir()
    dotenv.unlink()
    dotenv.symlink_to(theirs)

    assert _remove(dotenv).returncode == 0

    assert not dotenv.is_symlink()
    assert not dotenv.exists()
    assert theirs.is_dir()


def test_removal_fails_on_a_dotenv_that_is_a_directory(container):
    checkout, _ = container
    dotenv = checkout / ".env"
    dotenv.unlink()
    dotenv.mkdir()

    proc = _remove(dotenv)

    assert proc.returncode == 1
    assert f"cannot remove the credentials from {dotenv}: it is not a regular file" in proc.stderr
    assert dotenv.is_dir()


def test_an_install_writes_nothing_into_a_directory_a_dotenv_link_points_at(tmp_path, container):
    """Another account replaces .env with a link to a directory of its own: the install removes the link and writes a
    .env of this account's, and nothing of it -- no temporary file, no path -- lands in that directory."""
    checkout, workspace = container
    theirs = tmp_path / "their-dir"
    theirs.mkdir()
    theirs.chmod(0o777)
    (checkout / ".env").unlink()
    (checkout / ".env").symlink_to(theirs)

    proc = _install(_write_installer(tmp_path / "assets"), checkout, workspace, _CREDENTIALS)

    assert f"credentials kept out of {checkout}/.env, which another account can write" in proc.stdout
    assert list(theirs.iterdir()) == []
    assert not (checkout / ".env").is_symlink()
    assert _assignments(checkout / ".env")[_MARKER] == "kernel-agent-env"
    assert not set(_assignments(checkout / ".env")) & set(_CREDENTIAL_KEYS)
    assert _credentials_in(_kernel_env(workspace)) == _CREDENTIALS


def test_removal_renames_a_file_written_beside_dotenv(tmp_path, container, monkeypatch):
    """The kept lines are written in .env's own directory, so the rename over .env is one atomic step on one
    filesystem: a temporary file elsewhere (TMPDIR) would be copied across, and a reader could see part of it."""
    checkout, _ = container
    dotenv = checkout / ".env"
    shim = tmp_path / "shim"
    shim.mkdir()
    record = tmp_path / "mv-args"
    real_mv = shutil.which("mv")
    assert real_mv is not None
    (shim / "mv").write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" >> {shlex.quote(str(record))}\nexec {shlex.quote(real_mv)} "$@"\n',
        encoding="utf-8",
    )
    (shim / "mv").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")

    assert _remove(dotenv, "ANTHROPIC_AUTH_TOKEN").returncode == 0

    *_, source, target = record.read_text(encoding="utf-8").splitlines()
    assert target == str(dotenv)
    assert Path(source).parent == checkout
    assert Path(source).name.startswith(".env.")
    assert dotenv.read_bytes() == b"".join(_OTHER_LINES)


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
        _assert_only_what_it_lends_is_kept(dotenv)
        assert dotenv.stat().st_uid == _OTHER_UID
        their_bytes = their_file.read_bytes()

        our_tmp = _private_dir(root / "our-tmp", 0)
        our_workspace = root / "our-workspace"
        monkeypatch.setenv("TMPDIR", str(our_tmp))
        _install(installer, checkout, our_workspace, _CREDENTIALS)

        assert _credentials_in(_kernel_env(our_workspace)) == _CREDENTIALS
        _assert_only_what_it_lends_is_kept(dotenv)
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


def test_shell_and_preflight_load_the_same_from_a_private_checkout_reached_through_a_link(tmp_path, container):
    """``~/hl -> /scratch/me/hl``: install.sh keeps the logical path (REPO_ROOT), preflight resolves its own; the same
    .env, fed to both through the link, lends the same, everything a private .env lends, credentials included."""
    checkout, workspace = container
    link = tmp_path / "via-link"
    link.symlink_to(checkout)
    kept = {
        "HF_HOME": "/models/hf",
        "KERNEL_OPT_BACKEND_ORDER": "geak",
        "OPENAI_BASE_URL": "https://gateway.invalid/v1",
    }
    names = [*kept, *_CREDENTIAL_KEYS]
    env = _launch_env(link, workspace)
    for key in names:
        env.pop(key, None)
    report = 'load_dotenv_no_clobber; for k in "$@"; do printf "%s\\0" "${!k-}"; done'
    shell = subprocess.run(
        ["bash", "-c", '. "$0"; ' + report, RUNTIME_ENV_SH.as_posix(), *names],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    with mock.patch.dict(os.environ, env, clear=True):
        preflight_env._load_dotenv_fallback()
        preflight = [os.environ.get(key, "") for key in names]

    assert "not loaded from" not in shell.stderr
    assert shell.stdout.split("\0")[:-1] == preflight
    assert dict(zip(names, preflight, strict=True)) == {**kept, **_OLD_CREDENTIALS}


def test_an_install_through_the_users_own_link_keeps_the_credentials_in_dotenv(tmp_path, container):
    checkout, workspace = container
    link = tmp_path / "via-link"
    link.symlink_to(checkout)

    proc = _install(_write_installer(tmp_path / "assets"), link, workspace, _CREDENTIALS)

    assert "credentials kept out of" not in proc.stdout
    assert "ignored" not in proc.stderr
    persisted = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")
    assert {key: _assignments(checkout / ".env").get(key) for key in persisted} == {
        key: _CREDENTIALS[key] for key in persisted
    }
    assert _MARKER not in _assignments(checkout / ".env")


def test_root_in_a_container_on_the_users_bind_mounted_checkout_reads_its_dotenv_as_the_users(tmp_path, container):
    """The documented Docker (source) flow: ``docker run -v "$REPO_ROOT:$REPO_ROOT"`` with no ``--user``, then
    ``docker exec`` as root. The checkout and its 0600 .env are the user's; root, through the bind mount, loads them
    and keeps the user's credentials in .env, as it did before the shared-checkout rule."""
    if os.geteuid() != 0 or shutil.which("unshare") is None:
        pytest.skip("a bind mount in a private mount namespace needs root and unshare")
    checkout, workspace = container
    os.chown(checkout, _OTHER_UID, _OTHER_UID)
    os.chown(checkout / ".env", _OTHER_UID, _OTHER_UID)
    # Preflight, as root, judges the user's .env before the install has rewritten it.
    assert preflight_env._dotenv_not_private(checkout / ".env") is None
    mounted = tmp_path / "container-view" / "hyperloom"
    mounted.mkdir(parents=True)
    installer = _write_installer(tmp_path / "assets")
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ}
    env.update(REPO_ROOT=str(mounted), GEAK_REF=_PINNED_GEAK_REF, USER_DATA_PATH=str(workspace), **_no_credentials())
    probe = 'stat -c %u "$REPO_ROOT/.env"'
    argv = ["unshare", "-m", "--propagation", "private", "sh", "-c"]
    argv += ['mount --bind "$1" "$2" && ' + probe + ' && cd "$2" && exec bash "$3"', "-", str(checkout), str(mounted)]
    argv += [installer.as_posix()]

    proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[0] == str(_OTHER_UID)
    assert not (mounted / ".env").exists()
    assert "not loaded from" not in proc.stderr
    assert "ignored" not in proc.stderr
    assert "credentials kept out of" not in proc.stdout
    assert _credentials_in(_kernel_env(workspace)) == _OLD_CREDENTIALS
    persisted = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")
    assert {key: _assignments(checkout / ".env").get(key) for key in persisted} == {
        key: _OLD_CREDENTIALS[key] for key in persisted
    }
    assert _MARKER not in _assignments(checkout / ".env")
    assert _preflight_credentials(checkout, workspace) == _OLD_CREDENTIALS


def test_root_still_refuses_a_users_checkout_others_can_write(tmp_path, container):
    """Root takes the checkout's owner for its own, never a third account's .env nor a checkout its group can write."""
    if os.geteuid() != 0:
        pytest.skip("only root can give an entry to another account")
    checkout, _ = container
    dotenv = checkout / ".env"
    os.chown(checkout, _OTHER_UID, _OTHER_UID)
    os.chown(dotenv, _OTHER_UID, _OTHER_UID)
    checkout.chmod(0o775)
    said = f"{checkout} can be written by its group or others (mode 775)"
    shell = subprocess.run(
        ["bash", "-c", '. "$0"; runtime_env_dotenv_not_private "$1"', RUNTIME_ENV_SH.as_posix(), str(dotenv)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert shell == said
    assert preflight_env._dotenv_not_private(dotenv) == said
