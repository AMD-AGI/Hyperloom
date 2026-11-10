# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A checkout's ``.env`` another account can write lends only what preflight's allow list lends.

install.sh's shell loader (``load_dotenv_no_clobber`` in assets/runtime_env.sh) exported every assignment of ``.env``,
while preflight's (``_load_dotenv_fallback``) kept to ``is_allowed_dotenv_key``. On a checkout another account can write
-- one on shared storage -- that account could set ``LD_PRELOAD``, ``NODE_OPTIONS``, ``PYTHONPATH`` or ``BASH_ENV`` and
run code as this account during install.sh, or ``HTTPS_PROXY`` and read its traffic. Nor is refusing those enough:
``GEAK_REPO`` picks what install.sh clones and pip-installs, ``VLLM_ROCM_INDEX`` the index a framework install pulls
from, ``OPENAI_BASE_URL`` where this account's key goes. Such a ``.env`` now lends, to both loaders, only the few names
and values ``DOTENV_NOT_PRIVATE_LENT_VALUES`` lists (``is_lent_not_private_dotenv_assignment``), none of which names a
program, a package, a path or a host. Each refused name is said, never its value. A private ``.env`` is read as before
(see runtime_env.sh for why).

install_baremetal.sh, which writes credentials into ``.env`` too, keeps them out of one another account can write, as
install_kernel_tools.sh ``write_env_file`` does: they go to this container's kernel-agent env file alone.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from hyperloom.common import env_safety
from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.cli import preflight_env

from ._kernel_tools_harness import RUNTIME_ENV_SH
from .test_dotenv_credentials_private import (
    _CREDENTIALS,
    _MARKER,
    _OTHER_UID,
    _THIRD_UID,
    _assignments,
    _credentials_in,
    _launch_env,
    _run,
    _shared_root,
    _write_installer,
)
from .test_install_session_paths import _kernel_env

_ASSETS = RUNTIME_ENV_SH.parent
_BAREMETAL_SH = _ASSETS / "install_baremetal.sh"
_KERNEL_TOOLS_SH = _ASSETS / "install_kernel_tools.sh"

# What another account plants in a shared .env: loader and interpreter variables, traffic routing, git's program hooks,
# and the allow list's names that pick what an install fetches and runs or where this account's key goes.
_GIT_PROGRAM_KEYS = (
    "GIT_SSH_COMMAND",
    "GIT_SSH",
    "GIT_ASKPASS",
    "GIT_EXEC_PATH",
    "GIT_PROXY_COMMAND",
    "GIT_EDITOR",
    "GIT_CONFIG_GLOBAL",
)
_ROUTING_AND_PROGRAM_KEYS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "NO_PROXY", "HF_ENDPOINT", "PYTHON")
_ROUTING_AND_PROGRAM_KEYS += ("VIRTUAL_ENV", "KERNEL_AGENT_ENV", "GEAK_CLAUDE_BIN")
# Each value is something the install or a run would fetch, execute or send this account's key to.
_FETCH_AND_ENDPOINT_VALUES = {
    "GEAK_REPO": "https://planted-repo.invalid/geak.git",
    "GEAK_REF": "planted-branch",
    "VLLM_REPO": "https://planted-repo.invalid/vllm.git",
    "VLLM_ROCM_INDEX": "https://planted-index.invalid/simple",
    "AITER_REPO": "https://planted-repo.invalid/aiter.git",
    "SGLANG_REPO": "https://planted-repo.invalid/sglang.git",
    "OPENAI_BASE_URL": "http://planted-gateway.invalid/v1",
    "OPENAI_API_KEY": "sk-planted-openai",
    "OPENAI_CUSTOM_HEADERS": "X-Planted: 1",
    "GEAK_BASE_URL": "http://planted-gateway.invalid",
    "HYPERLOOM_DEPS_ROOT": "/planted/deps",
    "HYPERLOOM_DOCKER_TARGET_HOST": "planted-host.invalid",
    "KERNEL_OPT_BACKEND_ORDER": "geak",
    "HF_HOME": "/planted/hf",
    "USER_DATA_PATH": "/planted/workspace",
    # Written unquoted into the kernel-agent env file this account sources.
    "GEAK_CLAUDE_MODEL": "m'; touch /planted/model; '",
    # A listed name with a value that is not listed: FRAMEWORK names a script path.
    "FRAMEWORK": "../../planted/framework",
}


def _shell(script: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", f'. "$0"; {script}', RUNTIME_ENV_SH.as_posix(), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _shell_lent_values() -> dict[str, frozenset[str]]:
    listed = _shell('printf "%s\\n" "$RUNTIME_ENV_DOTENV_NOT_PRIVATE_LENT"').stdout.split()
    entries = [entry.split("=", 1) for entry in listed]
    assert len({name for name, _ in entries}) == len(entries)
    return {name: frozenset(values.split(",")) for name, values in entries}


def test_the_shell_copy_of_what_a_shared_dotenv_lends_equals_preflights():
    assert _shell_lent_values() == dict(env_safety.DOTENV_NOT_PRIVATE_LENT_VALUES)
    assert all(env_safety.is_allowed_dotenv_key(name) for name in env_safety.DOTENV_NOT_PRIVATE_LENT_VALUES)
    assert not set(env_safety.DOTENV_NOT_PRIVATE_LENT_VALUES) & {
        *env_safety.BLOCKED_UNTRUSTED_ENV_NAMES,
        *_FETCH_AND_ENDPOINT_VALUES.keys() - {"FRAMEWORK"},
        *(name.upper() for name in _ROUTING_AND_PROGRAM_KEYS),
    }
    # The values a shipped choice can take, so a framework added later is lent too.
    lent = env_safety.DOTENV_NOT_PRIVATE_LENT_VALUES
    assert lent["FRAMEWORK"] == frozenset(framework_registry.FRAMEWORKS)
    assert not [value for values in lent.values() for value in values if not re.fullmatch(r"[a-z0-9-]+", value)]


def _candidates() -> list[str]:
    names = {
        *env_safety.DOTENV_EXACT_ALLOWLIST,
        *env_safety.BLOCKED_UNTRUSTED_ENV_NAMES,
        *env_safety.DOTENV_NOT_PRIVATE_LENT_VALUES,
        *(prefix + "KNOB" for prefix in env_safety.DOTENV_PREFIX_ALLOWLIST),
        *(prefix.rstrip("_") for prefix in env_safety.DOTENV_PREFIX_ALLOWLIST),
        *_GIT_PROGRAM_KEYS,
        *_FETCH_AND_ENDPOINT_VALUES,
        "NODE_EXTRA_CA_CERTS",
        "PROMPT_COMMAND",
        "VLLM_PYTHON",
        "TRACELENS_INTERNAL_ROOT",
        "TL_EXTENSION",
        "HYPERLOOM",
        "_",
        "A1",
    }
    return sorted(names | {name.lower() for name in names} | {name.title() for name in names})


def test_the_shell_lends_exactly_what_preflight_lends_from_a_shared_dotenv():
    listed = sorted({value for values in env_safety.DOTENV_NOT_PRIVATE_LENT_VALUES.values() for value in values})
    values = [*listed, "", "planted", "vllm,sglang", "sglang,", ",", " vllm", "vllm ", "VLLM", "../vllm", "*", "1*"]
    pairs = [(name, value) for name in _candidates() for value in values]
    script = 'while [ $# -gt 0 ]; do runtime_env_dotenv_lent "$1" "$2" && printf "%s=%s\\n" "$1" "$2"; shift 2; done'
    lent = _shell(script, *(part for pair in pairs for part in pair)).stdout.splitlines()

    expected = [f"{n}={v}" for n, v in pairs if env_safety.is_lent_not_private_dotenv_assignment(n, v)]
    assert lent == expected
    assert {"FRAMEWORK=vllm", "HYPERLOOM_RUN_MODE=docker", "HF_HUB_OFFLINE=1"} <= set(lent)
    assert not [
        line for line in lent if line.split("=", 1)[0].upper() in _FETCH_AND_ENDPOINT_VALUES.keys() - {"FRAMEWORK"}
    ]


def test_the_shell_emits_the_credential_lines_write_env_file_emits():
    source = _KERNEL_TOOLS_SH.read_text(encoding="utf-8")
    match = re.search(r"^  _emit_credential_fallback\(\) \{\n.*?^  \}", source, re.MULTILINE | re.DOTALL)
    assert match
    value = "X-Tenant: a\nX-Team: ${TEAM}"
    theirs = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", match.group(0) + '\n_emit_credential_fallback "$1" "$2"', "-"]
        + ["ANTHROPIC_CUSTOM_HEADERS", value],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert _shell('runtime_env_emit_credential_fallback "$@"', "ANTHROPIC_CUSTOM_HEADERS", value).stdout == theirs


# How a checkout is made one another account can write. By its mode alone -- a 0777 checkout of this account's, which
# any account could have written into -- runs as any user, so CI's non-root run holds the allow list; by the other
# account (uid 65534) itself writing the .env and code needs root and setpriv.
_HOW = ("by-mode", "by-another-account")


def _shared_place(how: str, tmp_path: Path) -> Path:
    """Where the shared checkout goes: a directory every account can enter. The caller removes it."""
    if how == "by-another-account":
        return _shared_root()
    root = tmp_path / "shared"
    root.mkdir(mode=0o755)
    return root


def _our_dir(path: Path) -> Path:
    """A directory of this account's only it can enter (the container's own temporary directory)."""
    path.mkdir(mode=0o700)
    return path


def _append_as_planter(how: str, dotenv: Path, text: str) -> None:
    """Append ``text`` to the shared ``dotenv`` as whoever planted it."""
    if how == "by-mode":
        with dotenv.open("a", encoding="utf-8") as handle:
            handle.write(text)
        return
    subprocess.run(
        ["setpriv", f"--reuid={_OTHER_UID}", f"--regid={_OTHER_UID}", "--clear-groups", "--", "sh", "-c"]
        + [f'printf "%s" "$1" >> {dotenv}', "-", text],
        check=True,
    )


def _plant(root: Path, how: str = "by-another-account") -> tuple[Path, Path, dict[str, str]]:
    """A world-writable checkout whose .env and code another account wrote (``how``: see _HOW); the directory each
    planted program leaves a marker file in when it runs; and what was planted, by name."""
    checkout = root / "hyperloom"
    checkout.mkdir()
    checkout.chmod(0o777)
    planted = root / "planted"
    planted.mkdir()
    planted.chmod(0o777)
    markers = root / "markers"
    markers.mkdir()
    markers.chmod(0o777)
    hook = f"echo ran > {markers}/$1\n"
    values = {
        # A library that does not exist: ld.so names it on stderr in every program it is preloaded into.
        "LD_PRELOAD": f"{planted}/planted-preload.so",
        "LD_LIBRARY_PATH": str(planted),
        "NODE_OPTIONS": f"--require {planted}/planted.js",
        "NODE_EXTRA_CA_CERTS": f"{planted}/planted-ca.pem",
        # sitecustomize runs in every python started with it on the path.
        "PYTHONPATH": str(planted),
        "BASH_ENV": f"{planted}/bash_env",
        "ENV": f"{planted}/sh_env",
        "PROMPT_COMMAND": f"sh {planted}/hook prompt_command",
        **{key: f"sh {planted}/hook {key.lower()}" for key in _GIT_PROGRAM_KEYS},
        "HTTPS_PROXY": "http://planted-proxy.invalid:3128",
        "https_proxy": "http://planted-proxy.invalid:3129",
        "HTTP_PROXY": "http://planted-proxy.invalid:3130",
        "NO_PROXY": "planted.invalid",
        "HF_ENDPOINT": "https://planted-hub.invalid",
        "PYTHON": f"{planted}/python",
        "VIRTUAL_ENV": str(planted),
        "KERNEL_AGENT_ENV": f"{planted}/kernel-agent.env.sh",
        "GEAK_CLAUDE_BIN": f"{planted}/claude",
        **_FETCH_AND_ENDPOINT_VALUES,
        # Read in the loader's first pass, before the others: a listed name with a value that is not listed.
        "HYPERLOOM_RUN_MODE": "planted",
    }
    files = {
        "hook": hook,
        "sitecustomize.py": f"open({str(markers / 'pythonpath')!r}, 'w').write('ran')\n",
        "bash_env": f"echo ran > {markers}/bash_env\n",
        "sh_env": f"echo ran > {markers}/env\n",
        "planted.js": f"require('fs').writeFileSync({str(markers / 'node_options')!r}, 'ran')\n",
        "python": f"#!/bin/sh\necho ran > {markers}/python\n",
    }
    dotenv_text = "".join(f"{key}={value}\n" for key, value in values.items()) + "HF_HUB_OFFLINE=1\n"
    writer = (
        'cd "$1"; shift; while [ $# -gt 0 ]; do printf "%s" "$2" > "$1"; chmod 0755 "$1"; shift 2; done; '
        'printf "%s" "$DOTENV_TEXT" > "$CHECKOUT/.env"'
    )
    argv = ["sh", "-c", writer, "-", str(planted)] + [part for item in files.items() for part in item]
    if how == "by-another-account":
        argv = ["setpriv", f"--reuid={_OTHER_UID}", f"--regid={_OTHER_UID}", "--clear-groups", "--", *argv]
    subprocess.run(
        argv,
        env={"PATH": os.environ["PATH"], "DOTENV_TEXT": dotenv_text, "CHECKOUT": str(checkout)},
        check=True,
        timeout=60,
    )
    owner = _OTHER_UID if how == "by-another-account" else os.geteuid()
    assert (checkout / ".env").stat().st_uid == owner
    assert preflight_env._dotenv_not_private(checkout / ".env") == (
        f"{checkout} can be written by its group or others (mode 777)"
    )
    return checkout, markers, values


# Run where install.sh writes the kernel-agent env file: every program a planted variable would steer, started from
# the install's own shell, with what that shell exports recorded.
_RUN_WHAT_WOULD_RUN_PLANTED_CODE = """
env -0 > "$INSTALL_ENV_DUMP"
bash -c : ; sh -c : ; python3 -c pass
git --version > /dev/null
command -v node > /dev/null && node -e 0
"""


@pytest.mark.parametrize("how", _HOW)
def test_a_shared_dotenv_another_account_planted_lends_none_of_it_to_the_install(monkeypatch, tmp_path, how):
    root = _shared_place(how, tmp_path)
    try:
        checkout, markers, values = _plant(root, how)
        our_tmp = _our_dir(root / "our-tmp")
        workspace = root / "our-workspace"
        dump = root / "install-env"
        monkeypatch.setenv("TMPDIR", str(our_tmp))
        installer = _write_installer(root / "assets", _RUN_WHAT_WOULD_RUN_PLANTED_CODE)

        proc = _run(
            installer,
            checkout,
            {
                "USER_DATA_PATH": str(workspace),
                "INSTALL_ENV_DUMP": str(dump),
                "PYTHONPATH": "/opt/own/site",
                **_CREDENTIALS,
            },
        )

        assert proc.returncode == 0, proc.stderr
        assert sorted(entry.name for entry in markers.iterdir()) == []
        assert "planted-preload.so" not in proc.stderr
        exported = dict(entry.split("=", 1) for entry in dump.read_text(encoding="utf-8").split("\0") if entry)
        assert not {key for key in values if exported.get(key) == values[key]}
        assert exported["HF_HUB_OFFLINE"] == "1"
        warning = re.search(
            r"\[runtime-env WARN\] not loaded from (\S+), which another account can write .*", proc.stderr
        )
        assert warning, proc.stderr
        assert warning.group(1) == f"{checkout}/.env"
        assert proc.stderr.count("[runtime-env WARN] not loaded from") == 1
        # The run mode is read in the first pass, so it is named first.
        named = ["HYPERLOOM_RUN_MODE", *(key for key in values if key != "HYPERLOOM_RUN_MODE")]
        assert f": {' '.join(named)}; export any of them you need in this shell" in warning.group(0)
        assert not [value for value in values.values() if value in proc.stdout + proc.stderr]
        assert _credentials_in(_kernel_env(workspace)) == _CREDENTIALS
        # Nor does the install write there what such a .env would not lend: the kernel-agent env file carries it.
        assert "/opt/own/site" not in (checkout / ".env").read_text(encoding="utf-8")
        assert _assignments(_kernel_env(workspace))["PYTHONPATH"].endswith(":/opt/own/site")
    finally:
        shutil.rmtree(root)


@pytest.mark.parametrize("how", _HOW)
def test_preflight_lends_none_of_a_planted_shared_dotenv(capsys, tmp_path, how):
    root = _shared_place(how, tmp_path)
    try:
        checkout, _, values = _plant(root, how)
        with mock.patch.dict(os.environ, _launch_env(checkout, root / "workspace"), clear=True):
            for key in (*values, "HF_HUB_OFFLINE"):
                os.environ.pop(key, None)
            result = preflight_env._load_dotenv_fallback()
            loaded = {key: os.environ.get(key) for key in (*values, "HF_HUB_OFFLINE")}
        err = capsys.readouterr().err

        assert result["status"] == "applied"
        assert loaded == {**dict.fromkeys(values), "HF_HUB_OFFLINE": "1"}
        for key in values:
            assert f"ignoring .env key {key} from {checkout}/.env, which another account can write" in err
        assert not [value for value in values.values() if value in err]
    finally:
        shutil.rmtree(root)


@pytest.mark.parametrize("how", _HOW)
def test_a_shared_dotenv_never_names_the_workspace(tmp_path, how):
    """A ``hyperloom setup`` .env (one with a HYPERLOOM_RUN_MODE line) names the workspace and drops the caller's; one
    another account can write does not, though the line and its value are lent."""
    root = _shared_place(how, tmp_path)
    try:
        checkout = root / "hyperloom"
        checkout.mkdir()
        checkout.chmod(0o777)
        planted = {"USER_DATA_PATH": "/planted/workspace", "KERNEL_AGENT_ENV": "/planted/kernel-agent.env.sh"}
        text = "HYPERLOOM_RUN_MODE=baremetal\n" + "".join(f"{k}={v}\n" for k, v in planted.items())
        (checkout / ".env").write_text(text, encoding="utf-8")
        if how == "by-another-account":
            os.chown(checkout / ".env", _OTHER_UID, _OTHER_UID)
        workspace = tmp_path / "our-workspace"
        env = _launch_env(checkout, workspace)

        proc = _shell(
            "setup_dotenv_is_authoritative && echo authoritative; runtime_env_forget_stale_workspace; "
            'load_dotenv_no_clobber; printf "%s\\n" "$USER_DATA_PATH" "${KERNEL_AGENT_ENV-}" "$HYPERLOOM_RUN_MODE"',
            env=env,
        )
        with mock.patch.dict(os.environ, env, clear=True):
            selection = dict(preflight_env._workspace_selection())

        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.splitlines() == [str(workspace), "", "baremetal"]
        assert selection["USER_DATA_PATH"] == str(workspace)
        assert "KERNEL_AGENT_ENV" not in selection
    finally:
        shutil.rmtree(root)


def test_a_private_dotenv_is_read_as_before(tmp_path):
    """The decision: a .env only this account can write keeps lending what install_kernel_tools.sh and
    install_baremetal.sh write there and the documented shell flows read back, PYTHONPATH and the interpreter
    included; preflight keeps to its allow list there, as before."""
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o755)
    kept = {
        "PYTHONPATH": "/opt/own/site",
        "VLLM_PYTHON": "/opt/own/bin/python",
        "TRACELENS_INTERNAL_ROOT": "/opt/own/tl",
        "HTTPS_PROXY": "http://own-proxy.invalid:3128",
        "VIRTUAL_ENV": "/opt/own/venv",
    }
    (checkout / ".env").write_text("".join(f"{k}={v}\n" for k, v in kept.items()), encoding="utf-8")
    (checkout / ".env").chmod(0o600)
    env = {"PATH": os.environ["PATH"], "REPO_ROOT": str(checkout)}

    proc = _shell(
        'load_dotenv_no_clobber; for k in "$@"; do printf "%s=%s\\n" "$k" "${!k-}"; done; '
        'printf "refused=%s\\n" "$DOTENV_REFUSED_KEYS"',
        *kept,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [f"{k}={v}" for k, v in kept.items()] + ["refused="]
    assert "not loaded from" not in proc.stderr


# install_baremetal.sh as shipped, its framework, GPU and probe steps stubbed: what is left is the workspace choice,
# the credential phase and the runtime .env write.
_BAREMETAL_STUBS = """
base_preflight() { :; }
install_requested_framework() { :; }
apply_rocm_profiler_hotfix() { :; }
resolve_python() { return 1; }
resolve_installed_framework() { :; }
print_next_steps() { :; }
main --skip-base-check --yes
"""


def _baremetal(where: Path, tail: str = _BAREMETAL_STUBS) -> Path:
    text = _BAREMETAL_SH.read_text(encoding="utf-8")
    assert text.endswith('\nmain "$@"\n')
    where.mkdir(mode=0o755)
    shutil.copyfile(RUNTIME_ENV_SH, where / "runtime_env.sh")
    script = where / "install_baremetal.sh"
    script.write_text(text[: -len('main "$@"\n')] + tail, encoding="utf-8")
    return script


def _run_baremetal(script: Path, checkout: Path, workspace: Path, tmp: Path, **shell: str):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp), "TMPDIR": str(tmp), "REPO_ROOT": str(checkout)}
    env.update(USER_DATA_PATH=str(workspace), **shell)
    return subprocess.run(["bash", str(script)], cwd=checkout, env=env, capture_output=True, text=True, timeout=120)


_BAREMETAL_CREDENTIALS = {key: _CREDENTIALS[key] for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY")}
_BAREMETAL_CREDENTIALS["CLAUDE_CODE_OAUTH_TOKEN"] = _CREDENTIALS["CLAUDE_CODE_OAUTH_TOKEN"]


@pytest.mark.parametrize("how", _HOW)
def test_install_baremetal_writes_no_credential_into_a_shared_dotenv(monkeypatch, tmp_path, how):
    root = _shared_place(how, tmp_path)
    try:
        checkout, _, _ = _plant(root, how)
        # Another account's gateway, which a key read from the shell must never be paired with.
        _append_as_planter(how, checkout / ".env", "ANTHROPIC_BASE_URL=https://planted-gateway.invalid\n")
        tmp = _our_dir(root / "our-tmp")
        monkeypatch.setenv("TMPDIR", str(tmp))
        workspace = root / "our-workspace"
        script = _baremetal(root / "assets")
        # What an earlier install.sh wrote there: its other lines stay, its credential lines give way.
        env_file = _kernel_env(workspace)
        env_file.parent.mkdir(mode=0o700, parents=True)
        old_block = _shell('runtime_env_emit_credential_fallback "$@"', "ANTHROPIC_API_KEY", "sk-test-old-key").stdout
        env_file.write_text(f"#!/bin/sh\nexport MAGPIE_PATH='/opt/magpie'\n{old_block}", encoding="utf-8")
        env_file.chmod(0o600)

        proc = _run_baremetal(script, checkout, workspace, tmp, VIRTUAL_ENV="/opt/own/venv", **_BAREMETAL_CREDENTIALS)

        assert proc.returncode == 0, proc.stdout + proc.stderr
        text = (checkout / ".env").read_text(encoding="utf-8")
        assert "/opt/own/venv" not in text
        assert f"VIRTUAL_ENV is not written to {checkout}/.env, which another account can write" in proc.stderr
        assert not [value for value in _BAREMETAL_CREDENTIALS.values() if value in text]
        assert not {*_BAREMETAL_CREDENTIALS, "ANTHROPIC_AUTH_TOKEN"} & set(_assignments(checkout / ".env"))
        assert _assignments(checkout / ".env")[_MARKER] == "kernel-agent-env"
        assert env_file.stat().st_mode & 0o077 == 0
        assert {k: v for k, v in _credentials_in(env_file).items() if v} == _BAREMETAL_CREDENTIALS
        env_text = env_file.read_text(encoding="utf-8")
        assert "sk-test-old-key" not in env_text
        assert env_text.startswith("#!/bin/sh\nexport MAGPIE_PATH='/opt/magpie'\n")
        assert [env_text.count(f"export {key}=") for key in _BAREMETAL_CREDENTIALS] == [1, 1, 1]
        assert "credentials kept out of" in proc.stdout

        # The next run, from a shell that exports none, takes them from that file, never the planted gateway.
        rerun = _run_baremetal(script, checkout, workspace, tmp, ANTHROPIC_BASE_URL="")
        assert rerun.returncode == 0, rerun.stdout + rerun.stderr
        assert "loaded 3 missing credential(s) from" in rerun.stdout
        assert {k: v for k, v in _credentials_in(env_file).items() if v} == _BAREMETAL_CREDENTIALS
        assert not [value for value in _BAREMETAL_CREDENTIALS.values() if value in (checkout / ".env").read_text()]
    finally:
        shutil.rmtree(root)


@pytest.mark.parametrize("how", _HOW)
def test_install_baremetal_pairs_no_shell_key_with_a_shared_dotenvs_gateway(monkeypatch, tmp_path, how):
    root = _shared_place(how, tmp_path)
    try:
        checkout = root / "hyperloom"
        checkout.mkdir()
        checkout.chmod(0o777)
        (checkout / ".env").write_text("ANTHROPIC_BASE_URL=https://planted-gateway.invalid\n", encoding="utf-8")
        if how == "by-another-account":
            os.chown(checkout / ".env", _OTHER_UID, _OTHER_UID)
        tmp = _our_dir(root / "our-tmp")
        monkeypatch.setenv("TMPDIR", str(tmp))

        proc = _run_baremetal(
            _baremetal(root / "assets"),
            checkout,
            root / "our-workspace",
            tmp,
            ANTHROPIC_API_KEY=_CREDENTIALS["ANTHROPIC_API_KEY"],
            ANTHROPIC_BASE_URL="",
        )

        assert proc.returncode != 0
        assert "no usable LLM endpoint" in proc.stderr
        assert not list(tmp.glob("hyperloom-*/kernel-agent.env.sh"))
        assert _CREDENTIALS["ANTHROPIC_API_KEY"] not in (checkout / ".env").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(root)


def test_install_baremetal_keeps_a_private_checkouts_credentials_in_dotenv(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o755)
    tmp = tmp_path / "tmp"
    tmp.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(tmp))

    proc = _run_baremetal(
        _baremetal(tmp_path / "assets"), checkout, tmp_path / "workspace", tmp, **_BAREMETAL_CREDENTIALS
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assignments = _assignments(checkout / ".env")
    assert {key: assignments.get(key) for key in _BAREMETAL_CREDENTIALS} == _BAREMETAL_CREDENTIALS
    assert _MARKER not in assignments
    assert not list(tmp.glob("hyperloom-*/kernel-agent.env.sh"))


@pytest.mark.parametrize("how", _HOW)
def test_install_baremetal_reads_from_a_shared_dotenv_only_what_the_loader_lends(tmp_path, how):
    """Its own .env reads (the vLLM venv, the workspace, the backend order, the run mode) keep to runtime_env_dotenv_lent:
    a listed name with a listed value, nothing else."""
    root = _shared_place(how, tmp_path)
    try:
        checkout = root / "hyperloom"
        checkout.mkdir()
        checkout.chmod(0o777)
        (checkout / ".env").write_text(
            "VLLM_VENV_ROOT=/planted/venv\nUSER_DATA_PATH=/shared/workspace\nKERNEL_OPT_BACKEND_ORDER=geak\n"
            "HYPERLOOM_RUN_MODE=planted\nHYPERLOOM_FRAMEWORK_ENV=isolated\n",
            encoding="utf-8",
        )
        tail = (
            'DOTENV_NOT_PRIVATE="$(runtime_env_dotenv_not_private "$DOTENV")"\n'
            "for k in VLLM_VENV_ROOT USER_DATA_PATH KERNEL_OPT_BACKEND_ORDER HYPERLOOM_RUN_MODE HYPERLOOM_FRAMEWORK_ENV; do "
            'printf "%s=%s\\n" "$k" '
            '"$(read_dotenv_var "$k")"; done\n'
        )
        proc = _run_baremetal(_baremetal(root / "assets", tail), checkout, root / "ws", root)

        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.splitlines() == [
            "VLLM_VENV_ROOT=",
            "USER_DATA_PATH=",
            "KERNEL_OPT_BACKEND_ORDER=",
            "HYPERLOOM_RUN_MODE=",
            "HYPERLOOM_FRAMEWORK_ENV=isolated",
        ]
    finally:
        shutil.rmtree(root)


# What another account writes into a .env it can write in a checkout directory of this account's (0755, .env 0666):
# a loader, an endpoint, a fetch source, a credential, a planted line hidden in a quoted value's continuation and one
# behind a carriage return in a comment (a line to the shell, two to a reader of universal newlines), with a comment
# and the lent lines, which alone stay.
_PLANTED_IN_A_WRITABLE_DOTENV = (
    "# kept: a comment\n"
    "# a note\rHTTPS_PROXY=http://planted-proxy.invalid:3128\n"
    "FRAMEWORK=vllm\n"
    "NODE_OPTIONS=--require=/tmp/planted.js\n"
    "export LD_PRELOAD=/tmp/planted-preload.so\n"
    "OPENAI_BASE_URL=https://planted-gateway.invalid/v1\n"
    "GEAK_REPO=https://planted-repo.invalid/geak.git\n"
    "ANTHROPIC_API_KEY=sk-test-planted\n"
    'GEAK_NOTES="first\nPYTHONPATH=/tmp/planted"\n'
    "HF_HUB_OFFLINE='1'\n"
    "not an assignment\n"
)
_PLANTED_NAMES = (
    "NODE_OPTIONS",
    "LD_PRELOAD",
    "OPENAI_BASE_URL",
    "GEAK_REPO",
    "GEAK_NOTES",
    "PYTHONPATH",
    "HTTPS_PROXY",
)


def _writable_dotenv_in_a_private_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o755)
    dotenv = checkout / ".env"
    dotenv.write_text(_PLANTED_IN_A_WRITABLE_DOTENV, encoding="utf-8")
    dotenv.chmod(0o666)
    assert preflight_env._dotenv_not_private(dotenv) == f"{dotenv} can be written by its group or others (mode 666)"
    return checkout


def _assert_the_next_load_lends_none_of_it(checkout: Path, workspace: Path, framework: str = "vllm") -> None:
    """The install made .env this account's, so the next readers find it private: none of the planted lines is there
    for them to lend, in install.sh's loader or preflight's. ``framework`` is what the install left FRAMEWORK."""
    dotenv = checkout / ".env"
    assert preflight_env._dotenv_not_private(dotenv) is None
    text = dotenv.read_bytes().decode("utf-8")
    assert not [word for word in ("planted", "not an assignment", "#", "\r") if word in text]
    # Rebuilt from what it lends, each assignment written afresh.
    assert "HF_HUB_OFFLINE=1\n" in text.splitlines(keepends=True)
    assert [line for line in text.splitlines() if not re.fullmatch(r"[A-Z_]+=[a-z0-9-]*", line)] == []
    assert [
        key
        for key, value in _assignments(checkout / ".env").items()
        if not env_safety.is_lent_not_private_dotenv_assignment(key, value)
    ] == []
    env = _launch_env(checkout, workspace)
    for key in (*_PLANTED_NAMES, "ANTHROPIC_API_KEY"):
        env.pop(key, None)
    report = 'load_dotenv_no_clobber; for k in "$@"; do printf "%s=%s\\n" "$k" "${!k-}"; done'
    proc = _shell(report, *_PLANTED_NAMES, "FRAMEWORK", env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [f"{key}=" for key in _PLANTED_NAMES] + [f"FRAMEWORK={framework}"]
    with mock.patch.dict(os.environ, env, clear=True):
        preflight_env._load_dotenv_fallback()
        assert {key: os.environ.get(key) for key in _PLANTED_NAMES} == dict.fromkeys(_PLANTED_NAMES)
        assert os.environ.get("FRAMEWORK", "") == framework


def test_the_install_leaves_no_planted_line_in_the_dotenv_it_makes_this_accounts(tmp_path, monkeypatch):
    """write_env_file replaces a .env another account could write with a 0600 file of this account's; what that
    account planted there is gone first, so the private file it leaves lends none of it."""
    checkout = _writable_dotenv_in_a_private_checkout(tmp_path)
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("TMPDIR", str(_our_dir(tmp_path / "tmp")))

    proc = _run(_write_installer(tmp_path / "assets"), checkout, {"USER_DATA_PATH": str(workspace), **_CREDENTIALS})

    assert proc.returncode == 0, proc.stderr
    assert _assignments(checkout / ".env")[_MARKER] == "kernel-agent-env"
    assert _credentials_in(_kernel_env(workspace)) == _CREDENTIALS
    _assert_the_next_load_lends_none_of_it(checkout, workspace)


def test_install_baremetal_leaves_no_planted_line_in_the_dotenv_it_makes_this_accounts(tmp_path, monkeypatch):
    checkout = _writable_dotenv_in_a_private_checkout(tmp_path)
    workspace = tmp_path / "workspace"
    tmp = _our_dir(tmp_path / "tmp")
    monkeypatch.setenv("TMPDIR", str(tmp))

    proc = _run_baremetal(_baremetal(tmp_path / "assets"), checkout, workspace, tmp, **_BAREMETAL_CREDENTIALS)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _assignments(checkout / ".env")[_MARKER] == "kernel-agent-env"
    # Its own framework step, stubbed here, finds no framework and removes FRAMEWORK.
    _assert_the_next_load_lends_none_of_it(checkout, workspace, framework="")


# Lines of each shape the loader reads, lent or not: what the rewrite keeps must be what the loader lends.
_SHAPES = (
    "FRAMEWORK=vllm\n",
    "  export HYPERLOOM_RUN_MODE=docker\n",
    'HF_HUB_OFFLINE="1"\n',
    "HF_DATASETS_OFFLINE='0'   \n",
    "HYPERLOOM_LLM_MODE=openai\r\n",
    "HYPERLOOM_FRAMEWORK_ENV = shared\n",
    "FRAMEWORK=vllm # a comment\n",
    "FRAMEWORK=../planted\n",
    'HF_HUB_OFFLINE="1\n0"\n',
    "HF_HOME=/models/hf\n",
    "# FRAMEWORK=sglang\n",
    "\n",
    "   \n",
    "stray words\n",
    "HF_DATASETS_OFFLINE=1",
)


def test_the_rewrite_keeps_exactly_what_the_loader_lends_from_a_shared_dotenv(tmp_path):
    """runtime_env_keep_lent_dotenv_lines over a .env another account can write: the loader lends the same from the
    file before and after, and the file is rebuilt from those assignments alone, each written afresh as NAME=VALUE in
    its order: no comment, blank line or other byte of the old file stays."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    checkout.chmod(0o777)
    dotenv = checkout / ".env"
    dotenv.write_text("".join(_SHAPES), encoding="utf-8", newline="")
    names = sorted(env_safety.DOTENV_NOT_PRIVATE_LENT_VALUES) + ["HF_HOME"]
    env = {"PATH": os.environ["PATH"], "REPO_ROOT": str(checkout)}
    report = (
        'load_dotenv_no_clobber; for k in "$@"; do printf "%s=%s\\n" "$k" "${!k-}"; done; echo "[$DOTENV_REFUSED_KEYS]"'
    )

    before = _shell(report, *names, env=env)
    rewrite = _shell('runtime_env_keep_lent_dotenv_lines "$1"', str(dotenv), env=env)
    after = _shell(report, *names, env=env)

    assert rewrite.returncode == 0, rewrite.stderr
    assert before.returncode == after.returncode == 0, before.stderr + after.stderr
    assert after.stdout.splitlines()[:-1] == before.stdout.splitlines()[:-1]
    assert after.stdout.splitlines()[-1] == "[]"
    assert dotenv.read_bytes() == (
        b"FRAMEWORK=vllm\nHYPERLOOM_RUN_MODE=docker\nHF_HUB_OFFLINE=1\nHF_DATASETS_OFFLINE=0\n"
        b"HYPERLOOM_LLM_MODE=openai\nHYPERLOOM_FRAMEWORK_ENV=shared\nHF_DATASETS_OFFLINE=1\n"
    )
    assert dotenv.stat().st_mode & 0o777 == 0o600


# A long s: under a UTF-8 locale bash's ${1^^} and str.upper() both fold it to S, so this name folds to a lent one.
_LONG_S_NAME = "hf_data\u017fets_offline"


def test_a_name_that_folds_to_a_lent_one_is_neither_lent_nor_kept_by_the_rewrite(tmp_path):
    """A name that is not ASCII is refused before it is folded, so the rewrite drops it and the .env it leaves loads."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    checkout.chmod(0o777)
    dotenv = checkout / ".env"
    dotenv.write_text(f"FRAMEWORK=vllm\n{_LONG_S_NAME}=1\n", encoding="utf-8")
    env = {"PATH": os.environ["PATH"], "REPO_ROOT": str(checkout), "LC_ALL": "C.UTF-8"}

    folded = _shell('printf "%s" "${1^^}"', _LONG_S_NAME, env=env)
    lent = _shell('runtime_env_dotenv_lent "$1" 1', _LONG_S_NAME, env=env)
    rewrite = _shell('runtime_env_keep_lent_dotenv_lines "$1"', str(dotenv), env=env)
    load = _shell('load_dotenv_no_clobber && printf "%s" "$FRAMEWORK"', env=env)

    assert folded.stdout == "HF_DATASETS_OFFLINE"
    assert lent.returncode == 1
    assert rewrite.returncode == 0, rewrite.stderr
    assert dotenv.read_bytes() == b"FRAMEWORK=vllm\n"
    assert load.returncode == 0, load.stderr
    assert load.stdout == "vllm"


def _turkish_env(checkout: Path) -> dict[str, str]:
    env = {"PATH": os.environ["PATH"], "REPO_ROOT": str(checkout), "LC_ALL": "tr_TR.UTF-8"}
    if "LOCPATH" in os.environ:
        env["LOCPATH"] = os.environ["LOCPATH"]
    return env


def test_a_lent_name_with_an_i_is_lent_under_a_turkish_locale(tmp_path):
    """Under tr_TR ${1^^} folds i to a dotted capital I; the check folds in the C locale, and the caller's is kept."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    checkout.chmod(0o777)
    dotenv = checkout / ".env"
    dotenv.write_text("FRAMEWORK=vllm\nhf_hub_offline=1\nHf_Datasets_Offline=0\n", encoding="utf-8")
    env = _turkish_env(checkout)
    folded = _shell('printf "%s" "${1^^}"', "i", env=env)
    if folded.stdout != "\u0130":
        pytest.skip("the tr_TR.UTF-8 locale is not installed, so bash cannot fold i to a dotted capital I here")

    lent = _shell(
        'for n in "$@"; do runtime_env_dotenv_lent "$n" 1 && printf "%s\\n" "$n"; done; x=i; printf "%s" "${x^^}"',
        "hf_hub_offline",
        "HF_DATASETS_OFFLINE",
        "Hf_Datasets_Offline",
        env=env,
    )
    rewrite = _shell('runtime_env_keep_lent_dotenv_lines "$1"', str(dotenv), env=env)

    assert lent.returncode == 0, lent.stderr
    assert lent.stdout == "hf_hub_offline\nHF_DATASETS_OFFLINE\nHf_Datasets_Offline\n\u0130"
    assert rewrite.returncode == 0, rewrite.stderr
    assert dotenv.read_bytes() == b"FRAMEWORK=vllm\nhf_hub_offline=1\nHf_Datasets_Offline=0\n"


def test_preflight_lends_no_name_that_folds_to_a_lent_one(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    checkout.chmod(0o777)
    dotenv = checkout / ".env"
    dotenv.write_text("FRAMEWORK=vllm\n", encoding="utf-8")

    assert _LONG_S_NAME.upper() == "HF_DATASETS_OFFLINE"
    assert not env_safety.is_lent_not_private_dotenv_assignment(_LONG_S_NAME, "1")
    assert not env_safety.is_allowed_not_private_dotenv_key(_LONG_S_NAME)
    assert not env_safety.is_allowed_dotenv_key(_LONG_S_NAME)
    assert env_safety.is_lent_not_private_dotenv_assignment("hf_datasets_offline", "1")
    lent = preflight_env._allowed_dotenv_vars(dotenv, {"FRAMEWORK": "vllm", _LONG_S_NAME: "1"})
    assert lent == {"FRAMEWORK": "vllm"}


# Between the rewrite and the install's next write of .env, the other account appends a line to the file the rewrite
# left, as it would from an inotify watch on the checkout; each attempt is recorded.
_APPEND_AS_ANOTHER_ACCOUNT_AFTER = """
{wrapped}() {{
  _shipped_{wrapped} "$@" || return
  if setpriv --reuid={uid} --regid={uid} --clear-groups -- sh -c 'printf "%s\\n" "$1" >> "$2"' - "$LATE_LINE" "$DOTENV" \\
    2>/dev/null; then
    echo appended >> "$RACE_RECORD"
  else
    echo refused >> "$RACE_RECORD"
  fi
}}
"""


def _append_after(wrapped: str) -> str:
    return f'eval "_shipped_$(declare -f {wrapped})"\n' + _APPEND_AS_ANOTHER_ACCOUNT_AFTER.format(
        wrapped=wrapped, uid=_OTHER_UID
    )


@pytest.mark.parametrize("installer", ["install.sh", "install_baremetal.sh"])
@pytest.mark.parametrize("planted", [True, False], ids=["with-planted-lines", "lent-lines-only"])
def test_another_account_cannot_append_to_the_dotenv_the_rewrite_leaves(tmp_path, monkeypatch, installer, planted):
    """The rewrite leaves a file of mode 0600, even when it removes nothing: a line another account appends right after
    it is refused, so the install's next write never copies it into a .env the next reader finds private."""
    if os.geteuid() != 0 or shutil.which("setpriv") is None:
        pytest.skip("acting as a second account needs root and setpriv")
    root = _shared_root()
    try:
        checkout = root / "checkout"
        checkout.mkdir(mode=0o755)
        dotenv = checkout / ".env"
        text = _PLANTED_IN_A_WRITABLE_DOTENV if planted else "# kept: a comment\nFRAMEWORK=vllm\nHF_HUB_OFFLINE='1'\n"
        dotenv.write_text(text, encoding="utf-8")
        dotenv.chmod(0o666)
        workspace = tmp_path / "workspace"
        tmp = _our_dir(tmp_path / "tmp")
        monkeypatch.setenv("TMPDIR", str(tmp))
        record = tmp_path / "race-record"
        race = {"LATE_LINE": "NODE_OPTIONS=--require=/tmp/planted-late.js", "RACE_RECORD": str(record)}

        if installer == "install.sh":
            script = _write_installer(tmp_path / "assets", _append_after("runtime_env_keep_lent_dotenv_lines"))
            proc = _run(script, checkout, {"USER_DATA_PATH": str(workspace), **_CREDENTIALS, **race})
            framework = "vllm"
        else:
            script = _baremetal(tmp_path / "assets", _append_after("keep_lent_dotenv_lines") + _BAREMETAL_STUBS)
            proc = _run_baremetal(script, checkout, workspace, tmp, **_BAREMETAL_CREDENTIALS, **race)
            framework = ""

        assert proc.returncode == 0, proc.stdout + proc.stderr
        attempts = record.read_text(encoding="utf-8").split()
        assert attempts
        assert set(attempts) == {"refused"}
        assert dotenv.stat().st_mode & 0o777 == 0o600
        _assert_the_next_load_lends_none_of_it(checkout, workspace, framework=framework)
    finally:
        shutil.rmtree(root)


@pytest.mark.parametrize("owner", ["the-checkouts-owner", "a-third-account"])
def test_root_leaves_the_dotenv_it_rewrites_its_own(tmp_path, owner):
    """Root, on a user's checkout whose .env any account can write, replaces it with a 0600 file of root's, whoever
    owned the old one, as every later write of .env leaves it: nothing is chowned."""
    if os.geteuid() != 0:
        pytest.skip("only root can give an entry to another account")
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o755)
    os.chown(checkout, _OTHER_UID, _OTHER_UID)
    dotenv = checkout / ".env"
    dotenv.write_text("# a comment\nFRAMEWORK=vllm\n", encoding="utf-8")
    dotenv.chmod(0o666)
    uid = _OTHER_UID if owner == "the-checkouts-owner" else _THIRD_UID
    os.chown(dotenv, uid, uid)

    proc = _shell('runtime_env_keep_lent_dotenv_lines "$1"', str(dotenv))

    assert proc.returncode == 0, proc.stderr
    entry = dotenv.stat()
    assert entry.st_mode & 0o777 == 0o600
    assert (entry.st_uid, entry.st_gid) == (0, 0)
    assert dotenv.read_text(encoding="utf-8") == "FRAMEWORK=vllm\n"
    assert sorted(path.name for path in checkout.iterdir()) == [".env"]


# One .env, read by both loaders: carriage returns in a comment, before a line, inside a value, doubled at a line's end,
# in a quoted value over two CRLF lines and ending the file. A reader that split lines at a lone carriage return (as
# universal newlines do) would find OPENAI_BASE_URL and HF_DATASETS_OFFLINE assigned.
_CR_LADEN = (
    b"# a note\rOPENAI_BASE_URL=http://planted-gateway.invalid/v1\n"
    b"FRAMEWORK=vllm\r\n"
    b"\rHYPERLOOM_LLM_MODE=openai\n"
    b"HF_HUB_OFFLINE=1\rHF_DATASETS_OFFLINE=1\n"
    b"HYPERLOOM_RUN_MODE=baremetal\r\r\n"
    b'OPENAI_CUSTOM_HEADERS="X-A: 1\r\nX-B: 2"\r\n'
    b"HYPERLOOM_FRAMEWORK_ENV=shared\r"
)
_CR_NAMES = (
    "OPENAI_BASE_URL",
    "FRAMEWORK",
    "HYPERLOOM_LLM_MODE",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HYPERLOOM_RUN_MODE",
    "OPENAI_CUSTOM_HEADERS",
    "HYPERLOOM_FRAMEWORK_ENV",
)
_CR_LENT = {
    "FRAMEWORK": "vllm",
    "HYPERLOOM_LLM_MODE": "openai",
    "HYPERLOOM_RUN_MODE": "baremetal",
    "HYPERLOOM_FRAMEWORK_ENV": "shared",
}


def _both_loaders(checkout: Path) -> tuple[dict[str, str], dict[str, str]]:
    """What load_dotenv_no_clobber and preflight's _load_dotenv_fallback each take of _CR_NAMES from ``checkout``'s
    .env into an environment that has none of them, read as bytes so no carriage return is lost on the way."""
    env = {"PATH": os.environ["PATH"], "REPO_ROOT": str(checkout)}
    report = (
        'load_dotenv_no_clobber || exit 9; for k in "$@"; do [ -z "${!k+set}" ] || printf "%s=%s\\0" "$k" "${!k}"; done'
    )
    proc = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", f'. "$0"; {report}', RUNTIME_ENV_SH.as_posix(), *_CR_NAMES],
        env=env,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    shell = dict(item.split("=", 1) for item in proc.stdout.decode("utf-8").split("\0") if item)
    with mock.patch.dict(os.environ, env, clear=True):
        preflight_env._load_dotenv_fallback()
        python = {key: os.environ[key] for key in _CR_NAMES if key in os.environ}
    return shell, python


@pytest.mark.parametrize("mode", [0o600, 0o666], ids=["private", "writable-by-others"])
def test_the_shell_and_preflight_split_a_dotenvs_lines_alike(tmp_path, mode):
    """Both loaders end a line at a newline alone and drop one carriage return before it: from one .env full of
    carriage returns they take the same assignments, and neither finds one in a comment. A .env another account can
    write lends the same once the install rebuilt it, then private, to either."""
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o755)
    dotenv = checkout / ".env"
    dotenv.write_bytes(_CR_LADEN)
    dotenv.chmod(mode)

    shell, python = _both_loaders(checkout)

    assert python == shell
    if mode == 0o600:
        assert shell == {
            **_CR_LENT,
            "HF_HUB_OFFLINE": "1\rHF_DATASETS_OFFLINE=1",
            "OPENAI_CUSTOM_HEADERS": "X-A: 1\nX-B: 2",
        }
        return
    assert shell == _CR_LENT
    rewrite = _shell('runtime_env_keep_lent_dotenv_lines "$1"', str(dotenv))
    assert rewrite.returncode == 0, rewrite.stderr
    assert preflight_env._dotenv_not_private(dotenv) is None
    assert _both_loaders(checkout) == (_CR_LENT, _CR_LENT)


def _fake_command(bin_dir: Path, name: str, body: str) -> None:
    bin_dir.mkdir(exist_ok=True)
    command = bin_dir / name
    command.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    command.chmod(0o755)


def _rewrite_with(bin_dir: Path, dotenv: Path, **extra: str) -> subprocess.CompletedProcess:
    """runtime_env_keep_lent_dotenv_lines with the commands in ``bin_dir`` ahead of the real ones."""
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", **extra}
    return _shell('runtime_env_keep_lent_dotenv_lines "$1"', str(dotenv), env=env)


def _writable_dotenv(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    checkout.chmod(0o777)
    dotenv = checkout / ".env"
    dotenv.write_text("FRAMEWORK=vllm\nNODE_OPTIONS=--require=/tmp/planted.js\n", encoding="utf-8")
    dotenv.chmod(0o666)
    return dotenv


# The name the rewrite's new file gets, made known to the other account by a stand-in for mktemp.
_KNOWN_NAME = ".env.knownname0"


def test_the_rewrite_never_writes_through_a_link_at_its_new_files_name(tmp_path):
    """Another account that puts a link where the rewrite makes its new file finds it refused: the file is made
    there or not at all (O_EXCL), so a file of this account's the link names is never written."""
    dotenv = _writable_dotenv(tmp_path)
    victim = tmp_path / "victim"
    victim.write_text("this account's own file\n", encoding="utf-8")
    (dotenv.parent / _KNOWN_NAME).symlink_to(victim)
    _fake_command(tmp_path / "bin", "mktemp", f"echo {_KNOWN_NAME}")

    proc = _rewrite_with(tmp_path / "bin", dotenv)

    assert proc.returncode == 1
    assert f"cannot remove the credentials from {dotenv}" in proc.stderr
    assert victim.read_text(encoding="utf-8") == "this account's own file\n"
    assert "NODE_OPTIONS" in dotenv.read_text(encoding="utf-8")


def test_the_rewrite_refuses_a_new_file_that_is_not_a_regular_file_of_its_own(tmp_path):
    """A link at the new file's name to something other than a regular file (a pipe here) is opened by the shell's
    noclobber, and then refused before anything is written to it, given a mode or renamed."""
    dotenv = _writable_dotenv(tmp_path)
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe, 0o644)
    (dotenv.parent / _KNOWN_NAME).symlink_to(pipe)
    _fake_command(tmp_path / "bin", "mktemp", f"echo {_KNOWN_NAME}")
    reader = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)
    try:
        proc = _rewrite_with(tmp_path / "bin", dotenv)
        written = b""
        with contextlib.suppress(BlockingIOError):
            written = os.read(reader, 4096)
    finally:
        os.close(reader)

    assert proc.returncode == 1
    assert written == b""
    assert stat.S_IMODE(pipe.stat().st_mode) == 0o644
    assert dotenv.is_file() and not dotenv.is_symlink()
    assert "NODE_OPTIONS" in dotenv.read_text(encoding="utf-8")


def test_the_rewrite_finds_a_new_file_swapped_before_its_rename(tmp_path):
    """Another account that replaces the new file with its own between the write and the rename has put its file at
    .env; the rewrite finds that and fails rather than report a .env of this account's."""
    dotenv = _writable_dotenv(tmp_path)
    real_mv = shutil.which("mv")
    _fake_command(
        tmp_path / "bin",
        "mv",
        f'rm -f -- "$3" && printf "NODE_OPTIONS=--require=/tmp/swapped.js\\n" > "$3" && exec {real_mv} "$@"',
    )

    proc = _rewrite_with(tmp_path / "bin", dotenv)

    assert proc.returncode == 1
    assert f"cannot remove the credentials from {dotenv}" in proc.stderr


def test_the_rewrite_stays_in_the_directory_it_opened(tmp_path):
    """Another account that moves the checkout directory away and puts a link to a directory of its own at its name,
    while the rewrite runs, gets nothing there: the new file is made, written and renamed through the directory the
    rewrite opened."""
    dotenv = _writable_dotenv(tmp_path)
    checkout = dotenv.parent
    moved = tmp_path / "moved"
    theirs = tmp_path / "theirs"
    theirs.mkdir()
    _fake_command(
        tmp_path / "bin",
        "mktemp",
        f'mv -- "{checkout}" "{moved}" && ln -s -- "{theirs}" "{checkout}" && echo {_KNOWN_NAME}',
    )

    proc = _rewrite_with(tmp_path / "bin", dotenv)

    assert proc.returncode == 0, proc.stderr
    assert list(theirs.iterdir()) == []
    assert (moved / ".env").read_text(encoding="utf-8") == "FRAMEWORK=vllm\n"
    assert stat.S_IMODE((moved / ".env").stat().st_mode) == 0o600
    assert sorted(path.name for path in moved.iterdir()) == [".env"]
