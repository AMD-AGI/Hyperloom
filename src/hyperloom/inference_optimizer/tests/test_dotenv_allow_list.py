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

import os
import re
import shutil
import subprocess
from pathlib import Path
from unittest import mock


from hyperloom.common import env_safety
from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.cli import preflight_env
from hyperloom.orchestrator.actions.executors import benchmark_backend

from ._kernel_tools_harness import RUNTIME_ENV_SH
from .test_dotenv_credentials_private import (
    _CREDENTIALS,
    _MARKER,
    _OTHER_UID,
    _assignments,
    _credentials_in,
    _launch_env,
    _private_dir,
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
    # The values a shipped choice can take, so a framework or backend added later is lent too.
    lent = env_safety.DOTENV_NOT_PRIVATE_LENT_VALUES
    assert lent["FRAMEWORK"] == frozenset(framework_registry.FRAMEWORKS)
    assert lent["HYPERLOOM_BENCHMARK_BACKEND"] == benchmark_backend.KNOWN_BENCHMARK_BACKENDS
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
        "MAGPIE_PYTHON",
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
    assert not [line for line in lent if line.split("=", 1)[0].upper() in _FETCH_AND_ENDPOINT_VALUES.keys() - {"FRAMEWORK"}]


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


def _plant(root: Path) -> tuple[Path, Path, dict[str, str]]:
    """A world-writable checkout whose .env and code another account (uid 65534) wrote; the directory each planted
    program leaves a marker file in when it runs; and what was planted, by name."""
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
    argv = ["setpriv", f"--reuid={_OTHER_UID}", f"--regid={_OTHER_UID}", "--clear-groups", "--"]
    argv += ["sh", "-c", writer, "-", str(planted)] + [part for item in files.items() for part in item]
    subprocess.run(
        argv,
        env={"PATH": os.environ["PATH"], "DOTENV_TEXT": dotenv_text, "CHECKOUT": str(checkout)},
        check=True,
        timeout=60,
    )
    assert (checkout / ".env").stat().st_uid == _OTHER_UID
    return checkout, markers, values


# Run where install.sh writes the kernel-agent env file: every program a planted variable would steer, started from
# the install's own shell, with what that shell exports recorded.
_RUN_WHAT_WOULD_RUN_PLANTED_CODE = """
env -0 > "$INSTALL_ENV_DUMP"
bash -c : ; sh -c : ; python3 -c pass
git --version > /dev/null
command -v node > /dev/null && node -e 0
"""


def test_a_shared_dotenv_another_account_planted_lends_none_of_it_to_the_install(monkeypatch):
    root = _shared_root()
    try:
        checkout, markers, values = _plant(root)
        our_tmp = _private_dir(root / "our-tmp", 0)
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
        assert f": {' '.join(values)}; export any of them you need in this shell" in warning.group(0)
        assert not [value for value in values.values() if value in proc.stdout + proc.stderr]
        assert _credentials_in(_kernel_env(workspace)) == _CREDENTIALS
        # Nor does the install write there what such a .env would not lend: the kernel-agent env file carries it.
        assert "/opt/own/site" not in (checkout / ".env").read_text(encoding="utf-8")
        assert _assignments(_kernel_env(workspace))["PYTHONPATH"].endswith(":/opt/own/site")
    finally:
        shutil.rmtree(root)


def test_preflight_lends_none_of_a_planted_shared_dotenv(capsys):
    root = _shared_root()
    try:
        checkout, _, values = _plant(root)
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


def test_a_private_dotenv_is_read_as_before(tmp_path):
    """The decision: a .env only this account can write keeps lending what install_kernel_tools.sh and
    install_baremetal.sh write there and the documented shell flows read back, PYTHONPATH and the interpreter
    included; preflight keeps to its allow list there, as before."""
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o755)
    kept = {
        "PYTHONPATH": "/opt/own/site",
        "MAGPIE_PYTHON": "/opt/own/bin/python",
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


def test_install_baremetal_writes_no_credential_into_a_shared_dotenv(monkeypatch):
    root = _shared_root()
    try:
        checkout, _, _ = _plant(root)
        # Another account's gateway, which a key read from the shell must never be paired with.
        planted_gateway = "ANTHROPIC_BASE_URL=https://planted-gateway.invalid\n"
        subprocess.run(
            ["setpriv", f"--reuid={_OTHER_UID}", f"--regid={_OTHER_UID}", "--clear-groups", "--", "sh", "-c"]
            + [f'printf "%s" "$1" >> {checkout}/.env', "-", planted_gateway],
            check=True,
        )
        tmp = _private_dir(root / "our-tmp", 0)
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


def test_install_baremetal_pairs_no_shell_key_with_a_shared_dotenvs_gateway(monkeypatch):
    root = _shared_root()
    try:
        checkout = root / "hyperloom"
        checkout.mkdir()
        checkout.chmod(0o777)
        (checkout / ".env").write_text("ANTHROPIC_BASE_URL=https://planted-gateway.invalid\n", encoding="utf-8")
        os.chown(checkout / ".env", _OTHER_UID, _OTHER_UID)
        tmp = _private_dir(root / "our-tmp", 0)
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


def test_install_baremetal_reads_from_a_shared_dotenv_only_what_the_loader_lends():
    """Its own .env reads (the vLLM venv, the workspace, the backend order, the run mode) keep to runtime_env_dotenv_lent:
    a listed name with a listed value, nothing else."""
    root = _shared_root()
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
            'for k in VLLM_VENV_ROOT USER_DATA_PATH KERNEL_OPT_BACKEND_ORDER HYPERLOOM_RUN_MODE HYPERLOOM_FRAMEWORK_ENV; do '
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
