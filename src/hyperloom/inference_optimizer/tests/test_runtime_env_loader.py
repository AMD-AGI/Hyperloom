# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the shipped dotenv loader and runtime installer's environment boundary."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

from hyperloom.common.llm_config import parse_custom_headers

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_LOADER = _ASSETS / "runtime_env.sh"
_PYTHON_KEYS = ("PYTHON", "VIRTUAL_ENV", "INFERENCE_OPTIMIZER_FORCE_PYTHON")


def _run_shell(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"}}
    runner = tmp_path / "runner.sh"
    runner.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body + "\n", encoding="utf-8", newline="\n")
    return subprocess.run(
        ["bash", runner.as_posix()], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10
    )


def _load(tmp_path: Path, *, before: str = "", after: str = "") -> subprocess.CompletedProcess[str]:
    return _run_shell(
        tmp_path,
        f'REPO_ROOT="$PWD"\n{before}\n. {shlex.quote(_LOADER.as_posix())}\nload_dotenv_no_clobber\n{after}',
    )


def test_runtime_env_source_only_defines_functions(tmp_path: Path):
    (tmp_path / ".env").write_text("LOADER_TEST_VALUE=from-file\n", encoding="utf-8")
    proc = _run_shell(
        tmp_path,
        f'REPO_ROOT="$PWD"\nDOTENV_LOADED_COUNT=37\n'
        f". {shlex.quote(_LOADER.as_posix())}\n"
        '[ -z "${LOADER_TEST_VALUE+x}" ]\n[ "$DOTENV_LOADED_COUNT" = 37 ]\n'
        "declare -F load_dotenv_no_clobber >/dev/null",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == proc.stderr == ""


def test_runtime_env_preserves_assignment_syntax_and_gap_filling(tmp_path: Path):
    (tmp_path / ".env").write_bytes(
        b"  # comment\r\n\r\nnot an assignment\r\n"
        b"export LOADER_TEST_KEEP=from-file\r\n"
        b" LOADER_TEST_EMPTY = 'filled value' \r\n"
        b'LOADER_TEST_QUOTES="double value"\r\n'
        b"LOADER_TEST_HASH=literal # not a comment\r\n"
        b"LOADER_TEST_EQUALS=a=b\r\n"
        b"LOADER_TEST_BLANK=\r\n"
        b"LOADER_TEST_DUP=first\r\nLOADER_TEST_DUP=second\r\n"
        b"LOADER_TEST_LAST=without newline"
    )
    proc = _load(
        tmp_path,
        before="LOADER_TEST_KEEP=caller\nLOADER_TEST_EMPTY=",
        after="\n".join(
            [
                '[ "$LOADER_TEST_KEEP" = caller ]',
                '[ "$LOADER_TEST_EMPTY" = "filled value" ]',
                '[ "$LOADER_TEST_QUOTES" = "double value" ]',
                '[ "$LOADER_TEST_HASH" = "literal # not a comment" ]',
                '[ "$LOADER_TEST_EQUALS" = a=b ]',
                '[ "${LOADER_TEST_BLANK+x}" = x ] && [ -z "$LOADER_TEST_BLANK" ]',
                '[ "$LOADER_TEST_DUP" = first ]',
                '[ "$LOADER_TEST_LAST" = "without newline" ]',
                '[ "$DOTENV_LOADED_COUNT" = 7 ]',
                'bash -c \'[ "$LOADER_TEST_EMPTY" = "filled value" ]\'',
            ]
        ),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == proc.stderr == ""


@pytest.mark.parametrize("caller_mode", [None, "", "docker", "baremetal"])
@pytest.mark.parametrize("file_mode", ["docker", "baremetal"])
@pytest.mark.parametrize("caller_python", [None, "", "explicit"])
def test_runtime_env_python_isolation_uses_selected_mode_not_line_order(
    tmp_path: Path, caller_mode: str | None, file_mode: str, caller_python: str | None
):
    (tmp_path / ".env").write_text(
        "\n".join([*(f"{key}=host-{key}" for key in _PYTHON_KEYS), f"HYPERLOOM_RUN_MODE={file_mode}"]) + "\n",
        encoding="utf-8",
    )
    before = []
    if caller_mode is not None:
        before.append(f"HYPERLOOM_RUN_MODE={shlex.quote(caller_mode)}")
    if caller_python is not None:
        before.extend(f"{key}={shlex.quote(caller_python)}" for key in _PYTHON_KEYS)
    selected_mode = caller_mode or file_mode
    count = int(not caller_mode)
    checks = [f'[ "$HYPERLOOM_RUN_MODE" = {selected_mode} ]']
    for key in _PYTHON_KEYS:
        if selected_mode == "docker":
            expected = caller_python
        else:
            expected = caller_python or f"host-{key}"
            count += int(not caller_python)
        if expected is None:
            checks.append(f'[ -z "${{{key}+x}}" ]')
        else:
            checks.extend([f'[ "${{{key}+x}}" = x ]', f'[ "${{{key}}}" = {shlex.quote(expected)} ]'])
    checks.append(f'[ "$DOTENV_LOADED_COUNT" = {count} ]')
    proc = _load(tmp_path, before="\n".join(before), after="\n".join(checks))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == proc.stderr == ""


def test_runtime_env_mode_fallback_uses_existing_assignment_syntax(tmp_path: Path):
    (tmp_path / ".env").write_text(
        'PYTHON=/host/python\n export HYPERLOOM_RUN_MODE = "docker" \nHYPERLOOM_RUN_MODE=baremetal\n',
        encoding="utf-8",
    )
    proc = _load(
        tmp_path,
        after='[ "$HYPERLOOM_RUN_MODE" = docker ]\n[ -z "${PYTHON+x}" ]\n[ "$DOTENV_LOADED_COUNT" = 1 ]',
    )
    assert proc.returncode == 0, proc.stderr


def test_runtime_env_without_mode_keeps_baremetal_gap_fill(tmp_path: Path):
    (tmp_path / ".env").write_text("PYTHON=/host/python\n", encoding="utf-8")
    proc = _load(tmp_path, after='[ "$PYTHON" = /host/python ]\n[ -z "${HYPERLOOM_RUN_MODE+x}" ]')
    assert proc.returncode == 0, proc.stderr


def test_runtime_env_repeated_load_only_counts_missing_exports(tmp_path: Path):
    (tmp_path / ".env").write_text("HYPERLOOM_RUN_MODE=docker\nLOADER_TEST_VALUE=from-file\n", encoding="utf-8")
    proc = _load(
        tmp_path,
        after='[ "$DOTENV_LOADED_COUNT" = 2 ]\nload_dotenv_no_clobber\n'
        '[ "$DOTENV_LOADED_COUNT" = 0 ]\n[ "$LOADER_TEST_VALUE" = from-file ]',
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == proc.stderr == ""


@pytest.mark.parametrize("file_exists", [False, True])
def test_runtime_env_missing_or_empty_file_resets_count(tmp_path: Path, file_exists: bool):
    if file_exists:
        (tmp_path / ".env").touch()
    proc = _load(tmp_path, before="DOTENV_LOADED_COUNT=73", after='[ "$DOTENV_LOADED_COUNT" = 0 ]')
    assert proc.returncode == 0, proc.stderr


def test_runtime_env_does_not_execute_or_print_dotenv_values(tmp_path: Path):
    payload = "$(touch command-substitution-ran); `touch backtick-ran`; ${HOME}"
    (tmp_path / ".env").write_text(f'LOADER_TEST_SECRET="{payload}"\n', encoding="utf-8")
    proc = _load(tmp_path, after=f'[ "$LOADER_TEST_SECRET" = {shlex.quote(payload)} ]')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == proc.stderr == ""
    assert not (tmp_path / "command-substitution-ran").exists()
    assert not (tmp_path / "backtick-ran").exists()


@pytest.mark.parametrize(
    ("assignment", "expected_raw", "expected_headers"),
    [
        (
            r'OPENAI_CUSTOM_HEADERS="{\"X-Tenant\": \"acme\"}"',
            '{"X-Tenant": "acme"}',
            {"X-Tenant": "acme"},
        ),
        (
            r'OPENAI_CUSTOM_HEADERS="{\"Ocp-Apim-Subscription-Key\": \"mock-key\"}"',
            '{"Ocp-Apim-Subscription-Key": "mock-key"}',
            {"Ocp-Apim-Subscription-Key": "mock-key"},
        ),
        (
            "OPENAI_CUSTOM_HEADERS='X-Tenant: acme\nX-Label: first second'",
            "X-Tenant: acme\nX-Label: first second",
            {"X-Tenant": "acme", "X-Label": "first second"},
        ),
        (
            'OPENAI_CUSTOM_HEADERS="X-Tenant: acme\nX-Label: first second"',
            "X-Tenant: acme\nX-Label: first second",
            {"X-Tenant": "acme", "X-Label": "first second"},
        ),
        (
            'OPENAI_CUSTOM_HEADERS="{\n  \\"X-Tenant\\": \\"acme\\"\n}"',
            '{\n  "X-Tenant": "acme"\n}',
            {"X-Tenant": "acme"},
        ),
    ],
)
def test_runtime_env_quoted_headers_reach_real_header_parser(
    tmp_path: Path, assignment: str, expected_raw: str, expected_headers: dict[str, str]
):
    (tmp_path / ".env").write_text(assignment + "\n", encoding="utf-8", newline="\n")
    proc = _load(tmp_path, after='printf "%s" "$OPENAI_CUSTOM_HEADERS"')
    assert proc.returncode == 0, proc.stderr
    assert parse_custom_headers(proc.stdout, env={}) == expected_headers
    assert proc.stdout == expected_raw
    assert proc.stderr == ""


@pytest.mark.parametrize(
    ("quoted", "expected"),
    [
        (r'"C:\\Users\\operator\\data path"', r"C:\Users\operator\data path"),
        (r'"C:\runtime\new\folder\q"', r"C:\runtime\new\folder\q"),
        (r'"/workspace/a path/with\ spaces"', r"/workspace/a path/with\ spaces"),
        (r'"say \"hello\" and \\ goodbye"', 'say "hello" and \\ goodbye'),
        (r'"\$HOME \${HOME} \`literal\`"', "$HOME ${HOME} `literal`"),
        (r'"${HOME} $HOME $((2 + 2)) ~ *"', "${HOME} $HOME $((2 + 2)) ~ *"),
        ('"first\\\nsecond"', "firstsecond"),
        ('"first\\\\\nsecond"', "first\\\nsecond"),
        ('"first  \n  second\n"', "first  \n  second\n"),
        ("'first  \n  second\\\nthird'", "first  \n  second\\\nthird"),
        (r"'\"\\\$HOME\`literal\`'", r"\"\\\$HOME\`literal\`"),
        (r"\"\\\$HOME\`literal\`", r"\"\\\$HOME\`literal\`"),
    ],
)
def test_runtime_env_quote_escaping_preserves_safe_data(tmp_path: Path, quoted: str, expected: str):
    (tmp_path / ".env").write_text(f"LOADER_TEST_VALUE={quoted}\n", encoding="utf-8", newline="\n")
    proc = _load(tmp_path, after='printf "%s" "$LOADER_TEST_VALUE"')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected
    assert proc.stderr == ""


def test_runtime_env_escaped_commands_and_multiline_assignments_are_only_data(tmp_path: Path):
    expected = '$(touch escaped-command-ran)\n`touch escaped-backtick-ran`\nUSER_DATA_PATH=/wrong\n"quoted"'
    (tmp_path / ".env").write_text(
        'LOADER_TEST_VALUE="\\$(touch escaped-command-ran)\n'
        '\\`touch escaped-backtick-ran\\`\nUSER_DATA_PATH=/wrong\n\\"quoted\\""\n',
        encoding="utf-8",
        newline="\n",
    )
    proc = _load(
        tmp_path,
        before="USER_DATA_PATH=/selected",
        after='[ "$USER_DATA_PATH" = /selected ]\n[ "$DOTENV_LOADED_COUNT" = 1 ]\nprintf "%s" "$LOADER_TEST_VALUE"',
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == expected
    assert proc.stderr == ""
    assert not (tmp_path / "escaped-command-ran").exists()
    assert not (tmp_path / "escaped-backtick-ran").exists()


@pytest.mark.parametrize("quote", ["'", '"'])
def test_runtime_env_unterminated_quoted_value_fails_without_partial_export(tmp_path: Path, quote: str):
    secret = "mock-secret-not-for-diagnostics"
    (tmp_path / ".env").write_text(f"LOADER_TEST_VALUE={quote}{secret}\nnext line", encoding="utf-8", newline="\n")
    proc = _run_shell(
        tmp_path,
        f'REPO_ROOT="$PWD"\n. {shlex.quote(_LOADER.as_posix())}\n'
        'if load_dotenv_no_clobber; then exit 97; fi\n[ -z "${LOADER_TEST_VALUE+x}" ]',
    )
    assert proc.returncode == 0
    assert "unterminated quoted value" in proc.stderr
    assert secret not in proc.stdout + proc.stderr


@pytest.mark.parametrize("key", ["9INVALID", "INVALID-KEY", "INVALID KEY", "ARRAY[$(touch invalid-key-ran)]", ""])
def test_runtime_env_rejects_invalid_names_before_indirect_expansion(tmp_path: Path, key: str):
    secret = "dotenv-secret-must-not-be-printed"
    (tmp_path / ".env").write_text(f"{key}={secret}\n", encoding="utf-8")
    proc = _load(tmp_path, after="printf SHOULD_NOT_CONTINUE")
    assert proc.returncode != 0
    assert "invalid variable name" in proc.stderr.lower()
    assert secret not in proc.stdout + proc.stderr
    assert "SHOULD_NOT_CONTINUE" not in proc.stdout
    assert not (tmp_path / "invalid-key-ran").exists()


@pytest.mark.parametrize("authority", ["marker", "flag", "none"])
@pytest.mark.parametrize("file_root", ["/selected/session", "/different/session"])
def test_runtime_env_readonly_workspace_obeys_authority(tmp_path: Path, authority: str, file_root: str):
    prefix = "HYPERLOOM_RUN_MODE=baremetal\n" if authority == "marker" else ""
    (tmp_path / ".env").write_text(prefix + f"USER_DATA_PATH={file_root}\n", encoding="utf-8")
    before = "readonly USER_DATA_PATH=/selected/session"
    if authority == "flag":
        before += "\nHYPERLOOM_SETUP_ENV_AUTHORITATIVE=1"
    proc = _load(tmp_path, before=before, after='[ "$USER_DATA_PATH" = /selected/session ]')
    conflict = authority != "none" and file_root != "/selected/session"
    if conflict:
        assert proc.returncode != 0
        assert "USER_DATA_PATH" in proc.stderr
        assert "readonly" in proc.stderr.lower()
        assert file_root not in proc.stderr
    else:
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == proc.stderr == ""


def _installer_environment_block() -> str:
    text = (_ASSETS / "install.sh").read_text(encoding="utf-8")
    return text[text.index('REPO_ROOT="$(resolve_repo_root)"') : text.index("\nVLLM_IMAGE_SOURCE_ROOT=")]


@pytest.mark.parametrize("file_root", [None, "/selected/session", "/different/session"])
@pytest.mark.parametrize("readonly_root", [False, True])
def test_runtime_installer_workspace_initialization_handles_readonly(
    tmp_path: Path, file_root: str | None, readonly_root: bool
):
    (tmp_path / ".env").write_text(
        "HYPERLOOM_RUN_MODE=baremetal\n" + (f"USER_DATA_PATH={file_root}\n" if file_root is not None else ""),
        encoding="utf-8",
    )
    proc = _run_shell(
        tmp_path,
        f'_script_dir={shlex.quote(_ASSETS.as_posix())}\nresolve_repo_root() {{ printf "%s" "$PWD"; }}\n'
        f"{'readonly ' if readonly_root else ''}USER_DATA_PATH=/selected/session\n"
        "HYPERLOOM_RUNTIME_DIR=/old/runtime\nKERNEL_AGENT_ENV=/old/env\n"
        + _installer_environment_block()
        + '\n[ "$HYPERLOOM_RUNTIME_DIR" = "${USER_DATA_PATH}/runtime" ]\n'
        '[ "$KERNEL_AGENT_ENV" = "${USER_DATA_PATH}/runtime/kernel-agent.env.sh" ]\n'
        'printf "%s" "$USER_DATA_PATH"',
    )
    if readonly_root and file_root == "/different/session":
        assert proc.returncode != 0
        assert "USER_DATA_PATH" in proc.stderr
        assert "readonly" in proc.stderr.lower()
        assert "cannot unset" not in proc.stderr
    else:
        assert proc.returncode == 0, proc.stderr
        if readonly_root:
            assert proc.stdout == "/selected/session"
        elif file_root:
            assert proc.stdout == file_root
        else:
            assert proc.stdout != "/selected/session"
            assert "defaulting" in proc.stderr


def test_runtime_installer_rejects_empty_readonly_workspace(tmp_path: Path):
    proc = _run_shell(
        tmp_path,
        f'_script_dir={shlex.quote(_ASSETS.as_posix())}\nresolve_repo_root() {{ printf "%s" "$PWD"; }}\n'
        "readonly USER_DATA_PATH=\n" + _installer_environment_block(),
    )
    assert proc.returncode != 0
    assert "readonly USER_DATA_PATH is empty" in proc.stderr


def test_runtime_env_readonly_empty_same_value_is_not_assigned(tmp_path: Path):
    (tmp_path / ".env").write_text("HYPERLOOM_RUN_MODE=baremetal\nUSER_DATA_PATH=\n", encoding="utf-8")
    proc = _load(tmp_path, before="readonly USER_DATA_PATH=", after='[ "$DOTENV_LOADED_COUNT" = 1 ]')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == proc.stderr == ""


@pytest.mark.parametrize("python_pin", [None, "", "/missing/forced/python", "/"])
def test_runtime_installer_invalid_forced_python_never_falls_back(tmp_path: Path, python_pin: str | None):
    text = (_ASSETS / "install.sh").read_text(encoding="utf-8")
    resolver = text[text.index("resolve_python() {") : text.index("\nresolve_python\n")]
    pin = f"PYTHON={shlex.quote(python_pin)}\n" if python_pin is not None else ""
    proc = _run_shell(
        tmp_path,
        "log() { :; }\nwarn() { :; }\ndie() { printf '%s\\n' \"$*\" >&2; exit 99; }\n"
        "command() { printf FALLBACK_ATTEMPTED >&2; return 0; }\n"
        "INFERENCE_OPTIMIZER_FORCE_PYTHON=1\nCHECK_ONLY=0\nDRY_RUN=0\n" + pin + resolver + "\nresolve_python\n",
    )
    assert proc.returncode != 0
    assert "PYTHON" in proc.stderr
    assert "FALLBACK_ATTEMPTED" not in proc.stderr


def test_runtime_installer_valid_forced_python_is_preserved(tmp_path: Path):
    text = (_ASSETS / "install.sh").read_text(encoding="utf-8")
    resolver = text[text.index("resolve_python() {") : text.index("\nresolve_python\n")]
    proc = _run_shell(
        tmp_path,
        'PYTHON="$BASH"\nexpected="$PYTHON"\nINFERENCE_OPTIMIZER_FORCE_PYTHON=1\n'
        + resolver
        + '\nresolve_python\n[ "$PYTHON" = "$expected" ]',
    )
    assert proc.returncode == 0, proc.stderr
