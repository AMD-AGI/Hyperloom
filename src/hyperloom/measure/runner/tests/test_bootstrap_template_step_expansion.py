# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The start step survives a platform that expands variables in its text before ``sh`` runs it.

Two expansions are modelled. Kubernetes' own in a container's command and args (``$$`` to ``$``, ``$(NAME)`` to the
variable's value when it is defined, any other ``$`` kept), ported from ``third_party/forked/golang/expansion``. And a
shell that renders the step's text as an unquoted here-document, which expands ``$NAME``, ``${...}``, ``$$`` and
``$(...)`` whatever quotes surround them: run in a shell whose ``e`` holds an environment entry, it turns the earlier
plain-text step into exactly the boot log a hosted sandbox left.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from hyperloom.measure.runner.bootstrap import runner_start_shell_line, runner_template_payload, runner_template_step

#: The boot log of the hosted sandbox whose platform expanded the plain-text step.
_HOSTED_BOOT_LOG = (
    "mkdir: cannot create directory '': No such file or directory\n"
    "hyperloom-runner-bootstrap: bootstrap.sh or manifest of SINGULARITY_NAME=codeinterpreter is missing or not the "
    "one HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256 pins\n"
    "mkdir: cannot create directory '': No such file or directory\n"
)
_SHA = "0" * 40
_WAIT_SEC = 20.0


def k8s_expand(text: str, env: Mapping[str, str]) -> str:
    """Kubernetes' expansion of a container's command and args, as ``expansion.Expand`` with ``MappingFuncFor``."""
    out: list[str] = []
    checkpoint = cursor = 0
    while cursor < len(text):
        if text[cursor] == "$" and cursor + 1 < len(text):
            out.append(text[checkpoint:cursor])
            following = text[cursor + 1]
            advance = 1
            if following == "$":
                out.append("$")
            elif following == "(":
                closer = text.find(")", cursor + 2)
                if closer < 0:
                    out.append("$(")
                else:
                    name = text[cursor + 2 : closer]
                    out.append(env[name] if name in env else f"$({name})")
                    advance = closer - cursor
            else:
                out.append(f"${following}")
            cursor += advance
            checkpoint = cursor + 1
        cursor += 1
    return "".join(out) + text[checkpoint:]


def test_k8s_expand_follows_the_upstream_rules() -> None:
    env = {"VAR_A": "A", "EMPTY": ""}
    cases = {
        "$(VAR_A)": "A",
        "___$(VAR_A)___": "___A___",
        "$(EMPTY)": "",
        "$(UNDEFINED)": "$(UNDEFINED)",
        "$$(VAR_A)": "$(VAR_A)",
        "$$$$": "$$",
        "$VAR_A": "$VAR_A",
        "${VAR_A}": "${VAR_A}",
        "$(VAR_A": "$(VAR_A",
        "foo$": "foo$",
        "hl-start.$$": "hl-start.$",
    }
    assert {given: k8s_expand(given, env) for given in cases} == cases


def shell_render(text: str, env: Mapping[str, str], e: str | None) -> str:
    """``text`` as a shell renders it into an unquoted here-document, in a shell whose ``e`` is ``e``."""
    prefix = "" if e is None else f"e={e}; "
    rendered = subprocess.run(
        ["/bin/sh", "-c", f"{prefix}cat <<HLEOF\n{text}\nHLEOF\n"],
        env=dict(env),
        capture_output=True,
        text=True,
        check=True,
    )
    return rendered.stdout.removesuffix("\n")


def _pod(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """The pod's environment and the step's own paths, below ``root`` standing in for ``/in``, ``/out`` and so on."""
    paths = {
        "dirs": (str(root / "in"), str(root / "out"), str(root / "scratch")),
        "boot_log": str(root / "work" / "runner-boot.log"),
    }
    (root / "work").mkdir(parents=True)
    (root / "runner-cache").mkdir()
    (root / "tmp").mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "TMPDIR": str(root / "tmp"),
        "HYPERLOOM_IN_DIR": str(root / "in"),
        "HYPERLOOM_OUT_DIR": str(root / "out"),
        "HYPERLOOM_SCRATCH_DIR": str(root / "scratch"),
        "HYPERLOOM_RUNNER_CACHE": str(root / "runner-cache"),
        "HYPERLOOM_SOURCE_SHA": _SHA,
        "HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256": "a" * 64,
        "SINGULARITY_NAME": "codeinterpreter",
    }
    return env, paths  # type: ignore[return-value]


def _entry(root: Path, env: dict[str, str], bootstrap: str) -> None:
    """A cache entry whose ``bootstrap.sh`` is ``bootstrap``, its manifest pinned in ``env``."""
    entry = root / "runner-cache" / _SHA
    entry.mkdir()
    (entry / "bootstrap.sh").write_text(bootstrap)
    digest = hashlib.sha256(bootstrap.encode()).hexdigest()
    manifest = f'{{\n  "bootstrap_sha256": "{digest}",\n  "schema": 1\n}}\n'
    (entry / "manifest").write_text(manifest)
    env["HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256"] = hashlib.sha256(manifest.encode()).hexdigest()


_STUB = 'printf "%s\\n" "$@" >"$HYPERLOOM_OUT_DIR/started"\n'


def _run(step: str, env: Mapping[str, str]) -> int:
    return subprocess.run(["/bin/sh", "-c", step], env=dict(env), timeout=10, check=False).returncode


def _wait_for(path: Path) -> bool:
    deadline = time.monotonic() + _WAIT_SEC
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    return path.exists()


def _text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _status(root: Path) -> dict[str, str]:
    assert _wait_for(root / "out" / "status.json"), (root / "work" / "runner-boot.log").read_text()
    return json.loads((root / "out" / "status.json").read_text())


@pytest.mark.parametrize("pre", ["", 'cp "$HYPERLOOM_IN_DIR/../work/runner-boot.log" /dev/null || true'])
def test_the_step_has_no_character_a_platform_could_expand_or_unquote(pre: str) -> None:
    step = runner_template_step(pre=pre, env={"HYPERLOOM_SIGNING_KEY_FILE": "/app/work/k $x"})
    assert re.fullmatch(r"echo [A-Za-z0-9+/=]+ \| base64 -d \| sh", step), step
    assert "$" in runner_template_payload(pre=pre)


def test_the_step_is_its_own_expansion(tmp_path: Path) -> None:
    env, paths = _pod(tmp_path)
    step = runner_template_step(**paths)  # type: ignore[arg-type]
    assert k8s_expand(step, env) == step
    assert shell_render(step, env, e="SINGULARITY_NAME=codeinterpreter") == step


def _plain_step(paths: Mapping[str, object]) -> str:
    """The step as it was before it was encoded: the plain line, which the hosted platform rendered."""
    dirs = " ".join(map(shlex.quote, paths["dirs"]))  # type: ignore[call-overload]
    start = shlex.quote(runner_start_shell_line("--spec-wait-sec", "1800"))
    return f"mkdir -p {dirs} && (nohup sh -c {start} >{shlex.quote(str(paths['boot_log']))} 2>&1 &)"


def test_the_plain_step_rendered_by_a_shell_is_the_hosted_boot_log_line_for_line(tmp_path: Path) -> None:
    env, paths = _pod(tmp_path)
    plain = _plain_step(paths)
    assert k8s_expand(plain, env) != plain
    assert _run(shell_render(plain, env, e="SINGULARITY_NAME=codeinterpreter"), env) == 0
    boot = tmp_path / "work" / "runner-boot.log"
    deadline = time.monotonic() + _WAIT_SEC
    while _text(boot) != _HOSTED_BOOT_LOG and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _text(boot) == _HOSTED_BOOT_LOG
    assert not (tmp_path / "out" / "status.json").exists()  # so the application waited for its whole timeout


def test_kubernetes_expansion_alone_does_not_give_the_hosted_boot_log(tmp_path: Path) -> None:
    env, paths = _pod(tmp_path)
    assert _run(k8s_expand(_plain_step(paths), env), env) == 0
    status = _status(tmp_path)  # the line's own verify failure: it found the cache entry missing and said so
    assert "step verify" in status["message"]
    assert "SINGULARITY_NAME" not in (tmp_path / "work" / "runner-boot.log").read_text()


@pytest.mark.parametrize("expansion", ["none", "k8s", "shell"])
def test_the_step_starts_the_runner_under_every_expansion(tmp_path: Path, expansion: str) -> None:
    env, paths = _pod(tmp_path)
    _entry(tmp_path, env, _STUB)
    step = runner_template_step("--x", "a b", **paths)  # type: ignore[arg-type]
    if expansion == "k8s":
        step = k8s_expand(step, env)
    elif expansion == "shell":
        step = shell_render(step, env, e="SINGULARITY_NAME=codeinterpreter")
    began = time.monotonic()
    assert _run(step, env) == 0
    assert time.monotonic() - began < 5.0
    started = tmp_path / "out" / "started"
    assert _wait_for(started), (tmp_path / "work" / "runner-boot.log").read_text()
    assert started.read_text().splitlines() == ["--x", "a b", "--spec-wait-sec", "1800"]
    time.sleep(0.5)
    assert not (tmp_path / "out" / "status.json").exists()


@pytest.mark.parametrize("bootstrap", ["exec sleep 60\n", "sleep 60\nexit 3\n"])
def test_the_step_returns_at_once_when_its_output_is_a_pipe(tmp_path: Path, bootstrap: str) -> None:
    """A platform that reads the step's output to its end must not wait for the runner, nor for its start failure."""
    env, paths = _pod(tmp_path)
    _entry(tmp_path, env, f'touch "$HYPERLOOM_OUT_DIR/started"\n{bootstrap}')
    step = subprocess.Popen(
        ["/bin/sh", "-c", runner_template_step(**paths)],  # type: ignore[arg-type]
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    began = time.monotonic()
    try:
        step.communicate(timeout=10)
        assert step.returncode == 0 and time.monotonic() - began < 5.0
        assert _wait_for(tmp_path / "out" / "started"), (tmp_path / "work" / "runner-boot.log").read_text()
    finally:
        try:
            os.killpg(step.pid, signal.SIGKILL)  # the runner the step left behind, and its sleep
        except ProcessLookupError:
            # The group already exited on its own, so nothing is left to kill.
            pass
        step.communicate()


def test_a_missing_cache_entry_ends_the_test_with_a_start_failure(tmp_path: Path) -> None:
    env, paths = _pod(tmp_path)
    assert _run(runner_template_step(**paths), env) == 0  # type: ignore[arg-type]
    status = _status(tmp_path)
    assert status["error_class"] == "runner_start_failed" and status["state"] == "failed"
    assert "step verify" in status["message"]


def test_a_start_that_ends_without_a_status_leaves_one_with_the_boot_log_tail(tmp_path: Path) -> None:
    env, paths = _pod(tmp_path)
    _entry(tmp_path, env, 'echo "no \\"python3\\" here" >&2\nexit 3\n')
    assert _run(runner_template_step(**paths), env) == 0  # type: ignore[arg-type]
    status = _status(tmp_path)
    assert status["error_class"] == "runner_start_failed" and status["state"] == "failed"
    assert status["message"].startswith("runner start, exit 3 before status.json was written: ")
    assert status["message"].endswith(" no python3 here ")


def test_an_empty_out_dir_variable_falls_back_to_the_templates_out_dir(tmp_path: Path) -> None:
    env, paths = _pod(tmp_path)
    _entry(tmp_path, env, "exit 4\n")
    env["HYPERLOOM_OUT_DIR"] = ""
    assert _run(runner_template_step(**paths), env) == 0  # type: ignore[arg-type]
    assert "exit 4" in _status(tmp_path)["message"]


def test_a_failing_preparation_leaves_a_start_failure_and_fails_the_step(tmp_path: Path) -> None:
    env, paths = _pod(tmp_path)
    assert _run(runner_template_step(pre="false", **paths), env) == 1  # type: ignore[arg-type]
    status = _status(tmp_path)
    assert status["error_class"] == "runner_start_failed" and "exit 1" in status["message"]
    assert not (tmp_path / "out" / "started").exists()


def test_a_runner_status_is_never_overwritten(tmp_path: Path) -> None:
    env, paths = _pod(tmp_path)
    _entry(
        tmp_path,
        env,
        'printf "{\\"state\\": \\"failed\\", \\"error_class\\": \\"x\\"}" >"$HYPERLOOM_OUT_DIR/status.json"\nexit 2\n',
    )
    assert _run(runner_template_step(**paths), env) == 0  # type: ignore[arg-type]
    assert _status(tmp_path) == {"state": "failed", "error_class": "x"}
    time.sleep(0.5)
    assert _status(tmp_path) == {"state": "failed", "error_class": "x"}
