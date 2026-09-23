# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The documented launch recipes must not let ``.env`` override the caller."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]

# In-package docs ship in the wheel; examples/ only exists in a source checkout.
RECIPE_DOCS = (
    PKG_ROOT / "SKILL.md",
    PKG_ROOT / "references" / "operations.md",
    REPO_ROOT / "examples" / "hyperloom-custom-advanced" / "SKILL.md",
    REPO_ROOT / "examples" / "hyperloom-qwen3-8b-3h" / "SKILL.md",
    REPO_ROOT / "examples" / "hyperloom-qwen3-14b-fp8-12h" / "SKILL.md",
    REPO_ROOT / "examples" / "hyperloom-qwen3-14b-fp8-12h-forge" / "SKILL.md",
    REPO_ROOT / "examples" / "hyperloom-qwen3-14b-fp8-12h-atom" / "SKILL.md",
)

# Loads only credential vars, so the path-variable assertions above do not apply, but .env must still not outrank a
# credential the caller exported.
CREDENTIAL_ONLY_DOC = REPO_ROOT / "docs" / "how-to" / "optimize-custom-workload.md"

_FENCE = re.compile(r"^```(?:bash|sh)\s*$")
_FENCE_END = re.compile(r"^```\s*$")

# Lines belonging to the dotenv-load preamble.
_LOAD_LINE = (
    re.compile(r"^\s*$"),
    re.compile(r"^\s*#"),
    re.compile(r"^\s*cd\s"),
    re.compile(r"^\s*export\s+REPO_ROOT="),
    re.compile(r"/\.env"),
    re.compile(r"^\s*set\s+[-+]a\s*$"),
    re.compile(r"_dotenv_prev"),
)


def _bash_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if current is None:
            if _FENCE.match(line):
                current = []
            continue
        if _FENCE_END.match(line):
            blocks.append(current)
            current = None
            continue
        current.append(line)
    return blocks


def _extract_dotenv_load(doc: Path) -> str:
    """Return the leading dotenv-loading fragment of the doc's launch recipe."""
    for block in _bash_blocks(doc.read_text(encoding="utf-8")):
        if not any("/.env" in line for line in block):
            continue
        kept: list[str] = []
        for line in block:
            if not any(pattern.search(line) for pattern in _LOAD_LINE):
                break
            kept.append(line)
        fragment = "\n".join(kept)
        if "/.env" in fragment:
            return fragment
    raise AssertionError(f"no dotenv-loading bash block found in {doc}")


def _run_recipe(
    fragment: str,
    tmp_path: Path,
    exported: dict[str, str],
    *,
    dotenv_extra: str = "",
    observed: tuple[str, ...] = ("USER_DATA_PATH", "OPENAI_API_KEY", "ONLY_IN_DOTENV"),
) -> dict[str, str]:
    """Run the extracted fragment against a conflicting .env and report the result."""
    (tmp_path / ".env").write_text(
        "USER_DATA_PATH=/from/dotenv\nOPENAI_API_KEY=key-from-dotenv\nONLY_IN_DOTENV=filled\n" + dotenv_extra,
        encoding="utf-8",
    )
    script = tmp_path / "recipe.sh"
    values = " ".join(f'"${{{key}:-}}"' for key in observed)
    script.write_text(fragment + f'\nprintf "%s\\n" {values}\n', encoding="utf-8")

    # Minimal env: the developer shell leaks real credentials into pytest.
    run_env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "REPO_ROOT": str(tmp_path)}
    run_env.update(exported)
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=run_env,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return dict(zip(observed, proc.stdout.splitlines()[-len(observed) :]))


ATOM_DOC = REPO_ROOT / "examples" / "hyperloom-qwen3-14b-fp8-12h-atom" / "SKILL.md"


def _atom_runtime_load() -> str:
    blocks = _bash_blocks(ATOM_DOC.read_text(encoding="utf-8"))
    return "\n".join(
        next(
            block
            for block in blocks
            if any(line.startswith((". ", "source ")) and "kernel-agent.env.sh" in line for line in block)
            and any('PYTHON="$_atom_python"' in line and 'USER_DATA_PATH="$_atom_user_data"' in line for line in block)
        )
    )


def _atom_section_blocks(heading: str) -> list[list[str]]:
    section = re.search(
        rf"^### {re.escape(heading)}\n" + r"(.*?)(?=^#{2,3} |\Z)",
        ATOM_DOC.read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    assert section is not None, f"missing ATOM section: {heading}"
    return _bash_blocks(section.group(1))


def test_atom_run_mode_requires_user_choice() -> None:
    text = ATOM_DOC.read_text(encoding="utf-8")
    run_mode = " ".join(text.split("## Run Mode\n", 1)[1].split("### Execution shell", 1)[0].split())
    assert "Run Mode Resolution" in run_mode
    assert "ask the user to choose" in run_mode
    assert "Do not default to either mode" in run_mode
    assert "or an unset/empty mode" not in text
    readme = (REPO_ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    readme_entry = readme.split("- [`12h atom`]", 1)[1].split("- [`", 1)[0]
    assert "Docker as the default" not in readme_entry


@pytest.mark.parametrize("mode", ["baremetal", "docker"])
def test_atom_first_launch_runs_in_selected_context(tmp_path: Path, mode: str) -> None:
    """Execute the published mode, runtime and launch blocks without launching a GPU process."""
    mode_blocks = _atom_section_blocks("Baremetal" if mode == "baremetal" else "Docker container")
    launch = "\n".join(_atom_section_blocks("First launch")[0])
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    data = workspace / "user data"
    runtime = data / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "kernel-agent.env.sh").write_text(
        "export FRAMEWORK=vllm KERNEL_OPT_BACKEND_ORDER=stale PYTHON=/stale/python USER_DATA_PATH=/stale/data\n"
        "export RUNTIME_SENTINEL=loaded\n",
        encoding="utf-8",
    )
    model = workspace / "model files"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    selected = workspace / "selected python"
    run_log, launch_info = workspace / "run output.log", workspace / "launch info.json"
    observed = (
        "FRAMEWORK",
        "KERNEL_OPT_BACKEND_ORDER",
        "PYTHON",
        "USER_DATA_PATH",
        "RUNTIME_SENTINEL",
        "ATOM_CONTEXT",
        "CLAW_SESSION_ID",
        "HYPERLOOM_RUN_MODE",
    )
    probe = (
        "import json, os, sys; "
        f"json.dump(dict(argv=sys.argv[1:], env={{k:os.environ.get(k) for k in {observed!r}}}, "
        "cwd=os.getcwd(), stdin=sys.stdin.read()), open('launch.json', 'w')); "
        "print('optimizer stdout'); print('optimizer stderr', file=sys.stderr)"
    )
    selected.write_text(
        f'#!/usr/bin/env bash\nexec {shlex.quote(Path(sys.executable).as_posix())} -c {shlex.quote(probe)} "$@"\n',
        encoding="utf-8",
    )
    selected.chmod(0o755)
    exported = dict(
        USER_DATA_PATH=data.as_posix(),
        MODEL_PATH=model.as_posix(),
        KERNEL_OPT_BACKEND_ORDER="geak",
        PYTHON="/host-only/python",
        PATH="/host-only/bin:/usr/bin:/bin",
        VIRTUAL_ENV="/host-only/venv",
        REAL_PYTHON=Path(sys.executable).as_posix(),
        ATOM_CONTEXT="baremetal",
        CLAW_SESSION_ID="test-harness-session",
        HYPERLOOM_RUN_MODE=mode,
    )
    docker = """
docker() {
  MSYS2_ARG_CONV_EXCL='*' "$REAL_PYTHON" -c 'import json,sys; print(json.dumps(sys.argv[1:]), file=open("docker.jsonl", "a"))' "$@"
  [ "$1" != run ] || return 0
  [ "$1" = exec ] || return 90
  shift
  local workdir
  local -a forwarded=(PATH=/usr/bin:/bin "HOME=$HOME" ATOM_CONTEXT=docker)
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -i) shift ;;
      -w) workdir="$2"; shift 2 ;;
      -e|--env) forwarded+=("$2=${!2}"); shift 2 ;;
      *) break ;;
    esac
  done
  [ "$#" = 2 ] && [ "$1" = hyperloom-local ] && [ "$2" = bash ] || return 91
  (cd "$workdir" && env -i "${forwarded[@]}" bash)
}
"""
    payload = "set -e\n"
    if mode == "docker":
        payload += '[ -z "${PYTHON:-}" ]\n[ -z "${VIRTUAL_ENV:-}" ]\n[[ "$PATH" != /host-only/bin:* ]]\n'
    payload += _extract_dotenv_load(ATOM_DOC) + "\n"
    for key, value in {"PYTHON": selected, "RUN_LOG": run_log, "LAUNCH_INFO_FILE": launch_info}.items():
        payload += f"export {key}={shlex.quote(value.as_posix())}\n"
    payload += _atom_runtime_load() + "\n" + launch + "\n: > launch-finished\n"
    mode_script = "\n".join("\n".join(block) for block in mode_blocks)
    fragment = docker + mode_script
    fragment += (" <<'ATOM_LAUNCH'\n" + payload + "ATOM_LAUNCH\n") if mode == "docker" else "\n" + payload
    result = _run_recipe(
        fragment,
        workspace,
        exported,
        dotenv_extra=f"FRAMEWORK=sglang\nKERNEL_OPT_BACKEND_ORDER=forge\nHYPERLOOM_RUN_MODE={'docker' if mode == 'baremetal' else 'baremetal'}\n",
        observed=("REPO_ROOT", "HYPERLOOM_RUN_MODE"),
    )
    record = json.loads((workspace / "launch.json").read_text(encoding="utf-8"))
    argv = record["argv"]
    assert argv[:4] == ["-m", "hyperloom.inference_optimizer.cli", "--verbose", "optimize"]
    expected_flags = {
        "--model": model.as_posix(),
        "--framework": "atom",
        "--tp": "1",
        "--conc": "64",
        "--isl": "1024",
        "--osl": "1024",
        "--precision": "fp8",
        "--target-gain": "50",
        "--max-hours": "12",
        "--max-minutes-framework-pct": "0.43",
        "--max-minutes-kernel-pct": "0.42",
        "--launch-info-file": launch_info.as_posix(),
    }
    assert len(argv[4:]) == 2 * len(expected_flags)
    assert dict(zip(argv[4::2], argv[5::2])) == expected_flags
    assert record["env"] == dict(
        zip(
            observed,
            ("atom", "geak", selected.as_posix(), data.as_posix(), "loaded", mode, "test-harness-session", mode),
        )
    )
    assert Path(record["cwd"]) == workspace
    assert record["stdin"] == ""
    assert sorted(run_log.read_text(encoding="utf-8").splitlines()) == ["optimizer stderr", "optimizer stdout"]
    assert (workspace / "launch-finished").exists()
    if mode == "baremetal":
        assert result["HYPERLOOM_RUN_MODE"] == "baremetal"
        assert not (workspace / "docker.jsonl").exists()
    else:
        creation, entry = [json.loads(line) for line in (workspace / "docker.jsonl").read_text().splitlines()]
        assert creation[:2] == ["run", "-d"]
        assert "docker.io/rocm/atom-dev:v0.1.7-rc0" in creation
        for pair in (
            ["--device", "/dev/kfd"],
            ["--device", "/dev/dri"],
            ["-v", f"{result['REPO_ROOT']}:{result['REPO_ROOT']}"],
        ):
            assert any(creation[i : i + 2] == pair for i in range(len(creation) - 1))
        assert entry[:4] == ["exec", "-i", "-w", result["REPO_ROOT"]]
        assert entry[-2:] == ["hyperloom-local", "bash"]


@pytest.mark.parametrize("missing", ["MODEL_PATH", "PYTHON", "RUN_LOG", "LAUNCH_INFO_FILE"])
def test_atom_first_launch_requires_prepared_paths(tmp_path: Path, missing: str) -> None:
    launch = "\n".join(_atom_section_blocks("First launch")[0])
    exported = {
        "MODEL_PATH": (tmp_path / "model files").as_posix(),
        "PYTHON": "launch_probe",
        "RUN_LOG": (tmp_path / "run output.log").as_posix(),
        "LAUNCH_INFO_FILE": (tmp_path / "launch info.json").as_posix(),
    }
    del exported[missing]
    fragment = "launch_probe() { : > launch-reached; }\n" + launch
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_recipe(fragment, tmp_path, exported)
    assert missing in exc.value.stderr
    assert not (tmp_path / "launch-reached").exists()


def test_atom_recipe_stops_on_missing_runtime_env(tmp_path: Path) -> None:
    """A missing source must fail, not be hidden by the recipe's final printf."""
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_recipe(
            _atom_runtime_load(),
            tmp_path,
            {"USER_DATA_PATH": tmp_path.as_posix(), "PYTHON": "/selected/bin/python3"},
        )
    assert "kernel-agent.env.sh" in exc.value.stderr


def test_atom_runtime_install_accepts_readonly_user_data_path(tmp_path: Path) -> None:
    """The install invocation exports the fixed platform path without assigning it."""
    blocks = _bash_blocks(ATOM_DOC.read_text(encoding="utf-8"))
    install = "\n".join(next(b for b in blocks if 'bash "$INSTALL_SH"' in b))
    fragment = "readonly USER_DATA_PATH\nbash() { INSTALLER_REACHED=yes; }\n" + install
    result = _run_recipe(
        fragment,
        tmp_path,
        {"USER_DATA_PATH": tmp_path.as_posix(), "PYTHON": "/selected/bin/python3"},
        observed=("USER_DATA_PATH", "INSTALLER_REACHED"),
    )
    assert result == {"USER_DATA_PATH": tmp_path.as_posix(), "INSTALLER_REACHED": "yes"}


def test_atom_recipe_loads_runtime_below_readonly_parent(tmp_path: Path) -> None:
    """Run the documented child shell while preserving the platform isolation root."""
    text = ATOM_DOC.read_text(encoding="utf-8")
    shell = next(("\n".join(b) for b in _bash_blocks(text) if "bash --noprofile --norc" in b), None)
    assert shell is not None, "a readonly parent needs the documented child-shell boundary before sourcing env files"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "kernel-agent.env.sh").write_text(
        f"export USER_DATA_PATH={shlex.quote(tmp_path.as_posix())}\nexport RUNTIME_SENTINEL=loaded\n",
        encoding="utf-8",
    )
    child = "set -e\n" + _extract_dotenv_load(ATOM_DOC) + "\n" + _atom_runtime_load()
    child += '\n[ "$RUNTIME_SENTINEL" = loaded ]\n[ "$USER_DATA_PATH" = "$EXPECTED_DATA_PATH" ]\n'
    fragment = "set -e\nreadonly USER_DATA_PATH\n" + shell + " <<'ATOM_RECIPE'\n" + child + "ATOM_RECIPE\n"
    fragment += '[[ "$(declare -p USER_DATA_PATH)" == "declare -rx "* ]]\n'
    result = _run_recipe(
        fragment,
        tmp_path,
        {
            "USER_DATA_PATH": tmp_path.as_posix(),
            "EXPECTED_DATA_PATH": tmp_path.as_posix(),
            "PYTHON": "/selected/bin/python3",
        },
        observed=("USER_DATA_PATH",),
    )
    assert result["USER_DATA_PATH"] == tmp_path.as_posix()


def _write_atom_probe_modules(tmp_path: Path, help_exit: int = 0) -> None:
    for name in ("atom", "atom/entrypoints"):
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "atom/entrypoints/openai_server.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        'assert sys.argv[1:] == ["--help"]\nPath("help-called").touch()\n'
        f"raise SystemExit({help_exit})\n",
        encoding="utf-8",
    )
    (tmp_path / "torch.py").write_text(
        'from types import SimpleNamespace\nversion = SimpleNamespace(hip="test-rocm")\n__version__ = "test"\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize("help_exit", [0, 7], ids=["help-succeeds", "help-fails"])
def test_atom_recipe_requires_working_server_help(tmp_path: Path, help_exit: int) -> None:
    """Import success must not hide a broken framework CLI before setup starts."""
    _write_atom_probe_modules(tmp_path, help_exit)
    blocks = _bash_blocks(ATOM_DOC.read_text(encoding="utf-8"))
    selected_python = "\n".join(next(b for b in blocks if 'PYTHON="${PYTHON:-$(command -v python3)}"' in b))
    fragment = 'python3() { "$REAL_PYTHON" "$@"; }\n' + selected_python + "\n: > setup-reached\n"
    exported = {"PYTHON": Path(sys.executable).as_posix(), "REAL_PYTHON": Path(sys.executable).as_posix()}
    if help_exit:
        with pytest.raises(subprocess.CalledProcessError) as exc:
            _run_recipe(fragment, tmp_path, exported)
        assert exc.value.returncode == help_exit
        assert not (tmp_path / "setup-reached").exists()
    else:
        _run_recipe(fragment, tmp_path, exported)
        assert (tmp_path / "setup-reached").exists()
    assert (tmp_path / "help-called").exists()


@pytest.mark.parametrize(
    "case",
    ["path-cli", "explicit-cli", "bundle-only", "non-claude", "non-forge", "failed-cli", "timeout", "wrong-cli"],
)
def test_atom_recipe_checks_forge_cli_before_launch(tmp_path: Path, monkeypatch, case: str) -> None:
    """Use the real provider/CLI resolvers without starting an agent or calling its API."""
    from dataclasses import replace

    from kernelforge.agent_backends import claude, registry

    blocks = _bash_blocks(ATOM_DOC.read_text(encoding="utf-8"))
    block = next((b for b in blocks if any("resolve_claude_cli" in line for line in b)), None)
    assert block is not None, "the example must check the actual Forge executable before optimize"
    start, end = block.index("\"$PYTHON\" - <<'PY'") + 1, block.index("PY")
    source = "\n".join(block[start:end])
    monkeypatch.setattr(
        "os.environ",
        {
            "ANTHROPIC_API_KEY": "test-only",
            "KNOWLEDGE_STORE_MODE": "local",
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
        },
    )
    monkeypatch.setattr(registry, "_plugins_loaded", True)
    monkeypatch.setattr(
        registry,
        "_providers",
        {
            name: replace(
                provider, availability=lambda: True, credentialed=lambda _env, selected=name: selected == "claude"
            )
            for name, provider in registry._providers.items()
            if name in {"claude", "codex"}
        },
    )
    # Credential ranking should choose Claude without pinning FORGE_AGENT_BACKEND.
    if case == "non-claude":
        monkeypatch.setenv("FORGE_AGENT_BACKEND", "codex")
    if case == "non-forge":
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "geak")
    cli = (tmp_path / "cli tools" / "claude").as_posix()
    if case == "explicit-cli":
        monkeypatch.setenv("FORGE_AGENT_CLI", cli)
    bundle = tmp_path / "claude_agent_sdk" / "_bundled" / "claude"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("bundled CLI is not a resolver candidate", encoding="utf-8")
    monkeypatch.setattr(claude.shutil, "which", lambda name: cli if case == "path-cli" and name == "claude" else None)
    monkeypatch.setattr(claude.os.path, "isfile", lambda _path: False)
    calls = []

    def version_check(argv, **kwargs):
        calls.append(argv)
        assert argv[1:] == ["--version"]
        assert kwargs["timeout"] == 10
        if case == "bundle-only":
            raise FileNotFoundError("claude is absent")
        if case == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if case == "failed-cli":
            raise subprocess.CalledProcessError(7, argv)
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(argv, 0, "other tool" if case == "wrong-cli" else "Claude Code test", "")

    monkeypatch.setattr(subprocess, "run", version_check)
    if case in {"bundle-only", "failed-cli", "timeout", "wrong-cli"}:
        with pytest.raises(SystemExit, match="Forge Claude CLI"):
            exec(compile(source, str(ATOM_DOC), "exec"), {})
    else:
        exec(compile(source, str(ATOM_DOC), "exec"), {})
    if case in {"non-claude", "non-forge"}:
        assert calls == []
    else:
        assert len(calls) == 1
        assert calls[0][0] == (cli if case in {"path-cli", "explicit-cli"} else "claude")
    if case == "explicit-cli":
        assert claude.os.environ["FORGE_AGENT_CLI"] == cli


@pytest.mark.parametrize("provider", ("claude", "codex"))
def test_atom_forge_cli_check_controls_shell_launch(tmp_path: Path, provider: str) -> None:
    """A failing local version check must stop the documented shell before launch."""
    blocks = _bash_blocks(ATOM_DOC.read_text(encoding="utf-8"))
    check = "\n".join(next(b for b in blocks if any("resolve_claude_cli" in line for line in b)))
    exported = {
        "PYTHON": Path(sys.executable).as_posix(),
        "PYTHONPATH": (REPO_ROOT / "src").as_posix(),
        "FORGE_AGENT_BACKEND": provider,
        "FORGE_AGENT_CLI": Path(sys.executable).as_posix(),
        "USERPROFILE": str(tmp_path),
    }
    fragment = check + "\n: > launch-reached\n"
    if provider == "claude":
        with pytest.raises(subprocess.CalledProcessError) as exc:
            _run_recipe(fragment, tmp_path, exported)
        assert "Forge Claude CLI returned an unexpected version" in exc.value.stderr
        assert not (tmp_path / "launch-reached").exists()
    else:
        _run_recipe(fragment, tmp_path, exported)
        assert (tmp_path / "launch-reached").exists()


@pytest.mark.parametrize("source", ["dotenv", "runtime"])
@pytest.mark.parametrize("same_prefix", [False, True], ids=["conflicting-prefix", "same-prefix"])
def test_atom_recipe_checks_vllm_venv_root(tmp_path: Path, source: str, same_prefix: bool) -> None:
    """Old vLLM settings must not redirect ATOM's server to a different environment."""
    blocks = _bash_blocks(ATOM_DOC.read_text(encoding="utf-8"))
    selected_python = "\n".join(next(b for b in blocks if 'PYTHON="${PYTHON:-$(command -v python3)}"' in b))
    _write_atom_probe_modules(tmp_path)
    vllm_root = (Path(sys.prefix) / ".").as_posix() if same_prefix else (tmp_path / "other-venv").as_posix()
    setting = f"export VLLM_VENV_ROOT={shlex.quote(vllm_root)}\n"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "kernel-agent.env.sh").write_text(setting if source == "runtime" else "true\n", encoding="utf-8")
    fragment = 'python3() { "$REAL_PYTHON" "$@"; }\n' + _extract_dotenv_load(ATOM_DOC)
    fragment += "\n" + _atom_runtime_load() + "\n" + selected_python
    exported = {
        "USER_DATA_PATH": tmp_path.as_posix(),
        "PYTHON": Path(sys.executable).as_posix(),
        "REAL_PYTHON": Path(sys.executable).as_posix(),
    }
    kwargs = {"dotenv_extra": setting if source == "dotenv" else "", "observed": ("VLLM_VENV_ROOT",)}
    if same_prefix:
        assert _run_recipe(fragment, tmp_path, exported, **kwargs)["VLLM_VENV_ROOT"] == vllm_root
    else:
        with pytest.raises(subprocess.CalledProcessError) as exc:
            _run_recipe(fragment, tmp_path, exported, **kwargs)
        assert "VLLM_VENV_ROOT" in exc.value.stderr
        assert vllm_root in exc.value.stderr


@pytest.mark.parametrize(
    ("shell_backend", "dotenv_backend", "expected"),
    [("geak", "forge", "geak"), (None, "forge", "forge"), (None, "geak", "geak"), (None, None, "")],
    ids=["caller-wins", "dotenv-forge", "dotenv-geak", "cli-default"],
)
def test_atom_recipe_preserves_backend_selection(
    tmp_path: Path, shell_backend: str | None, dotenv_backend: str | None, expected: str
) -> None:
    """The final launch exports must not discard an explicit backend choice."""
    fragment = "set -e\n" + _extract_dotenv_load(ATOM_DOC) + "\n" + _atom_runtime_load()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "kernel-agent.env.sh").write_text(
        "export FRAMEWORK=vllm KERNEL_OPT_BACKEND_ORDER=stale PYTHON=/stale/python USER_DATA_PATH=/stale/data\n"
        "export RUNTIME_SENTINEL=loaded\n",
        encoding="utf-8",
    )
    exported = {"USER_DATA_PATH": tmp_path.as_posix(), "PYTHON": "/selected/bin/python3"}
    if shell_backend is not None:
        exported["KERNEL_OPT_BACKEND_ORDER"] = shell_backend
    dotenv = f"KERNEL_OPT_BACKEND_ORDER={dotenv_backend}\n" if dotenv_backend is not None else ""

    result = _run_recipe(
        fragment,
        tmp_path,
        exported,
        dotenv_extra=dotenv + "FRAMEWORK=sglang\n",
        observed=("FRAMEWORK", "KERNEL_OPT_BACKEND_ORDER", "PYTHON", "USER_DATA_PATH", "RUNTIME_SENTINEL"),
    )

    assert result == {
        "FRAMEWORK": "atom",
        "KERNEL_OPT_BACKEND_ORDER": expected,
        "PYTHON": "/selected/bin/python3",
        "USER_DATA_PATH": tmp_path.as_posix(),
        "RUNTIME_SENTINEL": "loaded",
    }


@pytest.mark.parametrize(
    "doc",
    [ATOM_DOC, REPO_ROOT / "src" / "hyperloom" / "skills" / "hyperloom-setup" / "SKILL.md"],
    ids=["atom-demo", "setup-skill"],
)
def test_atom_recipe_provides_direct_setup_commands(tmp_path: Path, doc: Path) -> None:
    """Execute both documented direct setup invocations without installing anything."""
    blocks = _bash_blocks(doc.read_text(encoding="utf-8"))
    setup_blocks = [
        "\n".join(block)
        for block in blocks
        if any("hyperloom.inference_optimizer.setup" in line for line in block)
        and not any("docker" in line for line in block)
        and (doc == ATOM_DOC or any("--frameworks atom" in line for line in block))
    ]
    assert len(setup_blocks) == 2, "direct mode needs a check-only command and a separate approved setup command"
    fragment = """
setup_args=()
python_probe() { setup_args+=("$*"); }
export PYTHON=python_probe
""" + "\n".join(setup_blocks)
    fragment += '\nCHECK_ARGS="${setup_args[0]}"\nINSTALL_ARGS="${setup_args[1]}"'
    result = _run_recipe(
        fragment, tmp_path, {"USER_DATA_PATH": "/selected/data"}, observed=("CHECK_ARGS", "INSTALL_ARGS")
    )

    for key, args in result.items():
        assert "-m hyperloom.inference_optimizer.setup" in args
        assert "--install-framework none" in args
        assert "--frameworks atom" in args
        assert "--require-frameworks" in args
        assert "--user-data-path /selected/data" in args
        assert ("--check-only" in args) == (key == "CHECK_ARGS")


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
def test_recipe_keeps_caller_user_data_path(doc: Path, tmp_path: Path) -> None:
    """A USER_DATA_PATH in .env must not overwrite the one the caller exported."""
    if not doc.exists():
        pytest.skip(f"{doc} not present in this layout")

    result = _run_recipe(
        _extract_dotenv_load(doc),
        tmp_path,
        {"USER_DATA_PATH": "/from/caller", "OPENAI_API_KEY": "key-from-caller"},
    )

    assert result["USER_DATA_PATH"] == "/from/caller"
    assert result["OPENAI_API_KEY"] == "key-from-caller"


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
def test_recipe_still_fills_missing_values(doc: Path, tmp_path: Path) -> None:
    """Protecting exported values must not stop .env from filling the gaps."""
    if not doc.exists():
        pytest.skip(f"{doc} not present in this layout")

    result = _run_recipe(_extract_dotenv_load(doc), tmp_path, {})

    assert result["USER_DATA_PATH"] == "/from/dotenv"
    assert result["OPENAI_API_KEY"] == "key-from-dotenv"
    assert result["ONLY_IN_DOTENV"] == "filled"


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
def test_recipe_lets_dotenv_fill_a_blank_export(doc: Path, tmp_path: Path) -> None:
    """An exported-but-empty value is a gap, matching install.sh's restore rule."""
    if not doc.exists():
        pytest.skip(f"{doc} not present in this layout")

    result = _run_recipe(_extract_dotenv_load(doc), tmp_path, {"USER_DATA_PATH": ""})

    assert result["USER_DATA_PATH"] == "/from/dotenv"


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
def test_a_failed_restore_names_the_variable(doc: Path, tmp_path: Path) -> None:
    """Suppressing eval's stderr hid the only precise diagnosis available."""
    if not doc.exists():
        pytest.skip(f"{doc} not present in this layout")

    fragment = _extract_dotenv_load(doc)
    (tmp_path / ".env").write_text("USER_DATA_PATH=/from/dotenv\n", encoding="utf-8")
    script = tmp_path / "recipe.sh"
    script.write_text("readonly LOCKED=locked\nexport LOCKED\n" + fragment + "\n", encoding="utf-8")

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "REPO_ROOT": str(tmp_path)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert "LOCKED" in proc.stderr, proc.stderr
    assert "readonly" in proc.stderr, proc.stderr


def test_credential_only_recipe_keeps_the_callers_key(tmp_path: Path) -> None:
    """install.sh snapshots the same credential vars for this exact reason."""
    if not CREDENTIAL_ONLY_DOC.exists():
        pytest.skip(f"{CREDENTIAL_ONLY_DOC} not present in this layout")

    fragment = _extract_dotenv_load(CREDENTIAL_ONLY_DOC)

    kept = _run_recipe(fragment, tmp_path, {"OPENAI_API_KEY": "key-from-caller"})
    assert kept["OPENAI_API_KEY"] == "key-from-caller"

    filled = _run_recipe(fragment, tmp_path, {})
    assert filled["OPENAI_API_KEY"] == "key-from-dotenv"
