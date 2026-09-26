# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Published recipes preserve caller choices and delegate environment checks to owners.

Import/help/ROCm and invalid-pin failures are exercised by test_setup_selected_python
and test_preflight_serving_framework; framework-aware PATH by test_derive_runtime_paths;
missing runtime and in-process precedence by test_preflight_auth_override; offline
Forge executable failures by kernelforge/tests/test_claude_cli_resolve.py.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
ATOM_DOC = REPO_ROOT / "examples" / "hyperloom-qwen3-14b-fp8-12h-atom" / "SKILL.md"
SETUP_DOC = REPO_ROOT / "src" / "hyperloom" / "skills" / "hyperloom-setup" / "SKILL.md"

# In-package docs ship in the wheel; examples/ only exists in a source checkout.
RECIPE_DOCS = (
    PKG_ROOT / "SKILL.md",
    PKG_ROOT / "references" / "operations.md",
    REPO_ROOT / "examples" / "hyperloom-custom-advanced" / "SKILL.md",
    REPO_ROOT / "examples" / "hyperloom-qwen3-8b-3h" / "SKILL.md",
    REPO_ROOT / "examples" / "hyperloom-qwen3-14b-fp8-12h" / "SKILL.md",
    REPO_ROOT / "examples" / "hyperloom-qwen3-14b-fp8-12h-forge" / "SKILL.md",
    ATOM_DOC,
)
CREDENTIAL_ONLY_DOC = REPO_ROOT / "docs" / "how-to" / "optimize-custom-workload.md"

_FENCE = re.compile(r"^```(?:bash|sh)\s*$")
_FENCE_END = re.compile(r"^```\s*$")


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


def _dotenv_loads(doc: Path) -> list[str]:
    """Extract real asset discovery and load, stopping before workload steps."""
    fragments = [
        "\n".join(block[: block.index("load_dotenv_no_clobber") + 1])
        for block in _bash_blocks(doc.read_text(encoding="utf-8"))
        if any("runtime_env.sh" in line for line in block) and "load_dotenv_no_clobber" in block
    ]
    assert fragments, f"no shared dotenv-loading bash block found in {doc}"
    return fragments


def _run_recipe(
    fragment: str,
    tmp_path: Path,
    exported: dict[str, str],
    *,
    dotenv_extra: str = "",
    observed: tuple[str, ...] = ("USER_DATA_PATH", "OPENAI_API_KEY", "ONLY_IN_DOTENV"),
    layout: str = "src",
) -> dict[str, str]:
    """Run the shipped asset via documented wheel/source discovery, without installing."""
    assets = tmp_path / layout / "hyperloom" / "inference_optimizer" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "install.sh").touch()
    (assets / "runtime_env.sh").write_bytes((PKG_ROOT / "assets" / "runtime_env.sh").read_bytes())
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
        timeout=30,
    )
    return dict(zip(observed, proc.stdout.splitlines()[-len(observed) :]))


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
    """Execute mode/selection/launch blocks up to the optimizer process boundary."""
    mode_blocks = _atom_section_blocks("Baremetal" if mode == "baremetal" else "Docker container")
    launch = "\n".join(_atom_section_blocks("First launch")[0])
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    data = workspace / "user data"
    runtime = data / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "kernel-agent.env.sh").write_text(
        "export FRAMEWORK=vllm\nexport KERNEL_OPT_BACKEND_ORDER=stale\nexport PYTHON=/stale/python\n"
        "export USER_DATA_PATH=/stale/data\nexport HYPERLOOM_KERNEL_AGENT_ROOT=/installed/kernel\n"
        "export MAGPIE_PATH=/installed/Magpie\n",
        encoding="utf-8",
    )
    model = workspace / "model files"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    selected = workspace / "selected python" / "python3"
    selected.parent.mkdir()
    run_log, launch_info = workspace / "run output.log", workspace / "launch info.json"
    observed = (
        "FRAMEWORK",
        "KERNEL_OPT_BACKEND_ORDER",
        "PYTHON",
        "USER_DATA_PATH",
        "MAGPIE_PATH",
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
    selected_bin = selected.parent.as_posix()
    if os.name == "nt":
        selected_bin = f"/{selected_bin[0].lower()}{selected_bin[2:]}"
    exported = dict(
        USER_DATA_PATH=data.as_posix(),
        MODEL_PATH=model.as_posix(),
        KERNEL_OPT_BACKEND_ORDER="geak",
        PYTHON=selected.as_posix() if mode == "baremetal" else "/host-only/python",
        PATH="/host-only/bin:/usr/bin:/bin",
        VIRTUAL_ENV="/host-only/venv",
        INFERENCE_OPTIMIZER_FORCE_PYTHON="1",
        REAL_PYTHON=Path(sys.executable).as_posix(),
        ATOM_CONTEXT="baremetal",
        CLAW_SESSION_ID="test-harness-session",
        HYPERLOOM_RUN_MODE=mode,
    )
    docker = (
        f"IMAGE_PATH={shlex.quote(selected_bin + ':/usr/bin:/bin')}\n"
        + """
docker() {
  MSYS2_ARG_CONV_EXCL='*' "$REAL_PYTHON" -c 'import json,sys; print(json.dumps(sys.argv[1:]), file=open("docker.jsonl", "a"))' "$@"
  [ "$1" != run ] || return 0
  [ "$1" = exec ] || return 90
  shift
  local workdir
  local -a forwarded=("PATH=$IMAGE_PATH" "HOME=$HOME" ATOM_CONTEXT=docker)
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
    )
    payload = "set -e\n"
    if mode == "docker":
        payload += '[ -z "${PYTHON:-}" ]\n[ -z "${VIRTUAL_ENV:-}" ]\n'
        payload += '[ -z "${INFERENCE_OPTIMIZER_FORCE_PYTHON:-}" ]\n[[ "$PATH" != /host-only/bin:* ]]\n'
    payload += "\n".join(_atom_section_blocks("Execution shell")[0]) + "\n"
    payload += "\n".join(_atom_section_blocks("Selected Python and Setup")[0]) + "\n"
    for key, value in {"RUN_LOG": run_log, "LAUNCH_INFO_FILE": launch_info}.items():
        payload += f"export {key}={shlex.quote(value.as_posix())}\n"
    payload += launch + "\n: > launch-finished\n"
    mode_script = "\n".join("\n".join(block) for block in mode_blocks)
    fragment = docker + mode_script
    fragment += (" <<'ATOM_LAUNCH'\n" + payload + "ATOM_LAUNCH\n") if mode == "docker" else "\n" + payload
    result = _run_recipe(
        fragment,
        workspace,
        exported,
        dotenv_extra=f"FRAMEWORK=sglang\nKERNEL_OPT_BACKEND_ORDER=forge\nHYPERLOOM_RUN_MODE={'docker' if mode == 'baremetal' else 'baremetal'}\n"
        "PYTHON=/host-only/dotenv/python\nVIRTUAL_ENV=/host-only/dotenv/venv\nINFERENCE_OPTIMIZER_FORCE_PYTHON=0\n",
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
            (
                "atom",
                "geak",
                selected.as_posix(),
                data.as_posix(),
                None,
                mode,
                "test-harness-session",
                mode,
            ),
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


def test_atom_runtime_install_accepts_readonly_user_data_path(tmp_path: Path) -> None:
    """The install invocation exports the fixed platform path without assigning it."""
    blocks = _bash_blocks(ATOM_DOC.read_text(encoding="utf-8"))
    install = "\n".join(next(b for b in blocks if 'bash "$INSTALL_SH"' in b))
    fragment = _dotenv_loads(ATOM_DOC)[0] + "\nreadonly USER_DATA_PATH\nbash() { INSTALLER_REACHED=yes; }\n" + install
    result = _run_recipe(
        fragment,
        tmp_path,
        {"USER_DATA_PATH": tmp_path.as_posix(), "PYTHON": "/selected/bin/python3"},
        observed=("USER_DATA_PATH", "INSTALLER_REACHED"),
    )
    assert result == {"USER_DATA_PATH": tmp_path.as_posix(), "INSTALLER_REACHED": "yes"}


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
@pytest.mark.parametrize("mode", ["docker", "baremetal"])
@pytest.mark.parametrize("caller", ["unset", "empty", "explicit"])
def test_recipe_dotenv_interpreter_settings_follow_source(tmp_path: Path, doc: Path, mode: str, caller: str) -> None:
    """Only Docker excludes interpreter settings originating in the mounted dotenv."""
    host = {
        "PYTHON": "/host-only/venv/bin/python3",
        "VIRTUAL_ENV": "/host-only/venv",
        "INFERENCE_OPTIMIZER_FORCE_PYTHON": "1",
    }
    current = {
        "PYTHON": "/selected env/python 'with quotes'",
        "VIRTUAL_ENV": "/selected env",
        "INFERENCE_OPTIMIZER_FORCE_PYTHON": "0",
    }
    exported = {"HYPERLOOM_RUN_MODE": mode, "USER_DATA_PATH": "/selected/data"}
    if caller != "unset":
        exported.update(current if caller == "explicit" else dict.fromkeys(host, ""))
    dotenv = "".join(f"export {key}={shlex.quote(value)}\n" for key, value in host.items())
    dotenv += f"HYPERLOOM_RUN_MODE={'baremetal' if mode == 'docker' else 'docker'}\nPATH=/host-only/bin\n"
    fragment = _dotenv_loads(doc)[0]
    fragment += '\nPYTHON_SET="${PYTHON+x}"\nVENV_SET="${VIRTUAL_ENV+x}"\n'
    fragment += 'FORCE_SET="${INFERENCE_OPTIMIZER_FORCE_PYTHON+x}"\n'
    result = _run_recipe(
        fragment,
        tmp_path,
        exported,
        dotenv_extra=dotenv,
        observed=(
            *host,
            "PATH",
            "HYPERLOOM_RUN_MODE",
            "USER_DATA_PATH",
            "ONLY_IN_DOTENV",
            "PYTHON_SET",
            "VENV_SET",
            "FORCE_SET",
        ),
    )
    expected = current if caller == "explicit" else dict.fromkeys(host, "") if mode == "docker" else host
    assert {key: result[key] for key in host} == expected
    assert result["PATH"] == "/usr/bin:/bin"
    assert result["HYPERLOOM_RUN_MODE"] == mode
    assert result["USER_DATA_PATH"] == "/selected/data"
    assert result["ONLY_IN_DOTENV"] == "filled"
    expected_set = "" if mode == "docker" and caller == "unset" else "x"
    assert [result[key] for key in ("PYTHON_SET", "VENV_SET", "FORCE_SET")] == [expected_set] * 3


@pytest.mark.parametrize("doc", [ATOM_DOC, SETUP_DOC], ids=["atom-demo", "setup-skill"])
@pytest.mark.parametrize("selection", ["activate-before", "activate-after", "default", "valid-pin", "invalid-pin"])
def test_atom_docker_recipe_passes_selected_python_to_setup(tmp_path: Path, doc: Path, selection: str) -> None:
    """Activation never sets PYTHON; the real recipe selects it and invokes setup."""
    bin_dir = tmp_path / "container env" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python3"
    python.write_text(
        '#!/usr/bin/env bash\n[ "$INFERENCE_OPTIMIZER_FORCE_PYTHON" = 1 ] || exit 91\n'
        'printf "%s\\n" "$@" > setup-argv\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    shell_bin = bin_dir.as_posix()
    if os.name == "nt":
        shell_bin = f"/{shell_bin[0].lower()}{shell_bin[2:]}"
    activation = f'export PATH={shlex.quote(shell_bin)}:"$PATH"\n'
    if selection != "default":
        activation += f"export VIRTUAL_ENV={shlex.quote(bin_dir.parent.as_posix())}\n"
    fragment = "set -e\n" + (activation if selection != "activate-after" else "")
    if selection in {"valid-pin", "invalid-pin"}:
        pin = python.as_posix() if selection == "valid-pin" else "/explicit-missing/python3"
        fragment += f"export PYTHON={shlex.quote(pin)} INFERENCE_OPTIMIZER_FORCE_PYTHON=0\n"
    fragment += _dotenv_loads(ATOM_DOC)[0] + "\n"
    if selection == "activate-after":
        fragment += activation
    blocks = _bash_blocks(doc.read_text(encoding="utf-8"))
    selected = next(block for block in blocks if "export INFERENCE_OPTIMIZER_FORCE_PYTHON=1" in block)
    check = next(block for block in blocks if any("setup --check-only" in line for line in block))
    fragment += "\n".join(selected) + "\n"
    if check is not selected:
        fragment += "\n".join(check) + "\n"
    fragment += ": > setup-finished\n"
    kwargs = dict(
        dotenv_extra="HYPERLOOM_RUN_MODE=docker\nPYTHON=/host/python\nVIRTUAL_ENV=/host/venv\n"
        "INFERENCE_OPTIMIZER_FORCE_PYTHON=1\nPATH=/host/bin\n",
        observed=("PYTHON", "VIRTUAL_ENV", "INFERENCE_OPTIMIZER_FORCE_PYTHON"),
    )
    exported = {"HYPERLOOM_RUN_MODE": "docker", "USER_DATA_PATH": "/selected/data"}
    if selection == "invalid-pin":
        with pytest.raises(subprocess.CalledProcessError) as failure:
            _run_recipe(fragment, tmp_path, exported, **kwargs)
        assert failure.value.returncode == 127
        assert "/explicit-missing/python3" in failure.value.stderr
        assert not (tmp_path / "setup-argv").exists()
        assert not (tmp_path / "setup-finished").exists()
    else:
        result = _run_recipe(fragment, tmp_path, exported, **kwargs)
        assert result["PYTHON"] == (python.as_posix() if selection == "valid-pin" else shell_bin + "/python3")
        assert result["VIRTUAL_ENV"] == ("" if selection == "default" else bin_dir.parent.as_posix())
        assert result["INFERENCE_OPTIMIZER_FORCE_PYTHON"] == "1"
        assert (tmp_path / "setup-argv").read_text().splitlines() == [
            "-m",
            "hyperloom.inference_optimizer.setup",
            "--check-only",
            "--",
            "--install-framework",
            "none",
            "--frameworks",
            "atom",
            "--require-frameworks",
            "--user-data-path",
            "/selected/data",
        ]
        assert (tmp_path / "setup-finished").exists()


@pytest.mark.parametrize(
    ("shell_backend", "dotenv_backend", "expected"),
    [("geak", "forge", "geak"), (None, "forge", "forge"), (None, "geak", "geak"), (None, None, "")],
    ids=["caller-wins", "dotenv-forge", "dotenv-geak", "cli-default"],
)
def test_atom_recipe_preserves_backend_selection(
    tmp_path: Path, shell_backend: str | None, dotenv_backend: str | None, expected: str
) -> None:
    """The workload selection must not discard an explicit kernel backend choice."""
    fragment = "\n".join(_atom_section_blocks("Execution shell")[0])
    exported = {"USER_DATA_PATH": tmp_path.as_posix(), "PYTHON": "/selected/bin/python3"}
    if shell_backend is not None:
        exported["KERNEL_OPT_BACKEND_ORDER"] = shell_backend
    dotenv = f"KERNEL_OPT_BACKEND_ORDER={dotenv_backend}\n" if dotenv_backend is not None else ""
    result = _run_recipe(
        fragment,
        tmp_path,
        exported,
        dotenv_extra=dotenv + "FRAMEWORK=sglang\n",
        observed=("FRAMEWORK", "KERNEL_OPT_BACKEND_ORDER", "PYTHON", "USER_DATA_PATH"),
    )
    assert result == {
        "FRAMEWORK": "atom",
        "KERNEL_OPT_BACKEND_ORDER": expected,
        "PYTHON": "/selected/bin/python3",
        "USER_DATA_PATH": tmp_path.as_posix(),
    }


@pytest.mark.parametrize("doc", [ATOM_DOC, SETUP_DOC], ids=["atom-demo", "setup-skill"])
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
        assert ("--yes" in args) == (key == "INSTALL_ARGS")


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
@pytest.mark.parametrize("layout", [".", "src"], ids=["wheel", "source"])
def test_recipe_keeps_caller_user_data_path(doc: Path, tmp_path: Path, layout: str) -> None:
    """Every shared preamble preserves non-empty exported paths and credentials."""
    if not doc.exists():
        pytest.skip(f"{doc} not present in this layout")
    for fragment in _dotenv_loads(doc):
        result = _run_recipe(
            fragment,
            tmp_path,
            {"USER_DATA_PATH": "/from/caller", "OPENAI_API_KEY": "key-from-caller"},
            layout=layout,
        )
        assert result["USER_DATA_PATH"] == "/from/caller"
        assert result["OPENAI_API_KEY"] == "key-from-caller"


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
def test_recipe_still_fills_missing_values(doc: Path, tmp_path: Path) -> None:
    if not doc.exists():
        pytest.skip(f"{doc} not present in this layout")
    result = _run_recipe(_dotenv_loads(doc)[0], tmp_path, {})
    assert result["USER_DATA_PATH"] == "/from/dotenv"
    assert result["OPENAI_API_KEY"] == "key-from-dotenv"
    assert result["ONLY_IN_DOTENV"] == "filled"


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
def test_recipe_lets_dotenv_fill_a_blank_export(doc: Path, tmp_path: Path) -> None:
    if not doc.exists():
        pytest.skip(f"{doc} not present in this layout")
    result = _run_recipe(_dotenv_loads(doc)[0], tmp_path, {"USER_DATA_PATH": ""})
    assert result["USER_DATA_PATH"] == "/from/dotenv"


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
@pytest.mark.parametrize("conflict", [False, True], ids=["same-root", "conflicting-root"])
def test_recipe_readonly_workspace_is_preserved_or_fails(doc: Path, tmp_path: Path, conflict: bool) -> None:
    """Readonly shell state needs no child shell; incompatible setup roots fail clearly."""
    root = "/selected/data" if conflict else "/from/dotenv"
    fragment = "readonly LOCKED=locked\nexport LOCKED\nreadonly USER_DATA_PATH\n" + _dotenv_loads(doc)[0]
    fragment += '\n[[ "$(declare -p USER_DATA_PATH)" == "declare -rx "* ]]\n: > load-reached\n'
    kwargs = dict(dotenv_extra="HYPERLOOM_RUN_MODE=baremetal\n", observed=("USER_DATA_PATH", "LOCKED"))
    if conflict:
        with pytest.raises(subprocess.CalledProcessError) as failure:
            _run_recipe(fragment, tmp_path, {"USER_DATA_PATH": root}, **kwargs)
        assert "USER_DATA_PATH" in failure.value.stderr
        assert "readonly" in failure.value.stderr.lower()
        assert not (tmp_path / "load-reached").exists()
    else:
        assert _run_recipe(fragment, tmp_path, {"USER_DATA_PATH": root}, **kwargs) == {
            "USER_DATA_PATH": root,
            "LOCKED": "locked",
        }
        assert (tmp_path / "load-reached").exists()


def test_credential_only_recipe_keeps_the_callers_key(tmp_path: Path) -> None:
    if not CREDENTIAL_ONLY_DOC.exists():
        pytest.skip(f"{CREDENTIAL_ONLY_DOC} not present in this layout")
    fragment = _dotenv_loads(CREDENTIAL_ONLY_DOC)[0]
    kept = _run_recipe(fragment, tmp_path, {"OPENAI_API_KEY": "key-from-caller"})
    assert kept["OPENAI_API_KEY"] == "key-from-caller"
    filled = _run_recipe(fragment, tmp_path, {})
    assert filled["OPENAI_API_KEY"] == "key-from-dotenv"
