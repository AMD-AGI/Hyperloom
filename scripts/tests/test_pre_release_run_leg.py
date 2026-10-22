# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for .github/pre-release/run-leg.sh, the single-leg release runner.

The runner is driven end to end with stub ``claude`` and ``pip`` executables on
PATH, so the setup/demo loop, the .env it hands the agent and the PASS/FAIL it
derives from state.json are exercised without a GPU or an LLM.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown.stop_reasons import SUCCESS_STOP_REASONS

ROOT = Path(__file__).resolve().parents[2]
PRE_RELEASE = ROOT / ".github/pre-release"
RUN_LEG = PRE_RELEASE / "run-leg.sh"
PROMPTS = PRE_RELEASE / "prompts/pre-release"
FAKE_KEY = "sk-test-not-a-real-key-7f3a"

# The release matrix: leg id -> (setup prompt, demo prompt, model).
LEGS = {
    "baremetal-sglang-12h": ("setup-baremetal-sglang.md", "demo-12h.md", "Qwen3-14B-FP8"),
    "baremetal-vllm-12h": ("setup-baremetal-vllm.md", "demo-12h.md", "Qwen3-14B-FP8"),
    "docker-sglang-12h": ("setup-image-sglang.md", "demo-12h.md", "Qwen3-14B-FP8"),
    "docker-vllm-12h": ("setup-image-vllm.md", "demo-12h.md", "Qwen3-14B-FP8"),
    "docker-sglang-forge-12h": ("setup-image-sglang.md", "demo-12h-forge.md", "Qwen3-14B-FP8"),
    "docker-vllm-forge-12h": ("setup-image-vllm.md", "demo-12h-forge.md", "Qwen3-14B-FP8"),
    "baremetal-atom-forge-12h": ("setup-baremetal-atom.md", "demo-12h-atom-forge.md", "Qwen3-14B-FP8"),
    "docker-atom-forge-12h": ("setup-image-atom.md", "demo-12h-atom-forge.md", "Qwen3-14B-FP8"),
    "baremetal-sglang-3h": ("setup-baremetal-sglang.md", "demo-3h.md", "Qwen3-8B"),
    "baremetal-vllm-3h": ("setup-baremetal-vllm.md", "demo-3h.md", "Qwen3-8B"),
    "docker-sglang-3h": ("setup-image-sglang.md", "demo-3h.md", "Qwen3-8B"),
    "docker-vllm-3h": ("setup-image-vllm.md", "demo-3h.md", "Qwen3-8B"),
    "baremetal-atom-3h": ("setup-baremetal-atom.md", "demo-3h.md", "Qwen3-8B"),
    "docker-atom-3h": ("setup-image-atom.md", "demo-3h.md", "Qwen3-8B"),
}

# Stub claude: turn 1 (--session-id) prints the setup completion line for the
# backend in .env; a resumed turn fed a demo prompt writes the session (manifest.json
# naming the inherited CLAW_SESSION_ID, and the state.json that STUB_STOP_REASON
# names). STUB_SETUP=never keeps setup from completing. STUB_OWN=none skips this
# run's session; STUB_FOREIGN=<stop_reason> also writes, later and with a future
# mtime, another run's session under the same root, as a shared root would hold.
# Every turn appends its background-task switch, CLAW_SESSION_ID and Experience KB
# home to turns.log.
STUB_CLAUDE = r"""#!/usr/bin/env bash
# run-leg.sh probes --version before it enters the workspace, so that call must not write .env into the caller's cwd.
[ "${1:-}" = --version ] && { echo "stub-claude 0.0.0"; exit 0; }
prompt="$(cat)"
echo "bg_disabled=${CLAUDE_CODE_DISABLE_BACKGROUND_TASKS:-unset} claw=${CLAW_SESSION_ID:-unset} kb=${HYPERLOOM_KB_HOME:-unset}" >> turns.log
echo "${HYPERLOOM_CODE_REVISION:-unset}" >> revisions.log
echo "$(command -v python3)" >> pythons.log
backend="$(sed -n 's/^FRAMEWORK=//p' .env)"
session() {  # session <dir> <claw id> <stop_reason>
  mkdir -p "$1/reports"
  printf '{"claw_session_id": "%s"}' "$2" > "$1/manifest.json"
  printf '{"stop_reason": "%s", "close_sequence_done": true}' "$3" > "$1/state.json"
  echo '{}' > "$1/reports/final.json"
}
case "$(printf '%s\n' "$prompt" | head -n1)" in
  "# "*demo*|"Your previous turn ended without"*)
    d="$USER_DATA_PATH/qwen/20260101T000000Z-own"
    [ "${STUB_OWN:-}" = none ] || session "$d" "${CLAW_SESSION_ID:-}" "$STUB_STOP_REASON"
    if [ -n "${STUB_FOREIGN:-}" ]; then
      f="$USER_DATA_PATH/qwen/20260101T000001Z-other"
      session "$f" other-claw-session "$STUB_FOREIGN"
      find "$f" -exec touch -d '+1 hour' {} +
    fi
    echo "launched; state.json at $d" ;;
  *)
    [ "${STUB_SETUP:-}" = never ] && { echo "still installing"; exit 0; }
    if [ -n "${STUB_VENV:-}" ]; then
      mkdir -p "$STUB_VENV/bin"; printf '#!/bin/sh\n' > "$STUB_VENV/bin/python3"; chmod +x "$STUB_VENV/bin/python3"
      echo "VIRTUAL_ENV=$STUB_VENV" >> .env
    fi
    echo "setup complete: baremetal/$backend" ;;
esac
"""

STUB_PIP = r"""#!/usr/bin/env bash
if [ "$1" = wheel ]; then
  while [ "$#" -gt 0 ]; do [ "$1" = --wheel-dir ] && { touch "$2/hyperloom_inference_optimizer-0-py3-none-any.whl"; }; shift; done
fi
exit 0
"""


def _exe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def leg_env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "claude", STUB_CLAUDE)
    _exe(bin_dir / "pip", STUB_PIP)
    models = tmp_path / "models"
    for name in ("Qwen3-8B", "Qwen3-14B-FP8"):
        (models / name).mkdir(parents=True)
        (models / name / "config.json").write_text("{}", encoding="utf-8")
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "CI_VERSION": "attempt-1",
        "LEG_WORK_DIR": str(tmp_path / "work"),
        "USER_DATA_PATH": str(tmp_path / "session"),
        "LEG_MODELS_DIR": str(models),
        "ANTHROPIC_API_KEY": FAKE_KEY,
        "ANTHROPIC_BASE_URL": "https://llm.example.test",
        "CLAUDE_MODEL": "test-model",
        "STUB_STOP_REASON": "sweep_done",
        "LEG_TURN_GAP_S": "0",
        "LEG_WAIT_INTERVAL_S": "1",
    }


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(RUN_LEG), *args], env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_every_matrix_leg_resolves_to_shipped_prompts(leg: str) -> None:
    result = _run(["--resolve", leg], {"PATH": os.environ["PATH"]})
    assert result.returncode == 0, result.stderr
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines())
    setup, demo, model = LEGS[leg]
    assert fields["setup_prompt"] == f".github/pre-release/prompts/pre-release/{setup}"
    assert fields["demo_prompt"] == f".github/pre-release/prompts/pre-release/{demo}"
    assert fields["model"] == model
    assert (PROMPTS / setup).is_file() and (PROMPTS / demo).is_file()


@pytest.mark.parametrize(
    "leg",
    ["docker-vllm-forge-3h", "baremetal-atom-12h", "docker-sglang-4h", "kube-vllm-3h", "docker-vllm-3h;id"],
)
def test_legs_outside_the_matrix_are_rejected(leg: str) -> None:
    result = _run(["--resolve", leg], {"PATH": os.environ["PATH"]})
    assert result.returncode == 2
    assert result.stdout == ""


def test_missing_required_env_is_named_and_nothing_runs(leg_env: dict[str, str], tmp_path: Path) -> None:
    env = {k: v for k, v in leg_env.items() if k not in {"ANTHROPIC_BASE_URL", "LEG_MODELS_DIR", "CI_VERSION"}}
    result = _run(["docker-vllm-3h"], env)
    assert result.returncode == 2
    for name in ("ANTHROPIC_BASE_URL", "CI_VERSION", "LEG_MODEL_PATH or LEG_MODELS_DIR"):
        assert name in result.stderr
    assert not (tmp_path / "work").exists()


def test_a_model_dir_without_config_json_is_refused(leg_env: dict[str, str], tmp_path: Path) -> None:
    (tmp_path / "models/Qwen3-8B/config.json").unlink()
    result = _run(["docker-vllm-3h"], leg_env)
    assert result.returncode == 2
    assert "config.json" in result.stderr


def test_clean_stop_reason_passes_and_the_key_does_not_outlive_the_leg(leg_env: dict[str, str]) -> None:
    result = _run(["docker-vllm-3h"], leg_env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: stop_reason=sweep_done" in result.stdout
    workspace = Path(leg_env["LEG_WORK_DIR"]) / "attempt-1/docker-vllm-3h"
    dotenv = workspace / ".env"
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o600
    lines = dotenv.read_text(encoding="utf-8").splitlines()
    assert "FRAMEWORK=vllm" in lines
    assert "HYPERLOOM_RUN_MODE=baremetal" in lines
    assert f"MODEL_PATH={leg_env['LEG_MODELS_DIR']}/Qwen3-8B" in lines
    assert f"USER_DATA_PATH={leg_env['USER_DATA_PATH']}" in lines
    assert not any(line.startswith("KERNEL_OPT_BACKEND_ORDER=") for line in lines)
    for text in (dotenv.read_text(encoding="utf-8"), result.stdout, result.stderr):
        assert FAKE_KEY not in text


def test_forge_leg_selects_the_forge_kernel_backend(leg_env: dict[str, str]) -> None:
    result = _run(["docker-sglang-forge-12h"], leg_env)
    assert result.returncode == 0, result.stdout + result.stderr
    dotenv = Path(leg_env["LEG_WORK_DIR"]) / "attempt-1/docker-sglang-forge-12h/.env"
    lines = dotenv.read_text(encoding="utf-8").splitlines()
    assert "KERNEL_OPT_BACKEND_ORDER=forge" in lines
    assert f"MODEL_PATH={leg_env['LEG_MODELS_DIR']}/Qwen3-14B-FP8" in lines


@pytest.mark.parametrize("stop_reason", ["baseline_failed", "signal", "enablement_failed"])
def test_unclean_stop_reason_fails_the_leg(leg_env: dict[str, str], stop_reason: str) -> None:
    result = _run(["baremetal-atom-3h"], {**leg_env, "STUB_STOP_REASON": stop_reason})
    assert result.returncode == 1
    assert f"FAIL: stop_reason={stop_reason}" in result.stdout


def test_closed_run_without_a_stop_reason_is_not_a_pass(leg_env: dict[str, str]) -> None:
    result = _run(["docker-vllm-3h"], {**leg_env, "STUB_STOP_REASON": "", "LEG_DEADLINE_S": "2"})
    assert result.returncode == 1
    assert "no terminal stop_reason within 2s" in result.stdout


def test_relative_workspace_is_refused(leg_env: dict[str, str]) -> None:
    result = _run(["docker-vllm-3h"], {**leg_env, "LEG_WORK_DIR": "work"})
    assert result.returncode == 2
    assert "LEG_WORK_DIR must be an absolute path" in result.stderr


def test_setup_that_never_completes_fails_after_its_turn_budget(leg_env: dict[str, str]) -> None:
    env = {**leg_env, "STUB_SETUP": "never", "LEG_SETUP_MAX_TURNS": "3"}
    result = _run(["baremetal-sglang-3h"], env)
    assert result.returncode == 1
    assert "setup incomplete after 3 turns" in result.stdout
    agent_log = Path(leg_env["LEG_WORK_DIR"]) / "attempt-1/baremetal-sglang-3h/agent.log"
    assert agent_log.read_text(encoding="utf-8").count("still installing") == 3


def test_every_claude_turn_runs_without_background_tasks(leg_env: dict[str, str]) -> None:
    # A turn's background shell ends with the turn: optimize must not be started there.
    env = {**leg_env, "CLAW_SESSION_ID": "claw-own", "STUB_SETUP": "never", "LEG_SETUP_MAX_TURNS": "2"}
    _run(["docker-vllm-3h"], env)
    turns = (Path(leg_env["LEG_WORK_DIR"]) / "attempt-1/docker-vllm-3h/turns.log").read_text(encoding="utf-8")
    assert [line.split(" kb=")[0] for line in turns.splitlines()] == ["bg_disabled=1 claw=claw-own"] * 2
    result = _run(["docker-vllm-3h"], {**leg_env, "CI_VERSION": "attempt-2", "CLAW_SESSION_ID": "claw-own"})
    assert result.returncode == 0, result.stdout + result.stderr
    turns = (Path(leg_env["LEG_WORK_DIR"]) / "attempt-2/docker-vllm-3h/turns.log").read_text(encoding="utf-8")
    assert [line.split(" kb=")[0] for line in turns.splitlines()] == ["bg_disabled=1 claw=claw-own"] * 2


def test_legs_sharing_a_session_root_keep_separate_experience_kb_homes(leg_env: dict[str, str]) -> None:
    # One Experience KB home serves one service: a second leg on the same root must
    # not be pointed at the first leg's home.
    homes = []
    for leg in ("docker-vllm-3h", "baremetal-sglang-3h"):
        result = _run([leg], {**leg_env, "CLAW_SESSION_ID": f"claw-{leg}"})
        assert result.returncode == 0, result.stdout + result.stderr
        workspace = Path(leg_env["LEG_WORK_DIR"]) / "attempt-1" / leg
        home = f"{workspace}/experience-kb"
        assert f"HYPERLOOM_KB_HOME={home}" in (workspace / ".env").read_text(encoding="utf-8").splitlines()
        assert {
            line.split(" kb=")[1] for line in (workspace / "turns.log").read_text(encoding="utf-8").splitlines()
        } == {home}
        homes.append(home)
    assert len(set(homes)) == 2


@pytest.mark.parametrize(("own", "foreign", "verdict"), [("signal", "sweep_done", 1), ("sweep_done", "signal", 0)])
def test_a_shared_session_root_judges_only_this_runs_session(
    leg_env: dict[str, str], own: str, foreign: str, verdict: int
) -> None:
    # Another run's newer session under the same root must not decide this leg.
    env = {**leg_env, "CLAW_SESSION_ID": "claw-own", "STUB_STOP_REASON": own, "STUB_FOREIGN": foreign}
    result = _run(["docker-vllm-3h"], env)
    assert result.returncode == verdict, result.stdout + result.stderr
    assert f"stop_reason={own}" in result.stdout
    assert "20260101T000000Z-own" in result.stdout
    run_log = Path(leg_env["LEG_WORK_DIR"]) / "attempt-1/docker-vllm-3h/run-leg.log"
    assert f"stop_reason={own}" in run_log.read_text(encoding="utf-8")


def test_another_runs_progress_does_not_hold_off_the_relaunch(leg_env: dict[str, str]) -> None:
    # Only a foreign session is being written: this leg has no optimizer, so it is
    # re-driven and then failed, instead of waiting out the whole deadline.
    env = {
        **leg_env,
        "CLAW_SESSION_ID": "claw-own",
        "STUB_OWN": "none",
        "STUB_FOREIGN": "sweep_done",
        "LEG_STALL_GRACE_S": "1",
        "LEG_DEMO_REDRIVES": "1",
        "LEG_DEADLINE_S": "20",
    }
    result = _run(["docker-vllm-3h"], env)
    assert result.returncode == 1
    assert "no state.json after 1 demo re-drive(s)" in result.stdout, result.stdout


def test_the_optimizer_is_told_the_checkouts_revision(leg_env: dict[str, str]) -> None:
    # The wheel installed into the workspace is not a checkout, so without this the
    # session's manifest carries no code revision.
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    result = _run(["docker-vllm-3h"], leg_env)
    assert result.returncode == 0, result.stdout + result.stderr
    revisions = (Path(leg_env["LEG_WORK_DIR"]) / "attempt-1/docker-vllm-3h/revisions.log").read_text(encoding="utf-8")
    assert set(revisions.split()) == {head.stdout.strip()}
    pinned = _run(["docker-vllm-3h"], {**leg_env, "CI_VERSION": "attempt-2", "HYPERLOOM_CODE_REVISION": "abc1234"})
    assert pinned.returncode == 0, pinned.stdout + pinned.stderr
    revisions = (Path(leg_env["LEG_WORK_DIR"]) / "attempt-2/docker-vllm-3h/revisions.log").read_text(encoding="utf-8")
    assert set(revisions.split()) == {"abc1234"}


def test_the_demo_launches_with_the_interpreter_setup_installed_into(leg_env: dict[str, str], tmp_path: Path) -> None:
    # Setup records its interpreter as VIRTUAL_ENV in .env; a PATH that dropped the
    # image's venv must not hand the demo a python3 without the framework.
    venv = tmp_path / "framework-venv"
    result = _run(["baremetal-sglang-3h"], {**leg_env, "STUB_VENV": str(venv)})
    assert result.returncode == 0, result.stdout + result.stderr
    pythons = (Path(leg_env["LEG_WORK_DIR"]) / "attempt-1/baremetal-sglang-3h/pythons.log").read_text(encoding="utf-8")
    setup_turn, demo_turn = pythons.splitlines()
    assert setup_turn != f"{venv}/bin/python3"
    assert demo_turn == f"{venv}/bin/python3"


def test_path_prefix_puts_the_images_interpreter_first_for_every_turn(leg_env: dict[str, str], tmp_path: Path) -> None:
    # A scheduler that replaces the image's PATH leaves `python3` resolving to a system
    # interpreter; the caller hands back the image's leading PATH entries.
    image_bin = tmp_path / "image-venv/bin"
    image_bin.mkdir(parents=True)
    # The runner reads JSON with python3 itself, so the image's interpreter must work.
    _exe(image_bin / "python3", f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    result = _run(["docker-atom-3h"], {**leg_env, "LEG_PATH_PREFIX": str(image_bin)})
    assert result.returncode == 0, result.stdout + result.stderr
    pythons = (Path(leg_env["LEG_WORK_DIR"]) / "attempt-1/docker-atom-3h/pythons.log").read_text(encoding="utf-8")
    assert pythons.splitlines() == [f"{image_bin}/python3", f"{image_bin}/python3"]
    assert f"python3={image_bin}/python3 path_prefix={image_bin}" in result.stdout
    unset = _run(["docker-atom-3h"], {**leg_env, "CI_VERSION": "attempt-2"})
    assert unset.returncode == 0, unset.stdout + unset.stderr
    pythons = (Path(leg_env["LEG_WORK_DIR"]) / "attempt-2/docker-atom-3h/pythons.log").read_text(encoding="utf-8")
    assert f"{image_bin}/python3" not in pythons


def test_runner_success_set_matches_the_optimizer() -> None:
    line = next(x for x in RUN_LEG.read_text(encoding="utf-8").splitlines() if x.startswith("CLEAN_STOP_REASONS="))
    assert set(line.split("=", 1)[1].strip('"').split()) == set(SUCCESS_STOP_REASONS)


# The leg definition is public and site-neutral: every host, address and path
# comes from the caller's env. nodejs.org is the one public download it names.
_HOSTNAME = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|ai|dev|cloud|internal|local)\b", re.IGNORECASE)
_IPV4 = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'=(`:-])(/[A-Za-z_][\w.-]*/)", re.MULTILINE)
_LEG_DEFINITION = [
    RUN_LEG,
    *sorted(PROMPTS.glob("setup-image-*.md")),
    PROMPTS / "setup-baremetal-atom.md",
    PROMPTS / "demo-12h-atom-forge.md",
]


@pytest.mark.parametrize("path", _LEG_DEFINITION, ids=lambda p: p.name)
def test_leg_definition_names_no_site_host_address_or_path(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert set(_HOSTNAME.findall(text)) <= {"nodejs.org"}
    assert _IPV4.findall(text) == []
    assert _ABSOLUTE_PATH.findall(text) == []
