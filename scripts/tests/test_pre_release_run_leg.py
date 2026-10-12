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
# backend in .env; a resumed turn fed a demo prompt writes the session state.json
# that STUB_STOP_REASON names. STUB_SETUP=never keeps setup from completing.
STUB_CLAUDE = r"""#!/usr/bin/env bash
prompt="$(cat)"
backend="$(sed -n 's/^FRAMEWORK=//p' .env)"
case "$(printf '%s\n' "$prompt" | head -n1)" in
  "# "*demo*)
    d="$USER_DATA_PATH/qwen/20260101T000000Z"; mkdir -p "$d"
    printf '{"stop_reason": "%s", "close_sequence_done": true}' "$STUB_STOP_REASON" > "$d/state.json"
    echo "launched; state.json at $d" ;;
  *)
    [ "${STUB_SETUP:-}" = never ] && { echo "still installing"; exit 0; }
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


def test_setup_that_never_completes_fails_after_its_turn_budget(leg_env: dict[str, str]) -> None:
    env = {**leg_env, "STUB_SETUP": "never", "LEG_SETUP_MAX_TURNS": "3"}
    result = _run(["baremetal-sglang-3h"], env)
    assert result.returncode == 1
    assert "setup incomplete after 3 turns" in result.stdout
    agent_log = Path(leg_env["LEG_WORK_DIR"]) / "attempt-1/baremetal-sglang-3h/agent.log"
    assert agent_log.read_text(encoding="utf-8").count("still installing") == 3


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
