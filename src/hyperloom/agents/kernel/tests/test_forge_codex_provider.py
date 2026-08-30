"""Provider selection for the forge-loop child under each credential shape.

KernelForge ships both a ``claude`` and a ``codex`` agent provider, and its
``Config.agent_backend`` defaults to ``auto``, which resolves to ``claude``. An
OpenAI-only deployment has no Anthropic credential and no Claude CLI auth, so
leaving the selection at ``auto`` sends every forge attempt to a provider that
cannot authenticate. These tests lock the selection to the credential shape and
lock out the silent provider fallback that would otherwise turn a missing Codex
SDK back into an unauthenticated Claude run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

try:  # tomllib is stdlib from 3.11; the ``ci`` extra pins tomli for 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on py3.10
    import tomli as tomllib  # type: ignore[no-redef]

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402

_PROVIDER_ENV = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "CODEX_MODEL",
    "CLAUDE_MODEL",
    "FORGE_CODEX_MODEL",
    "FORGE_CLAUDE_MODEL",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


def _use_openai_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/api/v1/llm-proxy/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "ak-openai")


def _use_anthropic_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/api/v1/llm-proxy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-anthropic")


def _capture_forge_loop_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Run ``_run_loop_via_cli`` with a stubbed child and return its argv."""
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("pass\n")
    driver.write_text("pass\n")
    experiments = tmp_path / "attempt" / "forge_experiments"
    experiments.mkdir(parents=True)
    captured: dict[str, list[str]] = {}

    class FakeProcess:
        pid = 4321
        returncode = 0

        def communicate(self, timeout=None):
            payload = {"baseline_ms": 2.0, "best_ms": 1.0}
            return f"__FORGE_RESULT__{json.dumps(payload)}__FORGE_RESULT__", ""

    def fake_popen(command, **_kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(forge_submit, "_apply_kernel_backend_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)

    forge_submit._run_loop_via_cli(
        worktree_kernel=str(kernel),
        driver=str(driver),
        workspace=str(workspace),
        snr_threshold=30.0,
        max_hours=1.0,
        branch="forge/test/provider",
        gpu_target="gfx950",
        gpu_type="mi355x",
        kernel_backend="triton",
        program_md_file="",
        invocation_spec_file="",
        experiments_dir=experiments,
        forge_log=tmp_path / "forge.log",
        timeout_s=60,
    )
    return captured["command"]


def _flag_value(command: list[str], flag: str) -> str:
    assert flag in command, f"{flag} missing from {command}"
    return command[command.index(flag) + 1]


def _capture_rewrite_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], dict[str, str]]:
    """Run ``_run_rewrite_via_cli`` with a stubbed child and return argv + env."""
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("pass\n")
    driver.write_text("pass\n")
    experiments = tmp_path / "attempt" / "forge_experiments"
    experiments.mkdir(parents=True)
    result_json = tmp_path / "attempt" / "rewrite_result.json"
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 4322
        returncode = 0

        def communicate(self, timeout=None):
            return '{"success": true}', ""

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env") or {}
        return FakeProcess()

    monkeypatch.setattr(forge_submit, "_apply_kernel_backend_env", lambda _env: None)
    monkeypatch.setattr(forge_submit.subprocess, "Popen", fake_popen)

    forge_submit._run_rewrite_via_cli(
        source_kernel=str(kernel),
        driver=str(driver),
        logical_op_name="test::op",
        source_entry="entry",
        source_language="triton",
        workspace=str(workspace),
        experiments_dir=experiments,
        result_json=result_json,
        target_functions=["entry"],
        shapes=[{"M": 8, "N": 8, "dtype": "fp16"}],
        invocation_spec_file="",
        driver_preparation=False,
        snr_threshold=30.0,
        gpu_target="gfx950",
        gpu_type="mi355x",
        max_hours=1.0,
        branch="forge/test/rewrite",
        framework="vllm",
        forge_log=tmp_path / "forge.log",
        timeout_s=60,
    )
    return captured["command"], captured["env"]


def test_openai_only_selects_the_codex_provider(tmp_path, monkeypatch):
    """OpenAI-only must pick codex explicitly instead of inheriting ``auto``."""
    _use_openai_only(monkeypatch)

    command = _capture_forge_loop_argv(tmp_path, monkeypatch)

    assert _flag_value(command, "--agent-backend") == "codex"


def test_openai_only_forwards_the_session_codex_model(tmp_path, monkeypatch):
    """The session's CODEX_MODEL wins over KernelForge's own codex default."""
    _use_openai_only(monkeypatch)
    monkeypatch.setenv("CODEX_MODEL", "gpt-5.5")

    command = _capture_forge_loop_argv(tmp_path, monkeypatch)

    assert _flag_value(command, "--model") == "gpt-5.5"


def test_openai_only_prefers_forge_codex_model_over_codex_model(tmp_path, monkeypatch):
    """FORGE_CODEX_MODEL is the rewrite counterpart of the fusion/codex override."""
    _use_openai_only(monkeypatch)
    monkeypatch.setenv("CODEX_MODEL", "gpt-orchestration")
    monkeypatch.setenv("FORGE_CODEX_MODEL", "gpt-forge-only")

    command = _capture_forge_loop_argv(tmp_path, monkeypatch)

    assert _flag_value(command, "--model") == "gpt-forge-only"


def test_openai_only_omits_the_model_flag_without_codex_model(tmp_path, monkeypatch):
    """With no CODEX_MODEL pinned, defer to the codex provider's own default."""
    _use_openai_only(monkeypatch)

    command = _capture_forge_loop_argv(tmp_path, monkeypatch)

    assert "--model" not in command


def test_anthropic_path_forwards_forge_claude_model(tmp_path, monkeypatch):
    """Claude-side rewrite must honor FORGE_CLAUDE_MODEL over CLAUDE_MODEL."""
    _use_anthropic_only(monkeypatch)
    monkeypatch.setenv("CLAUDE_MODEL", "claude-orchestration")
    monkeypatch.setenv("FORGE_CLAUDE_MODEL", "claude-forge-only")

    command = _capture_forge_loop_argv(tmp_path, monkeypatch)

    assert _flag_value(command, "--model") == "claude-forge-only"
    assert "--agent-backend" not in command


def test_anthropic_path_forwards_claude_model_without_forge_override(tmp_path, monkeypatch):
    """With no FORGE_CLAUDE_MODEL, rewrite still forwards CLAUDE_MODEL."""
    _use_anthropic_only(monkeypatch)
    monkeypatch.setenv("CLAUDE_MODEL", "claude-orchestration")

    command = _capture_forge_loop_argv(tmp_path, monkeypatch)

    assert _flag_value(command, "--model") == "claude-orchestration"


def test_flydsl_rewrite_openai_only_prefers_forge_codex_model(tmp_path, monkeypatch):
    """FlyDSL rewrite must resolve FORGE_CODEX_MODEL; KernelForge does not."""
    _use_openai_only(monkeypatch)
    monkeypatch.setenv("CODEX_MODEL", "gpt-orchestration")
    monkeypatch.setenv("FORGE_CODEX_MODEL", "gpt-forge-only")

    command, env = _capture_rewrite_argv(tmp_path, monkeypatch)

    assert env["FORGE_AGENT_BACKEND"] == "codex"
    assert env["FORGE_AGENT_FALLBACK_PROVIDER"] == "none"
    assert _flag_value(command, "--model") == "gpt-forge-only"


def test_flydsl_rewrite_anthropic_path_prefers_forge_claude_model(tmp_path, monkeypatch):
    """FlyDSL rewrite must pass FORGE_CLAUDE_MODEL via --model, not ambient env alone."""
    _use_anthropic_only(monkeypatch)
    monkeypatch.setenv("CLAUDE_MODEL", "claude-orchestration")
    monkeypatch.setenv("FORGE_CLAUDE_MODEL", "claude-forge-only")

    command, env = _capture_rewrite_argv(tmp_path, monkeypatch)

    assert "FORGE_AGENT_BACKEND" not in env
    assert _flag_value(command, "--model") == "claude-forge-only"


def test_flydsl_rewrite_omits_model_without_configured_ids(tmp_path, monkeypatch):
    """With no FORGE_*/CLAUDE/CODEX model pinned, defer to KernelForge defaults."""
    _use_anthropic_only(monkeypatch)

    command, _env = _capture_rewrite_argv(tmp_path, monkeypatch)

    assert "--model" not in command


def test_openai_only_disables_the_silent_claude_fallback(tmp_path, monkeypatch):
    """A missing Codex SDK must fail loudly, not degrade into unauthenticated Claude.

    KernelForge's ``agent_fallback_provider`` defaults to ``claude``, so without
    this flag an OpenAI-only run whose Codex SDK is absent silently produces a
    ClaudeBackend whose availability probe passes (it checks the binary, not the
    auth) and only fails at the first real turn, as "Not logged in".
    """
    _use_openai_only(monkeypatch)

    command = _capture_forge_loop_argv(tmp_path, monkeypatch)

    assert _flag_value(command, "--agent-fallback-provider") == "none"


def test_anthropic_only_leaves_provider_selection_untouched(tmp_path, monkeypatch):
    """The Anthropic-side shape keeps KernelForge's own resolution."""
    _use_anthropic_only(monkeypatch)

    command = _capture_forge_loop_argv(tmp_path, monkeypatch)

    assert "--agent-backend" not in command
    assert "--agent-fallback-provider" not in command


def test_both_sides_configured_leaves_provider_selection_untouched(tmp_path, monkeypatch):
    """With an Anthropic credential present, the claude kernel backend still works."""
    _use_openai_only(monkeypatch)
    _use_anthropic_only(monkeypatch)

    command = _capture_forge_loop_argv(tmp_path, monkeypatch)

    assert "--agent-backend" not in command


def test_openai_only_child_env_does_not_pin_the_claude_cli(monkeypatch):
    """OpenAI-only child env must not advertise a Claude CLI it cannot authenticate."""
    import _llm_stability_env

    monkeypatch.setattr(_llm_stability_env, "apply_llm_stability_env", lambda env: None)
    monkeypatch.setattr(forge_submit.shutil, "which", lambda _name: "/usr/local/bin/claude")
    _use_openai_only(monkeypatch)
    forge_submit._reset_knowledge_config_cache()

    env: dict[str, str] = {}
    forge_submit._apply_kernel_backend_env(env)

    assert "FORGE_CLAUDE_BIN" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_anthropic_only_child_env_still_pins_the_claude_cli(monkeypatch):
    """The Anthropic-side path keeps its existing claude CLI discovery."""
    import _llm_stability_env

    monkeypatch.setattr(_llm_stability_env, "apply_llm_stability_env", lambda env: None)
    monkeypatch.setattr(forge_submit.shutil, "which", lambda _name: "/usr/local/bin/claude")
    monkeypatch.setattr(forge_submit.os.path, "isfile", lambda path: path == "/usr/local/bin/claude")
    _use_anthropic_only(monkeypatch)
    forge_submit._reset_knowledge_config_cache()

    env: dict[str, str] = {}
    forge_submit._apply_kernel_backend_env(env)

    assert env["FORGE_CLAUDE_BIN"] == "/usr/local/bin/claude"


def test_install_sh_installs_the_codex_runtime():
    """install.sh must install the codex agent runtime, and verify it.

    Without it ``FORGE_AGENT_BACKEND=codex`` raises CodexUnavailableError
    ("Codex Python SDK is not installed"), which the provider fallback then
    converts into a silent Claude run.

    This used to assert on a ``kernelforge[claude,codex]`` install line, from
    when forge was a separate distribution installed from a checkout. forge now
    ships in this distribution: the editable path gets ``openai-codex`` through
    ``[test]`` -> ``[runtime]`` -> ``[llm]``, and the packaged-wheel path pulls
    the same extras by name. Both then run the same readiness probe.

    The assertion follows the extra rather than the pin. It used to look for the
    literal ``openai-codex>=0.144`` in install.sh, which only passed because
    install.sh restated pyproject's specifiers verbatim -- so the test was
    pinning the duplication instead of catching it, and would have gone green on
    a stale copy. What must hold is the *chain*: the packaged path names an
    extra, and that extra reaches openai-codex.
    """
    install_sh = Path(__file__).resolve().parents[3] / "inference_optimizer" / "assets" / "install.sh"
    text = install_sh.read_text(encoding="utf-8")

    assert "hyperloom-inference_optimizer[llm,forge]" in text, (
        "the packaged-wheel install path must install the llm+forge extras (the bare wheel ships no third-party deps)"
    )
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[4].parent / "pyproject.toml").read_text(encoding="utf-8")
    )
    extras = pyproject["project"]["optional-dependencies"]
    assert any(req.startswith("openai-codex") for req in extras["llm"]), (
        "the llm extra must carry the codex agent runtime; install.sh reaches it "
        "through [llm,forge] and no longer names it directly"
    )
    assert "import openai_codex" in text, (
        "install.sh must verify the codex runtime imports; a missing one silently "
        "downgrades an OpenAI-only deployment to a Claude run that dies on 'Not logged in'"
    )


def test_forge_loop_cli_accepts_the_provider_flags():
    """Guard the contract: these flags must exist in the installed KernelForge CLI.

    Passing an unknown option makes click exit 2 and every forge attempt REVERT,
    so a KernelForge upgrade that renames them must fail here, not in a session.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "kernelforge.cli", "forge-loop", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(f"kernelforge CLI unavailable (rc={proc.returncode})")
    for flag in ("--agent-backend", "--model", "--agent-fallback-provider"):
        assert flag in proc.stdout, flag
