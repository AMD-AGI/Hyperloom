from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "inference_optimizer" / "scripts" / "local_setup.sh"

# ``local_setup.sh`` honours these env vars when present. The optimizer
# session shell (and CI runners) routinely export them, pointing at host
# paths/gateways that have nothing to do with the per-test sandbox — e.g.
# a host ``INFERENCEX_PATH`` that doesn't exist makes the script ``die``.
# Strip them from the inherited environment so each test runs hermetically
# and only sees the variables it explicitly sets via ``env``.
_HOST_LEAK_VARS = (
    "REPO_ROOT",
    "USER_DATA_PATH",
    "HYPERLOOM_RUNTIME_DIR",
    "HYPERLOOM_DEPS_ROOT",
    "LOCAL_SETUP_ENV",
    "PRIMUS_CLAW_REPO",
    "INFERENCEX_REPO",
    "TRACELENS_REPO",
    "TRACELENS_REF",
    "OOB_SRC",
    "INFERENCEX_PATH",
    "TRACELENS_ROOT",
    "SAFE_API_KEY",
    "OPENAI_BASE_URL",
    "CURSOR_API_KEY",
)


def _clean_base_env() -> dict[str, str]:
    """Inherited env with all script-consumed vars stripped out."""
    run_env = os.environ.copy()
    for var in _HOST_LEAK_VARS:
        run_env.pop(var, None)
    return run_env


def _run_local_setup(tmp_path: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    run_env = _clean_base_env()
    run_env.update(
        {
            "USER_DATA_PATH": str(tmp_path / "session"),
            "HYPERLOOM_DEPS_ROOT": str(tmp_path / "deps"),
        }
    )
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _git_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    for rel, content in files.items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=path, check=True)
    return path


def test_local_setup_clones_missing_dependency_repos_and_writes_env(tmp_path: Path) -> None:
    secret = "ak-secret-that-must-not-be-written"
    remotes = tmp_path / "remotes"
    primus = _git_repo(remotes / "Primus-Claw", {"OOB/README.md": "oob\n"})
    inferencex = _git_repo(remotes / "InferenceX", {"README.md": "inferencex\n"})
    tracelens = _git_repo(remotes / "TraceLens-internal", {"README.md": "tracelens\n"})

    result = _run_local_setup(
        tmp_path,
        env={
            "PRIMUS_CLAW_REPO": str(primus),
            "INFERENCEX_REPO": str(inferencex),
            "TRACELENS_REPO": str(tracelens),
            "TRACELENS_REF": "HEAD",
            "SAFE_API_KEY": secret,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    env_file = tmp_path / "session" / "runtime" / "local-setup.env.sh"
    assert env_file.exists()
    env_text = env_file.read_text(encoding="utf-8")
    assert f"export REPO_ROOT='{REPO_ROOT}'" in env_text
    assert f"export OOB_SRC='{tmp_path / 'deps' / 'Primus-Claw' / 'OOB'}'" in env_text
    assert f"export INFERENCEX_PATH='{tmp_path / 'deps' / 'InferenceX'}'" in env_text
    assert f"export TRACELENS_ROOT='{tmp_path / 'deps' / 'TraceLens-internal'}'" in env_text
    assert secret not in env_text


def test_local_setup_respects_existing_dependency_paths(tmp_path: Path) -> None:
    existing_oob = tmp_path / "existing" / "Primus-Claw" / "OOB"
    existing_oob.mkdir(parents=True)
    remotes = tmp_path / "remotes"
    inferencex = _git_repo(remotes / "InferenceX", {"README.md": "inferencex\n"})
    tracelens = _git_repo(remotes / "TraceLens-internal", {"README.md": "tracelens\n"})

    result = _run_local_setup(
        tmp_path,
        env={
            "OOB_SRC": str(existing_oob),
            "INFERENCEX_REPO": str(inferencex),
            "TRACELENS_REPO": str(tracelens),
            "TRACELENS_REF": "HEAD",
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    env_text = (tmp_path / "session" / "runtime" / "local-setup.env.sh").read_text(encoding="utf-8")
    assert f"export OOB_SRC='{existing_oob}'" in env_text
    assert not (tmp_path / "deps" / "Primus-Claw").exists()


def test_local_setup_fails_for_missing_explicit_dependency_path(tmp_path: Path) -> None:
    result = _run_local_setup(
        tmp_path,
        env={"OOB_SRC": str(tmp_path / "missing" / "OOB")},
    )

    assert result.returncode != 0
    assert "OOB_SRC is set but does not exist" in result.stderr


def test_local_setup_dry_run_does_not_write_or_leak_secret(tmp_path: Path) -> None:
    secret = "ak-super-secret-value"
    result = _run_local_setup(
        tmp_path,
        "--dry-run",
        env={"SAFE_API_KEY": secret, "OPENAI_BASE_URL": "https://example.test/v1"},
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "SAFE_API_KEY: set" in result.stdout
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert not (tmp_path / "session" / "runtime" / "local-setup.env.sh").exists()
    assert "install.sh" not in result.stdout
    assert "Open this folder in Cursor" in result.stdout
    assert "source ${" not in result.stdout
    assert f"source '{tmp_path / 'session' / 'runtime' / 'local-setup.env.sh'}'" in result.stdout
    assert f"export USER_DATA_PATH='{tmp_path / 'session'}'" in result.stdout
    assert "@inference_optimizer/SKILL.md" in result.stdout
    assert "Optimize inference for this workload" in result.stdout


def test_local_setup_session_dir_rebases_default_deps_root(tmp_path: Path) -> None:
    session_dir = tmp_path / "custom-session"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--session-dir", str(session_dir)],
        cwd=REPO_ROOT,
        env=_clean_base_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    expected_deps = session_dir / "runtime" / "source-mirrors"
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"HYPERLOOM_DEPS_ROOT={expected_deps}" in result.stdout
    assert str(expected_deps / "Primus-Claw") in result.stdout
    assert "release/hyperloom_integration_0.3.1" in result.stdout
