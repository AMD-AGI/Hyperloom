# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "inference_optimizer" / "scripts" / "local_setup.sh"
IO_INSTALL = REPO_ROOT / "inference_optimizer" / "scripts" / "install.sh"
KA_INSTALL = REPO_ROOT / "kernel-agent" / "scripts" / "install.sh"
PREFLIGHT_KB = REPO_ROOT / "inference_optimizer" / "scripts" / "preflight_kb.sh"

# The default that all installers fall back to when USER_DATA_PATH is unset.
_DEFAULT_USER_DATA_PATH = "/workspace/hyperloom"
# Substring of the loud fallback notice each script prints to stderr.
_FALLBACK_WARNING = "USER_DATA_PATH not set"

# Strip these host-leaked env vars so each test runs hermetically.
_HOST_LEAK_VARS = (
    "REPO_ROOT",
    "USER_DATA_PATH",
    "HYPERLOOM_RUNTIME_DIR",
    "HYPERLOOM_DEPS_ROOT",
    "LOCAL_SETUP_ENV",
    "PRIMUS_CLAW_REPO",
    "INFERENCEX_REPO",
    "INFERENCEX_REF",
    "TRACELENS_REPO",
    "TRACELENS_REF",
    "OOB_SRC",
    "INFERENCEX_PATH",
    "TRACELENS_ROOT",
    "TRACELENS_INTERNAL_ROOT",
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
    tracelens_public = _git_repo(remotes / "TraceLens", {"README.md": "tracelens\n"})

    result = _run_local_setup(
        tmp_path,
        env={
            "PRIMUS_CLAW_REPO": str(primus),
            "INFERENCEX_REPO": str(inferencex),
            "INFERENCEX_REF": "HEAD",
            "TRACELENS_REPO": str(tracelens_public),
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
    assert f"export TRACELENS_ROOT='{tmp_path / 'deps' / 'TraceLens'}'" in env_text
    # Default is open-source-only: no internal extension.
    assert "TRACELENS_INTERNAL_ROOT" not in env_text
    assert "TL_EXTENSION" not in env_text
    assert not (tmp_path / "deps" / "TraceLens-internal").exists()
    assert "open-source-only" in result.stdout
    assert secret not in env_text


def test_local_setup_uses_internal_extension_when_root_set(tmp_path: Path) -> None:
    remotes = tmp_path / "remotes"
    primus = _git_repo(remotes / "Primus-Claw", {"OOB/README.md": "oob\n"})
    inferencex = _git_repo(remotes / "InferenceX", {"README.md": "inferencex\n"})
    tracelens_public = _git_repo(remotes / "TraceLens", {"README.md": "tracelens\n"})
    internal_checkout = _git_repo(
        tmp_path / "existing" / "TraceLens-internal", {"README.md": "internal\n"}
    )

    result = _run_local_setup(
        tmp_path,
        env={
            "PRIMUS_CLAW_REPO": str(primus),
            "INFERENCEX_REPO": str(inferencex),
            "INFERENCEX_REF": "HEAD",
            "TRACELENS_REPO": str(tracelens_public),
            "TRACELENS_REF": "HEAD",
            "TRACELENS_INTERNAL_ROOT": str(internal_checkout),
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    env_text = (tmp_path / "session" / "runtime" / "local-setup.env.sh").read_text(encoding="utf-8")
    assert f"export TRACELENS_INTERNAL_ROOT='{internal_checkout}'" in env_text
    assert "export TL_EXTENSION='TraceLens_internal'" in env_text


def test_local_setup_internal_missing_path_falls_back_to_oss_only(tmp_path: Path) -> None:
    # A non-existent TRACELENS_INTERNAL_ROOT must warn and fall back to OSS-only.
    remotes = tmp_path / "remotes"
    primus = _git_repo(remotes / "Primus-Claw", {"OOB/README.md": "oob\n"})
    inferencex = _git_repo(remotes / "InferenceX", {"README.md": "inferencex\n"})
    tracelens_public = _git_repo(remotes / "TraceLens", {"README.md": "tracelens\n"})

    result = _run_local_setup(
        tmp_path,
        env={
            "PRIMUS_CLAW_REPO": str(primus),
            "INFERENCEX_REPO": str(inferencex),
            "INFERENCEX_REF": "HEAD",
            "TRACELENS_REPO": str(tracelens_public),
            "TRACELENS_REF": "HEAD",
            "TRACELENS_INTERNAL_ROOT": str(tmp_path / "nope" / "TraceLens-internal"),
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    env_text = (tmp_path / "session" / "runtime" / "local-setup.env.sh").read_text(encoding="utf-8")
    assert "TRACELENS_INTERNAL_ROOT" not in env_text
    assert "TL_EXTENSION" not in env_text
    assert "falling back to open-source-only" in result.stderr


def test_local_setup_respects_existing_dependency_paths(tmp_path: Path) -> None:
    existing_oob = tmp_path / "existing" / "Primus-Claw" / "OOB"
    existing_oob.mkdir(parents=True)
    remotes = tmp_path / "remotes"
    inferencex = _git_repo(remotes / "InferenceX", {"README.md": "inferencex\n"})
    tracelens_public = _git_repo(remotes / "TraceLens", {"README.md": "tracelens\n"})

    result = _run_local_setup(
        tmp_path,
        env={
            "OOB_SRC": str(existing_oob),
            "INFERENCEX_REPO": str(inferencex),
            "INFERENCEX_REF": "HEAD",
            "TRACELENS_REPO": str(tracelens_public),
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


def test_local_setup_deps_root_stays_pod_local_under_session_dir(tmp_path: Path) -> None:
    # Deps root must NOT follow --session-dir: it stays on a pod-local base
    # (TMPDIR) so a shared session tree never collocates concurrent checkouts.
    session_dir = tmp_path / "custom-session"
    tmpdir = tmp_path / "podlocal"
    tmpdir.mkdir()
    env = _clean_base_env()
    env["TMPDIR"] = str(tmpdir)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--session-dir", str(session_dir)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    expected_deps = tmpdir / "hyperloom" / "open-source-repos"
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"HYPERLOOM_DEPS_ROOT={expected_deps}" in result.stdout
    assert str(session_dir / "runtime" / "open-source-repos") not in result.stdout
    assert str(expected_deps / "Primus-Claw") in result.stdout
    assert str(expected_deps / "TraceLens") in result.stdout
    assert str(expected_deps / "TraceLens-internal") not in result.stdout
    assert "0ebaa7109992b98b8f747a0fc0973e0f3b65d5d9" in result.stdout


def test_local_setup_explicit_deps_root_overrides_pod_local(tmp_path: Path) -> None:
    # An explicit HYPERLOOM_DEPS_ROOT still wins over the pod-local default.
    deps = tmp_path / "explicit-deps"
    env = _clean_base_env()
    env["HYPERLOOM_DEPS_ROOT"] = str(deps)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--session-dir", str(tmp_path / "sess")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert f"HYPERLOOM_DEPS_ROOT={deps}" in result.stdout
    assert str(deps / "Primus-Claw") in result.stdout


# install-harden: loud USER_DATA_PATH fallback notice + flock around the
# shared source-mirror clone/build region.

# ``--help`` runs the USER_DATA_PATH resolution + fallback notice then exits 0
# before any heavy install work — the fast hermetic way to test that path.
_HELP_SCRIPTS = {
    "inference_optimizer_install": IO_INSTALL,
    "kernel_agent_install": KA_INSTALL,
    "local_setup": SCRIPT,
}


def _run_help(
    script: Path, tmp_path: Path, user_data_path: str | None
) -> subprocess.CompletedProcess[str]:
    """Run ``bash <script> --help`` with a hermetic environment."""
    env = _clean_base_env()
    repo_stub = tmp_path / "repo_stub"
    repo_stub.mkdir(exist_ok=True)
    env["REPO_ROOT"] = str(repo_stub)
    env["MAGPIE_PYTHON"] = "python3"
    env["GEAK_RAG_INDEX_DEVICE"] = "cpu"
    if user_data_path is not None:
        env["USER_DATA_PATH"] = user_data_path
    return subprocess.run(
        ["bash", str(script), "--help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@pytest.mark.parametrize("script", list(_HELP_SCRIPTS.values()), ids=list(_HELP_SCRIPTS))
def test_install_warns_loudly_when_user_data_path_unset(script: Path, tmp_path: Path) -> None:
    result = _run_help(script, tmp_path, user_data_path=None)
    assert result.returncode == 0, result.stderr + result.stdout
    assert _FALLBACK_WARNING in result.stderr, result.stderr


@pytest.mark.parametrize("script", list(_HELP_SCRIPTS.values()), ids=list(_HELP_SCRIPTS))
def test_install_silent_when_user_data_path_set(script: Path, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    result = _run_help(script, tmp_path, user_data_path=str(data_root))
    assert result.returncode == 0, result.stderr + result.stdout
    assert _FALLBACK_WARNING not in result.stderr, result.stderr


@pytest.mark.parametrize("script", [IO_INSTALL, KA_INSTALL], ids=["inference_optimizer", "kernel_agent"])
def test_install_does_not_reference_default_path_when_set(script: Path, tmp_path: Path) -> None:
    # When USER_DATA_PATH is set, install.sh --help output must not reference the fallback root.
    data_root = tmp_path / "data"
    result = _run_help(script, tmp_path, user_data_path=str(data_root))
    assert result.returncode == 0, result.stderr + result.stdout
    combined = result.stdout + result.stderr
    assert _DEFAULT_USER_DATA_PATH not in combined, combined


@pytest.mark.parametrize(
    "script",
    [IO_INSTALL, KA_INSTALL, SCRIPT, PREFLIGHT_KB],
    ids=["inference_optimizer", "kernel_agent", "local_setup", "preflight_kb"],
)
def test_all_scripts_emit_user_data_path_fallback_notice(script: Path) -> None:
    # Grep-level guard (covers preflight_kb.sh, which has no --help entrypoint).
    text = script.read_text(encoding="utf-8")
    assert "_user_data_was_set" in text, script
    assert _FALLBACK_WARNING in text, script


@pytest.mark.parametrize("script", [IO_INSTALL, KA_INSTALL], ids=["inference_optimizer", "kernel_agent"])
def test_install_scripts_guard_mirror_writes_with_flock(script: Path) -> None:
    # Both install.sh entrypoints flock the mirror region and pass HYPERLOOM_INSTALL_LOCK_HELD to avoid self-deadlock.
    text = script.read_text(encoding="utf-8")
    assert ".install.lock" in text, script
    assert "exec 9>" in text, script
    assert "flock 9" in text, script
    assert "HYPERLOOM_INSTALL_LOCK_HELD" in text, script
    # The lock must live in the pod-local open-source root it guards, not on the
    # shared $HYPERLOOM_RUNTIME_DIR — otherwise separate pod roots block each other.
    assert 'exec 9>"${_open_source_root}/.install.lock"' in text, script
    assert '${HYPERLOOM_RUNTIME_DIR}/.install.lock' not in text, script


def test_flock_serializes_concurrent_critical_sections(tmp_path: Path) -> None:
    # Real serialization check of the installer idiom: two workers must not interleave.
    if shutil.which("flock") is None:
        pytest.skip("flock not available on this host")
    lock = tmp_path / ".install.lock"
    order = tmp_path / "order"
    worker_src = (
        'exec 9>"$1"\n'
        "flock 9\n"
        'echo "start-$2" >> "$3"\n'
        "sleep 0.5\n"
        'echo "end-$2" >> "$3"\n'
    )
    procs = [
        subprocess.Popen(["bash", "-c", worker_src, "worker", str(lock), tag, str(order)])
        for tag in ("A", "B")
    ]
    for proc in procs:
        assert proc.wait(timeout=30) == 0
    lines = order.read_text(encoding="utf-8").split()
    assert lines in (
        ["start-A", "end-A", "start-B", "end-B"],
        ["start-B", "end-B", "start-A", "end-A"],
    ), f"flock did not serialize critical sections: {lines}"


# pin-dependency-shas: install.sh must clone Magpie / InferenceX pinned to a
# commit SHA via the SHA-aware fetch-checkout dance, and the Magpie in-place
# patch must fail-loud on a genuine failure (strict default) while staying
# soft on a benign no-op. Grep/static-level guards on the script text.


def test_io_install_pins_magpie_and_inferencex_to_commit_sha() -> None:
    # Both deps pinned to a full 40-char SHA, operator-overridable; immune to HEAD drift (bugs.md §C #1).
    text = IO_INSTALL.read_text(encoding="utf-8")
    assert (
        '_open_source_root="${HYPERLOOM_OPEN_SOURCE_ROOT:-${TMPDIR:-/tmp}/hyperloom/open-source-repos}"'
        in text
    )
    assert 'MAGPIE_DIR="${MAGPIE_DIR:-${_open_source_root}/Magpie}"' in text
    assert 'INFERENCEX_DEFAULT_DIR="${INFERENCEX_DEFAULT_DIR:-${_open_source_root}/InferenceX}"' in text
    assert "export HYPERLOOM_OPEN_SOURCE_ROOT" not in text
    assert re.search(
        r'^MAGPIE_REF="\$\{MAGPIE_REF:-[0-9a-fA-F]{40}\}"', text, re.M
    ), "MAGPIE_REF must default to a full 40-char commit SHA and be overridable"
    assert re.search(
        r'^INFERENCEX_REF="\$\{INFERENCEX_REF:-[0-9a-fA-F]{40}\}"', text, re.M
    ), "INFERENCEX_REF must default to a full 40-char commit SHA and be overridable"


def test_io_install_uses_sha_aware_fetch_checkout_for_both_deps() -> None:
    # The SHA-aware fetch-checkout dance must be wired into both ensure_magpie and ensure_inferencex.
    text = IO_INSTALL.read_text(encoding="utf-8")
    assert "^[0-9a-fA-F]{7,40}$" in text, "missing raw-SHA detection regex"
    assert 'fetch --depth 1 origin "$ref"' in text, "missing shallow SHA fetch"
    assert "checkout -q FETCH_HEAD" in text, "missing detached SHA checkout"
    assert (
        'git_fetch_pinned "$MAGPIE_REPO" "$MAGPIE_DIR" "$MAGPIE_REF" "Magpie"' in text
    ), "ensure_magpie must clone via the pinned fetch-checkout helper"
    assert (
        'git_fetch_pinned "$INFERENCEX_REPO" "$INFERENCEX_PATH" "$INFERENCEX_REF" "InferenceX"'
        in text
    ), "ensure_inferencex must clone via the pinned fetch-checkout helper"
    # Regression guard: no unpinned clone of latest HEAD.
    assert 'git clone --depth 1 "$MAGPIE_REPO"' not in text
    assert 'git clone --depth 1 "$INFERENCEX_REPO"' not in text


def test_io_install_magpie_patch_strict_default_with_benign_no_op_soft() -> None:
    # A GENUINE atomic-patch failure (race unmitigated) must fail-loud by
    # default; a benign no-op (missing benchmarker tree) must still
    # warn-and-continue. Both PATCH_MAGPIE and MAGPIE_PATCH_STRICT overrides
    # are honoured and parsed as boolean-ish strings.
    text = IO_INSTALL.read_text(encoding="utf-8")
    assert "Magpie #C1 patch OK" in text, "success log must be preserved"
    assert "PATCH_MAGPIE=0" in text, "PATCH_MAGPIE=0 override must be preserved"
    # Genuine failure aborts (strict default).
    assert 'die "Magpie atomic-write patch GENUINELY failed' in text
    # Strict mode is downgradable to a warning via a falsy MAGPIE_PATCH_STRICT.
    assert "MAGPIE_PATCH_STRICT" in text
    assert 'is_falsy "${MAGPIE_PATCH_STRICT:-1}"' in text
    # Benign no-op (missing tree) still warns and continues.
    assert "Magpie atomic-write patch skipped" in text, "missing benign no-op warn"
    # Boolean-ish parsing helper replaces the brittle numeric -eq test.
    assert 'is_falsy "${PATCH_MAGPIE:-1}"' in text
