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
    # Default is open-source-only: the internal extension is NOT requested and
    # must not be cloned, exported, or activated via TL_EXTENSION.
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
    # An existing internal checkout provided via TRACELENS_INTERNAL_ROOT opts in.
    internal_checkout = _git_repo(
        tmp_path / "existing" / "TraceLens-internal", {"README.md": "internal\n"}
    )

    result = _run_local_setup(
        tmp_path,
        env={
            "PRIMUS_CLAW_REPO": str(primus),
            "INFERENCEX_REPO": str(inferencex),
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
    # TRACELENS_INTERNAL_ROOT set to a non-existent path: Hyperloom never clones
    # internal (no URL kept), so it must warn and fall back to open-source-only.
    remotes = tmp_path / "remotes"
    primus = _git_repo(remotes / "Primus-Claw", {"OOB/README.md": "oob\n"})
    inferencex = _git_repo(remotes / "InferenceX", {"README.md": "inferencex\n"})
    tracelens_public = _git_repo(remotes / "TraceLens", {"README.md": "tracelens\n"})

    result = _run_local_setup(
        tmp_path,
        env={
            "PRIMUS_CLAW_REPO": str(primus),
            "INFERENCEX_REPO": str(inferencex),
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
    assert str(expected_deps / "TraceLens") in result.stdout
    # Open-source-only by default: internal extension is not requested/cloned.
    assert str(expected_deps / "TraceLens-internal") not in result.stdout
    assert "c35c787ef31f0425fa0028a605ffc8c60a737c2c" in result.stdout


# ---------------------------------------------------------------------------
# install-harden (feature/xiaofei/install-path-and-flock):
#   A. loud USER_DATA_PATH fallback notice across all installers
#   B. flock around the shared source-mirror clone/build region
# ---------------------------------------------------------------------------

# Scripts that expose ``--help``. ``--help`` runs the top-of-file
# USER_DATA_PATH resolution (and its fallback notice) and then exits 0
# *before* any heavy / environment-coupled install work, so it is the fast,
# hermetic way to exercise exactly the fallback-notice code path. (A full
# ``--check-only`` / ``--dry-run`` run would clone Magpie, import
# inference_optimizer, chain to kernel-agent, etc. — far too heavy and
# flaky for a path-handling unit test.)
_HELP_SCRIPTS = {
    "inference_optimizer_install": IO_INSTALL,
    "kernel_agent_install": KA_INSTALL,
    "local_setup": SCRIPT,
}


def _run_help(
    script: Path, tmp_path: Path, user_data_path: str | None
) -> subprocess.CompletedProcess[str]:
    """Run ``bash <script> --help`` with a hermetic environment.

    ``REPO_ROOT`` points at an empty stub dir so no real ``.env`` is sourced;
    ``MAGPIE_PYTHON`` / ``GEAK_RAG_INDEX_DEVICE`` are pinned so kernel-agent's
    top-level probes do not shell out to rocm-smi or import Magpie/torch.
    ``USER_DATA_PATH`` is exported only when ``user_data_path`` is provided;
    otherwise it stays stripped (by ``_clean_base_env``) so the unset
    fallback branch fires.
    """
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
    # local_setup.sh is intentionally excluded: its --help text documents the
    # default path, so a literal-string check there is meaningless. The two
    # install.sh entrypoints do not mention it in --help output, so when
    # USER_DATA_PATH is set their output must not reference the fallback root.
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
    # All four installers must detect the unset case BEFORE applying the
    # default and warn loudly. Grep-level so it also covers preflight_kb.sh,
    # which has no --help entrypoint to drive functionally.
    text = script.read_text(encoding="utf-8")
    assert "_user_data_was_set" in text, script
    assert _FALLBACK_WARNING in text, script


@pytest.mark.parametrize("script", [IO_INSTALL, KA_INSTALL], ids=["inference_optimizer", "kernel_agent"])
def test_install_scripts_guard_mirror_writes_with_flock(script: Path) -> None:
    # Both install.sh entrypoints serialize the source-mirror clone/build
    # region with an fd-based flock on $HYPERLOOM_RUNTIME_DIR/.install.lock and
    # hand the lock to the chained installer via HYPERLOOM_INSTALL_LOCK_HELD so
    # the inference_optimizer -> kernel-agent chain cannot self-deadlock.
    text = script.read_text(encoding="utf-8")
    assert ".install.lock" in text, script
    assert "exec 9>" in text, script
    assert "flock 9" in text, script
    assert "HYPERLOOM_INSTALL_LOCK_HELD" in text, script


def test_flock_serializes_concurrent_critical_sections(tmp_path: Path) -> None:
    # Fast, real serialization check of the exact idiom the installers use
    # (`exec 9>LOCK; flock 9; <work>`): two workers contending on one lock
    # must run their critical sections one-at-a-time (no interleaving).
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


# ---------------------------------------------------------------------------
# pin-dependency-shas (feature/xiaofei/pin-dependency-shas):
#   inference_optimizer/scripts/install.sh must clone Magpie / InferenceX
#   pinned to a commit SHA via the SHA-aware fetch-checkout dance (mirroring
#   the GEAK_REF pin in kernel-agent/scripts/install.sh), and the Magpie
#   in-place patch step must be fail-soft (warn, not die) so a pinned
#   upstream-atomic Magpie does not abort every install.
#
# These are grep/static-level guards on the script text (the established
# pattern for the install.sh entrypoints — see
# test_install_scripts_guard_mirror_writes_with_flock). A full DRY_RUN run is
# intentionally not driven here: it imports inference_optimizer, resolves a
# ROCm python, and runs the torch gate, which is too heavy/host-coupled for a
# hermetic unit test (the DRY_RUN log is exercised manually in the PR notes).
# ---------------------------------------------------------------------------


def test_io_install_pins_magpie_and_inferencex_to_commit_sha() -> None:
    # Both deps are pinned to a full 40-char commit SHA and stay operator-
    # overridable via the ``${VAR:-<sha>}`` form (mirrors GEAK_REF). Pinning a
    # SHA — not a branch — is what makes a fresh install reproducible and
    # immune to upstream force-push / HEAD drift (the bugs.md §C #1 root cause).
    text = IO_INSTALL.read_text(encoding="utf-8")
    assert re.search(
        r'^MAGPIE_REF="\$\{MAGPIE_REF:-[0-9a-fA-F]{40}\}"', text, re.M
    ), "MAGPIE_REF must default to a full 40-char commit SHA and be overridable"
    assert re.search(
        r'^INFERENCEX_REF="\$\{INFERENCEX_REF:-[0-9a-fA-F]{40}\}"', text, re.M
    ), "INFERENCEX_REF must default to a full 40-char commit SHA and be overridable"


def test_io_install_uses_sha_aware_fetch_checkout_for_both_deps() -> None:
    # The clone helper must use the GEAK-style SHA-aware fetch-checkout dance
    # (a raw SHA cannot be passed to ``git clone --branch``) and must be wired
    # into BOTH ensure_magpie and ensure_inferencex. The old bare
    # ``git clone --depth 1 <repo>`` of latest HEAD must be gone.
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
    # Regression guard: no unpinned ``clone --depth 1 <repo>`` of latest HEAD.
    assert 'git clone --depth 1 "$MAGPIE_REPO"' not in text
    assert 'git clone --depth 1 "$INFERENCEX_REPO"' not in text


def test_io_install_magpie_patch_is_fail_soft() -> None:
    # Defense in depth: with MAGPIE_REF pinned to an upstream-atomic commit the
    # in-place patcher finds no legacy block and returns False. That must
    # warn-and-continue (no-op), not die and abort the install. The
    # PATCH_MAGPIE=0 hard-skip override must be preserved.
    text = IO_INSTALL.read_text(encoding="utf-8")
    assert "Magpie atomic-write patch did not apply" in text, "missing fail-soft warn"
    assert "Magpie #C1 patch OK" in text, "success log must be preserved"
    assert "PATCH_MAGPIE=0" in text, "PATCH_MAGPIE=0 override must be preserved"
    # The previous fail-loud ``die`` on patch failure must be gone.
    assert 'die "Magpie atomic-write patch failed' not in text
