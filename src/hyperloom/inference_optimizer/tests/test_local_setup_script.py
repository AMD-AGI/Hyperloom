# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "local_setup.sh"
IO_INSTALL = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"
KA_INSTALL = REPO_ROOT / "src" / "hyperloom" / "agents" / "kernel" / "scripts" / "install.sh"
BAREMETAL = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install_baremetal.sh"
PREFLIGHT_KB = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "preflight_kb.sh"

# The default that all installers fall back to when USER_DATA_PATH is unset.
_DEFAULT_USER_DATA_PATH = "/workspace/hyperloom"
# Substring of the loud fallback notice each script prints to stderr.
_FALLBACK_WARNING = "USER_DATA_PATH not set"
_CANONICAL_KERNEL_AGENT_ROOT = "src/hyperloom/agents/kernel"
_CANONICAL_KERNEL_AGENT_INSTALL = f"{_CANONICAL_KERNEL_AGENT_ROOT}/scripts/install.sh"

# Strip these host-leaked env vars so each test runs hermetically.
_HOST_LEAK_VARS = (
    "REPO_ROOT",
    "USER_DATA_PATH",
    "HYPERLOOM_RUNTIME_DIR",
    "HYPERLOOM_DEPS_ROOT",
    "LOCAL_SETUP_ENV",
    "KERNEL_FORGE_REPO",
    "FORGE_PATH",
    "KERNEL_FORGE_ROOT",
    "KERNEL_FORGE_PATH",
    "SAFE_API_KEY",
    "OPENAI_BASE_URL",
)


def _clean_base_env() -> dict[str, str]:
    """Inherited env with all script-consumed vars stripped out."""
    run_env = os.environ.copy()
    for var in _HOST_LEAK_VARS:
        run_env.pop(var, None)
    return run_env


def test_skill_guidance_uses_in_tree_kernel_agent_paths() -> None:
    """Agent-facing launch docs must not recreate the retired sibling checkout path."""
    guidance_files = set(REPO_ROOT.rglob("SKILL.md"))
    guidance_files.update((REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "references").glob("*.md"))
    guidance_files.update((REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets").glob("*.example"))
    assert guidance_files

    offenders: list[str] = []
    stale_fragments = (
        "$REPO_ROOT/" + "kernel-agent",
        "${REPO_ROOT}/" + "kernel-agent",
        "kernel-agent" + "/scripts/install.sh",
        "kernel-agent" + "/tools/",
        "kernel-agent" + "/skills/",
    )
    for path in sorted(guidance_files):
        text = path.read_text(encoding="utf-8")
        if any(fragment in text for fragment in stale_fragments):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "stale kernel-agent path guidance in: " + ", ".join(offenders)

    inference_skill = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "SKILL.md"
    kernel_skill = REPO_ROOT / "src" / "hyperloom" / "agents" / "kernel" / "SKILL.md"
    setup_example = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "setup_env.sh.example"
    assert _CANONICAL_KERNEL_AGENT_INSTALL in inference_skill.read_text(encoding="utf-8")
    assert _CANONICAL_KERNEL_AGENT_INSTALL in kernel_skill.read_text(encoding="utf-8")
    assert f'export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/{_CANONICAL_KERNEL_AGENT_ROOT}"' in setup_example.read_text(
        encoding="utf-8"
    )


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
    forge = _git_repo(remotes / "KernelForge", {"OOB/README.md": "oob\n"})

    result = _run_local_setup(
        tmp_path,
        env={
            "KERNEL_FORGE_REPO": str(forge),
            "SAFE_API_KEY": secret,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    env_file = tmp_path / "session" / "runtime" / "local-setup.env.sh"
    assert env_file.exists()
    env_text = env_file.read_text(encoding="utf-8")
    assert f"export REPO_ROOT='{REPO_ROOT}'" in env_text
    assert f"export FORGE_PATH='{tmp_path / 'deps' / 'KernelForge'}'" in env_text
    assert f"export KERNEL_FORGE_ROOT='{tmp_path / 'deps' / 'KernelForge'}'" in env_text
    assert "INFERENCEX_PATH" not in env_text
    assert "TRACELENS_ROOT" not in env_text
    # The env file must export the canonical open-source-root key so
    # install.sh / paths / handler / tool resolve the SAME default root.
    assert f"export HYPERLOOM_OPEN_SOURCE_ROOT='{tmp_path / 'deps'}'" in env_text
    assert secret not in env_text


def test_local_setup_respects_existing_dependency_paths(tmp_path: Path) -> None:
    existing_forge = tmp_path / "existing" / "KernelForge"
    existing_forge.mkdir(parents=True)

    result = _run_local_setup(
        tmp_path,
        env={
            "FORGE_PATH": str(existing_forge),
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    env_text = (tmp_path / "session" / "runtime" / "local-setup.env.sh").read_text(encoding="utf-8")
    assert f"export FORGE_PATH='{existing_forge}'" in env_text
    assert not (tmp_path / "deps" / "KernelForge").exists()


def test_local_setup_dry_run_does_not_write_or_leak_secret(tmp_path: Path) -> None:
    secret = "ak-super-secret-value"
    result = _run_local_setup(
        tmp_path,
        "--dry-run",
        env={"SAFE_API_KEY": secret, "OPENAI_BASE_URL": "https://example.test/v1"},
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert not (tmp_path / "session" / "runtime" / "local-setup.env.sh").exists()
    assert "install.sh" not in result.stdout
    assert "Open this folder in Cursor" in result.stdout
    assert "source ${" not in result.stdout
    assert f"source '{tmp_path / 'session' / 'runtime' / 'local-setup.env.sh'}'" in result.stdout
    assert f"export USER_DATA_PATH='{tmp_path / 'session'}'" in result.stdout
    assert "@src/hyperloom/inference_optimizer/SKILL.md" in result.stdout
    assert "Optimize inference for this workload" in result.stdout


def test_local_setup_deps_root_stays_pod_local_under_session_dir(tmp_path: Path) -> None:
    # Deps root must NOT follow --session-dir: it stays on a pod-local base
    # so a shared session tree never collocates concurrent checkouts. The base
    # is a non-ephemeral pod-internal dir (NOT /tmp).
    session_dir = tmp_path / "custom-session"
    deps_base = tmp_path / "podlocal"
    deps_base.mkdir()
    env = _clean_base_env()
    env["HYPERLOOM_OPEN_SOURCE_ROOT"] = str(deps_base)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--session-dir", str(session_dir)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    expected_deps = deps_base
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"HYPERLOOM_DEPS_ROOT={expected_deps}" in result.stdout
    assert str(session_dir / "runtime" / "open-source-repos") not in result.stdout
    assert str(expected_deps / "KernelForge") in result.stdout


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
    assert str(deps / "KernelForge") in result.stdout


def test_local_setup_custom_deps_root_env_exports_matching_open_source_root(tmp_path: Path) -> None:
 # with ONLY --deps-root set (no HYPERLOOM_OPEN_SOURCE_ROOT), the written
    # env must export HYPERLOOM_OPEN_SOURCE_ROOT == the custom deps root so a
    # sourcing consumer (install/paths/handler/tool) resolves the SAME default.
    remotes = tmp_path / "remotes"
    forge = _git_repo(remotes / "KernelForge", {"OOB/README.md": "oob\n"})
    custom_deps = tmp_path / "custom-deps"
    run_env = _clean_base_env()
    run_env.update(
        {
            "USER_DATA_PATH": str(tmp_path / "session"),
            "KERNEL_FORGE_REPO": str(forge),
        },
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "--deps-root", str(custom_deps)],
        cwd=REPO_ROOT,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    env_text = (tmp_path / "session" / "runtime" / "local-setup.env.sh").read_text(encoding="utf-8")
    assert f"export HYPERLOOM_DEPS_ROOT='{custom_deps}'" in env_text
    assert f"export HYPERLOOM_OPEN_SOURCE_ROOT='{custom_deps}'" in env_text
    assert f"export FORGE_PATH='{custom_deps / 'KernelForge'}'" in env_text


def test_local_setup_exports_open_source_root_in_current_shell(tmp_path: Path) -> None:
 # HYPERLOOM_OPEN_SOURCE_ROOT must be exported into the running process
    # (not just the env file) so a same-shell install/optimize that skips
    # sourcing still sees the custom deps root. --dry-run + a probe that echoes
    # the exported var back out.
    custom_deps = tmp_path / "custom-deps"
    env = _clean_base_env()
    env["HYPERLOOM_DEPS_ROOT"] = str(custom_deps)
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{SCRIPT}" --dry-run >/dev/null 2>&1 || true; '
            'echo "PROBE_OSR=${HYPERLOOM_OPEN_SOURCE_ROOT:-UNSET}"',
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert f"PROBE_OSR={custom_deps}" in result.stdout, result.stdout + result.stderr


# install-harden: loud USER_DATA_PATH fallback notice + flock around the
# shared source-mirror clone/build region.

# ``--help`` runs the USER_DATA_PATH resolution + fallback notice then exits 0
# before any heavy install work — the fast hermetic way to test that path.
_HELP_SCRIPTS = {
    "inference_optimizer_install": IO_INSTALL,
    "kernel_agent_install": KA_INSTALL,
    "local_setup": SCRIPT,
}


def _run_help(script: Path, tmp_path: Path, user_data_path: str | None) -> subprocess.CompletedProcess[str]:
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


def test_ka_install_repo_root_fallback_tracks_src_layout() -> None:
    """The standalone kernel-agent installer must find the repo root from its in-tree path."""
    text = KA_INSTALL.read_text(encoding="utf-8")
    assert 'KERNEL_AGENT_ROOT="${KERNEL_AGENT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"' in text
    assert 'REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"' in text
    assert 'REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"' not in text
    assert (KA_INSTALL.parent / "..").resolve() == REPO_ROOT / _CANONICAL_KERNEL_AGENT_ROOT
    assert (KA_INSTALL.parent / "../../../../..").resolve() == REPO_ROOT


def test_baremetal_defaults_kernel_backend_order_to_geak() -> None:
    """Bare-metal installs default to whole-pipeline GEAK so forge is opt-in."""
    text = BAREMETAL.read_text(encoding="utf-8")
    assert 'export KERNEL_OPT_BACKEND_ORDER="${KERNEL_OPT_BACKEND_ORDER:-geak}"' in text
    assert 'printf \'export KERNEL_OPT_BACKEND_ORDER=%q\\n\' "$KERNEL_OPT_BACKEND_ORDER"' in text
    assert "kernel_backend_order_includes_forge()" in text
    assert "skipping local_setup.sh" in text
    assert "forge backend requested" in text
    assert 'if kernel_backend_order_includes_forge; then' in text


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


def test_ka_install_classifies_tracelens_override_by_path_not_env_presence() -> None:
 # the persistent kernel-agent env re-exports the
    # resolved default TRACELENS_ROOT, so a presence-only classifier
    # (${TRACELENS_ROOT:+1}) would treat the default as an operator override and
    # skip managed clone/realign. Override must be decided by comparing against
    # the pod-local default path.
    text = KA_INSTALL.read_text(encoding="utf-8")
    assert '_tracelens_root_was_set="${TRACELENS_ROOT:+1}"' not in text, (
        "presence-only override classifier must be removed"
    )
    assert '_tracelens_default_root="$(_canonicalize_path "${_open_source_root}/TraceLens")"' in text
    assert '[ "$(_canonicalize_path "${TRACELENS_ROOT}")" != "${_tracelens_default_root}" ]' in text, (
        "override must be classified by canonicalized path inequality with the default"
    )


# Behavioral coverage of ensure_tracelens(): extract the function verbatim from
# install.sh and run it against a real local git remote with heavy deps stubbed,
# so the default-path clone+pin and stale-SHA realign are exercised end-to-end.
_ENSURE_TRACELENS_SHIM = """
set -euo pipefail
log(){ :; }
warn(){ :; }
die(){ echo "DIE:$*" >&2; exit 1; }
run(){ "$@"; }
verify_die(){ echo "VDIE:$*" >&2; exit 1; }
_pip_install_editable(){ return 0; }
_tracelens_internal_enabled(){ return 1; }
TraceLens_generate_perf_report_pytorch_inference(){ :; }
DRY_RUN=0
CHECK_ONLY=0
TRACELENS_INTERNAL_ROOT=""
__ENSURE_TRACELENS_FUNC__
ensure_tracelens
"""


def _extract_ensure_tracelens() -> str:
    text = KA_INSTALL.read_text(encoding="utf-8")
    match = re.search(r"^ensure_tracelens\(\) \{.*?^\}", text, re.S | re.M)
    assert match, "could not extract ensure_tracelens() from install.sh"
    body = match.group(0)
    # Guard the brittle non-greedy `^}` capture against silent truncation: the
    # function must include the atomic clone tail. If someone adds a column-0 `}`
    # mid-function, this fails loudly instead of testing a partial body.
    assert 'mv "$_tl_tmp"' in body, "ensure_tracelens() extraction truncated early"
    return body


def _extract_tracelens_classify_block() -> str:
    # The top-level override classifier (_canonicalize_path + _tracelens_root_was_set
    # + the TRACELENS_ROOT default assignment) that ensure_tracelens() consumes.
    text = KA_INSTALL.read_text(encoding="utf-8")
    match = re.search(
        r"^_canonicalize_path\(\) \{.*?^TRACELENS_ROOT=\"\$\{TRACELENS_ROOT:-\$\{_open_source_root\}/TraceLens\}\"",
        text,
        re.S | re.M,
    )
    assert match, "could not extract the TRACELENS override classifier from install.sh"
    return match.group(0)


def _allow_sha_fetch(repo: Path) -> None:
    # Local file remotes reject on-demand SHA fetches unless explicitly allowed;
    # install.sh pins via `fetch origin <sha>`.
    subprocess.run(
        ["git", "-C", str(repo), "config", "uploadpack.allowReachableSHA1InWant", "true"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "uploadpack.allowAnySHA1InWant", "true"],
        check=True,
    )


def _two_commit_tracelens_remote(tmp_path: Path) -> tuple[Path, str, str]:
    remote = _git_repo(tmp_path / "remote" / "TraceLens", {"README.md": "v1\n"})
    old_sha = subprocess.run(
        ["git", "-C", str(remote), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    (remote / "README.md").write_text("v2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(remote), "commit", "-aqm", "v2"], check=True)
    pin_sha = subprocess.run(
        ["git", "-C", str(remote), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    _allow_sha_fetch(remote)
    return remote, old_sha, pin_sha


def _run_ensure_tracelens(tmp_path: Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = _ENSURE_TRACELENS_SHIM.replace("__ENSURE_TRACELENS_FUNC__", _extract_ensure_tracelens())
    env = _clean_base_env()
    env.update(extra_env)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_ka_ensure_tracelens_clones_and_pins_missing_default(tmp_path: Path) -> None:
 # default path (TRACELENS_ROOT=<default>, not an override) that
    # is MISSING must be cloned and pinned to TRACELENS_REF, not fail.
    remote, _old_sha, pin_sha = _two_commit_tracelens_remote(tmp_path)
    default_root = tmp_path / "deps" / "TraceLens"

    result = _run_ensure_tracelens(
        tmp_path,
        {
            "TRACELENS_ROOT": str(default_root),
            "_tracelens_root_was_set": "",
            "TRACELENS_REPO": str(remote),
            "TRACELENS_REF": pin_sha,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    head = subprocess.run(
        ["git", "-C", str(default_root), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert head == pin_sha, f"missing default not cloned+pinned: {head} != {pin_sha}"


def test_ka_ensure_tracelens_realigns_stale_default(tmp_path: Path) -> None:
 # default path already present but on a stale SHA must be
    # fetched/checked out to TRACELENS_REF.
    remote, old_sha, pin_sha = _two_commit_tracelens_remote(tmp_path)
    default_root = tmp_path / "deps" / "TraceLens"
    default_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(default_root)], check=True)
    subprocess.run(["git", "-C", str(default_root), "checkout", "-q", old_sha], check=True)

    result = _run_ensure_tracelens(
        tmp_path,
        {
            "TRACELENS_ROOT": str(default_root),
            "_tracelens_root_was_set": "",
            "TRACELENS_REPO": str(remote),
            "TRACELENS_REF": pin_sha,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    head = subprocess.run(
        ["git", "-C", str(default_root), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert head == pin_sha, f"stale default not realigned: {head} != {pin_sha}"


def test_ka_ensure_tracelens_override_fails_on_incomplete_checkout(tmp_path: Path) -> None:
 # a non-default operator override that exists but lacks .git
    # (half-done clone) must fail fast, not be silently accepted.
    remote, _old_sha, pin_sha = _two_commit_tracelens_remote(tmp_path)
    override = tmp_path / "operator" / "TraceLens"
    override.mkdir(parents=True)
    (override / "README.md").write_text("not a git tree\n", encoding="utf-8")

    result = _run_ensure_tracelens(
        tmp_path,
        {
            "TRACELENS_ROOT": str(override),
            "_tracelens_root_was_set": "1",
            "TRACELENS_REPO": str(remote),
            "TRACELENS_REF": pin_sha,
        },
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert "not a git checkout" in result.stderr, result.stderr


# Integration: run the top-level override classifier (which computes
# _tracelens_root_was_set from the canonicalized TRACELENS_ROOT) TOGETHER with
# ensure_tracelens(), so a regression in the classifier block — not just the
# function body — is caught end-to-end.
_CLASSIFY_ENSURE_SHIM = """
set -euo pipefail
log(){ :; }
warn(){ :; }
die(){ echo "DIE:$*" >&2; exit 1; }
run(){ "$@"; }
verify_die(){ echo "VDIE:$*" >&2; exit 1; }
_pip_install_editable(){ return 0; }
_tracelens_internal_enabled(){ return 1; }
TraceLens_generate_perf_report_pytorch_inference(){ :; }
DRY_RUN=0
CHECK_ONLY=0
TRACELENS_INTERNAL_ROOT=""
_open_source_root="${HL_OPEN_SOURCE_ROOT}"
__CLASSIFY_BLOCK__
__ENSURE_TRACELENS_FUNC__
ensure_tracelens
echo "WAS_SET=[${_tracelens_root_was_set:-}]"
"""


def _run_classify_and_ensure(
    tmp_path: Path, open_source_root: Path, extra_env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    script = (
        _CLASSIFY_ENSURE_SHIM
        .replace("__CLASSIFY_BLOCK__", _extract_tracelens_classify_block())
        .replace("__ENSURE_TRACELENS_FUNC__", _extract_ensure_tracelens())
    )
    env = _clean_base_env()
    env["HL_OPEN_SOURCE_ROOT"] = str(open_source_root)
    env.update(extra_env)
    return subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_ka_classifier_treats_default_env_as_managed_and_realigns(tmp_path: Path) -> None:
    # End-to-end: TRACELENS_ROOT=<default> (trailing slash) must be classified as
    # NOT-an-override by the top-level block, so ensure_tracelens realigns the
    # stale default checkout to TRACELENS_REF.
    remote, old_sha, pin_sha = _two_commit_tracelens_remote(tmp_path)
    open_source_root = tmp_path / "deps"
    default_root = open_source_root / "TraceLens"
    default_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(default_root)], check=True)
    subprocess.run(["git", "-C", str(default_root), "checkout", "-q", old_sha], check=True)

    result = _run_classify_and_ensure(
        tmp_path,
        open_source_root,
        {
            "TRACELENS_ROOT": str(default_root) + "/",
            "TRACELENS_REPO": str(remote),
            "TRACELENS_REF": pin_sha,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "WAS_SET=[]" in result.stdout, result.stdout
    head = subprocess.run(
        ["git", "-C", str(default_root), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert head == pin_sha, f"default-in-env not realigned by classifier+ensure: {head} != {pin_sha}"


def test_ka_classifier_treats_nondefault_as_override(tmp_path: Path) -> None:
    # End-to-end: a non-default TRACELENS_ROOT is classified as an override
    # (_tracelens_root_was_set=1); an existing git checkout is adopted as-is and
    # NOT realigned.
    remote, _old_sha, pin_sha = _two_commit_tracelens_remote(tmp_path)
    open_source_root = tmp_path / "deps"
    override = tmp_path / "operator" / "TraceLens"
    override.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(override)], check=True)
    first_sha = subprocess.run(
        ["git", "-C", str(override), "rev-list", "--max-parents=0", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(override), "checkout", "-q", first_sha], check=True)

    result = _run_classify_and_ensure(
        tmp_path,
        open_source_root,
        {
            "TRACELENS_ROOT": str(override),
            "TRACELENS_REPO": str(remote),
            "TRACELENS_REF": pin_sha,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "WAS_SET=[1]" in result.stdout, result.stdout
    head = subprocess.run(
        ["git", "-C", str(override), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert head == first_sha, "override must be adopted as-is, not realigned"


def test_ka_classifier_clones_and_pins_missing_default(tmp_path: Path) -> None:
    # End-to-end through the classifier: a MISSING default (TRACELENS_ROOT=<default>)
    # is NOT-an-override and ensure_tracelens must clone+pin it to TRACELENS_REF.
    remote, _old_sha, pin_sha = _two_commit_tracelens_remote(tmp_path)
    open_source_root = tmp_path / "deps"
    default_root = open_source_root / "TraceLens"

    result = _run_classify_and_ensure(
        tmp_path,
        open_source_root,
        {
            "TRACELENS_ROOT": str(default_root),
            "TRACELENS_REPO": str(remote),
            "TRACELENS_REF": pin_sha,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "WAS_SET=[]" in result.stdout, result.stdout
    head = subprocess.run(
        ["git", "-C", str(default_root), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert head == pin_sha, f"missing default not cloned+pinned via classifier: {head} != {pin_sha}"


def test_ka_ensure_tracelens_managed_rebuilds_non_git_tree(tmp_path: Path) -> None:
 # a MANAGED default path that exists but is NOT a git tree
    # (half-done/crashed clone) is dropped and rebuilt+pinned, so it never
    # lingers as an unusable tree. Mirrors the runtime self-heal
    # (_ensure_tracelens_checkout move-aside+re-clone) and local_setup.sh; the
    # earlier warn-and-preserve asymmetry is removed.
    remote, _old_sha, pin_sha = _two_commit_tracelens_remote(tmp_path)
    default_root = tmp_path / "deps" / "TraceLens"
    default_root.mkdir(parents=True)
    (default_root / "README.md").write_text("not a git tree\n", encoding="utf-8")

    result = _run_ensure_tracelens(
        tmp_path,
        {
            "TRACELENS_ROOT": str(default_root),
            "_tracelens_root_was_set": "",
            "TRACELENS_REPO": str(remote),
            "TRACELENS_REF": pin_sha,
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    # Rebuilt: now a real git checkout pinned to TRACELENS_REF, stale tree gone.
    assert (default_root / ".git").exists()
    head = subprocess.run(
        ["git", "-C", str(default_root), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert head == pin_sha, f"managed non-git tree not rebuilt+pinned: {head} != {pin_sha}"


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
    assert "${HYPERLOOM_RUNTIME_DIR}/.install.lock" not in text, script


def test_flock_serializes_concurrent_critical_sections(tmp_path: Path) -> None:
    # Real serialization check of the installer idiom: two workers must not interleave.
    if shutil.which("flock") is None:
        pytest.skip("flock not available on this host")
    lock = tmp_path / ".install.lock"
    order = tmp_path / "order"
    worker_src = 'exec 9>"$1"\nflock 9\necho "start-$2" >> "$3"\nsleep 0.5\necho "end-$2" >> "$3"\n'
    procs = [subprocess.Popen(["bash", "-c", worker_src, "worker", str(lock), tag, str(order)]) for tag in ("A", "B")]
    for proc in procs:
        assert proc.wait(timeout=30) == 0
    lines = order.read_text(encoding="utf-8").split()
    assert lines in (
        ["start-A", "end-A", "start-B", "end-B"],
        ["start-B", "end-B", "start-A", "end-A"],
    ), f"flock did not serialize critical sections: {lines}"


# pin-dependency-refs: install.sh must clone Magpie / InferenceX pinned to a
# deterministic ref, and Magpie must support commit SHA and tag refs.


def test_io_install_pins_magpie_release_commit_and_inferencex_commit_sha() -> None:
    # Magpie and InferenceX both default to full 40-char SHAs and avoid HEAD drift.
    text = IO_INSTALL.read_text(encoding="utf-8")
    assert '_open_source_root="${HYPERLOOM_OPEN_SOURCE_ROOT:-/opt/hyperloom/open-source-repos}"' in text
    assert 'MAGPIE_PATH="${MAGPIE_PATH:-${_open_source_root}/Magpie}"' in text
    assert 'INFERENCEX_DEFAULT_DIR="${INFERENCEX_DEFAULT_DIR:-${_open_source_root}/InferenceX}"' in text
    assert "export HYPERLOOM_OPEN_SOURCE_ROOT" not in text
    assert re.search(r'^MAGPIE_REF="\$\{MAGPIE_REF:-[0-9a-fA-F]{40}\}"', text, re.M), (
        "MAGPIE_REF must default to the public Magpie release commit SHA and be overridable"
    )
    assert re.search(r'^INFERENCEX_REF="\$\{INFERENCEX_REF:-[0-9a-fA-F]{40}\}"', text, re.M), (
        "INFERENCEX_REF must default to a full 40-char commit SHA and be overridable"
    )


def test_io_install_uses_sha_aware_fetch_checkout_for_both_deps() -> None:
    # SHA refs use fetch-checkout; tag refs use git clone --branch.
    text = IO_INSTALL.read_text(encoding="utf-8")
    assert "^[0-9a-fA-F]{7,40}$" in text, "missing raw-SHA detection regex"
    assert 'fetch --depth 1 origin "$ref"' in text, "missing shallow SHA fetch"
    assert "checkout -q FETCH_HEAD" in text, "missing detached SHA checkout"
    assert 'git clone --depth 1 --branch "$ref"' in text, "missing tag ref clone path"
    assert 'git_fetch_pinned "$MAGPIE_REPO" "$MAGPIE_PATH" "$MAGPIE_REF" "Magpie"' in text, (
        "ensure_magpie must clone via the pinned fetch-checkout helper"
    )
    assert 'git_fetch_pinned "$INFERENCEX_REPO" "$INFERENCEX_PATH" "$INFERENCEX_REF" "InferenceX"' in text, (
        "ensure_inferencex must clone via the pinned fetch-checkout helper"
    )
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


def test_baremetal_wheel_check_only_first_run_succeeds(tmp_path: Path) -> None:
    # Standalone wheel mode: a first-run --check-only (wheel not installed yet)
    # must report the missing package and exit 0, not die trying to locate
    # site-packages assets that do not exist yet.
    fake_py = tmp_path / "fakepython"
    # Stub interpreter: make `import hyperloom.inference_optimizer` (and its
    # importlib.util.find_spec probe) look unavailable, so bootstrap_wheel_install
    # takes the "package missing" path.
    fake_py.write_text(
        "#!/usr/bin/env bash\n"
        "exec /usr/bin/env python3 -c '\n"
        "import sys, runpy\n"
        "code = sys.argv[1] if len(sys.argv) > 1 else \"\"\n"
        "code = code.replace(\"hyperloom.inference_optimizer\", \"hyperloom._absent_io_\")\n"
        "exec(compile(code, \"<stub>\", \"exec\"))\n"
        "' \"$@\"\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)

    env = _clean_base_env()
    env["HYPERLOOM_INSTALL_SOURCE"] = "wheel"
    env["INFERENCE_OPTIMIZER_FORCE_PYTHON"] = "1"
    env["PYTHON"] = str(fake_py)
    env["USER_DATA_PATH"] = str(tmp_path / "udp")

    result = subprocess.run(
        ["bash", str(BAREMETAL), "--check-only", "--skip-base-check", "--yes"],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "cannot locate packaged assets" not in combined
    assert "hyperloom.inference_optimizer missing" in combined
