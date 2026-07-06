# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Static guard: only back-compat surfaces in :data:`ALLOWED_FILES` may mention ``extra_sglang_args`` (renamed to ``extra_server_args``)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# Repo root: this file's great-great-grandparent.
_REPO_ROOT = Path(__file__).resolve().parents[2]


# Allowlist: repo-relative POSIX path -> justification. Meant to shrink, not grow.
ALLOWED_FILES: dict[str, str] = {
    # Canonical compat helper (tree-reform.MD §7/P2.1): the single reader of
    # the legacy key now lives in hyperloom.common; the compat + sub-agent
    # shims re-export it and no longer carry the literal themselves.
    "src/hyperloom/common/payload_aliases.py": "canonical payload-aliases compat helper (tree-reform.MD §7)",
    # Compat helper modules (docstrings naming both keys; re-export the
    # canonical helper above).
    "inference_optimizer/compat/__init__.py": "compat package docstring naming the rename",
    "inference_optimizer/compat/payload_aliases.py": "compat re-export shim docstring names the rename",
    # Sub-agent reader site (kernel-agent) that falls back to the legacy key.
    "kernel-agent/tools/kernel_optimization.py": "kernel-agent reader site falls back to candidate_extra_sglang_args for legacy envelopes",
    # Back-compat injection points in production code (renamed kwarg
    # alias on GridVariant, walk-and-rewrite on SharedState loader).
    "inference_optimizer/actions/_meta/replay_warm_recipe.yaml": "warm-replay internal action schema mirrors RecipeKB best_config "
    "field names",
    "inference_optimizer/orchestrator/action_executors/_grid_base.py": "GridVariant(extra_sglang_args=...) back-compat kwarg (tree-reform.MD P2.2: GridVariant extracted from _grid_runner.py to the _grid_base sibling)",
    "inference_optimizer/orchestrator/shared_state.py": "_migrate_legacy_extra_sglang_args_keys walker + state.json transform",
    "inference_optimizer/orchestrator/research_hints.py": "priors-match scorer builds a token blob from variant fields and "
    "reads the legacy extra_sglang_args key alongside the canonical "
    "extra_server_args so pre-rename variant dicts still match",
    "inference_optimizer/orchestrator/optimization_journal.py": "journal classification reads existing stack/variant args fields",
    # legacy v0.6 breakdown reader walks raw optimization_stack which can
    # carry the pre-rename candidate_extra_sglang_args field; the emitted
    # key is the canonical extra_server_args.
    "inference_optimizer/breakdown/legacy_collectors.py": "legacy v0.6 reader: raw optimization_stack carries pre-rename "
    "candidate_extra_sglang_args (breakdown loads state without the "
    "SharedState key migration)",
    # CI transform reads legacy-keyed session_breakdown.json artefacts.
    "ci/transform_to_session_summary_v2.py": "legacy session-breakdown JSON reader (operator-side back-compat)",
    "ci/test_ci_transform_v2.py": "unit tests assert the ci legacy reader migrates extra_sglang_args "
    "-> extra_server_args",
    # Prompt / orientation text that names both keys explicitly so the
    # LLM and any human reader of the prompt knows the alias exists
    # for one release.
    # Pytest marker registration mentions the legacy name in the
    # marker's description.
    "pyproject.toml": "pytest marker description naming the legacy alias",
    # Test files: compat helper coverage + back-compat regression
    # tests + this guard itself + the per-sub-agent shim tests.
    "inference_optimizer/tests/test_payload_aliases.py": "compat helper test surface",
    "inference_optimizer/tests/test_back_compat_legacy_field_name.py": "back-compat regression tests",
    "inference_optimizer/tests/test_coordinator_kb_writes.py": "recipe write-back regression test asserts best_config / "
    "what_worked emit the KB-legacy extra_sglang_args field read "
    "from canonical state",
    "inference_optimizer/tests/test_no_legacy_writer_sites.py": "this guard's allowlist + docstring",
    "inference_optimizer/tests/test_optimization_journal.py": "journal tests cover stack entries carrying legacy args field",
    "inference_optimizer/tests/test_warm_replay.py": "warm-replay tests mirror RecipeKB best_config field names",
    "kernel-agent/tools/test_payload_aliases_shim.py": "kernel-agent shim test surface",
    "robustness-agent/tests/test_payload_aliases_shim.py": "robustness-agent shim test surface",
    # Readers that funnel external-envelope payloads through
    # ``read_extra_server_args`` mention the legacy key inline in the
    # surrounding code comment to document why the call goes through
    # the helper.
    "inference_optimizer/orchestrator/coordinator.py":
        "comments explain the read_extra_server_args call at the LLM "
        "intent / sub-agent envelope read boundaries",
    # tree-reform.MD P2.2 3b-1: Coordinator method clusters extracted into
    # collaborator objects; the read_extra_server_args envelope-read comments /
    # KB best_config arg handling moved with them.
    "inference_optimizer/orchestrator/_resume.py": "resume/replay collaborator carries the coordinator envelope-read surface",
    "inference_optimizer/orchestrator/_writeback.py": "writeback collaborator carries the KB best_config / recipe-attrs arg surface",
    "inference_optimizer/orchestrator/_gating.py": "gating collaborator carries the sequence-denial envelope-read surface",
    "inference_optimizer/orchestrator/_dispatcher.py": "dispatcher collaborator carries the delegate/dispatch envelope-read surface",
    "inference_optimizer/orchestrator/_proposals.py": "proposals collaborator carries the KB recipe best_config arg surface",
    "inference_optimizer/orchestrator/_advisory.py": "advisory collaborator carries the proposed-variant arg surface",
    "inference_optimizer/orchestrator/_inline_actions.py": "inline-actions collaborator carries the inline delegate envelope-read surface",
    "inference_optimizer/orchestrator/_conversation.py": "conversation collaborator carries the inbox/context envelope-read surface",
    "inference_optimizer/orchestrator/_maintenance.py": "maintenance collaborator carries the checkpoint/observation arg surface",
    # tree-reform.MD P2.2 3b-2: Coordinator phase-handler clusters extracted
    # into collaborator objects; the read_extra_server_args envelope-read
    # comments / warm-recipe KB best_config arg handling moved with them.
    "inference_optimizer/orchestrator/_phase_machine.py": "phase-machine handler carries the cumulative-merge helper import",
    "inference_optimizer/orchestrator/_phase_prelude.py": "prelude handler carries the warm-recipe KB best_config/extra_sglang_args read+merge surface",
    "inference_optimizer/orchestrator/_phase_sweep.py": "sweep handler carries the cumulative-merge helper import",
    "inference_optimizer/orchestrator/_phase_close.py": "close handler carries the cumulative-merge helper import",
    "inference_optimizer/orchestrator/_phase_internal.py": "internal-tasks handler carries the cumulative-merge helper import",
    "inference_optimizer/orchestrator/_phase_kernel_stack.py": "kernel-stack handler carries the cumulative-merge helper import",
    "inference_optimizer/orchestrator/_phase_kernel.py": "kernel handler carries the cumulative-merge helper import",
    "inference_optimizer/orchestrator/_phase_explore.py": "explore handler carries the cumulative-merge helper import",
    "inference_optimizer/orchestrator/_phase_framework.py": "framework handler carries the cumulative-merge helper import",
    "inference_optimizer/orchestrator/result_recorder.py":
        "result-recording / fact-synthesis methods extracted verbatim from "
        "coordinator.py (phase 1B); same read_extra_server_args envelope-read "
        "boundaries as the coordinator they came from",
    "inference_optimizer/orchestrator/coordinator_helpers.py":
        "holds the extracted _merge_cumulative_extra_sglang_args helper "
        "that merges the legacy KB best_config arg stacks",
    "inference_optimizer/orchestrator/_kernel_decisions.py":
        "tree-reform.MD P2.2: kernel-decision write-owner functions extracted "
        "from kernel_request_handlers.py to this sibling module (re-exported "
        "back); carries _resolve_kernel_patch_identity's read_extra_server_args "
        "call + docstring naming the legacy extra_sglang_args alias",
    "inference_optimizer/orchestrator/kernel_request_handlers.py":
        "comments explain the read_extra_server_args call at the "
        "integrate_patch sub-agent envelope read boundary; also holds the "
        "kernel-decision write-owner functions folded back from the former "
        "shared_state_kernel.py (phase 6C), carrying the same legacy "
        "extra_sglang_args read surface",
    "robustness-agent/tests/test_signals_repeated_payload.py":
        "regression test that legacy + canonical envelopes hash to "
        "the same fingerprint",

    # Regression tests parametrise the ``_load_materialized_workload_metadata``
    # reader over (sglang, vllm, atom) including stray-EXTRA_SGLANG_ARGS
    # cases for atom YAMLs, so the test source mentions the legacy name
    # by design.
    "inference_optimizer/tests/test_kernel_request_handlers_units.py": "test_server_args_read_from_per_framework_env_key + "
    "test_atom_server_args_not_read_from_extra_sglang_args "
    "parametrise the materialised metadata reader",
    # Watermark-refresh profile executor inherits current_best's launch
    # args through the canonical channel; the comment + params key name
    # the legacy alias for the downstream reader's back-compat path.
    "inference_optimizer/orchestrator/action_executors/profile.py": "watermark-refresh inheritance comments name the legacy "
    "extra_sglang_args channel for the downstream reader",
    "inference_optimizer/tests/test_profile_and_kernel_handlers.py": "profile/kernel handler tests exercise the legacy "
    "extra_sglang_args inheritance channel",
    # Back-compat regression for the cumulative-merge helper, which is
    # named after the legacy field it dedupes.
    "inference_optimizer/tests/test_extra_sglang_args_merge.py": "cumulative extra_sglang_args merge/dedupe back-compat tests",
    # T0 fallback reads best_config via read_extra_server_args (which falls
    # back to the legacy key); dispatcher comment names the alias.
    "inference_optimizer/orchestrator/cortex_t0.py": "warm-start config extraction reads legacy extra_sglang_args "
    "via read_extra_server_args fallback for older recipe rows",
    "inference_optimizer/recipe_kb/dispatcher.py": "v2-to-arbor projection reads body.extra_sglang_args for "
    "legacy kb-extract recipes that lack body.best_config",
}


# Files under these top-level prefixes are skipped entirely.
_SKIP_DIRECTORIES: tuple[str, ...] = (
    # Plan / migration narrative trees (slated for deletion). These describe
    # the legacy key as a removal target, not a live writer site.
    "atom_plan/",
    "code_cleansing_plan/",
    ".git/",
    "node_modules/",
    "__pycache__/",
    # Local build / venv trees (not source of truth for writer-site policy).
    "build/",
    ".venv/",
    # setuptools build metadata (regenerated SOURCES.txt mirrors file
    # names, not source content — not a real writer site).
    "hyperloom_inference_optimizer.egg-info/",
)


_LEGACY_PATTERN = re.compile(r"extra_sglang_args")


def _git_tracked_files() -> set[str] | None:
    """Repo-relative POSIX paths tracked by git, or ``None`` when git is unavailable.

    The guard is about *repo* writer sites, so untracked local artifacts
    (gitignored scratch notes, ``reference_sessions/`` runtime dumps, etc.)
    must not trip it. ``None`` falls back to a raw filesystem scan.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "-z"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    return {p for p in (proc.stdout or "").split("\0") if p}


def _iter_repo_files() -> list[Path]:
    """All non-binary, non-skipped, git-tracked repo files we want to scan."""
    tracked = _git_tracked_files()
    out: list[Path] = []
    for path in _REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        # Only scan tracked source; skip untracked / gitignored local artifacts.
        if tracked is not None and rel not in tracked:
            continue
        if any(rel.startswith(skip) for skip in _SKIP_DIRECTORIES):
            continue
        # Limit to text-shaped suffixes so we don't accidentally try to
        # read a binary fixture (none currently exist but be defensive).
        if path.suffix not in {
            ".py",
            ".md",
            ".yaml",
            ".yml",
            ".toml",
            ".sh",
            ".txt",
            ".json",
            ".cfg",
            ".ini",
        }:
            continue
        out.append(path)
    return out


def _files_with_legacy_key() -> set[str]:
    """Repo-relative POSIX paths of files containing
    ``extra_sglang_args``."""
    hits: set[str] = set()
    for path in _iter_repo_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _LEGACY_PATTERN.search(text):
            hits.add(path.relative_to(_REPO_ROOT).as_posix())
    return hits


# Guards
def test_no_legacy_writer_sites_outside_allowlist():
    """Every file mentioning ``extra_sglang_args`` must be on the allowlist."""
    actual = _files_with_legacy_key()
    allowed = set(ALLOWED_FILES.keys())
    unexpected = sorted(actual - allowed)
    assert not unexpected, (
        "Files mentioning the legacy 'extra_sglang_args' field name "
        "outside the back-compat allowlist:\n  "
        + "\n  ".join(unexpected)
        + "\n\nEither (a) rename to 'extra_server_args' (preferred), "
        "or (b) if this is a deliberate back-compat surface, add the "
        "file to ALLOWED_FILES with a one-line justification."
    )


def test_allowlist_is_minimal():
    """Every allowlist entry must actually contain a legacy-key hit (no dead rows)."""
    actual = _files_with_legacy_key()
    dead_entries = sorted(set(ALLOWED_FILES.keys()) - actual)
    assert not dead_entries, (
        "ALLOWED_FILES entries that no longer contain "
        "'extra_sglang_args' (dead allowlist rows — remove them):\n  " + "\n  ".join(dead_entries)
    )


def test_allowlist_paths_resolve():
    """Sanity: every allowlist key must point at an existing file."""
    missing = [p for p in ALLOWED_FILES if not (_REPO_ROOT / p).exists()]
    assert not missing, "ALLOWED_FILES entries pointing at non-existent paths:\n  " + "\n  ".join(missing)


@pytest.mark.parametrize("path,_reason", sorted(ALLOWED_FILES.items()))
def test_allowlist_entries_have_justification(path: str, _reason: str):
    """Every allowlist value must be a non-empty justification string."""
    reason = ALLOWED_FILES[path]
    assert reason and reason.strip(), (
        f"ALLOWED_FILES[{path!r}] has an empty justification; add a "
        f"one-line explanation of why this file is allowed to mention "
        f"the legacy key."
    )
