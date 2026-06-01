"""Static guard: only the explicit back-compat surfaces are allowed to
mention ``extra_sglang_args``.

The payload-surface field ``extra_sglang_args`` was renamed to
``extra_server_args``. The legacy name is kept alive as a *read-only*
deprecation alias. Every in-repo reference to the legacy string falls
into exactly one of the categories enumerated in
:data:`ALLOWED_FILES`:

* The compat helper modules (Hyperloom + per-sub-agent shims).
* The SharedState / GridVariant back-compat code paths.
* Tests that explicitly assert on the deprecation alias behaviour.
* Prompt / SKILL / explanatory text that names both keys so the LLM
  knows the alias exists for one release.

Any *new* reference to the legacy name outside this allowlist is a
regression. When the deprecation alias is finally removed the
allowlist shrinks to the empty set and this guard becomes an absolute
"no `extra_sglang_args` anywhere" assertion until retired.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Repo root resolution. The test file lives at
# ``<repo>/inference_optimizer/tests/test_no_legacy_writer_sites.py``;
# the repo root is its great-great-grandparent.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Allowlist. Keys are repo-relative POSIX paths; values explain why the
# file is allowed to mention the legacy key. Adding a new entry MUST
# include a justification — the allowlist is meant to shrink, not grow.
# ---------------------------------------------------------------------------
ALLOWED_FILES: dict[str, str] = {
    # Compat helper modules (the only canonical reader of the legacy
    # key). Mentions are mechanical: constants, docstrings, error
    # messages naming both keys.
    "inference_optimizer/compat/__init__.py":
        "compat package docstring naming the rename",
    "inference_optimizer/compat/payload_aliases.py":
        "compat helper for the legacy alias",

    # Sub-agent shims — duplicated by design (sub-agents are
    # standalone packages, see framework_agent.repo_map).
    "kernel-agent/tools/_payload_aliases.py":
        "kernel-agent payload-aliases shim",
    "kernel-agent/tools/kernel_optimization.py":
        "kernel-agent reader site falls back to candidate_extra_sglang_args for legacy envelopes",
    "robustness-agent/src/robustness_agent/_payload_aliases.py":
        "robustness-agent payload-aliases shim",

    # Back-compat injection points in production code (renamed kwarg
    # alias on GridVariant, walk-and-rewrite on SharedState loader).
    "inference_optimizer/actions/_meta/replay_warm_recipe.yaml":
        "warm-replay internal action schema mirrors RecipeKB best_config "
        "field names",
    "inference_optimizer/orchestrator/action_executors/_grid_runner.py":
        "GridVariant(extra_sglang_args=...) back-compat kwarg",
    "inference_optimizer/orchestrator/shared_state.py":
        "_migrate_legacy_extra_sglang_args_keys walker + state.json transform",
    "inference_optimizer/orchestrator/action_executors/_explore_roofline_filter.py":
        "roofline filter reads canonical extra_server_args with a "
        "read-only legacy extra_sglang_args fallback for pre-rename "
        "variant objects",
    "inference_optimizer/orchestrator/optimization_journal.py":
        "journal classification reads existing stack/variant args fields",
    # The materializer keeps the legacy name in its env-routing docstring
    # because the per-framework env names (EXTRA_SGLANG_ARGS / VLLM_ /
    # ATOM_) are intentionally unchanged.
    "inference_optimizer/orchestrator/action_executors/_workload_envs.py":
        "docstring naming the renamed payload field for context",
    # CI transform reads legacy-keyed session_breakdown.json artefacts.
    "ci/transform_to_session_summary_v2.py":
        "legacy session-breakdown JSON reader (operator-side back-compat)",

    # Prompt / orientation text that names both keys explicitly so the
    # LLM and any human reader of the prompt knows the alias exists
    # for one release.
    "inference_optimizer/orchestrator/system_prompts/prompt_builder.py":
        "explanatory paragraph naming the legacy alias",
    "inference_optimizer/cli.py":
        "warm-replay executor registration comment names RecipeKB "
        "best_config field names",

    # Pytest marker registration mentions the legacy name in the
    # marker's description.
    "pyproject.toml":
        "pytest marker description naming the legacy alias",

    # Test files: compat helper coverage + back-compat regression
    # tests + this guard itself + the per-sub-agent shim tests.
    "inference_optimizer/tests/test_payload_aliases.py":
        "compat helper test surface",
    "inference_optimizer/tests/test_back_compat_legacy_field_name.py":
        "back-compat regression tests",
    "inference_optimizer/tests/test_explore_roofline_filter.py":
        "roofline filter tests exercise GridVariant.extra_sglang_args",
    "inference_optimizer/tests/test_grid_runner.py":
        "GridVariant back-compat tests exercise extra_sglang_args kwarg",
    "inference_optimizer/tests/test_coordinator_kb_writes.py":
        "recipe write-back regression test asserts best_config / "
        "what_worked emit the KB-legacy extra_sglang_args field read "
        "from canonical state",
    "inference_optimizer/tests/test_no_legacy_writer_sites.py":
        "this guard's allowlist + docstring",
    "inference_optimizer/tests/test_optimization_journal.py":
        "journal tests cover stack entries carrying legacy args field",
    "inference_optimizer/tests/test_warm_replay.py":
        "warm-replay tests mirror RecipeKB best_config field names",
    "kernel-agent/tools/test_payload_aliases_shim.py":
        "kernel-agent shim test surface",
    "robustness-agent/tests/test_payload_aliases_shim.py":
        "robustness-agent shim test surface",

    # Readers that funnel external-envelope payloads through
    # ``read_extra_server_args`` mention the legacy key inline in the
    # surrounding code comment to document why the call goes through
    # the helper.
    "inference_optimizer/orchestrator/coordinator.py":
        "comments explain the read_extra_server_args call at the LLM "
        "intent / sub-agent envelope read boundaries",
    "inference_optimizer/orchestrator/kernel_request_handlers.py":
        "comments explain the read_extra_server_args call at the "
        "integrate_patch sub-agent envelope read boundary",
    "robustness-agent/src/robustness_agent/signals/repeated_payload.py":
        "_normalise_extra_server_args_key uses the shim to fold "
        "legacy-keyed envelopes into the same fingerprint",
    "robustness-agent/tests/test_signals_repeated_payload.py":
        "regression test that legacy + canonical envelopes hash to "
        "the same fingerprint",

    # Regression tests parametrise the ``_load_materialized_workload_metadata``
    # reader over (sglang, vllm, atom) including stray-EXTRA_SGLANG_ARGS
    # cases for atom YAMLs, so the test source mentions the legacy name
    # by design.
    "inference_optimizer/tests/test_kernel_request_handlers_units.py":
        "test_server_args_read_from_per_framework_env_key + "
        "test_atom_server_args_not_read_from_extra_sglang_args "
        "parametrise the materialised metadata reader",

    # SKILL.md names the legacy extra_sglang_args alias as compat-surface
    # context in the IR-8 entry.
    "inference_optimizer/SKILL.md":
        "IR-8 entry names the legacy extra_sglang_args alias as "
        "compat-surface context",

    # Watermark-refresh profile executor inherits current_best's launch
    # args through the canonical channel; the comment + params key name
    # the legacy alias for the downstream reader's back-compat path.
    "inference_optimizer/orchestrator/action_executors/profile.py":
        "watermark-refresh inheritance comments name the legacy "
        "extra_sglang_args channel for the downstream reader",
    "inference_optimizer/tests/test_profile_and_kernel_handlers.py":
        "profile/kernel handler tests exercise the legacy "
        "extra_sglang_args inheritance channel",
    # Back-compat regression for the cumulative-merge helper, which is
    # named after the legacy field it dedupes.
    "inference_optimizer/tests/test_extra_sglang_args_merge.py":
        "cumulative extra_sglang_args merge/dedupe back-compat tests",

}


# Files under these top-level prefixes are skipped entirely.
_SKIP_DIRECTORIES: tuple[str, ...] = (
    # Plan / migration narrative tree (slated for deletion).
    "atom_plan/",
    ".git/",
    "node_modules/",
    "__pycache__/",
    # setuptools build metadata (regenerated SOURCES.txt mirrors file
    # names, not source content — not a real writer site).
    "hyperloom_inference_optimizer.egg-info/",
)


_LEGACY_PATTERN = re.compile(r"extra_sglang_args")


def _iter_repo_files() -> list[Path]:
    """All non-binary, non-skipped repo files we want to scan."""
    out: list[Path] = []
    for path in _REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if any(rel.startswith(skip) for skip in _SKIP_DIRECTORIES):
            continue
        # Limit to text-shaped suffixes so we don't accidentally try to
        # read a binary fixture (none currently exist but be defensive).
        if path.suffix not in {
            ".py", ".md", ".yaml", ".yml", ".toml", ".sh", ".txt",
            ".json", ".cfg", ".ini",
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


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_no_legacy_writer_sites_outside_allowlist():
    """Every file mentioning ``extra_sglang_args`` must be on the
    allowlist. Catches a future regression where a new writer site
    accidentally emits the legacy key, or where a previously-renamed
    site gets reverted."""
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
    """Every allowlist entry must actually contain a legacy-key hit.
    Forces the allowlist to evolve with the code — once a back-compat
    surface is removed, the corresponding allowlist entry must go too."""
    actual = _files_with_legacy_key()
    dead_entries = sorted(set(ALLOWED_FILES.keys()) - actual)
    assert not dead_entries, (
        "ALLOWED_FILES entries that no longer contain "
        "'extra_sglang_args' (dead allowlist rows — remove them):\n  "
        + "\n  ".join(dead_entries)
    )


def test_allowlist_paths_resolve():
    """Sanity: every allowlist key must point at an existing file.
    Catches typos / paths that move."""
    missing = [
        p for p in ALLOWED_FILES
        if not (_REPO_ROOT / p).exists()
    ]
    assert not missing, (
        "ALLOWED_FILES entries pointing at non-existent paths:\n  "
        + "\n  ".join(missing)
    )


@pytest.mark.parametrize("path,_reason", sorted(ALLOWED_FILES.items()))
def test_allowlist_entries_have_justification(path: str, _reason: str):
    """Every allowlist value must be a non-empty justification string.
    The empty / placeholder reason is a code smell — it signals that
    the entry was added without a clear rationale."""
    reason = ALLOWED_FILES[path]
    assert reason and reason.strip(), (
        f"ALLOWED_FILES[{path!r}] has an empty justification; add a "
        f"one-line explanation of why this file is allowed to mention "
        f"the legacy key."
    )
