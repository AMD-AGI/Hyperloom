# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Remote KB publication state reported in a forge-loop result."""

from __future__ import annotations


def _initial_remote_publication_state(warm: dict) -> dict:
    """Seed publication authority from an already-materialized KB warm-start."""
    commit = str(warm.get("applied_commit") or "")
    solution = str(warm.get("solution_slug") or "")
    if warm.get("applied") and commit and solution:
        return {
            "status": "published",
            "state": "materialized_from_remote",
            "source": "existing_warm_start_solution",
            "solution_slug": solution,
            "best_commit": commit,
            "pending_commit": "",
            "last_attempted_commit": "",
            "published_commit": commit,
            "last_result": {
                "written": False,
                "reason": "existing_warm_start_solution",
                "solution": solution,
            },
        }
    return {
        "status": "not_attempted",
        "state": "not_attempted",
        "source": "",
        "solution_slug": "",
        "best_commit": "",
        "pending_commit": "",
        "last_attempted_commit": "",
        "published_commit": "",
        "last_result": None,
    }


def _warm_start_publication_covers(state: dict, commit: str) -> bool:
    """Whether ``commit`` is already represented by the consumed KB solution."""
    return bool(
        commit
        and state.get("source") == "existing_warm_start_solution"
        and state.get("published_commit") == commit
        and not state.get("pending_commit")
    )


def _record_remote_publication_result(
    state: dict,
    *,
    commit: str,
    result: dict,
) -> None:
    """Apply one campaign publication attempt without erasing prior authority."""
    state["best_commit"] = commit
    state["last_result"] = result
    if result.get("written"):
        state["published_commit"] = commit
        state["pending_commit"] = ""
        state["status"] = "published"
        state["state"] = "published"
        state["source"] = "campaign_publication"
        state["solution_slug"] = str(result.get("solution") or "")
        return
    reason = str(result.get("reason") or "error")
    # Settled outcomes: the store reached a decision, so retrying cannot change
    # it. Everything else is treated as transient and stays pending.
    if reason in {
        "not_configured",
        "missing_gpu_type",
        "no_improvement_over_reuse",
        "empty_diff",
        "not_better_than_kb",
    }:
        state["pending_commit"] = ""
        state["status"] = reason
        state["state"] = reason
        state["source"] = "campaign_publication"
        return
    state["status"] = "pending_retry"
    state["state"] = "pending_retry"
    state["source"] = "campaign_publication"


def _remote_publication_view(state: dict, best_commit: str) -> dict:
    """Return the authoritative local-versus-remote best publication status."""
    local_best = str(best_commit or "")
    published = str(state.get("published_commit") or "")
    pending = str(state.get("pending_commit") or "")
    return {
        key: value
        for key, value in {
            **state,
            "best_commit": local_best,
            "local_best_commit": local_best,
            "pending_commit": pending,
            "published_commit": published,
            "latest_best_published": bool(local_best and published == local_best and not pending),
        }.items()
        if key != "last_result"
    }
