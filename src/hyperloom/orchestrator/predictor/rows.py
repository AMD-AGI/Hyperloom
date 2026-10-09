# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn one predictor answer into rows for the untested-proposal queue.

The service samples several completions and returns every distinct proposal in
sampling order, so ranking happens here: rows are ordered by how many samples
proposed them, a knob family takes one slot, and at most :data:`MAX_QUEUED` rows
are queued. Each row is reduced to what it changes on top of the current
champion, and a delta this session has already benched or queued is dropped.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from hyperloom.inference_optimizer.grid_server_args import remove_server_args
from hyperloom.orchestrator.actions.executors._proposal_identity import content_fingerprint
from hyperloom.orchestrator.predictor.client import Action, Prediction

#: Attribution label on predictor rows, the explore variants made from them, and their attempts.
PROVENANCE = "primatune"
#: ``domain`` of a predictor round in ``specialist_rounds``; nothing dispatches on it.
QUEUE_DOMAIN = "primatune"
#: Specialist rounds carry no priority, so predictor rows sort ahead of them in the queue the Coordinator benches from.
QUEUE_PRIORITY = 1
#: Rows one answer may queue: orchestration's grid ceiling.
MAX_QUEUED = 6
#: Characters of the service's rationale a row keeps.
RATIONALE_CHARS = 600

#: Env prefixes owned by the other serving stack; the service repairs flags, not envs.
_FOREIGN_ENV_PREFIXES: dict[str, tuple[str, ...]] = {"vllm": ("SGLANG_",), "sglang": ("VLLM_",)}
#: Flags whose value is a JSON config: the value, not the name, says which knobs move.
_STRUCTURED_FLAGS = frozenset({"--compilation-config"})


def _flags_text(server_args: Mapping[str, Any]) -> str:
    """Render flags as a CLI fragment; a JSON value becomes one compact token."""
    parts: list[str] = []
    for flag, value in server_args.items():
        if value is False:
            continue
        parts.append(flag)
        if isinstance(value, (dict, list)):
            parts.append(json.dumps(value, separators=(",", ":")))
        elif value is not True and str(value if value is not None else "").strip():
            parts.append(str(value).strip())
    return " ".join(parts)


def _change(
    server_args: Mapping[str, Any], envs: Mapping[str, str], base_args: str, base_envs: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """The part of a proposal the champion does not already set to the same value.

    The service answers with a full launch recipe, so on a deep stack most of a
    proposal restates the champion. Only an exact ``(flag, value)`` echo goes --
    the same flag with another value is the change itself -- and the comparison
    uses the tokenizer the launch path composes with.
    """
    kept = {
        flag: value
        for flag, value in server_args.items()
        if remove_server_args(_flags_text({flag: value}), [base_args])
    }
    kept_envs = {key: value for key, value in envs.items() if key not in base_envs or base_envs[key] != value}
    return kept, kept_envs


def _family(server_args: Mapping[str, Any], envs: Mapping[str, str]) -> frozenset[str]:
    """Which knobs a row moves, ignoring values -- except for a JSON config flag, whose value names the knobs."""
    names = {
        f"{flag}={_flags_text({flag: value})}" if flag in _STRUCTURED_FLAGS else flag
        for flag, value in server_args.items()
    }
    return frozenset(names | {f"env:{key}" for key in envs})


def _sample_key(server_args: Any, envs: Any, source_change: Any) -> tuple[tuple, tuple, str]:
    """The identity the service de-duplicated its samples on, so a vote lands on the proposal it collapsed into."""
    args = server_args if isinstance(server_args, dict) else {}
    env = envs if isinstance(envs, dict) else {}
    return (
        tuple(sorted((str(k), str(v)) for k, v in args.items())),
        tuple(sorted((str(k), str(v)) for k, v in env.items())),
        str(source_change or ""),
    )


def _votes(prediction: Prediction) -> dict[tuple[tuple, tuple, str], int]:
    """How many raw samples (``meta["candidates"]``) proposed each distinct proposal."""
    counts: dict[tuple[tuple, tuple, str], int] = {}
    for sample in prediction.meta.get("candidates") or []:
        if isinstance(sample, dict):
            key = _sample_key(sample.get("server_args"), sample.get("envs"), sample.get("source_change"))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _reason(action: Action, flags: Mapping[str, Any], envs: Mapping[str, str], votes: int, samples: Any) -> str:
    """The vote share and the service's rationale; without a rationale, the knobs the row moves."""
    if action.rationale:
        share = f"{votes}/{samples}" if isinstance(samples, int) and samples > 0 else f"{votes} vote(s)"
        return f"PrimaTune {share}: {action.rationale[:RATIONALE_CHARS]}"
    knobs = sorted(flags) + sorted(f"env:{name}" for name in envs)
    return "predictor: " + ", ".join(knobs[:4]) + (f" (+{len(knobs) - 4})" if len(knobs) > 4 else "")


def _champion(state: Any) -> tuple[str, dict[str, str]]:
    best = state.current_best if isinstance(state.current_best, dict) else {}
    args = str(best.get("effective_extra_server_args") or best.get("extra_server_args") or "").strip()
    envs = best.get("extra_envs")
    return args, ({str(k): str(v) for k, v in envs.items()} if isinstance(envs, dict) else {})


def proposal_rows(prediction: Prediction, *, key: str, state: Any) -> list[dict[str, Any]]:
    """Queue rows for one answer, most-voted first; empty when every proposal is a duplicate."""
    foreign = _FOREIGN_ENV_PREFIXES.get(str(state.framework or "").strip().lower(), ())
    base_args, base_envs = _champion(state)
    blocked = state.benched_fingerprints() | {row["fingerprint"] for row in state.untested_proposal_rows()}
    votes = _votes(prediction)
    samples = prediction.meta.get("samples")
    candidates: list[tuple[frozenset[str], dict[str, Any]]] = []
    for index, action in enumerate(prediction.config_actions):
        envs = {str(k): str(v) for k, v in action.envs.items() if not str(k).upper().startswith(foreign)}
        flags, envs = _change(action.server_args, envs, base_args, base_envs)
        row: dict[str, Any] = {"name": f"primatune-{key}-{index}", "extra_args": _flags_text(flags), "extra_envs": envs}
        fingerprint = content_fingerprint(row)
        if not (flags or envs) or fingerprint in blocked:
            continue
        blocked.add(fingerprint)
        row_votes = votes.get(_sample_key(action.server_args, action.envs, action.source_change), 0)
        row.update(provenance=PROVENANCE, reason=_reason(action, flags, envs, row_votes, samples), votes=row_votes)
        if isinstance(samples, int):
            row["samples"] = samples
        candidates.append((_family(flags, envs), row))
    # Stable sort: the service's sampling order breaks ties.
    candidates.sort(key=lambda item: -item[1]["votes"])
    rows: list[dict[str, Any]] = []
    families: set[frozenset[str]] = set()
    for family, row in candidates:
        if family in families:
            continue
        families.add(family)
        rows.append(row)
        if len(rows) == MAX_QUEUED:
            break
    return rows
