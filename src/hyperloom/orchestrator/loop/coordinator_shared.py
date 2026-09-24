# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Constants, records and inbox rendering shared by the Coordinator and its collaborators.

They sit below ``loop.coordinator`` so a collaborator can import them without importing the module that
constructs it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hyperloom.common.prompt_safety import defang_prompt_structure as _defang_prompt_structure
from hyperloom.common.prompt_safety import flatten_for_prompt as _flatten_for_inbox
from hyperloom.orchestrator.actions.executors._grid_server_args import tokenize_server_args_preserving_json

from ..state.failure_evidence import UNMEASURED_OUTCOMES, render_failure_line
from .coordinator_helpers import serialize_verdict_advisory

if TYPE_CHECKING:
    from ..bus.message_bus import Message

log = logging.getLogger(__name__)


# Recipe snapshot severity tags (schema has no fixed enum).
_SEVERITY_CRASH: str = "crash"
_SEVERITY_REGRESS: str = "regress"

# Bounded transient-failure auto-retry for specialist dispatches (infra-only).
SPECIALIST_AUTO_RETRY_MAX: int = 2
# Combined baseline-failure backstop: fast-fail after this many TOTAL baseline failures.
_BASELINE_MAX_TOTAL_FAILURES: int = 3
# Unified authored-lane max attempts (apply-failure retries + Critic reauthor).
_AUTHORED_LANE_MAX_ATTEMPTS: int = 3
# Default min TRANSFER confidence a warm-replay champion must clear to be enqueued.
_DEFAULT_WARM_REPLAY_MIN_CONFIDENCE: float = 0.7
# Default resume-drift floor (%): a re-measured current_best below this fraction of its recorded tput is flagged as
# drift.
_DEFAULT_RESUME_DRIFT_FLOOR_PCT: float = 95.0


def _extract_enablement_launch_log(result_payload: dict[str, Any] | None) -> str:
    """Extract launch/traceback text from a failed baseline result payload."""
    if not isinstance(result_payload, dict):
        return ""
    parts: list[str] = []
    for key in ("error", "stderr", "log_tail", "log_excerpt", "traceback", "reason"):
        val = result_payload.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, (list, tuple)):
            joined = "\n".join(str(x) for x in val if str(x).strip())
            if joined.strip():
                parts.append(joined.strip())
    return "\n".join(parts).strip()


def _framework_config_levers_from_done(
    done_payload: dict[str, Any] | None,
    *,
    levers_ride_with_patches: bool = False,
) -> dict[str, Any]:
    """Extract a config-lever set from a FRAMEWORK specialist deliverable.

    Args:
        done_payload: The specialist's ``specialist_done`` payload.
        levers_ride_with_patches: Whether a lever delivered alongside a patch
            belongs to the patch's round. True for ENABLEMENT, where the pair is
            jointly what makes the model boot; False while optimizing, where a
            patch is its own outcome and a lever is judged on its own.
    """
    if not isinstance(done_payload, dict):
        return {}
    proposals = done_payload.get("proposal_set") or []
    if not isinstance(proposals, list):
        return {}
    # A patch deliverable otherwise takes precedence: a lever that merely
    # *accompanies* a patch is not a config-only outcome. ``atomic`` remains the
    # specialist's own way to say the two are inseparable, but it cannot be the
    # only way -- it is a model-authored boolean, and the same specialist has
    # emitted ``atomic: false`` on a lever whose own reason read "required to
    # boot at all once the patch lands". Enablement therefore decides this from
    # the lane it is running, not from the deliverable's self-description.
    patches = done_payload.get("patches_written") or []
    if isinstance(patches, list) and patches and not levers_ride_with_patches:
        proposals = [e for e in proposals if isinstance(e, dict) and e.get("atomic") is True]
        if not proposals:
            return {}
    for entry in proposals:
        if not isinstance(entry, dict):
            continue
        extra_envs: dict[str, str] = {}
        envs = entry.get("extra_envs")
        if isinstance(envs, dict):
            for k, v in envs.items():
                key = str(k).strip()
                if key:
                    extra_envs[key] = str(v)
        args = entry.get("extra_args")
        extra_server_args = ""
        if isinstance(args, str) and args.strip():
            parsed_args = tokenize_server_args_preserving_json(args)
            if parsed_args is None:
                log.warning(
                    "FRAMEWORK config lever %r has server args unsupported by "
                    "Magpie's unquoted argv transport; dropping the args%s",
                    entry.get("name"),
                    " while preserving its environment overrides" if extra_envs else "",
                )
                if not extra_envs:
                    continue
            else:
                extra_server_args = parsed_args[0]
        elif isinstance(args, (list, tuple)):
            arg_tokens = [str(a) for a in args if str(a).strip()]
            if any(any(ch.isspace() for ch in token) for token in arg_tokens):
                log.warning(
                    "FRAMEWORK config lever %r has a whitespace-bearing argv token; dropping the args%s",
                    entry.get("name"),
                    " while preserving its environment overrides" if extra_envs else "",
                )
                if not extra_envs:
                    continue
            else:
                parsed_args = tokenize_server_args_preserving_json(" ".join(arg_tokens))
                if parsed_args is None:
                    log.warning(
                        "FRAMEWORK config lever %r has unparseable server args; dropping the args%s",
                        entry.get("name"),
                        " while preserving its environment overrides" if extra_envs else "",
                    )
                    if not extra_envs:
                        continue
                else:
                    extra_server_args = parsed_args[0]
        if extra_server_args or extra_envs:
            return {
                "extra_server_args": extra_server_args,
                "extra_envs": extra_envs,
            }
    return {}


# Hard-trigger thresholds: optimisation rounds a domain may go without a specialist dispatch / a KEEP before the
# Coordinator force-dispatches one.
FORCE_STALLED_SPECIALIST_ROUNDS: int = 8
FORCE_STALLED_KEEP_ROUNDS: int = 12


# Result keys surfaced in delegated_result inbox line; first match wins per group.
_OUTCOME_GAIN_KEYS: tuple[str, ...] = (
    "validated_gain_pct",
    "gain_pct",
    "predicted_gain_pct",
    "delta_pct",
)
_OUTCOME_TPUT_KEYS: tuple[str, ...] = (
    "tokens_per_s",
    "tput",
    "throughput",
    "tput_tok_s",
)
_OUTCOME_STATUS_KEYS: tuple[str, ...] = ("status", "verdict", "outcome", "runner_status")
# Notes rendered per inbox line.
_OUTCOME_NOTES_MAX: int = 3


def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    """Return ``d[k]`` for the first ``k`` in ``keys`` present + non-None."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _defang_alert_payload(value: Any) -> Any:
    """Recursively defang string leaves of an alert payload (keys untouched)."""
    if isinstance(value, str):
        return _defang_prompt_structure(value)
    if isinstance(value, dict):
        return {k: _defang_alert_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_defang_alert_payload(v) for v in value]
    return value


def _format_inbox_event(m: "Message", *, max_variant_rows: int = 3) -> str:
    """Render one inbox ``Message`` as a compact, high-signal line."""
    topic = (m.topic or "").strip()
    payload = m.payload if isinstance(m.payload, dict) else {}
    # Canonical inbox header ordering that downstream parsers anchor on.
    if getattr(m, "msg_id", None):
        head = f"seq={m.seq} msg_id={m.msg_id} from={m.from_agent} topic={topic}"
    else:
        head = f"seq={m.seq} from={m.from_agent} topic={topic}"

    if topic == "delegated_result":
        kind = payload.get("kind")
        state = payload.get("state")
        error = payload.get("error")
        result = payload.get("result")
        parts = [head, f"kind={kind!r}", f"state={state!r}"]
        notes: list[Any] = []
        if isinstance(result, dict):
            status = _first_present(result, _OUTCOME_STATUS_KEYS)
            gain = _first_present(result, _OUTCOME_GAIN_KEYS)
            tput = _first_present(result, _OUTCOME_TPUT_KEYS)
            kept = result.get("kept")
            if status is not None:
                parts.append(f"status={status!r}")
            if kept is not None:
                parts.append(f"kept={kept!r}")
            if gain is not None:
                parts.append(f"gain={gain}")
            if tput is not None:
                parts.append(f"tput={tput}")
            # Executors that never raise report the failure inside the result envelope, leaving the top-level error
            # None.
            if not error:
                error = result.get("error")
            raw_notes = result.get("notes")
            if isinstance(raw_notes, list):
                # patch_safety_numeric is the Critic's artifact; it is not a lever here.
                notes = [n for n in raw_notes if n and not str(n).startswith("patch_safety_numeric:")][
                    :_OUTCOME_NOTES_MAX
                ]
            done = result.get("specialist_done") if kind == "specialist" else None
            if isinstance(done, dict):
                summary = str(done.get("summary") or "").strip()
                if summary:
                    parts.append(f"summary={summary[:400]!r}")
                if done.get("confidence") is not None:
                    parts.append(f"confidence={done['confidence']}")
                for label, key in (("findings", "new_findings"), ("questions", "residual_questions")):
                    items = done.get(key)
                    if isinstance(items, list) and items:
                        parts.append(f"{label}={len(items)}")
        if error:
            parts.append(f"error={str(error)[:200]!r}")
        if notes:
            shown = "; ".join(str(n) for n in notes)
            parts.append(f"notes={shown[:300]!r}")
        header_line = " ".join(parts)
        if max_variant_rows <= 0 or not isinstance(result, dict):
            return header_line
        pvos = result.get("per_variant_outcomes")
        if not isinstance(pvos, list):
            return header_line
        failures = [
            v for v in pvos if isinstance(v, dict) and str(v.get("outcome") or "").upper() in UNMEASURED_OUTCOMES
        ]
        if not failures:
            return header_line
        lines = [header_line]
        for vo in failures[:max_variant_rows]:
            row = dict(vo)
            row["error_excerpt"] = _flatten_for_inbox(vo.get("error_excerpt") or vo.get("reason") or "")
            lines.append("  failure: " + render_failure_line(row, excerpt_chars=120))
        elided = len(failures) - max_variant_rows
        if elided > 0:
            lines.append(f"  (+{elided} more failures; pull get_variant_failures)")
        return "\n".join(lines)

    if topic in ("policy_denial", "denial") or (topic == "observation" and payload.get("kind") == "policy_denial"):
        return (
            f"{head} action={payload.get('action_name')!r} "
            f"rule={payload.get('rule')!r} "
            f"hint={str(payload.get('hint') or '')[:140]!r}"
        )

    if topic == "review_verdict":
        parts = [
            f"{head} target={payload.get('target_proposal_msg_id')!r} "
            f"verdict={payload.get('verdict')!r} "
            f"reasoning={str(payload.get('reasoning') or '')[:140]!r}"
        ]
        advisory = serialize_verdict_advisory(payload)
        required_evidence = advisory.get("required_evidence")
        if required_evidence:
            shown = "; ".join(str(item) for item in required_evidence[:3])
            parts.append(f"required_evidence[{len(required_evidence)}]={shown[:140]!r}")
        risks = advisory.get("risks")
        if risks:
            parts.append(f"risks={len(risks)}")
        advice_text = advisory.get("advice_text")
        if advice_text:
            parts.append(f"advice={advice_text[:140]!r}")
        return " ".join(parts)

    if topic == "observation":
        kind = payload.get("kind")
        if kind is not None:
            return f"{head} kind={kind!r} payload={payload}"

    if topic == "alert":
        # Alert payloads can embed attacker-influenceable server.log excerpts; defang string leaves so a log line
        # can't inject prompt structure.
        return f"{head} payload={_defang_alert_payload(payload)}"

    return f"{head} payload={payload}"


@dataclass
class PendingProposal:
    """A propose_action intent waiting for Critic Review."""

    proposal_msg_id: str
    from_agent: str
    action_name: str
    predicted_gain_pct: float
    payload: dict[str, Any]
    decided: bool = False
    verdict: str | None = None  # approve / reject / redirect / advise / needs_review
