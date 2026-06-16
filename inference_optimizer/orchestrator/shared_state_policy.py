# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Policy-denial write-owner functions extracted from :class:`SharedState`.

Part of the SharedState behavior-offload (phase 2). The policy-denial streak
bookkeeping and its summary belong to the PolicyGate decision domain; they live
here as free functions taking ``state`` first. ``SharedState`` keeps forwarding
shims so existing callers are unchanged.
"""

from __future__ import annotations

from typing import Any

from .shared_state import _now_iso

def record_policy_denial(
    state,
    *,
    action_name: str,
    rule: str,
    hint: str,
    intent_type: str,
    tick: int,
    intent_payload: dict[str, Any] | None = None,
) -> int:
    """Append a PolicyGate denial row and bump the per-(action, rule) streak.

    Records a capped rolling history entry and increments the
    consecutive-denial counter keyed by ``"<action_name>:<rule>"``.

    Args:
        action_name (str): The action the denied intent targeted (empty
            is normalized to ``"*"`` in the streak key).
        rule (str): The PolicyGate rule id that fired.
        hint (str): Human-readable remediation hint surfaced to the LLM.
        intent_type (str): The denied intent's type.
        tick (int): The Coordinator tick at which the denial occurred.
        intent_payload (dict[str, Any] | None): Optional intent payload;
            when present, its sorted keys are recorded for context.

    Returns:
        int: The new consecutive-denial streak value for this
            (action, rule) pair.
    """
    key = f"{action_name or '*'}:{rule}"
    streak = int(state.policy_denial_streak.get(key, 0)) + 1
    state.policy_denial_streak[key] = streak
    entry = {
        "tick": int(tick),
        "action_name": action_name or "",
        "rule": rule,
        "hint": hint or "",
        "intent_type": intent_type,
        "streak": streak,
        "ts": _now_iso(),
    }
    if intent_payload:
        entry["intent_payload_keys"] = sorted(intent_payload.keys())
    history = list(state.policy_denial_history or [])
    history.append(entry)
    if len(history) > state._POLICY_DENIAL_HISTORY_CAP:
        history = history[-state._POLICY_DENIAL_HISTORY_CAP :]
    state.policy_denial_history = history
    return streak


def reset_policy_denial_streak(state, action_name: str) -> None:
    """Clear all consecutive-denial streaks for a given action.

    Drops every ``policy_denial_streak`` entry whose key begins with
    ``"<action_name>:"`` — called when the action finally succeeds so a
    later denial starts a fresh streak.

    Args:
        action_name (str): The action whose streaks should be reset; a
            falsy value is a no-op.
    """
    if not action_name:
        return
    prefix = f"{action_name}:"
    state.policy_denial_streak = {
        k: v
        for k, v in (state.policy_denial_streak or {}).items()
        if not k.startswith(prefix)
    }


def to_policy_denial_summary(state, *, top_k: int = 6) -> str:
    """Render the most recent PolicyGate denials for prompt injection.

    Args:
        top_k (int): Maximum number of newest denial rows to render.

    Returns:
        str: A ``=== Recent policy denials ===`` block, or ``""`` when
            no denials have been recorded.
    """
    if not state.policy_denial_history:
        return ""
    rows = list(state.policy_denial_history)[-top_k:]
    lines = [
        "=== Recent policy denials "
        f"(newest last, total={len(state.policy_denial_history)}) ==="
    ]
    for r in rows:
        lines.append(
            f"  tick={r.get('tick')} action={r.get('action_name')!r} "
            f"rule={r.get('rule')!r} streak={r.get('streak')} "
            f"hint={str(r.get('hint') or '')[:140]!r}"
        )
    return "\n".join(lines)

