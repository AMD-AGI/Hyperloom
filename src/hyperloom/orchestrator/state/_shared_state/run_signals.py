# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Policy denials and the session's stop reason."""

from __future__ import annotations

from typing import Any
from hyperloom_common.env import env_bool
from hyperloom_common.timeutil import now_iso


class _RunSignalsMixin:
    """Records policy denials and their streak, and commits the session's stop reason."""

    _POLICY_DENIAL_HISTORY_CAP = 50

    def record_policy_denial(
        self,
        *,
        action_name: str,
        rule: str,
        hint: str,
        intent_type: str,
        intent_payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a PolicyGate denial row, stamped with the current tick, and bump the per-(action, rule) streak.

        Records a capped rolling history entry and increments the
        consecutive-denial counter keyed by ``"<action_name>:<rule>"``.

        Args:
            action_name (str): The action the denied intent targeted (empty
                is normalized to ``"*"`` in the streak key).
            rule (str): The PolicyGate rule id that fired.
            hint (str): Human-readable remediation hint surfaced to the LLM.
            intent_type (str): The denied intent's type.
            intent_payload (dict[str, Any] | None): Optional intent payload;
                when present, its sorted keys are recorded for context.

        Returns:
            int: The new consecutive-denial streak value for this
                (action, rule) pair.
        """
        key = f"{action_name or '*'}:{rule}"
        streak = int(self.policy_denial_streak.get(key, 0)) + 1
        self.policy_denial_streak[key] = streak
        entry = {
            "tick": int(self.tick or 0),
            "action_name": action_name or "",
            "rule": rule,
            "hint": hint or "",
            "intent_type": intent_type,
            "streak": streak,
            "ts": now_iso(),
        }
        if intent_payload:
            entry["intent_payload_keys"] = sorted(intent_payload.keys())
        history = list(self.policy_denial_history or [])
        history.append(entry)
        if len(history) > self._POLICY_DENIAL_HISTORY_CAP:
            history = history[-self._POLICY_DENIAL_HISTORY_CAP :]
        self.policy_denial_history = history
        return streak

    def reset_policy_denial_streak(self, action_name: str) -> None:
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
        self.policy_denial_streak = {
            k: v for k, v in (self.policy_denial_streak or {}).items() if not k.startswith(prefix)
        }

    # stop_reason ENUM validator
    def set_stop_reason(
        self,
        value: str,
        *,
        strict: bool | None = None,
    ) -> str:
        """Validated writer for :attr:`stop_reason` (Inv-8.3 closed vocab): values outside ``STOP_REASON_VOCAB`` map to ``\"unknown\"`` (lenient) or raise (``strict=True``, default env ``INFERENCE_OPTIMIZER_STRICT_STOP_REASON``). Returns value written."""
        from hyperloom.inference_optimizer.breakdown.stop_reasons import STOP_REASON_VOCAB, is_valid_stop_reason

        text = str(value or "").strip()
        if not text:
            self.stop_reason = ""
            self.stop_ts = ""
            return ""
        if is_valid_stop_reason(text):
            return self._commit_stop_reason(text)
        if strict is None:
            strict = env_bool("INFERENCE_OPTIMIZER_STRICT_STOP_REASON")
        if strict:
            raise ValueError(f"stop_reason={text!r} not in STOP_REASON_VOCAB ({sorted(STOP_REASON_VOCAB)!r})")
        # Lenient: map to "unknown" and warn.
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "stop_reason=%r not in STOP_REASON_VOCAB; mapped to 'unknown' "
            ". Set "
            "INFERENCE_OPTIMIZER_STRICT_STOP_REASON=1 to fail-fast.",
            text,
        )
        return self._commit_stop_reason("unknown")

    def _commit_stop_reason(self, reason: str) -> str:
        """Write a validated stop reason, stamping the end time on the first one."""
        self.stop_reason = reason
        if not self.stop_ts:
            self.stop_ts = now_iso()
        return reason
