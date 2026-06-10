# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Local payload-schema check mirroring upstream ``PolicyGate.validate_intent``.

PolicyGate is the server-side source of truth (rejects with ``PolicyDenied``
+ a ``policy_denied`` inbox observation), but that costs ≥1 silent tick.
:class:`PolicyAware` runs the same checks locally and raises
:class:`PolicyViolation` so unit tests catch schema drift immediately. The
mirrored schema is a subset limited to robustness-emittable intents;
``tests/test_role_contract.py`` cross-checks the upstream table when importable.
"""

from __future__ import annotations

from typing import Any

from ..role.envelope import (
    ALERT_SEVERITIES,
    CORE_STATE_FIELDS,
    Intent,
    IntentType,
    KILL_TASK_ALLOWED_SCOPES,
    PAYLOAD_REQUIRED,
    ROBUSTNESS_ALLOWED_INTENTS,
    ROBUSTNESS_DELEGATE_ACTIONS,
    ROBUSTNESS_ONLY_INTENTS,
    ROBUSTNESS_STATE_FIELDS,
)


class PolicyViolation(ValueError):
    """Raised when an intent fails local PolicyGate-equivalent checks.

    Attributes:
        rule: short identifier matching upstream ``PolicyDenied.rule``
            (``role`` / ``payload`` / ``state_field`` / ``kill_scope`` /
            ``robustness_only_source`` / ``delegate_action``).
        hint: optional one-line corrective suggestion.
    """

    def __init__(self, reason: str, *, rule: str, hint: str | None = None):
        """Initialise the violation with a reason, rule id, and optional hint.

        Args:
            reason (str): Human-readable description of the violation.
            rule (str): Short rule identifier mirroring upstream
                ``PolicyDenied.rule``.
            hint (str | None): Optional one-line corrective suggestion.
        """
        super().__init__(reason)
        self.rule = rule
        self.hint = hint


class PolicyAware:
    """Local validator for intents the robustness reactor is about to emit.

    Construct once and reuse; the validator is stateless. Use either
    :meth:`assert_payload_complete` (raise on first violation) or
    :meth:`validate_all` (collect every violation for diagnostics).
    """

    def assert_payload_complete(self, intent: Intent) -> None:
        """Raise :class:`PolicyViolation` if the intent is not emit-safe.

        Order of checks matches upstream ``PolicyGate.validate_intent``:
        role allowlist -> required fields -> per-intent extra rules.

        Args:
            intent (Intent): The intent about to be emitted.

        Raises:
            PolicyViolation: If the intent fails any local PolicyGate check.
        """
        self._check_role(intent)
        self._check_required_fields(intent)
        self._check_per_intent(intent)

    def validate_all(self, intent: Intent) -> list[PolicyViolation]:
        """Return every violation without raising.

        Used in unit tests to assert exhaustive coverage of a malformed
        intent. The reactor calls :meth:`assert_payload_complete` only.

        Args:
            intent (Intent): The intent to validate.

        Returns:
            list[PolicyViolation]: Every violation found; empty when the intent
            is emit-safe.
        """
        violations: list[PolicyViolation] = []
        for check in (self._check_role, self._check_required_fields, self._check_per_intent):
            try:
                check(intent)
            except PolicyViolation as exc:
                violations.append(exc)
        return violations

    # -- internal checks -------------------------------------------------

    def _check_role(self, intent: Intent) -> None:
        """Verify the intent type is in the robustness role allowlist.

        Args:
            intent (Intent): The intent to check.

        Raises:
            PolicyViolation: If the role may not emit this intent type.
        """
        if intent.type not in ROBUSTNESS_ALLOWED_INTENTS:
            raise PolicyViolation(
                f"role=robustness cannot emit intent_type={intent.type.value!r}",
                rule="role",
                hint="see ROBUSTNESS_ALLOWED_INTENTS in role.envelope",
            )

    def _check_required_fields(self, intent: Intent) -> None:
        """Ensure all payload fields required for the intent type are present.

        Args:
            intent (Intent): The intent to check.

        Raises:
            PolicyViolation: If a required payload field is missing.
        """
        required = PAYLOAD_REQUIRED.get(intent.type, ())
        payload = intent.payload or {}
        for field_name in required:
            if field_name not in payload:
                raise PolicyViolation(
                    f"intent_type={intent.type.value!r} missing required "
                    f"payload field {field_name!r}",
                    rule="payload",
                    hint=f"required fields: {required!r}",
                )

    def _check_per_intent(self, intent: Intent) -> None:
        """Run the extra, type-specific payload checks for an intent.

        Args:
            intent (Intent): The intent to check.

        Raises:
            PolicyViolation: If the intent-type-specific check fails.
        """
        payload = intent.payload or {}
        if intent.type == IntentType.ALERT:
            self._check_alert(payload)
        elif intent.type == IntentType.ESCALATE_STRATEGY_CHANGE:
            self._check_escalate(payload)
        elif intent.type == IntentType.KILL_TASK:
            self._check_kill_task(payload)
        elif intent.type == IntentType.FORCE_DISPATCH:
            self._check_force_dispatch(payload)
        elif intent.type == IntentType.PRUNE_BRANCH:
            self._check_prune_branch(payload)
        elif intent.type == IntentType.DELEGATE:
            self._check_delegate(payload)
        elif intent.type == IntentType.UPDATE_STATE:
            self._check_update_state(payload)
        elif intent.type == IntentType.SEND_MESSAGE:
            self._check_send_message(payload)

        if intent.type in ROBUSTNESS_ONLY_INTENTS:
            # ``source`` is not validatable locally; upstream PolicyGate
            # enforces source=robustness via ROBUSTNESS_ONLY_SOURCE_ALLOWLIST.
            pass

    def _check_alert(self, payload: dict[str, Any]) -> None:
        """Validate an ``alert`` payload's severity and summary.

        Args:
            payload (dict[str, Any]): The alert intent payload.

        Raises:
            PolicyViolation: If the severity is unknown or the summary empty.
        """
        severity = str(payload.get("severity", "")).strip()
        if severity not in ALERT_SEVERITIES:
            raise PolicyViolation(
                f"alert.severity={severity!r} not in "
                f"{sorted(ALERT_SEVERITIES)!r}",
                rule="payload",
            )
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            raise PolicyViolation(
                "alert.summary must be a non-empty string",
                rule="payload",
            )

    def _check_escalate(self, payload: dict[str, Any]) -> None:
        """Validate an ``escalate_strategy_change`` payload.

        Args:
            payload (dict[str, Any]): The escalate intent payload.

        Raises:
            PolicyViolation: If reason or next_action_hint is empty, or the
                optional severity is invalid.
        """
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise PolicyViolation(
                "escalate_strategy_change.reason must be non-empty",
                rule="payload",
            )
        hint = str(payload.get("next_action_hint", "")).strip()
        if not hint:
            raise PolicyViolation(
                "escalate_strategy_change.next_action_hint must be non-empty",
                rule="payload",
            )
        severity = payload.get("severity")
        if severity is not None and severity not in ALERT_SEVERITIES:
            raise PolicyViolation(
                f"escalate severity={severity!r} not in "
                f"{sorted(ALERT_SEVERITIES)!r}",
                rule="payload",
            )

    def _check_kill_task(self, payload: dict[str, Any]) -> None:
        """Validate a ``kill_task`` payload, including its scope.

        Args:
            payload (dict[str, Any]): The kill_task intent payload.

        Raises:
            PolicyViolation: If task_id/reason is empty or the scope is not in
                the robustness-allowed scope set.
        """
        task_id = str(payload.get("task_id", "")).strip()
        if not task_id:
            raise PolicyViolation("kill_task.task_id must be non-empty", rule="payload")
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise PolicyViolation("kill_task.reason must be non-empty", rule="payload")
        scope = str(payload.get("scope", "task")).strip()
        if scope not in KILL_TASK_ALLOWED_SCOPES:
            raise PolicyViolation(
                f"kill_task.scope={scope!r} not in "
                f"{sorted(KILL_TASK_ALLOWED_SCOPES)!r}",
                rule="kill_scope",
                hint="upstream v0.6 keeps server / process kills out per IR-5",
            )

    def _check_force_dispatch(self, payload: dict[str, Any]) -> None:
        """Validate a ``force_dispatch`` payload.

        Args:
            payload (dict[str, Any]): The force_dispatch intent payload.

        Raises:
            PolicyViolation: If task_id or reason is empty.
        """
        task_id = str(payload.get("task_id", "")).strip()
        if not task_id:
            raise PolicyViolation(
                "force_dispatch.task_id must be non-empty", rule="payload"
            )
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise PolicyViolation(
                "force_dispatch.reason must be non-empty", rule="payload"
            )

    def _check_prune_branch(self, payload: dict[str, Any]) -> None:
        """Validate a ``prune_branch`` payload.

        Args:
            payload (dict[str, Any]): The prune_branch intent payload.

        Raises:
            PolicyViolation: If family or reason is empty.
        """
        family = str(payload.get("family", "")).strip()
        if not family:
            raise PolicyViolation(
                "prune_branch.family must be non-empty", rule="payload"
            )
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise PolicyViolation(
                "prune_branch.reason must be non-empty", rule="payload"
            )

    def _check_delegate(self, payload: dict[str, Any]) -> None:
        """Validate a ``delegate`` payload's action name against the allowlist.

        Args:
            payload (dict[str, Any]): The delegate intent payload.

        Raises:
            PolicyViolation: If action_name is empty or not allowed for the
                robustness role.
        """
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyViolation(
                "delegate.action_name must be non-empty", rule="payload"
            )
        if action_name not in ROBUSTNESS_DELEGATE_ACTIONS:
            raise PolicyViolation(
                f"delegate.action_name={action_name!r} not allowed for "
                f"robustness; allowed: "
                f"{sorted(ROBUSTNESS_DELEGATE_ACTIONS)!r}",
                rule="delegate_action",
                hint="kernel-owned actions go via REQUEST(target_agent='kernel')",
            )

    def _check_update_state(self, payload: dict[str, Any]) -> None:
        """Validate an ``update_state`` payload's field allowlist.

        Args:
            payload (dict[str, Any]): The update_state intent payload.

        Raises:
            PolicyViolation: If changes is not a non-empty dict, touches core
                state fields, or includes fields outside the robustness
                allowlist.
        """
        changes = payload.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise PolicyViolation(
                "update_state.changes must be a non-empty dict",
                rule="payload",
            )
        core_fields = sorted(set(changes.keys()) & CORE_STATE_FIELDS)
        if core_fields:
            raise PolicyViolation(
                "update_state cannot mutate core state fields: "
                f"{core_fields!r}",
                rule="state_field",
            )
        non_robust = sorted(set(changes.keys()) - ROBUSTNESS_STATE_FIELDS)
        if non_robust:
            raise PolicyViolation(
                "update_state contains fields outside robustness allowlist: "
                f"{non_robust!r}",
                rule="state_field",
                hint=f"allowed: {sorted(ROBUSTNESS_STATE_FIELDS)!r}",
            )

    def _check_send_message(self, payload: dict[str, Any]) -> None:
        """Validate a ``send_message`` payload's topic.

        Unknown topics are not rejected (upstream soft-degrades them to
        ``observation``), only an empty topic is a violation.

        Args:
            payload (dict[str, Any]): The send_message intent payload.

        Raises:
            PolicyViolation: If the topic is empty.
        """
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            raise PolicyViolation(
                "send_message.topic must be non-empty", rule="payload"
            )
        # Unknown topics are not rejected (upstream soft-degrades to observation).
