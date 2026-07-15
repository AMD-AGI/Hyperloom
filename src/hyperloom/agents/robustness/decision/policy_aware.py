# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Local payload-schema check mirroring upstream ``PolicyGate.validate_intent``.

PolicyGate is the server-side source of truth (rejects with ``PolicyDenied``
+ a ``policy_denied`` inbox observation), but that costs ≥1 silent tick.
:class:`PolicyAware` runs the same checks locally and raises
:class:`PolicyViolation` so unit tests catch schema drift immediately. The
mirrored schema is a subset limited to robustness-emittable intents.

The per-intent rules are not re-listed here: this module only orders the three
gate stages (role -> required fields -> per-intent) and dispatches the last one
through the single ``role.envelope.INTENT_SPEC`` table. ``PolicyViolation`` and
the validators live next to the intent contract in ``role.envelope``;
``tests/test_role_contract.py`` cross-checks that table against the upstream one.
"""

from __future__ import annotations

from ..role.envelope import (
    INTENT_SPEC,
    PAYLOAD_REQUIRED,
    Intent,
    PolicyViolation,
    ROBUSTNESS_ALLOWED_INTENTS,
)


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

        Reads the required-field tuple from the single ``PAYLOAD_REQUIRED``
        map (itself derived from ``INTENT_SPEC``).

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
                    f"intent_type={intent.type.value!r} missing required payload field {field_name!r}",
                    rule="payload",
                    hint=f"required fields: {required!r}",
                )

    def _check_per_intent(self, intent: Intent) -> None:
        """Run the type-specific payload validator from the spec table.

        Dispatches through ``INTENT_SPEC`` so there is no parallel if/elif
        chain to keep in sync with the required-field map. Intent types with
        no spec entry (never emitted by robustness) have no extra rules.

        Args:
            intent (Intent): The intent to check.

        Raises:
            PolicyViolation: If the intent-type-specific validator fails.
        """
        spec = INTENT_SPEC.get(intent.type)
        if spec is not None:
            spec.validator(intent.payload or {})


__all__ = ["PolicyAware"]
