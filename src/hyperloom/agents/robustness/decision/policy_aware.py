# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local payload-schema check mirroring upstream ``PolicyGate.validate_intent``."""

from __future__ import annotations

from ..role.envelope import (
    INTENT_SPEC,
    PAYLOAD_REQUIRED,
    Intent,
    PolicyViolation,
    ROBUSTNESS_ALLOWED_INTENTS,
)


class PolicyAware:
    """Local validator for intents the robustness reactor is about to emit."""

    def assert_payload_complete(self, intent: Intent) -> None:
        """Raise :class:`PolicyViolation` if the intent is not emit-safe."""
        self._check_role(intent)
        self._check_required_fields(intent)
        self._check_per_intent(intent)

    def _check_role(self, intent: Intent) -> None:
        """Verify the intent type is in the robustness role allowlist."""
        if intent.type not in ROBUSTNESS_ALLOWED_INTENTS:
            raise PolicyViolation(
                f"role=robustness cannot emit intent_type={intent.type.value!r}",
                rule="role",
                hint="see ROBUSTNESS_ALLOWED_INTENTS in role.envelope",
            )

    def _check_required_fields(self, intent: Intent) -> None:
        """Ensure all payload fields required for the intent type are present."""
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
        """Run the type-specific payload validator from the spec table."""
        spec = INTENT_SPEC.get(intent.type)
        if spec is not None:
            spec.validator(intent.payload or {})


__all__ = ["PolicyAware"]
