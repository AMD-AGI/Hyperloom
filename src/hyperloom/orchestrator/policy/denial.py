# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The exception every policy rule raises to reject an intent."""

from __future__ import annotations

__all__ = ["PolicyDenied"]


class PolicyDenied(RuntimeError):
    """Intent rejected by PolicyGate.

    Attributes:
        rule: short identifier of the rule that fired.
        hint: optional one-line agent-actionable suggestion.
    """

    def __init__(self, reason: str, *, rule: str | None = None, hint: str | None = None):
        """Initialise the denial with a human-readable reason and metadata.

        Args:
            reason (str): human-readable explanation passed to the base
                ``RuntimeError``; surfaced in logs and the policy_denied
                observation event.
            rule (str | None): short identifier of the rule that fired,
                used by the Coordinator to classify the denial. Defaults
                to ``None``.
            hint (str | None): optional one-line, agent-actionable
                suggestion describing the canonical fix. Defaults to
                ``None``.
        """
        super().__init__(reason)
        self.rule = rule
        self.hint = hint
