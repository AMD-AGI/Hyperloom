# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's stop reason."""

from __future__ import annotations

from hyperloom_common.env import env_bool
from hyperloom_common.timeutil import now_iso


class _RunSignalsMixin:
    """Commits the session's stop reason."""

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
