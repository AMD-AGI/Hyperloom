"""Compatibility shim — intent schema lives in ``protocol.intent``."""

from inference_optimizer.protocol.intent import (
    Intent,
    IntentType,
    NoIntentEmitted,
    validate_envelope,
)

__all__ = [
    "Intent",
    "IntentType",
    "NoIntentEmitted",
    "validate_envelope",
]
