# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recipe projection: the ordered enablement replay contract and its sufficiency.

``recipe_steps`` serializes the setup -> build -> patch ordering enablement
executes but never recorded; ``replay_sufficiency`` is the authoritative verdict
on whether that recipe can actually be replayed.
"""

from .credentials import (
    classify_credential_class,
    classify_credential_value,
    detect_credential_channels,
    installer_class,
    sanitize_command_text,
)
from .steps import build_recipe_steps
from .sufficiency import evaluate_replay_sufficiency, read_status

__all__ = [
    "build_recipe_steps",
    "classify_credential_class",
    "classify_credential_value",
    "detect_credential_channels",
    "evaluate_replay_sufficiency",
    "installer_class",
    "read_status",
    "sanitize_command_text",
]
