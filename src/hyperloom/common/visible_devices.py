# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Visible-device mask names and parsing, shared by every layer that reads one.

The same tuple of mask variables and the same "split on ``,``/``;``, keep the
first occurrence of each non-negative int" parser had accumulated five separate
copies (``bus/gpu_pool``, ``policy/gate``, ``actions/executors/_ray_serving``,
``common/env_safety``, ``loop/coordinator_helpers``), and their empty-mask
semantics had already drifted apart. This module is the single definition; it
imports nothing outside the standard library so the pure-helper layers can use
it without dragging in the SQLite connection ``gpu_pool`` owns.

Two var tuples, deliberately distinct:

* :data:`COUNTING_VISIBLE_DEVICE_VARS` — what ``gpu_pool`` / ``gate`` consult to
  answer "how many GPUs does this process have". Identical to what those layers
  always used; widening it would change GPU accounting repo-wide. It is DERIVED
  from the chain below by subtracting an explicit exclusion set, so the two can
  never drift: a new var is counted unless someone names it as uncounted, and a
  test asserts every chain member is classified.
* :data:`VISIBLE_DEVICE_VARS` — the full ROCm pin-resolution chain, used when
  answering "where is this run pinned". ``HSA_VISIBLE_DEVICES`` is ROCr's legacy
  name and ``GPU_DEVICE_ORDINAL`` is the legacy HIP-level filter; a run pinned
  with either is really pinned, and omitting them left it reported as unpinned.

The ROCr-level and HIP-level groups are exposed separately because the
distinction is load-bearing: a ROCr-level mask renumbers the devices the child
sees (so ids inside it are LOGICAL), while a HIP-level mask indexes into
whatever ROCr already exposed (so its ids are absolute unless a ROCr mask is
also in force).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "COUNTING_VISIBLE_DEVICE_VARS",
    "GPU_MASK_ENV_NAMES",
    "HIP_LEVEL_VARS",
    "ROCR_LEVEL_VARS",
    "VISIBLE_DEVICE_VARS",
    "effective_mask_tokens",
    "is_rocr_level",
    "mask_tokens",
    "parse_device_list",
]

#: ROCr-level masks: these slice the device set and renumber it ``0..N-1``.
#: ``HSA_VISIBLE_DEVICES`` is the legacy spelling; ``ROCR_VISIBLE_DEVICES``
#: wins when both are set.
ROCR_LEVEL_VARS: tuple[str, ...] = (
    "ROCR_VISIBLE_DEVICES",
    "HSA_VISIBLE_DEVICES",
)

#: HIP-level masks: these index INTO whatever ROCr exposed.
HIP_LEVEL_VARS: tuple[str, ...] = (
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)

#: Full pin-resolution precedence: ROCr level before HIP level, canonical
#: spelling before legacy within each level.
VISIBLE_DEVICE_VARS: tuple[str, ...] = ROCR_LEVEL_VARS + HIP_LEVEL_VARS

#: Vars the capacity-counting layers deliberately do NOT read.
#:
#: Both are legacy aliases of a var already in the counting set, and both are
#: honoured only when their modern spelling is absent — so counting them would
#: not find a GPU the modern spelling missed, it would only change the answer on
#: hosts that happen to export the legacy name. That is a repo-wide GPU
#: accounting change, not a bugfix, so it stays out until someone makes it
#: deliberately.
_UNCOUNTED_VISIBLE_DEVICE_VARS: frozenset[str] = frozenset(
    {
        "HSA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    }
)

#: The subset the capacity-counting layers read — ``gpu_pool``, ``policy.gate``,
#: ``actions.executors._ray_serving``. DERIVED from
#: :data:`VISIBLE_DEVICE_VARS` rather than re-listed, so a var added to the
#: precedence chain is counted by default and can only be left out by naming it
#: in :data:`_UNCOUNTED_VISIBLE_DEVICE_VARS`. The two tuples cannot silently
#: drift apart: ``test_visible_devices.py`` asserts every chain member is
#: classified exactly once.
COUNTING_VISIBLE_DEVICE_VARS: tuple[str, ...] = tuple(
    var for var in VISIBLE_DEVICE_VARS if var not in _UNCOUNTED_VISIBLE_DEVICE_VARS
)

#: Every name that selects hardware rather than tuning it (superset of the
#: precedence chain), for env scrubbing.
GPU_MASK_ENV_NAMES: frozenset[str] = frozenset(VISIBLE_DEVICE_VARS)


def is_rocr_level(var: str) -> bool:
    """Does ``var`` slice and renumber the device set (rather than index it)?

    Args:
        var: A visible-devices env var name.

    Returns:
        ``True`` for the ROCr-level masks, whose member ids are logical
        positions from the child's point of view.
    """
    return str(var or "") in ROCR_LEVEL_VARS


def mask_tokens(raw: Any) -> list[str]:
    """Split a visible-devices mask into its device tokens.

    Tokens are NOT required to be numeric: ROCm accepts GPU UUID masks
    (``ROCR_VISIBLE_DEVICES=GPU-a1b2c3,GPU-d4e5f6``), and those still say how
    MANY devices the child will see, which is all the logical-index arithmetic
    needs.

    Args:
        raw: A ``,``/``;``-separated mask, or an already-parsed YAML sequence.

    Returns:
        Non-empty tokens in order, duplicates preserved.
    """
    if isinstance(raw, (list, tuple)):
        parts = [str(p) for p in raw]
    else:
        parts = str(raw if raw is not None else "").replace(";", ",").split(",")
    return [tok for tok in (p.strip() for p in parts) if tok]


def _is_negative_ordinal(tok: str) -> bool:
    """Is ``tok`` a negative device ordinal, i.e. a token that names no device?

    Written as a positive test rather than ``int(tok) < 0`` in a ``try`` so the
    non-numeric case (a GPU UUID) is an ordinary ``False`` rather than a
    swallowed ``ValueError``.

    Args:
        tok: One already-stripped mask token.

    Returns:
        ``True`` only for a leading ``-`` followed by digits.
    """
    return tok.startswith("-") and tok[1:].isdigit()


def effective_mask_tokens(raw: Any) -> list[str]:
    """The devices a mask actually exposes, in the order the runtime sees them.

    :func:`mask_tokens` is the LITERAL split; this is the *effective* set. ROCm
    exposes ``ROCR_VISIBLE_DEVICES="3,3,2"`` as two devices, not three, and
    drops a negative ordinal. Deriving both the device COUNT and the forwarded
    id list from this one function is what keeps them from disagreeing: counting
    literal tokens inflates the count (logical index 2 of a 2-device set is
    invalid), while re-serializing from the parsed ints deflates it.

    Non-numeric tokens are kept — a UUID mask names real devices — so this
    cannot filter out a genuinely invalid non-numeric entry; it removes only the
    two forms that are unambiguously not extra devices.

    Args:
        raw: A ``,``/``;``-separated mask, or an already-parsed YAML sequence.

    Returns:
        Tokens with duplicates and negative ordinals removed, first-seen order.
    """
    out: list[str] = []
    for tok in mask_tokens(raw):
        if tok in out or _is_negative_ordinal(tok):
            continue
        out.append(tok)
    return out


def parse_device_list(raw: Any) -> list[int]:
    """Parse a visible-devices mask into absolute NUMERIC device ids.

    Args:
        raw: A ``,``/``;``-separated mask (``"4,5,6,7"``) or a YAML sequence;
            ``None`` and malformed entries are tolerated.

    Returns:
        Unique non-negative ids in first-seen order; ``[]`` for an empty mask
        and for a well-formed but non-numeric one (e.g. a UUID mask), which is
        why callers that need a device COUNT must use
        :func:`effective_mask_tokens`.
    """
    out: list[int] = []
    for tok in effective_mask_tokens(raw):
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out
