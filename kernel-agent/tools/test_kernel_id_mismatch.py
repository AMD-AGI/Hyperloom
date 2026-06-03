"""kernel_id mismatch handling at the kernel-agent boundary.

The Orchestration LLM occasionally supplies a ``kernel_id`` that does not
match any TraceLens candidate (e.g. it echoes an operator name like
``aiter.silu_and_mul`` or a hallucinated ``kn001`` instead of the real
``k001``). Previously ``find_candidate`` raised ``KeyError`` on the first
lookup, crashing the entire kernel-optimization subprocess before GEAK was
ever invoked.

Contract:

* ``find_candidate`` resolves by ``kernel_id`` / ``name`` exactly, then via a
  normalized match (case-insensitive, ``kn``/``rn`` prefix folded to ``k``).
* When nothing matches it returns ``None`` instead of raising, so the caller
  can skip the kernel gracefully rather than aborting the whole run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_optimization as ko  # noqa: E402


CANDIDATES = [
    {"kernel_id": "k001", "name": "aten::mm"},
    {"kernel_id": "k002", "name": "sgl_kernel::silu_and_mul"},
    {"kernel_id": "k010", "name": "aiter::rmsnorm"},
]


def test_exact_id_match():
    assert ko.find_candidate(CANDIDATES, "k002")["name"] == "sgl_kernel::silu_and_mul"


def test_name_match():
    assert ko.find_candidate(CANDIDATES, "aten::mm")["kernel_id"] == "k001"


def test_normalized_prefix_match():
    # LLM hallucinated ``kn001`` / ``rn010``; fold the synthetic prefix back to
    # the real ``k`` numbering.
    assert ko.find_candidate(CANDIDATES, "kn001")["kernel_id"] == "k001"
    assert ko.find_candidate(CANDIDATES, "rn010")["kernel_id"] == "k010"


def test_unknown_id_returns_none_not_raise():
    # Pure hallucination that maps to nothing must NOT crash the subprocess.
    assert ko.find_candidate(CANDIDATES, "aiter.silu_and_mul") is None
    assert ko.find_candidate(CANDIDATES, "framework_sglang_silu_and_mul_m64") is None
