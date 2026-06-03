"""Map a local model path to the InferenceX upstream's display name.

InferenceX (https://inferencex.semianalysis.com) refers to models by
short human names (``MiniMax-M2.5``, ``DeepSeek-R1-0528``), but local
weights typically live at HuggingFace-style paths like
``/wekafs/models/MiniMaxAI-MiniMax-M2.5``. This module owns the
translation.

Hard rules:

* The mapping is **best-effort**. When we are not confident, we return
  ``None`` and the caller gracefully skips target_analysis. Never raise.
* The known-models list is hardcoded here (it changes ~monthly). We
  intentionally do NOT hit ``/filters`` at runtime to keep
  target_analysis at < 250 ms total.
* Matching is case-insensitive; vendor prefixes from common HF repo
  conventions (``MiniMaxAI-``, ``deepseek-ai-``, ``meta-llama-``, ...)
  are stripped before comparison.

If you add a new model to the upstream you must add it here (and the
unit test in ``tests/test_baseline_comparison.py`` will catch
out-of-sync drift).
"""

from __future__ import annotations

import re
from pathlib import Path


KNOWN_INFERENCEX_MODELS: tuple[str, ...] = (
    "DeepSeek-R1-0528",
    "GLM-5",
    "gpt-oss-120b",
    "Llama-3.3-70B-Instruct-FP8",
    "Qwen-3.5-397B-A17B",
    "Kimi-K2.5",
    "MiniMax-M2.5",
)

_VENDOR_PREFIX_RE = re.compile(
    r"^(MiniMaxAI[-_]|deepseek-ai[-_]|deepseek[-_]|meta-llama[-_]|"
    r"Qwen[-_]|moonshotai[-_]|openai[-_]|google[-_]|microsoft[-_]|"
    r"zhipuai[-_]|THUDM[-_])",
    re.IGNORECASE,
)


def to_inferencex_name(model_path_or_name: str) -> str | None:
    """Translate a local path / HF repo string into an InferenceX display name.

    Returns the canonical name from :data:`KNOWN_INFERENCEX_MODELS` if a
    match is found, ``None`` otherwise. Caller treats ``None`` as
    "skip target_analysis for this run" — never as an error.

    Matching algorithm:

    1. Take the basename (``Path.name``).
    2. Strip a leading vendor prefix.
    3. Case-insensitive exact match against the known list.

    Args:
        model_path_or_name (str): A local weights path, HuggingFace repo
            string, or bare model name to translate.

    Returns:
        str | None: The canonical InferenceX display name when a confident
            match is found, otherwise ``None``.
    """
    if not model_path_or_name:
        return None
    raw = str(model_path_or_name).strip()
    if not raw:
        return None

    candidate = Path(raw).name if ("/" in raw or "\\" in raw) else raw
    stripped = _VENDOR_PREFIX_RE.sub("", candidate, count=1)
    needle = stripped.casefold()

    for known in KNOWN_INFERENCEX_MODELS:
        if known.casefold() == needle:
            return known

    return None


__all__ = [
    "KNOWN_INFERENCEX_MODELS",
    "to_inferencex_name",
]
