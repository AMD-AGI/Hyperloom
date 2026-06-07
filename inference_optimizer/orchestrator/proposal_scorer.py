# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ProposalScorer — purely advisory multi-model scorer for specialist proposals.

Scores each variant with one or more LLM models (advisory only; never
sorts/ranks/auto-selects, never touches Critic/PolicyGate). Each model scored
independently via ``asyncio.gather``.

Output schema (``score`` return value)::

    {"scale": "0-10", "models": {"<slug>": {"<name>": {"score": <0-10>, "reason": "<str>"}}}, "errors": {...}}

Test seam: pass ``client_factory`` to bypass real client construction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .backends.base import parse_call_timeout_env

log = logging.getLogger(__name__)


DEFAULT_SCORER_MODELS: tuple[str, ...] = (
    "claude-opus-4-8",
    "gpt-5.5",
    # Kimi K2.6 / Gemini 3.1 Pro are reasoning models needing the larger 4096 token cap; Gemini MUST carry the ``gemini/`` prefix (bare slug routes to a broken Vertex ADC path).
    "dvue-aoai-005-Kimi-K2.6",
    "gemini/gemini-3.1-pro-preview",
)

# Soft cap on proposals fed to each model so a pathological set can't blow up the prompt.
_MAX_PROPOSALS_SCORED: int = 16
_MAX_FIELD_CHARS: int = 600


_SCORING_INSTRUCTIONS = """
You are scoring candidate serving-config variants proposed by a perf
specialist for an LLM-inference optimization run on AMD GPUs.

For EACH proposal below, output a single composite score from 0 to 10:
how likely / how much this variant improves serving THROUGHPUT for the
stated gap, grounded ONLY in the evidence shown. 10 = very likely a
large win; 0 = irrelevant or likely harmful. Be calibrated, not generous.

Output ONLY one compact JSON object, no prose, no markdown fence needed:

{"scores": {"<proposal_name>": {"score": <0-10 number>, "reason": "<= 15 words"}}}

Rules:
- One entry per proposal, keyed by its exact "name".
- "reason" MUST be <= 15 words. Keep it terse to keep the reply short.
- Do not add keys other than "score" and "reason".
""".strip()


# Match a fenced ```json``` block (preferred) then a bare top-level "scores" object.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{.*?\"scores\".*\})", re.DOTALL)


def _extract_scores_json(text: str) -> dict[str, Any] | None:
    """Pull the first valid ``{"scores": {...}}`` object out of a reply."""
    if not text:
        return None
    for m in _FENCED_JSON_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "scores" in data:
            return data
    for m in _BARE_JSON_RE.finditer(text):
        candidate = m.group(1)
        for end in range(len(candidate), 0, -1):
            try:
                data = json.loads(candidate[:end])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "scores" in data:
                return data
            break  # parsed but wrong shape; don't keep shrinking
    return None


def _clip(value: Any, *, limit: int = _MAX_FIELD_CHARS) -> str:
    s = "" if value is None else str(value)
    return s if len(s) <= limit else (s[:limit] + "…")


def _coerce_score(raw: Any) -> float | None:
    """Coerce a model-emitted score into a clamped [0, 10] float."""
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN
        return None
    return max(0.0, min(10.0, val))


def _normalise_model_scores(
    parsed: dict[str, Any], *, proposal_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Project parsed ``{"scores": {...}}`` onto known names (drop unknowns, clamp [0,10], truncate reasons)."""
    out: dict[str, dict[str, Any]] = {}
    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        return out
    known = set(proposal_names)
    for name, entry in scores.items():
        key = str(name)
        if key not in known or not isinstance(entry, dict):
            continue
        score = _coerce_score(entry.get("score"))
        if score is None:
            continue
        out[key] = {
            "score": score,
            "reason": _clip(entry.get("reason"), limit=160),
        }
    return out


@dataclass
class ProposalScorer:
    """Advisory multi-model scorer (see module docstring)."""

    models: tuple[str, ...] = DEFAULT_SCORER_MODELS
    api_key_env: str = "ANTHROPIC_AUTH_TOKEN"  # AMD proxy; accepts OPENAI too
    base_url_env: str = "OPENAI_BASE_URL"
    # 4096 so reasoning raters (e.g. Kimi K2.6) can finish reasoning and still emit the scores JSON.
    max_completion_tokens: int = 4096
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_PROPOSAL_SCORER_TIMEOUT_SEC",
            default=120.0,
        )
    )

    # Test seam — set to bypass real OpenAI client construction.
    client_factory: Callable[[], Any] | None = None

    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.models = tuple(
            m for m in (str(x).strip() for x in (self.models or ())) if m
        )
        if self.client_factory is not None:
            self._client = self.client_factory()
            return
        # Lazy: construct on first use so an unconfigured env degrades per-call, not at boot.
        self._client = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai SDK not installed; run `pip install openai>=1.50`"
            ) from exc
        api_key = (
            os.environ.get(self.api_key_env)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} not set in env (ProposalScorer cannot auth)"
            )
        base_url = (
            os.environ.get(self.base_url_env)
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
        )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    def _build_prompt(
        self, *, gap: dict[str, Any], proposals: list[dict[str, Any]],
    ) -> str:
        """Build ONE group-scoring prompt covering every proposal."""
        lines: list[str] = ["=== Gap ==="]
        lines.append(f"domain: {_clip(gap.get('domain'), limit=80)}")
        lines.append(
            f"gap_canonical_id: {_clip(gap.get('gap_canonical_id'), limit=160)}"
        )
        symptom = gap.get("gap_symptom") or gap.get("summary")
        if symptom:
            lines.append(f"symptom: {_clip(symptom)}")
        evidence = gap.get("gap_evidence")
        if evidence:
            lines.append(f"evidence: {_clip(json.dumps(evidence, sort_keys=True))}")

        lines.append("")
        lines.append("=== Proposals to score ===")
        for i, p in enumerate(proposals):
            name = str(p.get("name") or f"proposal_{i}")
            lines.append(f"- name: {name}")
            if p.get("extra_args"):
                lines.append(f"  extra_args: {_clip(p.get('extra_args'))}")
            if p.get("extra_envs"):
                lines.append(
                    f"  extra_envs: {_clip(json.dumps(p.get('extra_envs'), sort_keys=True))}"
                )
            if p.get("reason"):
                lines.append(f"  reason: {_clip(p.get('reason'))}")
            if p.get("kb_evidence"):
                lines.append(
                    f"  kb_evidence: {_clip(json.dumps(p.get('kb_evidence'), sort_keys=True))}"
                )
        return "\n".join(lines)

    async def _score_one_model(
        self, model: str, prompt: str, proposal_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Score every proposal with a single model (raises on failure; caller records the per-model error)."""
        client = self._ensure_client()
        full_prompt = f"{prompt}\n\n{_SCORING_INSTRUCTIONS}"
        messages = [{"role": "user", "content": full_prompt}]
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=self.max_completion_tokens,
                ),
                timeout=self.call_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"timed out after {self.call_timeout_s:.0f}s"
            ) from exc
        text = (resp.choices[0].message.content or "")
        parsed = _extract_scores_json(text)
        if parsed is None:
            raise RuntimeError(
                f"no parseable scores JSON (reply_chars={len(text)})"
            )
        return _normalise_model_scores(parsed, proposal_names=proposal_names)

    async def score(
        self, *, gap: dict[str, Any], proposals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Score ``proposals`` against ``gap`` with every configured model (per-model failures land in ``errors``, never raised)."""
        proposals = [p for p in (proposals or []) if isinstance(p, dict)]
        if not proposals or not self.models:
            return {"scale": "0-10", "models": {}, "errors": {}}
        if len(proposals) > _MAX_PROPOSALS_SCORED:
            proposals = proposals[:_MAX_PROPOSALS_SCORED]
        proposal_names = [
            str(p.get("name") or f"proposal_{i}")
            for i, p in enumerate(proposals)
        ]
        prompt = self._build_prompt(gap=gap, proposals=proposals)

        results = await asyncio.gather(
            *(
                self._score_one_model(m, prompt, proposal_names)
                for m in self.models
            ),
            return_exceptions=True,
        )
        models_out: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for model, res in zip(self.models, results):
            if isinstance(res, BaseException):
                errors[model] = f"{type(res).__name__}: {str(res)[:200]}"
                log.warning(
                    "ProposalScorer: model=%s failed: %r", model, res,
                )
                continue
            if res:
                models_out[model] = res
            else:
                errors[model] = "no usable scores returned"
        return {"scale": "0-10", "models": models_out, "errors": errors}


__all__ = ["ProposalScorer", "DEFAULT_SCORER_MODELS", "_extract_scores_json"]
