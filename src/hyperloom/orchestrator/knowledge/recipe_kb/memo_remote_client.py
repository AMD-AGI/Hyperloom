"""Read-side recipe client backed by a MeMo MEM (Memory-as-a-Model) service.

The MEM answers recipe questions closed-book over an OpenAI-compatible endpoint:
no retrieval, no KB round-trip. It is a *lossy* projection of the recipe KB it
was trained on, so every row it produces carries ``authority=INFERRED`` and a
``provenance.source="memo"`` marker.

Three contracts keep the lossy source from corrupting measured knowledge:

  * **Read-only.** Only the read half of the duck-typed remote interface is
    implemented (``get_recipe`` / ``search``). The local store stays the sole
    write target.
  * **No cross-model search.** ``search`` always returns ``[]``. The MEM only
    knows facts keyed by a full identity tuple, so answering a relaxed label
    match (the T0 L2/L3 cascade) would be fabrication. Returning empty lets the
    caller fall through to a backend holding real rows.
  * **Last resort.** :class:`RecipeKB` consults its remote *before* the local
    store and falls through only on a remote miss, so a remote answer shadows
    the local row unconditionally. That ordering is right for an authoritative
    central service but backwards for a lossy MEM, which would mask measured
    rows. Passing ``local_guard`` restores the intended precedence: the MEM
    abstains for any identity the local store can already answer.
  * **No config by default.** ``best_config`` is withheld unless
    ``MEMO_KB_ALLOW_CONFIG`` is set. T0 stamps confidence 1.0 on any exact
    identity match regardless of ``authority``, and the PRELUDE warm-replay gate
    admits a config on confidence alone — so a fabricated launch recipe would be
    applied to a live server. Withholding it makes that gate skip with
    ``best_config_empty`` while the throughput still flows for observation.

The MEM does not abstain: for an identity it never saw during training it still
emits a confident, fabricated number. Callers that need a capability boundary
must gate on an external manifest; this client does not attempt to guess.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from hyperloom.inference_optimizer.recipe_snapshot_constants import recipe_canonical_id
from hyperloom.orchestrator.knowledge.recipe_kb import RemoteRecipeClientError
from hyperloom.orchestrator.knowledge.recipe_kb.canonical_id import cid_to_path_components

log = logging.getLogger(__name__)

DEFAULT_CONFIDENCE = 0.5
DEFAULT_MAX_TOKENS = 160
DEFAULT_TIMEOUT_SEC = 10.0

# SGLang still emits an empty <think></think> block even with thinking disabled.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# "... is 19703.6 tokens/second." / "... reached 19703.6 tok/s."
_THROUGHPUT_RE = re.compile(
    r"(?<![\d.])(?P<value>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:tokens?/second|tokens?\s+per\s+second|tok/s)\b",
    re.IGNORECASE,
)
# Launch recipes arrive in one backtick span, optionally "args | env: K=V K2=V2".
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_ENV_SPLIT_RE = re.compile(r"\|\s*env:\s*", re.IGNORECASE)
_ENV_PAIR_RE = re.compile(r"(?P<key>[A-Z][A-Z0-9_]*)=(?P<value>\S+)")


def hyperloom_model_slug(raw: str) -> str:
    """Slug a model reference the way ``canonical_id`` does.

    Round-trips through the public canonical-id helpers rather than
    reimplementing the rule, so the two can never drift apart.

    Args:
        raw: A model path, HF id, or bare name.

    Returns:
        The model slug that would appear in a canonical id (basename,
        lowercased).
    """
    cid = recipe_canonical_id(
        model=raw,
        hardware="",
        framework_name="",
        framework_version="",
        precision="",
    )
    return cid_to_path_components(cid)[0]


def memo_model_key(raw: str) -> str:
    """Spell a model reference the way the MEM's training data keyed it.

    ``canonical_id`` slugs a model to its basename, so ``Qwen/Qwen3-32B`` and
    ``mergebench/Qwen3-32B`` both collapse to ``qwen3-32b``. The MEM instead
    flattened the separator, giving ``qwen-qwen3-32b`` and
    ``mergebench-qwen3-32b`` — distinct keys holding distinct numbers. Asking
    with the basename therefore does not merely miss: it can return a different
    model's throughput, and the MEM never signals the substitution.

    An ``org/name`` reference keeps the org. A filesystem path cannot: the org
    is absent from the input, so the basename is all that is recoverable, which
    also matches the un-prefixed keys the MEM holds for such models.

    Args:
        raw: A model path, HF id, or bare name.

    Returns:
        The model spelling to put in a MEM question, or ``""`` when ``raw`` is
        empty.
    """
    value = (raw or "").strip().rstrip("/")
    if not value:
        return ""
    if not value.startswith("/") and value.count("/") == 1:
        org, _, name = value.partition("/")
        value = f"{org}-{name}" if org and name else value.rsplit("/", 1)[-1]
    elif "/" in value:
        value = value.rsplit("/", 1)[-1]
    cleaned = value.lower()
    for ch in (" ", "\t"):
        cleaned = cleaned.replace(ch, "_")
    return cleaned


def strip_think(text: str) -> str:
    """Drop the empty thinking block the server prepends.

    Args:
        text: Raw completion text.

    Returns:
        The text with any ``<think>...</think>`` block removed and trimmed.
    """
    return _THINK_RE.sub("", text or "").strip()


def parse_throughput(answer: str) -> float:
    """Extract a single throughput value from a MEM answer.

    Exactly one unit-qualified number is required: zero means the MEM declined,
    more than one means the answer is ambiguous. Both are unusable.

    Args:
        answer: Completion text, already stripped of the thinking block.

    Returns:
        Throughput in tokens/second, or ``0.0`` when not exactly one
        unit-qualified value is present.
    """
    matches = _THROUGHPUT_RE.findall(answer or "")
    if len(matches) != 1:
        return 0.0
    try:
        return float(matches[0].replace(",", ""))
    except ValueError:
        return 0.0


def parse_best_config(answer: str) -> dict[str, Any]:
    """Extract server args and env vars from a MEM answer.

    Reads the first backtick span and splits it on ``| env:``; anything after
    the span is prose.

    Args:
        answer: Completion text, already stripped of the thinking block.

    Returns:
        A ``best_config`` dict with ``extra_server_args`` and ``extra_envs``,
        or an empty dict when no usable span is present.
    """
    spans = _BACKTICK_RE.findall(answer or "")
    if not spans:
        return {}
    parts = _ENV_SPLIT_RE.split(spans[0], maxsplit=1)
    extra_server_args = parts[0].strip()
    env_part = parts[1] if len(parts) > 1 else ""
    extra_envs = {m.group("key"): m.group("value") for m in _ENV_PAIR_RE.finditer(env_part)}
    if not extra_server_args and not extra_envs:
        return {}
    return {"extra_server_args": extra_server_args, "extra_envs": extra_envs}


def _throughput_question(ctx: dict[str, str]) -> str:
    """Build the closed-book throughput question for one identity tuple.

    The wording matches the MEM's training distribution; paraphrasing measurably
    lowers hit rate, so it must not be reworded casually.

    Args:
        ctx: Identity fields (model / hardware / framework / framework_version /
            precision).

    Returns:
        The question to send to the MEM.
    """
    return (
        f"What is the best measured serving throughput for {ctx['model']} on "
        f"{ctx['hardware']} with {ctx['framework']} {ctx['framework_version']} "
        f"at {ctx['precision']} precision?"
    )


def _best_config_question(ctx: dict[str, str]) -> str:
    """Build the closed-book best-config question for one identity tuple.

    Args:
        ctx: Identity fields, as in :func:`_throughput_question`.

    Returns:
        The question to send to the MEM.
    """
    return (
        f"What is the best recorded configuration for {ctx['model']} on "
        f"{ctx['hardware']} with {ctx['framework']} {ctx['framework_version']} "
        f"at {ctx['precision']} precision?"
    )


class MemoRemoteRecipeClient:
    """Duck-typed read-side recipe client served by a MeMo MEM endpoint.

    Implements the surface the dispatcher expects of a remote: ``enabled`` /
    ``get_recipe`` / ``search`` / ``close``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "dummy",
        model: str = "",
        confidence: float = DEFAULT_CONFIDENCE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        local_guard: Any = None,
        allow_config: bool = False,
    ) -> None:
        """Configure the MEM client.

        Args:
            base_url: OpenAI-compatible base URL of the MEM service.
            api_key: Bearer token; local vLLM / SGLang ignore the value.
            model: Served model name; resolved from the endpoint when empty.
            confidence: Confidence stamped on every row this client returns.
            max_tokens: Generation cap per question.
            timeout: Per-request timeout in seconds.
            local_guard: Optional store with a ``get_recipe`` method. When it
                already holds the identity, the MEM abstains so its lossy answer
                cannot shadow a real row. See the module docstring.
            allow_config: Emit ``best_config``. Off by default because a
                fabricated launch recipe would clear the warm-replay gate on
                confidence alone. See the module docstring.
        """
        self._base_url = (base_url or "").strip()
        self._api_key = api_key or "dummy"
        self._model = (model or "").strip()
        self._confidence = max(0.0, min(1.0, float(confidence)))
        self._max_tokens = int(max_tokens)
        self._timeout = float(timeout)
        self._local_guard = local_guard
        self._allow_config = bool(allow_config)
        self._model_hint = ""
        self._client: Any = None

    def set_model_hint(self, raw_model: str) -> None:
        """Record the run's unslugged model reference.

        ``canonical_id`` keeps only the model basename, but the MEM was keyed on
        the org-qualified spelling (see :func:`memo_model_key`). The caller knows
        the original reference; handing it over lets questions use the spelling
        the MEM actually learned.

        Args:
            raw_model: The model path or HF id as the operator supplied it.
        """
        self._model_hint = (raw_model or "").strip()

    def _question_model_name(self, cid_model: str) -> str:
        """Choose the model spelling to put in a MEM question.

        The hint describes a single model, but reads arrive for many identities
        (the T0 relaxed-match cascade, config-donor probes). Applying a run's
        hint to an unrelated identity would ask about the wrong model, so the
        hint is used only when it slugs to the identity being read.

        Args:
            cid_model: The model slug decoded from the canonical id.

        Returns:
            The org-qualified spelling when the hint matches this identity,
            otherwise ``cid_model`` unchanged.
        """
        if not self._model_hint:
            return cid_model
        try:
            if hyperloom_model_slug(self._model_hint) != cid_model:
                return cid_model
        except Exception:  # noqa: BLE001 - a bad hint must not break the read
            log.debug("memo_recipe: unusable model hint %r", self._model_hint)
            return cid_model
        return memo_model_key(self._model_hint) or cid_model

    def set_local_guard(self, local_guard: Any) -> None:
        """Attach the store whose rows the MEM must not shadow.

        Args:
            local_guard: Store exposing ``get_recipe(canonical_id=...)``.
        """
        self._local_guard = local_guard

    def _shadowed_by_local(self, canonical_id: str) -> bool:
        """Whether the guard store already answers this identity usefully.

        Existence is not enough. The T0 anchor seeds a bare local row (identity
        plus tracing tags, no champion config or metrics) *before* running its
        warm-start cascade, so a presence check would let that placeholder shadow
        the MEM on every read and the MEM could never contribute. Only a row
        Hyperloom itself considers actionable may take precedence.

        Args:
            canonical_id: Canonical recipe identity.

        Returns:
            ``True`` when the guard holds an actionable row, meaning the MEM must
            abstain. A guard failure returns ``False`` so reads still proceed.
        """
        if self._local_guard is None:
            return False
        try:
            row = self._local_guard.get_recipe(canonical_id=canonical_id)
        except Exception:  # noqa: BLE001 - a broken guard must not block reads
            log.warning("memo_recipe: local guard lookup failed", exc_info=True)
            return False
        if not isinstance(row, dict) or not row:
            return False
        # Deferred: cortex_t0 imports this package at module level.
        from hyperloom.orchestrator.knowledge.cortex_t0 import _recipe_is_actionable

        return bool(_recipe_is_actionable(row))

    @property
    def enabled(self) -> bool:
        """Whether the client is configured well enough to serve reads.

        Returns:
            ``True`` when a base URL is set.
        """
        return bool(self._base_url)

    def close(self) -> None:
        """Release the underlying HTTP client. Idempotent."""
        self._client = None

    def get_recipe(
        self,
        *,
        canonical_id: str,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Answer one identity tuple closed-book from the MEM.

        Args:
            canonical_id: Canonical recipe identity to look up.
            version: Ignored; the MEM carries no version history. A non-``None``
                value yields ``None`` so the caller falls through to a real store.

        Returns:
            A flat arbor-shape row, or ``None`` when the guard store already
            answers the identity, or when the MEM produced neither a throughput
            value nor a replayable config.
        """
        if not self.enabled or version is not None:
            return None
        if self._shadowed_by_local(canonical_id):
            return None
        try:
            model, hardware, framework_name, _mt, _arch, framework_version, precision = (
                cid_to_path_components(canonical_id)
            )
        except Exception:  # noqa: BLE001 - malformed cid: let a real store try
            log.debug("memo_recipe: unparseable canonical_id %r", canonical_id)
            return None

        # Only the question uses the MEM's spelling; the returned row keeps the
        # canonical slug so Hyperloom's keying is untouched.
        ctx = {
            "model": self._question_model_name(model),
            "hardware": hardware,
            "framework": framework_name,
            "framework_version": framework_version,
            "precision": precision,
        }
        throughput = parse_throughput(self._ask(_throughput_question(ctx)))
        best_config = (
            parse_best_config(self._ask(_best_config_question(ctx)))
            if self._allow_config
            else {}
        )
        if not throughput and not best_config:
            return None

        return {
            "canonical_id": canonical_id,
            "model": model,
            "hardware": hardware,
            "framework_name": framework_name,
            "framework_version": framework_version,
            "precision": precision,
            "best_config": best_config,
            "best_throughput": throughput,
            "authority": "INFERRED",
            "confidence": self._confidence,
            "provenance": {
                "source": "memo",
                "model": self._model,
                "measured": False,
                "note": (
                    "Generated closed-book by a MeMo MEM; a lossy projection of the "
                    "recipe KB. Never persist this row as measured."
                ),
            },
        }

    def search(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Always empty: the MEM cannot serve relaxed label matches.

        The T0 L2/L3 cascade drops ``model`` or ``framework_version`` to find a
        sibling recipe. A closed-book model asked such a question fabricates an
        answer, so this client declines and lets the caller fall through.

        Args:
            **kwargs: Accepted and ignored (``label_match`` / ``limit`` /
                ``prefer`` / ...).

        Returns:
            An empty list, always.
        """
        return []

    def _ask(self, question: str) -> str:
        """Send one question to the MEM and return the cleaned completion.

        Args:
            question: The user-role prompt.

        Returns:
            Completion text with the thinking block stripped, or ``""`` on any
            per-request failure (a MEM miss must not fail the read).

        Raises:
            RemoteRecipeClientError: If the client cannot be constructed.
        """
        client = self._ensure_client()
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": question}],
                max_tokens=self._max_tokens,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return strip_think(response.choices[0].message.content or "")
        except Exception:  # noqa: BLE001 - degrade to a miss, never break T0
            log.warning("memo_recipe: completion failed", exc_info=True)
            return ""

    def _ensure_client(self) -> Any:
        """Lazily build the OpenAI-compatible client and resolve the model name.

        Returns:
            The cached client instance.

        Raises:
            RemoteRecipeClientError: If the OpenAI SDK is missing or the served
                model name cannot be resolved.
        """
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RemoteRecipeClientError(
                "the openai package is required for the MeMo recipe remote"
            ) from exc
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
        )
        if not self._model:
            try:
                self._model = self._client.models.list().data[0].id
            except Exception as exc:  # noqa: BLE001
                raise RemoteRecipeClientError(
                    f"cannot resolve the served MeMo model at {self._base_url}"
                ) from exc
        return self._client


def build_memo_remote_from_env() -> MemoRemoteRecipeClient | None:
    """Construct a MEM client from ``MEMO_KB_URL`` and friends.

    Returns:
        A configured client, or ``None`` when ``MEMO_KB_URL`` is unset so the
        caller keeps its existing remote untouched.
    """
    base_url = (os.environ.get("MEMO_KB_URL", "") or "").strip()
    if not base_url:
        return None
    raw_conf = (os.environ.get("MEMO_KB_CONFIDENCE", "") or "").strip()
    try:
        confidence = float(raw_conf) if raw_conf else DEFAULT_CONFIDENCE
    except ValueError:
        confidence = DEFAULT_CONFIDENCE
    raw_timeout = (os.environ.get("MEMO_KB_TIMEOUT_SEC", "") or "").strip()
    try:
        timeout = float(raw_timeout) if raw_timeout else DEFAULT_TIMEOUT_SEC
    except ValueError:
        timeout = DEFAULT_TIMEOUT_SEC
    allow_config = (os.environ.get("MEMO_KB_ALLOW_CONFIG", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return MemoRemoteRecipeClient(
        base_url=base_url,
        api_key=(os.environ.get("MEMO_KB_TOKEN", "") or "dummy").strip() or "dummy",
        model=(os.environ.get("MEMO_KB_MODEL", "") or "").strip(),
        confidence=confidence,
        timeout=timeout,
        allow_config=allow_config,
    )


__all__ = [
    "MemoRemoteRecipeClient",
    "build_memo_remote_from_env",
    "parse_best_config",
    "parse_throughput",
    "strip_think",
]
