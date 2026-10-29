# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Probe the gateway model catalog and resolve the orchestration, Codex and critic models before a session starts."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from common import llm_config
from common.env import env_bool
from common.llm_config import CLAUDE_OAUTH_TOKEN_ENV
from common.llm_headers import parse_custom_headers
from .backends import (
    critic_review_target,
    critic_on_codex_side,
)
from .credentials import (
    _CLAUDE_ALLOWED_MODELS,
    _CODEX_FALLBACK_MODELS,
    _CATALOG_RETRY_DELAYS_SEC,
)


log = logging.getLogger("hyperloom.inference_optimizer.cli")

# Per-attempt read timeout for the gateway /models catalog probe.
try:
    _CATALOG_REQUEST_TIMEOUT_SEC = float(
        os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_TIMEOUT_SEC", "5.0") or "5.0"
    )
except (TypeError, ValueError):
    _CATALOG_REQUEST_TIMEOUT_SEC = 5.0


# Sentinel returned by _probe_llm_catalog when the gateway has no /models route (HTTP 404/405).
_CATALOG_NO_MODELS_ENDPOINT: frozenset[str] = frozenset()


def _probe_llm_catalog(
    *,
    base_url: str,
    api_key: str,
) -> set[str] | frozenset[str] | None:
    """Probe ``<base_url>/models`` with retry (gateway flakes); return set of model ids or None."""

    if not base_url:
        return None

    try:
        import httpx  # type: ignore[import-not-found]
    except ImportError:
        # httpx should already be installed; return None so the caller decides.
        print(
            "Preflight: WARNING — httpx not importable, skipping catalog "
            "probe. _ensure_python_sdks should have installed it."
        )
        return None

    probe_url = base_url.rstrip("/") + "/models"
    headers = _catalog_probe_headers(base_url=base_url, api_key=api_key)

    delays = (0.0, *_CATALOG_RETRY_DELAYS_SEC)
    last_err: str = ""
    for i, delay in enumerate(delays):
        if delay > 0:
            time.sleep(delay)
        try:
            resp = httpx.get(
                probe_url,
                headers=headers,
                timeout=_CATALOG_REQUEST_TIMEOUT_SEC,
            )
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            print(f"Preflight: catalog probe attempt {i + 1}/{len(delays)} failed: {last_err}")
            continue
        if resp.status_code in (404, 405):
            # The endpoint has no /models route; not a transient/auth error, so stop retrying and signal "no catalog
            # endpoint" distinctly.
            print(
                f"Preflight: catalog probe got HTTP {resp.status_code} for "
                f"{probe_url}; endpoint exposes no /models route"
            )
            return _CATALOG_NO_MODELS_ENDPOINT
        if resp.status_code != 200:
            last_err = f"HTTP {resp.status_code}: {(resp.text or '')[:200]}"
            print(f"Preflight: catalog probe attempt {i + 1}/{len(delays)} got {last_err}")
            continue
        try:
            data = resp.json()
        except ValueError as exc:
            last_err = f"JSON decode: {exc}"
            print(f"Preflight: catalog probe attempt {i + 1}/{len(delays)} returned non-JSON: {last_err}")
            continue
        ids: set[str] = set()
        for model in data.get("data") or []:
            if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                continue
            model_id = model["id"]
            ids.add(model_id)
            ids.add(_catalog_compare_model_id(model_id))
        if not ids:
            last_err = "empty data[]"
            continue
        return ids

    print(f"Preflight: catalog probe exhausted {len(delays)} attempts ({last_err}); cannot validate model availability")
    return None


def _catalog_probe_headers(*, base_url: str, api_key: str) -> dict[str, str]:
    """Build headers for a direct ``<base_url>/models`` probe."""
    openai_base = (os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
    probe = (base_url or "").strip().rstrip("/")
    probing_openai = bool(openai_base) and (probe == openai_base or probe.startswith(openai_base + "/"))
    env_name = "OPENAI_CUSTOM_HEADERS" if probing_openai else "ANTHROPIC_CUSTOM_HEADERS"
    headers = parse_custom_headers(os.environ.get(env_name))
    if api_key and not any(name.lower() == "authorization" for name in headers):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _catalog_compare_model_id(model_id: str) -> str:
    """Normalize catalog IDs for preflight comparison only."""
    text = str(model_id or "").strip()
    lowered = text.lower()
    if text.lower().startswith("claude-"):
        return lowered.replace(".", "-")
    return lowered


def _same_gateway(anthropic_url: str, openai_url: str) -> bool:
    """True when both protocol sides are served by one gateway."""
    if not anthropic_url or not openai_url:
        return False
    if anthropic_url == openai_url:
        return True
    from urllib.parse import urlsplit

    return urlsplit(anthropic_url).netloc == urlsplit(openai_url).netloc


def _codex_model_should_follow_claude() -> bool:
    """True when the operator supplied only Anthropic config."""
    return llm_config.is_anthropic_only()


def _claude_model_should_follow_codex() -> bool:
    """True when the operator supplied only OpenAI-compatible config."""
    return llm_config.is_openai_only()


def _catalog_probe_has_no_credential() -> bool:
    """True when a Claude subscription token is the only credential available."""
    if not os.environ.get(CLAUDE_OAUTH_TOKEN_ENV, "").strip():
        return False
    # Any other Anthropic-side form can authenticate the probe, so the list of what disqualifies "oauth-only" is the
    # registry minus the token itself, derived rather than restated.
    bearer_capable = (
        *(n for n in llm_config.ANTHROPIC_CREDENTIAL_ENV_ORDER if n != CLAUDE_OAUTH_TOKEN_ENV),
        "OPENAI_API_KEY",
    )
    return not any(os.environ.get(name, "").strip() for name in bearer_capable)


def _custom_orch_model_allowed() -> bool:
    """Whether orchestration may use a model outside the AMD Claude allowlist; only an explicit false-token denies."""
    return env_bool("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", True)


def _critic_agent_runtime_needed(critic_choice: str) -> bool:
    """Whether the selected critic path will actually instantiate critic-agent."""
    return critic_choice == "agent"


def _reject_unusable_claude_model(chosen: str, *, allow_custom: bool) -> None:
    """Refuse a model id the static allowlist cannot accept, before any gateway is asked."""
    if not allow_custom and chosen not in _CLAUDE_ALLOWED_MODELS:
        print(
            f"ERROR: --claude-model={chosen!r} is not allowed. "
            f"Orchestration model must be one of "
            f"{list(_CLAUDE_ALLOWED_MODELS)} (best first). Refusing to start. "
            f"For a non-AMD gateway, set "
            f"INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1 to use a custom "
            f"orchestration model validated against your gateway catalog.",
            file=sys.stderr,
        )
        sys.exit(2)
    if allow_custom and not chosen:
        print(
            "ERROR: --claude-model is empty but "
            "custom orchestration model support is enabled; pass an explicit "
            "model id. Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)


def _catalog_probe_candidates(
    resolved_urls: tuple[str, str],
    *,
    claude_follows_codex: bool,
) -> list[tuple[str, str]]:
    """The gateways worth asking for a model catalog, in the order they are asked.

    The Claude catalog must come from the Anthropic side unless the Claude
    model follows Codex, in which case the OpenAI side is the authority.
    """
    anthropic_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not anthropic_url:
        anthropic_url = resolved_urls[0]
    if not openai_url:
        openai_url = resolved_urls[1]
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    # OpenAI-side key only.
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    candidates: list[tuple[str, str]] = []
    if claude_follows_codex:
        if openai_url:
            candidates.append((openai_url, openai_key))
        elif anthropic_url:
            candidates.append((anthropic_url, anthropic_key))
        return candidates
    # A keyless Anthropic candidate can only fail — a subscription token is not a catalog credential — so it
    # must not consume the one probe slot and leave a configured OpenAI gateway unverified.
    if anthropic_url and anthropic_key:
        candidates.append((anthropic_url, anthropic_key))
    if openai_url and ((anthropic_url and _same_gateway(anthropic_url, openai_url)) or not candidates):
        # One gateway serving both protocols, or the only side left once a keyless Anthropic side was skipped
        # above.
        candidates.append((openai_url, openai_key))
    return candidates


def _probe_claude_catalog(
    resolved_urls: tuple[str, str],
    *,
    chosen: str,
    claude_follows_codex: bool,
):
    """The gateway's model catalog, and whether a probe was made at all.

    ``(None, False)`` means the probe was skipped because the only credential
    cannot authenticate one -- the static allowlist gate has already run, and
    it is the only check the probe would have added.
    """
    override_url = os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
    if not override_url and _catalog_probe_has_no_credential():
        print(
            "Preflight: catalog probe skipped: oauth-only credential "
            f"(CLAUDE_CODE_OAUTH_TOKEN); cannot verify --claude-model={chosen!r}. Proceeding."
        )
        return None, False
    if override_url:
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        return _probe_llm_catalog(base_url=override_url, api_key=api_key), True
    catalog_ids: set[str] | frozenset[str] | None = None
    seen_urls: set[str] = set()
    for cand_url, cand_key in _catalog_probe_candidates(resolved_urls, claude_follows_codex=claude_follows_codex):
        if not cand_url or cand_url in seen_urls:
            continue
        seen_urls.add(cand_url)
        catalog_ids = _probe_llm_catalog(base_url=cand_url, api_key=cand_key)
        # A missing /models route is the only answer worth re-asking on the other side, and only because a
        # dual-protocol gateway lists its models there.
        if catalog_ids is not _CATALOG_NO_MODELS_ENDPOINT:
            break
    return catalog_ids, True


def _validate_and_resolve_claude_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str],
    *,
    claude_follows_codex: bool | None = None,
) -> set[str] | None:
    """Gate Claude model selection against the gateway catalog; mutates ``args.claude_model``."""
    if claude_follows_codex is None:
        claude_follows_codex = _claude_model_should_follow_codex()
    chosen = (args.claude_model or "").strip()
    # Custom orchestration models are enabled by default; the gateway catalog probe below is the sole gate.
    allow_custom = _custom_orch_model_allowed()
    _reject_unusable_claude_model(chosen, allow_custom=allow_custom)
    catalog_ids, probed = _probe_claude_catalog(
        resolved_urls,
        chosen=chosen,
        claude_follows_codex=claude_follows_codex,
    )
    if not probed:
        return None

    if catalog_ids is _CATALOG_NO_MODELS_ENDPOINT:
        # The gateway has no /models route; the model cannot be verified here, so proceed rather than refuse.
        print(
            f"Preflight: WARNING — gateway has no /models route (HTTP 404/405); "
            f"cannot verify --claude-model={chosen!r}. Proceeding."
        )
        return None

    if catalog_ids is None:
        # Auth/network/server/non-JSON/empty-catalog failure: genuinely unverifiable.
        if allow_custom:
            print(
                f"Preflight: WARNING — gateway catalog unreachable; cannot verify "
                f"--claude-model={chosen!r}. Proceeding with custom orchestration "
                f"model support enabled (trusting the operator id)."
            )
            return None
        print(
            "ERROR: gateway catalog unreachable after retries; cannot "
            "verify Claude model availability. Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    if chosen in catalog_ids or _catalog_compare_model_id(chosen) in catalog_ids:
        print(f"Preflight: Claude model {chosen!r} confirmed in gateway catalog")
        return catalog_ids

    # For non-allowlisted custom ids the AMD fallback is meaningless; fail clearly on a catalog miss.
    if allow_custom and chosen not in _CLAUDE_ALLOWED_MODELS:
        print(
            f"ERROR: --claude-model={chosen!r} not present in gateway catalog "
            f"(custom orchestration model support enabled; catalog has "
            f"{sorted(catalog_ids)[:20]}). Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Walk the allowlist in order so it acts as a real preference ladder: a gateway that carries opus-4-8 but not the
    # newer default must land on 4-8, not skip two generations down to the last entry.
    for candidate in _CLAUDE_ALLOWED_MODELS:
        if candidate == chosen:
            continue
        if candidate in catalog_ids:
            print(f"Preflight: WARNING — {chosen!r} not in gateway catalog; falling back to {candidate!r}")
            args.claude_model = candidate
            return catalog_ids

    print(
        f"ERROR: none of the allowed Claude models {list(_CLAUDE_ALLOWED_MODELS)!r} "
        f"present in gateway catalog "
        f"(catalog has {sorted(m for m in catalog_ids if m.startswith('claude-'))}). "
        f"Refusing to start.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _resolve_models_for_run(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str],
    *,
    claude_follows_codex: bool | None = None,
    codex_follows_claude: bool | None = None,
) -> None:
    """Resolve both model ids against the gateway before any session work."""
    if claude_follows_codex is None:
        claude_follows_codex = _claude_model_should_follow_codex()
    if codex_follows_claude is None:
        codex_follows_claude = _codex_model_should_follow_claude()

    if claude_follows_codex:
        # codex_model is about to become the orchestration model, so it needs the ladder whatever the critic backend
        # is.
        _smoke_test_codex_model(args, resolved_urls)
        args.claude_model = args.codex_model

    # Hard-gate the Claude model (mutates args.claude_model on fallback; sys.exit(2) on failure).
    _validate_and_resolve_claude_model(args, resolved_urls, claude_follows_codex=claude_follows_codex)

    if codex_follows_claude:
        args.codex_model = args.claude_model

    _probe_critic_review_model(
        args, codex_follows_claude=codex_follows_claude, claude_follows_codex=claude_follows_codex
    )


_CRITIC_PROBE_TIMEOUT_SEC = 60.0


def _probe_critic_review_model(
    args: argparse.Namespace, *, codex_follows_claude: bool, claude_follows_codex: bool = False
) -> None:
    """Send the critic's model one real request before the session starts; exit rc=2 when it cannot answer.

    A catalog listing only proves a gateway names a model, not that its upstream serves it, and the critic has no
    fallback model: a review path that fails every call must stop the launch rather than run a session without a
    critic.
    """
    if _resolve_critic_choice(args) != "agent":
        return
    try:
        protocol, model = critic_review_target(
            args.critic_protocol,
            orchestration_on_codex=critic_on_codex_side(
                codex_follows_claude=codex_follows_claude, claude_follows_codex=claude_follows_codex
            ),
            claude_model=args.claude_model,
            codex_model=args.codex_model,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    messages = [{"role": "user", "content": "Reply with OK."}]
    last_error: BaseException | None = None
    for delay in (0.0, *_CATALOG_RETRY_DELAYS_SEC):
        if delay:
            time.sleep(delay)
        try:
            if protocol == "anthropic":
                llm_config.anthropic_completion(
                    request=llm_config.AnthropicCompletionRequest(
                        model=model, messages=messages, max_tokens=16, timeout_s=_CRITIC_PROBE_TIMEOUT_SEC
                    ),
                    component="critic",
                    operation="preflight",
                )
            else:
                llm_config.chat_completion(
                    llm_config.get_openai_client(timeout=_CRITIC_PROBE_TIMEOUT_SEC),
                    component="critic",
                    operation="preflight",
                    model=model,
                    messages=messages,
                    max_completion_tokens=16,
                )
        except Exception as exc:  # any failure means the review path is unusable; exits 2 after the last retry
            log.warning("critic model %r probe failed over the %s protocol: %s", model, protocol, exc, exc_info=True)
            last_error = exc
            continue
        print(f"Preflight: critic model {model!r} answered over the {protocol} protocol")
        return
    print(
        f"ERROR: critic model {model!r} did not answer over the {protocol} protocol: {last_error!r}\n"
        f"  The critic reviews with the orchestration model unless --critic-protocol selects the other side "
        f"(then CLAUDE_MODEL for anthropic, CODEX_MODEL for openai); there is no fallback model.\n"
        f"  Fix that model or its credential, or pass --critic-mock to run without a critic. Refusing to start.",
        file=sys.stderr,
    )
    sys.exit(2)


def _smoke_test_codex_model(
    args: argparse.Namespace,
    resolved_urls: tuple[str, str],
) -> None:
    """WARN-only catalog check for ``--codex-model`` on an OpenAI-only launch, where it becomes the orchestration model.

    Steps down the ladder before Coordinator starts.
    """
    openai_url = os.environ.get("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "").strip()
    if not openai_url:
        openai_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not openai_url:
        openai_url = resolved_urls[1]
    # OpenAI-side key only.
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    catalog_ids = _probe_llm_catalog(base_url=openai_url, api_key=openai_key)
    if catalog_ids is None:
        # WARN-only path: don't block startup just because the OpenAI catalog is unreachable (the Claude gate already
        # validated reachability).
        print(
            "Preflight: WARNING — OpenAI-side catalog unreachable; skipping "
            "--codex-model verification (CodexBackend may fail at first turn)."
        )
        return

    chosen = (args.codex_model or "").strip()
    if chosen in catalog_ids:
        print(f"Preflight: Codex model {chosen!r} confirmed in gateway catalog")
        return

    if chosen in _CODEX_FALLBACK_MODELS:
        for candidate in _CODEX_FALLBACK_MODELS:
            if candidate == chosen:
                continue
            if candidate in catalog_ids:
                print(
                    f"Preflight: WARNING — codex model {chosen!r} not in gateway catalog; falling back to {candidate!r}"
                )
                args.codex_model = candidate
                return

    print(
        f"Preflight: WARNING — codex model {chosen!r} not in gateway catalog "
        f"({sorted(m for m in catalog_ids if m.startswith('gpt-'))}); "
        f"CodexBackend will fail at first turn. Pass --codex-model with a "
        f"value in the catalog (known-good ids, newest first: "
        f"{list(_CODEX_FALLBACK_MODELS)})."
    )


# Default critic backend; override via env or --critic-mock/--critic-agent.
DEFAULT_CRITIC_BACKEND = os.environ.get(
    "INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND",
    "agent",
)
_VALID_CRITIC_BACKENDS = ("mock", "agent")


def _resolve_choice(
    attr: str,
    default: str,
    valid: tuple[str, ...],
    flag_hint: str,
    *,
    args: argparse.Namespace,
) -> tuple[str, bool]:
    """Resolve a backend choice from CLI args with validation and fallback to default."""
    chosen = getattr(args, attr, None)
    explicit = chosen is not None
    if chosen is None:
        chosen = default
    if chosen not in valid:
        print(
            f"ERROR: {attr.replace('_', ' ')} {chosen!r} not in {valid!r} (set by {flag_hint})",
            file=sys.stderr,
        )
        sys.exit(2)
    return chosen, explicit


def _resolve_critic_choice(args: argparse.Namespace) -> str:
    """Resolve the active critic backend choice (arg → DEFAULT_CRITIC_BACKEND); hard-fails on invalid."""
    chosen, _ = _resolve_choice(
        "critic_backend",
        DEFAULT_CRITIC_BACKEND,
        _VALID_CRITIC_BACKENDS,
        "--critic-mock / --critic-agent or INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND",
        args=args,
    )
    return chosen
