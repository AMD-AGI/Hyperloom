# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic dual-read between the cortex substrate and gbrain.

Two knowledge sources steer a specialist's config choices:

* **substrate** (the new cortex knowledge graph) — directional lever priors
  from ``/v2/reasoning/levers`` (``beneficial`` / ``neutral`` / ``harmful``
  per knob, with calibration + provenance), already warmed into the specialist
  params as ``substrate_levers``.
* **gbrain** (the legacy RecipeKB graph) — the warm-start champion recipe
  (``warm_start_recipe``), whose ``best_config`` is a concrete proposal of CLI
  args + env levers, already warmed into the params.

Rather than re-implement cortex's knob resolution in Hyperloom (fragile, and it
would drift from the substrate), we let the substrate *judge* the gbrain recipe:
POST the recipe's ``best_config`` to ``/v2/reasoning/assess`` and read back per
-lever verdicts (``confirmed`` / ``conflicts`` / ``deviated`` / ``no_basis``).
That assessment IS the deterministic comparison — computed by the authoritative
side — and it tells us, lever by lever, whether the two sources agree.

:func:`build_dual_read` folds the forward levers, the gbrain proposal, and the
assessment into one structured digest with explicit source attribution and a
single ``verdict`` / ``selected_source``. Everything here is best-effort and
advisory: a failed assess call degrades to a substrate-only (or gbrain-only)
digest and never blocks dispatch.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from ..compat.payload_aliases import CANONICAL_KEY  # "extra_server_args"

log = logging.getLogger(__name__)

ASSESS_PATH = "/v2/reasoning/assess"
DEFAULT_TIMEOUT_SEC = 3.0
_ENVS_KEY = "extra_envs"

# Substrate /assess per-lever statuses that mean the gbrain recipe lever
# contradicts the substrate's measured evidence.
_CONFLICT_STATUSES = frozenset({"conflicts", "deviated"})
_CONFIRM_STATUSES = frozenset({"confirmed"})


class SubstrateAssessClient:
    """Minimal sync POST client for ``/v2/reasoning/assess`` (best-effort).

    Sibling of :class:`substrate_levers_client.SubstrateLeversClient`; shares
    the same env config (``CORTEX_KB_URL`` / ``CORTEX_KB_HTTP_TIMEOUT_SEC`` /
    ``KB_SERVICE_TOKEN``) and the urllib-only transport so it installs in the
    minimal container. All failures return ``None``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        if not base_url:
            raise ValueError("SubstrateAssessClient: base_url is required")
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("KB_SERVICE_TOKEN") or ""
        self.timeout_sec = float(timeout_sec)

    @classmethod
    def from_env(cls) -> "SubstrateAssessClient | None":
        """Build a client from env, or ``None`` when no cortex KB is configured."""
        url = (os.environ.get("CORTEX_KB_URL") or "").strip()
        if not url:
            return None
        try:
            timeout = float(os.environ.get("CORTEX_KB_HTTP_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC)))
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SEC
        return cls(url, timeout_sec=timeout)

    def assess(
        self,
        *,
        focus: dict[str, Any],
        params: dict[str, Any] | None = None,
        envs: dict[str, Any] | None = None,
        args: str = "",
    ) -> dict[str, Any] | None:
        """POST one proposal to ``/v2/reasoning/assess``.

        Returns the decoded assessment (``focus`` / ``seed`` / ``reasonable`` /
        ``rationale`` / ``summary`` / ``verdicts``), or ``None`` on any error
        (best-effort; never raises).
        """
        if not isinstance(focus, dict) or not str(focus.get("model") or "").strip():
            return None
        body = {"focus": focus, "params": params or None, "envs": envs or None, "args": args or ""}
        url = f"{self.base_url}{ASSESS_PATH}"
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                payload = resp.read().decode("utf-8") or "{}"
                return json.loads(payload) if payload else {}
        except (urllib.error.URLError, ValueError, OSError) as exc:
            log.warning("substrate dual-read: KB assess call failed (%s): %r", url, exc)
            return None


def recipe_best_config(warm_start_recipe: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the gbrain warm recipe's champion ``best_config`` (or ``{}``)."""
    if not isinstance(warm_start_recipe, dict):
        return {}
    recipe = warm_start_recipe.get("recipe")
    if not isinstance(recipe, dict):
        return {}
    best = recipe.get("best_config")
    return best if isinstance(best, dict) else {}


def recipe_assess_inputs(warm_start_recipe: dict[str, Any] | None) -> dict[str, Any]:
    """Project the gbrain recipe ``best_config`` into ``/assess`` inputs.

    Returns ``{"args": <cli string>, "envs": {<env levers>}}`` from the
    canonical ``extra_server_args`` + nested ``extra_envs`` keys. Empty when the
    recipe carries no champion config.
    """
    best = recipe_best_config(warm_start_recipe)
    if not best:
        return {}
    args = str(best.get(CANONICAL_KEY) or "").strip()
    envs_raw = best.get(_ENVS_KEY)
    envs = {str(k): v for k, v in envs_raw.items()} if isinstance(envs_raw, dict) else {}
    out: dict[str, Any] = {}
    if args:
        out["args"] = args
    if envs:
        out["envs"] = envs
    return out


def focus_from(
    substrate_levers: dict[str, Any] | None,
    warm_start_recipe: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the assess focus, preferring the substrate digest's echoed focus.

    Falls back to the gbrain recipe's model/hardware so an assess is still
    possible when only the recipe is warmed.
    """
    if isinstance(substrate_levers, dict):
        focus = substrate_levers.get("focus")
        if isinstance(focus, dict) and str(focus.get("model") or "").strip():
            return {k: v for k, v in focus.items() if v not in (None, "", "unknown")}
    recipe = (warm_start_recipe or {}).get("recipe") if isinstance(warm_start_recipe, dict) else None
    if isinstance(recipe, dict):
        candidate = {
            "model": recipe.get("model") or recipe.get("model_name") or "",
            "hardware": recipe.get("hardware") or recipe.get("hw") or "",
        }
        return {k: v for k, v in candidate.items() if v not in (None, "", "unknown")}
    return {}


def _trim_levers(levers: Any, limit: int = 20) -> list[dict[str, Any]]:
    if not isinstance(levers, list):
        return []
    return [lv for lv in levers if isinstance(lv, dict)][:limit]


def build_dual_read(
    *,
    substrate_levers: dict[str, Any] | None,
    warm_start_recipe: dict[str, Any] | None,
    assess: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fold the two sources + the substrate assessment into one digest.

    Pure / deterministic given its inputs (no IO). The ``verdict`` and
    ``selected_source`` make the source attribution explicit for the trace:

    * ``agree``         — substrate confirms the gbrain recipe levers.
    * ``conflict``      — substrate flags gbrain levers as harmful/deviated.
    * ``substrate_only``— only the substrate has an opinion (no recipe).
    * ``gbrain_only``   — only the recipe exists, substrate has no basis.
    * ``no_basis``      — both present but substrate cannot judge the recipe.
    * ``no_data``       — neither source carries usable signal.
    """
    levers_dig = substrate_levers if isinstance(substrate_levers, dict) else {}
    levers = levers_dig.get("levers") if isinstance(levers_dig.get("levers"), list) else []
    has_substrate = bool(levers)

    best_config = recipe_best_config(warm_start_recipe)
    has_recipe = bool(best_config)

    assess_dig = assess if isinstance(assess, dict) else {}
    verdicts = assess_dig.get("verdicts") if isinstance(assess_dig.get("verdicts"), list) else []
    reasonable = str(assess_dig.get("reasonable") or "")

    conflicts: list[dict[str, Any]] = []
    confirmations: list[dict[str, Any]] = []
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        status = str(v.get("status") or "")
        row = {
            "lever": v.get("lever"),
            "knob": v.get("knob"),
            "polarity": v.get("polarity"),
            "status": status,
            "predicted_factor": v.get("predicted_factor"),
            "note": v.get("note"),
        }
        if status in _CONFLICT_STATUSES:
            conflicts.append(row)
        elif status in _CONFIRM_STATUSES:
            confirmations.append(row)

    if conflicts:
        verdict, selected = "conflict", "substrate"
    elif has_recipe and reasonable == "supported":
        verdict, selected = "agree", "both_agree"
    elif has_recipe and confirmations:
        verdict, selected = "agree", "both_agree"
    elif has_recipe and not has_substrate:
        verdict, selected = "gbrain_only", "gbrain"
    elif has_recipe and reasonable in ("insufficient_basis", ""):
        verdict, selected = "no_basis", "substrate" if has_substrate else "gbrain"
    elif has_substrate and not has_recipe:
        verdict, selected = "substrate_only", "substrate"
    else:
        verdict, selected = "no_data", "none"

    recipe_env = warm_start_recipe if isinstance(warm_start_recipe, dict) else {}
    recipe_inner = recipe_env.get("recipe") if isinstance(recipe_env.get("recipe"), dict) else {}

    return {
        "focus": assess_dig.get("focus") or levers_dig.get("focus") or {},
        "verdict": verdict,
        "selected_source": selected,
        "substrate": {
            "seed": levers_dig.get("seed"),
            "summary": levers_dig.get("summary") or {},
            "lever_count": len(levers),
            "levers": _trim_levers(levers),
        },
        "gbrain_recipe": {
            "tier": recipe_env.get("tier"),
            "confidence": recipe_env.get("confidence"),
            "canonical_id": recipe_inner.get("canonical_id"),
            "best_config": best_config,
        },
        "assessment": {
            "reasonable": reasonable or None,
            "rationale": assess_dig.get("rationale"),
            "summary": assess_dig.get("summary") or {},
            "verdict_count": len(verdicts),
        },
        "conflicts": conflicts,
        "confirmations": confirmations,
    }


def compute_dual_read(
    *,
    substrate_levers: dict[str, Any] | None,
    warm_start_recipe: dict[str, Any] | None,
    client: "SubstrateAssessClient | None" = None,
) -> dict[str, Any]:
    """Run the substrate↔gbrain dual-read end-to-end (best-effort).

    Builds an assess client from env when one is not supplied, asks the
    substrate to judge the gbrain recipe's ``best_config``, and folds the
    result with :func:`build_dual_read`. Returns ``{}`` only when there is no
    usable signal from either source; otherwise returns the digest (the assess
    half may be empty if the call failed — fail-soft).
    """
    has_levers = isinstance(substrate_levers, dict) and bool(substrate_levers.get("levers"))
    inputs = recipe_assess_inputs(warm_start_recipe)
    if not has_levers and not inputs:
        return {}

    assess: dict[str, Any] | None = None
    if inputs:
        focus = focus_from(substrate_levers, warm_start_recipe)
        if focus.get("model"):
            try:
                if client is None:
                    client = SubstrateAssessClient.from_env()
                if client is not None:
                    assess = client.assess(
                        focus=focus, envs=inputs.get("envs"), args=inputs.get("args", "")
                    )
            except Exception as exc:  # noqa: BLE001 — advisory, never a gate
                log.warning("substrate dual-read: assess failed: %r", exc)
                assess = None

    digest = build_dual_read(
        substrate_levers=substrate_levers,
        warm_start_recipe=warm_start_recipe,
        assess=assess,
    )
    return digest if digest.get("verdict") != "no_data" else {}


__all__ = [
    "SubstrateAssessClient",
    "ASSESS_PATH",
    "DEFAULT_TIMEOUT_SEC",
    "recipe_best_config",
    "recipe_assess_inputs",
    "focus_from",
    "build_dual_read",
    "compute_dual_read",
]
