"""Shared T0 (PRELUDE) Cortex anchor — KB_design §3.2 §5.1 / §3.13 M1 §5.1.

The T0 anchor is the boot-time Cortex KB ritual every session must
run before EXPLORE starts:

1. ``session begin`` (sync) — mints the Cortex ``session_id``;
   reused across resume via ``.kb_sid`` + ``SharedState.cortex_session_id``.
2. ``propose_point`` of the registered ``recipe`` kind (canonical id
   ``recipe:{slug(model)}:{slug(hw)}``) — idempotent across sessions
   for the same workload/hw pair; KB-explorer / warm-start can index it.
3. ``find_recipe`` snapshot → ``.kb_warm.json`` +
   ``SharedState.warm_start_recipe``.
4. ``traps`` snapshot → ``.kb_pitfalls.json`` +
   ``SharedState.warm_start_pitfalls``.

Historically this lived inside :func:`cli._bootstrap_cortex_kb` (a
~150-line cli helper). KB_gaps/Gap-12 covers the consequence: an
SDK / integration-test caller that constructs
:class:`Coordinator` directly bypasses the cli path and ends up with
an empty warm_start surface. v0.8 §3.2 §5.1 says T0 belongs to the
PRELUDE phase the Coordinator owns; the canonical entry point is
still cli (preserves fail-fast on Cortex outages, prints the boot
banner the operator expects), but the same ritual is exposed here so
the Coordinator can run a *defensive fallback* when it detects
``cortex_kb`` is wired but ``cortex_session_id`` is still empty.

The two callers parameterise the helper via two knobs:

* ``fail_fast``: cli passes ``True`` so a Cortex outage exits the
  process; Coordinator passes ``False`` so SDK callers degrade to
  ``warm_start={}`` instead of crashing inside a long-running
  reactor loop.
* ``on_status``: cli supplies ``print`` so the operator sees the
  ``Cortex KB        : session_id=...`` banner; Coordinator
  supplies a ``log.info`` shim so the same line lands in the
  session log without polluting stdout.

The helper is the **single source of truth** for the T0 ritual; any
future change (e.g. extra ``propose_point`` for ``sweep_grid``)
lives here and both entry points pick it up automatically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .. import cortex_kb_constants as C
from ..cortex_kb_client import (
    CortexKBClient,
    CortexKBError,
    recipe_canonical_id,
)


log = logging.getLogger(__name__)


@dataclass
class T0Result:
    """Outcome of one :func:`run_t0_anchor` invocation.

    ``status`` ∈ {``"ok"``, ``"resumed"``, ``"skipped_disabled"``,
    ``"skipped_already"``, ``"failed_session_begin"``}. The two
    ``skipped_*`` outcomes are the no-op paths: ``skipped_disabled``
    when the client is disabled (``--degraded-kb`` / SDK without
    Cortex), ``skipped_already`` when ``cortex_session_id`` was
    already non-empty on entry (e.g. the cli T0 ran and the
    Coordinator's fallback finds nothing to do).
    """

    status: str
    session_id: str = ""
    workload: str = ""
    hw: str = ""
    warm_present: bool = False
    traps_present: bool = False
    error: str = ""


def _default_status_emitter(line: str) -> None:
    """Default ``on_status`` — log at INFO."""
    log.info("%s", line)


def run_t0_anchor(
    client: CortexKBClient,
    shared_state: Any,
    *,
    workload: str,
    hw: str,
    image_digest: str = "",
    stack_fingerprint: Mapping[str, str] | None = None,
    extra_attrs: Mapping[str, Any] | None = None,
    resume: bool = False,
    fail_fast: bool = False,
    on_status: Callable[[str], None] | None = None,
    session_dir: Path | None = None,
    save_state: bool = True,
) -> T0Result:
    """Run the T0 Cortex anchor.

    Mutates ``shared_state`` in place (sets ``cortex_session_id``,
    ``warm_start_ts``, ``warm_start_recipe``, ``warm_start_pitfalls``)
    and, when ``save_state=True``, persists via
    ``shared_state.save(session_dir)``.

    Failure handling:

    * ``client.enabled is False`` → no-op, returns
      ``T0Result(status='skipped_disabled')``.
    * ``shared_state.cortex_session_id`` already non-empty → no-op
      (returns ``skipped_already``); we trust the prior anchor and
      keep the existing warm_start fields. Resume callers that
      *want* to refresh the warm_start surface should set
      ``shared_state.cortex_session_id = ""`` before invoking us.
    * :class:`CortexKBError` on ``session_begin``:
        - ``fail_fast=True``: re-raise so cli can ``sys.exit(2)``.
        - ``fail_fast=False``: log warning + return
          ``failed_session_begin``; downstream stays workable with
          an empty warm_start.
    * ``find_recipe`` / ``traps`` failures are *always* non-fatal
      (warm_start is M5+ consumption,
      stale data is preferable to a crashed PRELUDE).

    Args
    ----
    client:
        The :class:`CortexKBClient` instance. Caller constructed it
        with the right ``session_dir`` / ``kb_url`` / ``enabled``
        flags; we don't reconstruct.
    shared_state:
        The live :class:`SharedState` whose ``cortex_session_id`` /
        ``warm_start_*`` fields we write.
    workload / hw:
        Identifiers used for ``recipe_canonical_id`` + ``find_recipe``
        + ``traps`` lookup. Mandatory; callers fall back to manifest
        keys (cli) or shared_state attributes (Coordinator).
    image_digest, stack_fingerprint, extra_attrs:
        Pass-through fields for ``session begin`` ``attrs``.
    resume:
        ``True`` when we are picking up a pre-existing session
        (``.kb_sid`` on disk or non-empty
        ``shared_state.cortex_session_id``). Influences the status
        banner but no T0 step is skipped just because resume=True.
    fail_fast:
        See class docstring.
    on_status:
        Banner emitter. Defaults to ``log.info``; cli supplies
        ``print``.
    session_dir:
        Used by ``save_state`` and the ``.kb_sid`` discovery fall-
        back. Defaults to ``client.session_dir``.
    save_state:
        When ``True`` (default), call ``shared_state.save(session_dir)``
        at the end so the writes survive a process crash.

    Returns
    -------
    :class:`T0Result` capturing the outcome.
    """
    emit = on_status or _default_status_emitter
    sd = session_dir or client.session_dir

    if not client.enabled:
        emit("Cortex KB        : DISABLED (--degraded-kb)")
        return T0Result(status="skipped_disabled", workload=workload, hw=hw)

    sid = (getattr(shared_state, "cortex_session_id", "") or "").strip()
    sid_file = client.sid_path
    if not sid and sid_file.exists():
        try:
            sid = sid_file.read_text(encoding="utf-8").strip()
        except OSError:
            sid = ""

    workload = (workload or "").strip() or "unknown_model"
    hw = (hw or "").strip() or "unknown_gpu"
    canonical = recipe_canonical_id(workload, hw)

    began_now = False
    if not sid:
        try:
            sid = client.session_begin(
                workload=workload,
                hw=hw,
                image_digest=image_digest,
                stack_fingerprint=stack_fingerprint,
                extra_attrs=extra_attrs,
            )
        except CortexKBError as exc:
            if fail_fast:
                raise
            log.warning(
                "Cortex T0 skipped: session_begin failed (%s); "
                "warm_start will stay empty for this session.",
                exc,
            )
            return T0Result(
                status="failed_session_begin",
                workload=workload, hw=hw,
                error=str(exc),
            )
        shared_state.cortex_session_id = sid
        shared_state.warm_start_ts = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )
        began_now = True
    else:
        # Reuse path — either a prior cli T0 already ran (so we are
        # the Coordinator fallback no-op) or this is a resume that
        # carried .kb_sid forward.
        shared_state.cortex_session_id = sid
        if resume:
            emit(f"Cortex KB        : resumed session_id={sid}")

    # Mint the recipe anchor — best-effort; KB validates canonical_id
    # against the registered ``recipe`` kind schema, so this is the
    # idempotent (model, hardware) anchor downstream warm-start /
    # kb-explorer rely on.
    # PR-A10: stamp model_class / model_family / framework into the
    # recipe anchor so future T4 (same-class) / T3 (same-family)
    # fallback lookups can succeed across sessions.
    from ..cortex_kb_client import model_family as _model_family
    _model_class = (extra_attrs or {}).get("model_class") if isinstance(extra_attrs, Mapping) else ""
    _framework = (extra_attrs or {}).get("framework") if isinstance(extra_attrs, Mapping) else ""
    _attrs = {
        "model":        workload,
        "hardware":     hw,
        "model_family": _model_family(workload),
        "isl":          getattr(shared_state, "last_profile_args", "") or None,
    }
    if _model_class:
        _attrs["model_class"] = str(_model_class)
    if _framework:
        _attrs["framework"] = str(_framework)
    try:
        client.propose_point(
            canonical_id=canonical,
            kind=C.KIND_RECIPE,
            authority=C.AUTHORITY_EXPERIENTIAL,
            attrs=_attrs,
            evidence=[
                f"log:hyperloom-session-{getattr(shared_state, 'session_id', '')}",
            ],
        )
    except CortexKBError as exc:
        log.warning("propose_point recipe anchor failed: %s", exc)

    # warm_start_recipe — non-fatal. PR-A10: graceful fallback ladder
    # so a cold-start session for a model with no exact KB recipe can
    # still pick up a same-family / same-class / same-hw prior.
    model_class = (extra_attrs or {}).get("model_class") if isinstance(extra_attrs, Mapping) else None
    framework = (extra_attrs or {}).get("framework") if isinstance(extra_attrs, Mapping) else None
    warm_point: dict[str, Any] = {}
    warm_tier: str = "miss"
    warm_conf: float = 0.0
    try:
        warm_point, warm_tier, warm_conf = client.find_recipe_with_fallback(
            workload=workload,
            hw=hw,
            model_class=str(model_class) if model_class else None,
            framework=str(framework) if framework else None,
        )
    except CortexKBError as exc:
        log.info("find_recipe_with_fallback non-fatal failure: %s", exc)
    # Keep a legacy raw text envelope on disk so existing readers
    # (kb_explorer, breakdown collectors) keep working; new readers
    # should prefer shared_state.warm_start_recipe["tier"] /
    # ["confidence"] / ["recipe"].
    warm_text = json.dumps(
        {"points": [warm_point] if warm_point else []}, sort_keys=True,
    )
    try:
        warm_path = sd / "runtime" / "cortex" / ".kb_warm.json"
        warm_path.parent.mkdir(parents=True, exist_ok=True)
        warm_path.write_text(
            json.dumps(
                {
                    "workload": workload, "hw": hw,
                    "tier": warm_tier, "confidence": warm_conf,
                    "recipe": warm_point, "raw": warm_text,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        shared_state.warm_start_recipe = {
            "workload": workload, "hw": hw,
            "tier": warm_tier, "confidence": warm_conf,
            "recipe": warm_point, "raw": warm_text,
        }
    except OSError as exc:
        log.warning("warm_start snapshot write failed: %s", exc)

    # warm_start_pitfalls — non-fatal.
    traps_text = ""
    try:
        traps_text = client.traps(symptom=f"{workload} {hw}")
    except CortexKBError as exc:
        log.info("traps non-fatal failure: %s", exc)
    try:
        pit_path = sd / "runtime" / "cortex" / ".kb_pitfalls.json"
        pit_path.parent.mkdir(parents=True, exist_ok=True)
        pit_path.write_text(
            json.dumps(
                {"workload": workload, "hw": hw, "raw": traps_text},
                indent=2,
            ),
            encoding="utf-8",
        )
        if traps_text.strip():
            shared_state.warm_start_pitfalls = [{"raw": traps_text}]
    except OSError as exc:
        log.warning("warm_start_pitfalls snapshot write failed: %s", exc)

    if save_state:
        try:
            shared_state.save(sd)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "Cortex T0: SharedState.save failed (sid=%s, workload=%s)",
                sid, workload,
            )

    # PR-A10: warm_present reflects whether the fallback ladder actually
    # found a usable record (i.e. tier != "miss" and confidence > 0). The
    # old `bool(warm_text.strip())` check fired on every 200 OK including
    # empty `{"points":[]}` responses, which misled operators into
    # thinking KB warm-start was working when it was silently empty.
    # traps_present keeps the legacy semantics because the traps payload
    # is still flat JSON text.
    warm_present = bool(warm_point) and warm_conf > 0.0
    traps_present = bool((traps_text or "").strip())
    if began_now:
        warm_label = (
            f"hit:{warm_tier}@{warm_conf:.2f}" if warm_present else
            "seed_only" if warm_point else
            "empty"
        )
        emit(
            f"Cortex KB        : session_id={sid} workload={canonical} "
            f"(warm={warm_label}, "
            f"traps={'hit' if traps_present else 'empty'})"
        )
    return T0Result(
        status="ok" if began_now else "skipped_already" if not resume else "resumed",
        session_id=sid,
        workload=workload,
        hw=hw,
        warm_present=warm_present,
        traps_present=traps_present,
    )


__all__ = [
    "T0Result",
    "run_t0_anchor",
]
