"""Shared T0 (PRELUDE) Cortex anchor — KB warm-start only.

The T0 anchor is the boot-time Cortex KB ritual every session must
run before EXPLORE starts. With the hypothesize/verify protocol
retired (no more KB-side session), T0 is now a pure **read** plus
a small recipe-anchor backfill:

1. ``find_recipe_with_fallback`` snapshot → ``.kb_warm.json`` +
   ``SharedState.warm_start_recipe`` (graceful fallback ladder:
   exact → same-family → same-class → same-hw → cross-hw).
2. ``traps`` snapshot → ``.kb_pitfalls.json`` +
   ``SharedState.warm_start_pitfalls``.
3. (Optional) ``propose_point(kind=recipe)`` to backfill model-family
   / model-class / framework attrs onto the recipe anchor when they
   are missing — keeps PR-A10's same-family fallback queryable for
   future sessions. The first KEEP / CLOSE update_recipe call will
   create the anchor on demand anyway, so this step is best-effort.

The two callers parameterise the helper via two knobs:

* ``fail_fast``: cli passes ``True`` so a Cortex outage exits the
  process; Coordinator passes ``False`` so SDK callers degrade to
  ``warm_start={}`` instead of crashing inside a long-running
  reactor loop.
* ``on_status``: cli supplies ``print`` so the operator sees the
  ``Cortex KB        : recipe=...`` banner; Coordinator supplies a
  ``log.info`` shim so the same line lands in the session log
  without polluting stdout.

``cortex_session_id`` on SharedState is still maintained (as the
hyperloom-local session identifier carried into fact-write
``source_session_id`` attrs for traceability) but no longer
corresponds to a KB-side session: it now uses the session_dir
basename when present, otherwise falls back to the file-based
``.kb_sid`` snapshot from earlier runs.
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
    ``"skipped_already"``}. ``skipped_disabled`` is the no-op path
    when the client is disabled (``--degraded-kb`` / SDK without
    Cortex); ``skipped_already`` when ``cortex_session_id`` was
    already non-empty on entry (Coordinator's fallback no-op after
    the cli T0 already ran).
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
    * The KB session begin protocol was retired; this helper no
      longer fail-fasts on session creation. ``fail_fast`` is kept
      as a no-op for back-compat with cli callers.
    * ``find_recipe`` / ``traps`` failures are *always* non-fatal —
      warm_start is a hint, stale data is preferable to a crashed
      PRELUDE.

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

    # Hyperloom-local session id (carried into fact-write attrs for
    # traceability). The Cortex KB session protocol was retired, so
    # ``cortex_session_id`` no longer corresponds to a remote sid;
    # it is now whatever uniquely identifies this run.
    sid = (getattr(shared_state, "cortex_session_id", "") or "").strip()
    if not sid and sd is not None:
        sid = Path(sd).name

    workload = (workload or "").strip() or "unknown_model"
    hw = (hw or "").strip() or "unknown_gpu"

    if sid:
        shared_state.cortex_session_id = sid
        if resume:
            emit(f"Cortex KB        : resumed session_id={sid}")
    began_now = not getattr(shared_state, "warm_start_ts", "")
    if began_now:
        shared_state.warm_start_ts = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )

    # Backfill metadata attrs onto the recipe anchor so PR-A10
    # fallback queries can match it earlier. Best-effort: the first
    # KEEP / CLOSE update_recipe will create the point on demand
    # anyway, so this backfill just pre-stamps the searchable tags.
    #
    # Tags written here:
    #   * model / hardware / model_family            — always
    #   * model_class / framework                    — from extra_attrs (cli)
    #   * framework_version                          — from stack_fingerprint
    #   * precision / tp / conc / isl / osl /        — from SharedState (cli)
    #     max_model_len
    from ..cortex_kb_client import model_family as _model_family
    _extra: Mapping[str, Any] = extra_attrs if isinstance(extra_attrs, Mapping) else {}
    _model_class = str(_extra.get("model_class") or "").strip()
    _framework = str(_extra.get("framework") or "").strip()
    _attrs: dict[str, Any] = {
        "model":        workload,
        "hardware":     hw,
        "model_family": _model_family(workload),
    }
    if _model_class:
        _attrs["model_class"] = _model_class
    if _framework:
        _attrs["framework"] = _framework
    # framework_version — lift from stack_fingerprint
    # (``{rocm, sglang, vllm, aiter}`` versions, populated by manifest).
    fp: Mapping[str, Any] = stack_fingerprint if isinstance(stack_fingerprint, Mapping) else {}
    if _framework in ("sglang", "vllm"):
        version = str(fp.get(_framework) or "").strip()
        if version and version != "unknown":
            _attrs["framework_version"] = version
    rocm_v = str(fp.get("rocm") or "").strip()
    if rocm_v and rocm_v != "unknown":
        _attrs["rocm_version"] = rocm_v
    aiter_v = str(fp.get("aiter") or "").strip()
    if aiter_v and aiter_v != "unknown":
        _attrs["aiter_version"] = aiter_v
    # Workload-shape tags — read from SharedState (cli already wrote
    # them via _seed_shared_state). Skipping empty / zero values so
    # KB query filter doesn't match "tp=0" placeholders.
    for src_attr, dst_key in (
        ("precision",     "precision"),
        ("tp",            "tp"),
        ("conc",          "conc"),
        ("isl",           "isl"),
        ("osl",           "osl"),
        ("max_model_len", "max_model_len"),
    ):
        v = getattr(shared_state, src_attr, None)
        if v not in (None, "", 0):
            _attrs[dst_key] = v
    try:
        client.update_recipe(
            model=workload,
            hardware=hw,
            extra_attrs=_attrs,
        )
    except CortexKBError as exc:
        log.warning("update_recipe T0 backfill failed: %s", exc)
    except Exception:  # noqa: BLE001 — defensive
        log.exception("update_recipe T0 backfill raised unexpectedly")

    # warm_start_recipe — non-fatal. PR-A10: graceful fallback ladder
    # so a cold-start session for a model with no exact KB recipe can
    # still pick up a same-family / same-class / same-hw prior.
    model_class = (extra_attrs or {}).get("model_class") if isinstance(extra_attrs, Mapping) else None
    framework = (extra_attrs or {}).get("framework") if isinstance(extra_attrs, Mapping) else None
    # Workload-shape filters (T2 fallback tier — same precision + tp
    # beats same-family alone).
    _precision = str(getattr(shared_state, "precision", "") or "").strip()
    _tp = int(getattr(shared_state, "tp", 0) or 0)
    warm_point: dict[str, Any] = {}
    warm_tier: str = "miss"
    warm_conf: float = 0.0
    try:
        warm_point, warm_tier, warm_conf = client.find_recipe_with_fallback(
            workload=workload,
            hw=hw,
            model_class=str(model_class) if model_class else None,
            framework=str(framework) if framework else None,
            precision=_precision or None,
            tp=_tp if _tp > 0 else None,
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
            f"Cortex KB        : session_id={sid} "
            f"workload={recipe_canonical_id(workload, hw)} "
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
