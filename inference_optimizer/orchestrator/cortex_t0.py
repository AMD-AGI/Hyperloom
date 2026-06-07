# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared T0 (PRELUDE) Cortex anchor — KB warm-start only.

The T0 anchor is the boot-time KB ritual every session runs before
EXPLORE starts. Under the v2 RecipeKB design it is a pure **read**
plus a small recipe-anchor metadata backfill:

1. warm-start via a single ``RecipeKB.get_recipe(canonical_id)`` →
   ``.kb_warm.json`` + ``SharedState.warm_start_recipe``. Remote goes
   through the one ``/recipes/search`` route (5-tuple ``label_match``;
   the server does any exact-vs-relative fallback); local is an exact
   read of the on-disk recipe.json. The hit is classified ``exact``
   (returned ``canonical_id`` equals the request) or ``relative`` (the
   server returned a neighbouring 5-tuple), else ``miss``.
2. embedded ``pitfalls`` / ``lessons`` → ``.kb_pitfalls.json`` /
   ``.kb_lessons.json`` + ``SharedState.warm_start_pitfalls`` /
   ``warm_start_lessons`` — read straight off the warm-start recipe
   row (1:1 with the recipe under v2; no separate query).
3. a single ``put_recipe`` metadata stamp (read-modify-write against
   the LOCAL store) so this 5-tuple's row carries the latest tracing
   tuple. The first KEEP / CLOSE write refreshes best_config anyway,
   so this step is best-effort.

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
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ..recipe_kb import RecipeKB, recipe_canonical_id
from ..recipe_snapshot_constants import detect_framework_version


log = logging.getLogger(__name__)


@dataclass
class T0Result:
    """Outcome of one :func:`run_t0_anchor` invocation.

    ``status`` ∈ {``"ok"``, ``"resumed"``, ``"skipped_disabled"``,
    ``"skipped_already"``}.

    * ``skipped_disabled`` — client is disabled (``--degraded-kb``
      / SDK without Cortex).
    * ``skipped_already`` — ``cortex_session_id`` AND
      ``warm_start_ts`` were both set on entry (Coordinator's
      fallback no-op after the cli T0 already ran).
    * ``ok`` — first anchor of this process.
    * ``resumed`` — refreshed warm-start on a ``resume=True`` call.
    """

    status: str
    session_id: str = ""
    workload: str = ""
    hw: str = ""
    warm_present: bool = False
    pitfalls_present: bool = False
    lessons_present: bool = False
    error: str = ""


def _default_status_emitter(line: str) -> None:
    """Default ``on_status`` — log at INFO."""
    log.info("%s", line)


def run_t0_anchor(
    kb: RecipeKB,
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
    """Run the T0 recipe-snapshot anchor.

    Mutates ``shared_state`` in place (sets ``cortex_session_id``,
    ``warm_start_ts``, ``warm_start_recipe``, ``warm_start_pitfalls``)
    and, when ``save_state=True``, persists via
    ``shared_state.save(session_dir)``.

    Reads / writes go through the :class:`RecipeKB` dispatcher.
    Identity uses the v2 5-tuple canonical_id
    (model + hardware + framework + framework_version + precision);
    no fallback ladder — a cold-start session for a model whose
    exact 5-tuple has no prior recipe simply gets ``warm_present =
    False`` and the optimizer cold-starts.

    Failure handling:

    * Already-anchored short-circuit (``cortex_session_id`` AND
      ``warm_start_ts`` both set, ``resume=False``) → no-op,
      returns ``T0Result(status='skipped_already')``.
    * ``resume=True`` callers intentionally bypass the short-circuit
      so a refreshed warm-start surface is fetched (the local store
      may have grown new recipes since the original run, and the
      central kb-service may have caught up too).
    * ``fail_fast`` is kept as a no-op for back-compat with cli
      callers.
    * Read failures (remote unhealthy / disabled) are absorbed by
      the dispatcher and degrade silently to local-only — warm_start
      is a hint, stale data is preferable to a crashed PRELUDE.

    Args
    ----
    kb:
        The :class:`RecipeKB` dispatcher instance. Caller
        constructed it with the right ``--local-kb-root`` /
        ``--cortex-kb-url`` / ``--degraded-kb`` settings via
        ``cli._build_recipe_kb_dispatcher``.
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
        back. Caller MUST pass an explicit value — the dispatcher
        doesn't carry a session_dir of its own.
    save_state:
        When ``True`` (default), call ``shared_state.save(session_dir)``
        at the end so the writes survive a process crash.

    Returns
    -------
    :class:`T0Result` capturing the outcome.
    """
    emit = on_status or _default_status_emitter
    if session_dir is None:
        raise ValueError("run_t0_anchor requires an explicit session_dir")
    sd = Path(session_dir)

    # Hyperloom-local session id (carried into fact-write attrs for
    # traceability). The Cortex KB session protocol was retired, so
    # ``cortex_session_id`` no longer corresponds to a remote sid;
    # it is now whatever uniquely identifies this run.
    sid = (getattr(shared_state, "cortex_session_id", "") or "").strip()
    if not sid and sd is not None:
        sid = Path(sd).name

    workload = (workload or "").strip() or "unknown_model"
    hw = (hw or "").strip() or "unknown_gpu"

    # Short-circuit when this session has already been anchored in
    # the current process. Both cli._bootstrap_cortex_kb (canonical
    # entry) and Coordinator._ensure_cortex_t0_anchored (SDK fallback)
    # call us in sequence on a normal fresh launch; the second call
    # should NOT re-issue the recipe-anchor write + warm-start ladder
    # (those are 7+ KB HTTP requests in the worst case). Detect prior
    # anchoring via ``warm_start_ts`` — written below on the first
    # successful pass, present on resume from state.json, absent on a
    # cold start.
    #
    # Resume callers (``resume=True``) intentionally bypass the
    # short-circuit so a refreshed warm-start surface is fetched
    # (the live KB may have grown new recipes since the original run).
    if (
        sid
        and not resume
        and (getattr(shared_state, "warm_start_ts", "") or "").strip()
    ):
        shared_state.cortex_session_id = sid
        emit(f"Cortex KB        : already anchored session_id={sid}")
        return T0Result(
            status="skipped_already",
            session_id=sid,
            workload=workload,
            hw=hw,
            warm_present=bool(getattr(shared_state, "warm_start_recipe", {})),
            pitfalls_present=bool(getattr(shared_state, "warm_start_pitfalls", [])),
            lessons_present=bool(getattr(shared_state, "warm_start_lessons", [])),
        )

    if sid:
        shared_state.cortex_session_id = sid
        if resume:
            emit(f"Cortex KB        : resumed session_id={sid}")
    began_now = not getattr(shared_state, "warm_start_ts", "")
    if began_now:
        shared_state.warm_start_ts = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )

    # Backfill metadata onto the recipe anchor so subsequent reads
    # (warm-start) and the CLOSE-time update_recipe see the operator-
    # tracing fields (model_class, image_digest, claw_session_id,
    # ...). T0 only stamps metadata — best_config / best_throughput /
    # what_worked etc. stay whatever they were (preserved across the
    # read-modify-write below). The CLOSE-time hook in coordinator
    # rewrites them with measured values.
    _extra: Mapping[str, Any] = (
        extra_attrs if isinstance(extra_attrs, Mapping) else {}
    )
    _model_class = str(_extra.get("model_class") or "").strip()
    _framework   = str(_extra.get("framework")   or "").strip()
    _precision   = str(getattr(shared_state, "precision", "") or "").strip()
    fp: Mapping[str, Any] = (
        stack_fingerprint if isinstance(stack_fingerprint, Mapping) else {}
    )
    # framework_version: prefer SharedState (CLI explicit), else lift
    # from stack_fingerprint, else auto-detect via importlib.
    _fw_version = str(getattr(shared_state, "framework_version", "") or "").strip()
    if not _fw_version and _framework in ("sglang", "vllm"):
        _fw_version = str(fp.get(_framework) or "").strip()
        if _fw_version == "unknown":
            _fw_version = ""
    if not _fw_version and _framework:
        _fw_version = detect_framework_version(_framework)

    # Operator-traceability + workload-shape tags — written into
    # ``extras`` so they round-trip through the arbor schema's
    # free-form key support. Skipping empty / zero values.
    _extras: dict[str, Any] = {}
    if _model_class:
        _extras["model_class"] = _model_class
    # Architecture-identity tags from the model's config.json (carried on
    # SharedState by ``cli._load_model_config_tags``). Stamped so a
    # fine-tuned model's recipe records the same architecture as its base.
    _architectures = getattr(shared_state, "model_architectures", None) or []
    if isinstance(_architectures, list):
        _arch_list = [str(a).strip() for a in _architectures if str(a or "").strip()]
        if _arch_list:
            _extras["architectures"] = _arch_list
    _model_type = str(getattr(shared_state, "model_type", "") or "").strip()
    if _model_type:
        _extras["model_type"] = _model_type
    rocm_v = str(fp.get("rocm") or "").strip()
    if rocm_v and rocm_v != "unknown":
        _extras["rocm_version"] = rocm_v
    aiter_v = str(fp.get("aiter") or "").strip()
    if aiter_v and aiter_v != "unknown":
        _extras["aiter_version"] = aiter_v
    if image_digest and image_digest != "unknown":
        _extras["image_digest"] = str(image_digest).strip()
    for src_key in ("claw_session_id", "sandbox_user_id"):
        v = str(_extra.get(src_key) or "").strip()
        if v:
            _extras[src_key] = v
    for src_attr, dst_key in (
        ("tp",            "tp"),
        ("ep",            "ep"),
        ("conc",          "conc"),
        ("isl",           "isl"),
        ("osl",           "osl"),
        ("max_model_len", "max_model_len"),
    ):
        v = getattr(shared_state, src_attr, None)
        if v not in (None, "", 0):
            _extras[dst_key] = v
    if "ep" not in _extras:
        raw_ep = (os.environ.get("EP") or "").strip()
        try:
            n = int(raw_ep) if raw_ep else 0
        except ValueError:
            n = 0
        if n > 0:
            _extras["ep"] = n
    raw_pp = (os.environ.get("PP") or "").strip()
    try:
        pp_n = int(raw_pp) if raw_pp else 0
    except ValueError:
        pp_n = 0
    if pp_n > 0:
        _extras["pp"] = pp_n

    # Build the canonical_id from the 5-tuple we just resolved. sglang
    # 0.4.5 / sglang 0.5.x / vllm 0.6.0 each get their own recipe row
    # (precision is the strongest "secondary" identity dim — fp8 vs
    # bf16 changes optimal tp / ep wholesale).
    cid = recipe_canonical_id(
        model=workload, hardware=hw,
        framework=_framework or "",
        framework_version=_fw_version or "",
        precision=_precision or "",
    )

    # Persist the resolved framework + framework_version back onto
    # SharedState so the CLOSE/KEEP write path
    # (coordinator._workload_canonical_id) derives an IDENTICAL cid.
    # T0 may have lifted framework_version from the stack fingerprint
    # or importlib auto-detect; the coordinator has neither source, so
    # without this write-back its re-derivation could differ and the
    # KEEP/REVERT/CLOSE writes would land on a different recipe row
    # than the one warm-start just anchored.
    if _framework:
        shared_state.framework = _framework
    if _fw_version:
        shared_state.framework_version = _fw_version

    # Read-modify-write: load the LOCAL payload (if any) so this T0
    # metadata stamp doesn't clobber best_config / sessions /
    # what_worked / etc. that the prior CLOSE wrote locally. We read the
    # LOCAL store (not the remote-first dispatcher) for the same reason
    # ``coordinator._kb_amend_recipe`` does: the row we must not lose is
    # the one this operator wrote locally; a remote-first read could
    # merge an older central row and downgrade local data on the put.
    try:
        live = kb.local.get_recipe(canonical_id=cid) or {}
    except Exception as exc:  # noqa: BLE001 — defensive
        log.info("T0 anchor local get_recipe non-fatal failure: %s", exc)
        live = {}

    # Merge prior extras with the new ones we want to stamp; new
    # values win (T0 always carries the freshest operator-tracing
    # tuple, by design — see the ``most recent tracing tuple``
    # comment in the prior implementation).
    merged_extras: dict[str, Any] = {}
    prior_extras = {
        k: v for k, v in (live or {}).items()
        if k not in {
            "canonical_id", "version", "created_at", "updated_at",
            "model", "hardware", "framework", "framework_version",
            "precision",
            "best_config", "best_throughput",
            "what_worked", "what_failed", "remaining_gaps",
            "prs_tested", "pitfalls", "lessons",
            "last_profiled", "stack_fingerprint", "sessions",
            "authority", "confidence", "evidence_refs", "provenance",
        }
    }
    merged_extras.update(prior_extras)
    merged_extras.update(_extras)

    # Stack fingerprint — preserve prior values where a key is
    # present locally but not stamped this round.
    sfp_payload: dict[str, str] = dict(live.get("stack_fingerprint") or {})
    if isinstance(fp, Mapping):
        for fp_key in ("vllm_version", "aiter_commit", "rocm_version"):
            new = str(fp.get(fp_key.replace("_version", "").replace("_commit", "")) or "").strip()
            if new and new != "unknown":
                sfp_payload[fp_key] = new

    try:
        kb.put_recipe(
            canonical_id=cid,
            model=workload, hardware=hw,
            framework=_framework or "", framework_version=_fw_version or "",
            precision=_precision or "",
            best_config=dict(live.get("best_config") or {}),
            best_throughput=float(live.get("best_throughput") or 0.0),
            what_worked=list(live.get("what_worked") or []),
            what_failed=list(live.get("what_failed") or []),
            remaining_gaps=list(live.get("remaining_gaps") or []),
            prs_tested=list(live.get("prs_tested") or []),
            pitfalls=list(live.get("pitfalls") or []),
            lessons=list(live.get("lessons") or []),
            last_profiled=str(live.get("last_profiled") or ""),
            stack_fingerprint=sfp_payload,
            sessions=list(live.get("sessions") or []),
            extras=merged_extras,
            provenance={
                "source":       "hyperloom-inference-optimizer",
                "generator":    "t0_anchor",
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="microseconds",
                ),
                "details":      {"sid": sid},
            },
        )
    except Exception:  # noqa: BLE001 — defensive
        log.exception("T0 anchor put_recipe raised unexpectedly")

    # warm_start_recipe — ONE call through the dispatcher. Remote goes
    # via the single /recipes/search route (5-tuple label_match) and
    # the SERVER decides exact-vs-relative match + fallback; local is
    # an exact arbor-style read of this 5-tuple's recipe.json. We
    # classify the hit by comparing the returned row's canonical_id to
    # the requested one: equal => exact (this precise 5-tuple existed);
    # different => the server returned a relative (fallback) match for a
    # neighbouring 5-tuple.
    warm_point: dict[str, Any] = {}
    warm_tier: str = "miss"
    warm_conf: float = 0.0
    try:
        row = kb.get_recipe(canonical_id=cid)
    except Exception as exc:  # noqa: BLE001 — dispatcher absorbs RemoteRecipeClientError
        log.info("warm-start get_recipe non-fatal failure: %s", exc)
        row = None
    if isinstance(row, dict) and row:
        warm_point = row
        if str(row.get("canonical_id") or "") == cid:
            warm_tier = "exact"
            warm_conf = 1.0
        else:
            # Server-side fallback returned a neighbouring 5-tuple's
            # recipe. Still useful as a prior; confidence sits at the
            # warm-replay trigger threshold (0.7) so it is verified-
            # before-applied rather than trusted blindly.
            warm_tier = "relative"
            warm_conf = 0.7
    # Keep the on-disk warm.json envelope shape stable so existing
    # readers (kb_explorer, breakdown collectors) keep working;
    # new readers should prefer shared_state.warm_start_recipe.
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

    # warm_start_pitfalls / warm_start_lessons — under the v2 design
    # these are embedded fields of the recipe row (one row per
    # 5-tuple), so we just read them out of the warm_point we
    # already loaded. No separate query, no cross-recipe ranking
    # ladder — the user explicitly asked for exact-5-tuple-only
    # semantics.
    pitfalls_list: list[dict[str, Any]] = list(warm_point.get("pitfalls") or [])
    lessons_list:  list[dict[str, Any]] = list(warm_point.get("lessons") or [])
    try:
        pit_path = sd / "runtime" / "cortex" / ".kb_pitfalls.json"
        pit_path.parent.mkdir(parents=True, exist_ok=True)
        pit_path.write_text(
            json.dumps(
                {
                    "workload":  workload,
                    "hw":        hw,
                    "framework": _framework or "",
                    "pitfalls":  pitfalls_list,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if pitfalls_list:
            shared_state.warm_start_pitfalls = pitfalls_list
    except OSError as exc:
        log.warning("warm_start_pitfalls snapshot write failed: %s", exc)
    try:
        les_path = sd / "runtime" / "cortex" / ".kb_lessons.json"
        les_path.parent.mkdir(parents=True, exist_ok=True)
        les_path.write_text(
            json.dumps(
                {
                    "workload":  workload,
                    "hw":        hw,
                    "framework": _framework or "",
                    "lessons":   lessons_list,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if lessons_list:
            shared_state.warm_start_lessons = lessons_list
    except OSError as exc:
        log.warning("warm_start_lessons snapshot write failed: %s", exc)

    if save_state:
        try:
            shared_state.save(sd)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "Cortex T0: SharedState.save failed (sid=%s, workload=%s)",
                sid, workload,
            )

    # warm_present reflects whether the fallback ladder actually
    # found a usable record (i.e. tier != "miss" and confidence > 0). The
    # old `bool(warm_text.strip())` check fired on every 200 OK including
    # empty `{"points":[]}` responses, which misled operators into
    # thinking KB warm-start was working when it was silently empty.
    # traps_present keeps the legacy semantics because the traps payload
    # is still flat JSON text.
    warm_present = bool(warm_point) and warm_conf > 0.0
    pitfalls_present = bool(pitfalls_list)
    lessons_present = bool(lessons_list)
    if began_now:
        warm_label = (
            f"hit:{warm_tier}@{warm_conf:.2f}" if warm_present else
            "seed_only" if warm_point else
            "empty"
        )
        emit(
            f"Recipe KB        : session_id={sid} "
            f"workload={cid} "
            f"(warm={warm_label}, "
            f"pitfalls={len(pitfalls_list)}, "
            f"lessons={len(lessons_list)})"
        )
    return T0Result(
        status="ok" if began_now else "skipped_already" if not resume else "resumed",
        session_id=sid,
        workload=workload,
        hw=hw,
        warm_present=warm_present,
        pitfalls_present=pitfalls_present,
        lessons_present=lessons_present,
    )


__all__ = [
    "T0Result",
    "run_t0_anchor",
]
