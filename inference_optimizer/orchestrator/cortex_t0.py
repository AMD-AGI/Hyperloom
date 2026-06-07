# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared T0 (PRELUDE) Cortex anchor — KB warm-start only."""

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
    """Outcome of one :func:`run_t0_anchor` invocation. ``status`` ∈ {ok, resumed, skipped_disabled, skipped_already}."""

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

    Mutates ``shared_state`` in place (warm_start_* fields) and persists when
    ``save_state=True``. ``session_dir`` is required. Returns a :class:`T0Result`.
    """
    emit = on_status or _default_status_emitter
    if session_dir is None:
        raise ValueError("run_t0_anchor requires an explicit session_dir")
    sd = Path(session_dir)

    # Hyperloom-local session id (Cortex KB session protocol retired).
    sid = (getattr(shared_state, "cortex_session_id", "") or "").strip()
    if not sid and sd is not None:
        sid = Path(sd).name

    workload = (workload or "").strip() or "unknown_model"
    hw = (hw or "").strip() or "unknown_gpu"

    # Short-circuit when already anchored (via ``warm_start_ts``); resume=True bypasses.
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

    # Backfill operator-tracing metadata; T0 only stamps metadata (best_config preserved, rewritten at CLOSE).
    _extra: Mapping[str, Any] = (
        extra_attrs if isinstance(extra_attrs, Mapping) else {}
    )
    _model_class = str(_extra.get("model_class") or "").strip()
    _framework   = str(_extra.get("framework")   or "").strip()
    _precision   = str(getattr(shared_state, "precision", "") or "").strip()
    fp: Mapping[str, Any] = (
        stack_fingerprint if isinstance(stack_fingerprint, Mapping) else {}
    )
    # framework_version: SharedState > stack_fingerprint > importlib auto-detect.
    _fw_version = str(getattr(shared_state, "framework_version", "") or "").strip()
    if not _fw_version and _framework in ("sglang", "vllm"):
        _fw_version = str(fp.get(_framework) or "").strip()
        if _fw_version == "unknown":
            _fw_version = ""
    if not _fw_version and _framework:
        _fw_version = detect_framework_version(_framework)

    # Operator-traceability + workload-shape tags into ``extras`` (skip empty/zero).
    _extras: dict[str, Any] = {}
    if _model_class:
        _extras["model_class"] = _model_class
    # Architecture-identity tags from config.json (records base architecture for fine-tunes).
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

    # Build canonical_id from the resolved 5-tuple (precision is a strong identity dim).
    cid = recipe_canonical_id(
        model=workload, hardware=hw,
        framework=_framework or "",
        framework_version=_fw_version or "",
        precision=_precision or "",
    )

    # Persist framework + framework_version onto SharedState so CLOSE/KEEP derives an identical cid.
    if _framework:
        shared_state.framework = _framework
    if _fw_version:
        shared_state.framework_version = _fw_version

    # Read-modify-write the LOCAL store so the stamp doesn't clobber best_config / sessions / what_worked.
    try:
        live = kb.local.get_recipe(canonical_id=cid) or {}
    except Exception as exc:  # noqa: BLE001 — defensive
        log.info("T0 anchor local get_recipe non-fatal failure: %s", exc)
        live = {}

    # Merge prior extras; new values win.
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

    # Stack fingerprint — preserve prior values not stamped this round.
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

    # warm_start_recipe — one dispatcher call; equal canonical_id => exact, else relative.
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
            # Neighbouring 5-tuple fallback; confidence 0.7 (verified-before-applied).
            warm_tier = "relative"
            warm_conf = 0.7
    # Keep warm.json envelope shape stable; new readers prefer shared_state.warm_start_recipe.
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

    # warm_start_pitfalls / warm_start_lessons are embedded recipe-row fields; read from warm_point.
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

    # warm_present = usable record (tier != "miss" and confidence > 0).
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
