# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json

from ...session.paths import is_path_within
from ._common import (
    _rel,
    _to_float,
    _to_int,
)


def _geak_result_kernel_specs(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return authored kernel specs from ``result.json``, both lanes, env excluded."""
    try:
        from hyperloom.orchestrator.loop.coordinator_helpers import (
            _geak_accepted_kernel_specs,
        )

        return _geak_accepted_kernel_specs(result)
    except Exception:  # pragma: no cover - offline replay without orchestrator
        out: list[dict[str, Any]] = []
        kernels = result.get("accepted_kernels") or []
        heads = result.get("accepted_heads") or []
        if not isinstance(kernels, list):
            kernels = []
        if not isinstance(heads, list):
            heads = []
        for lane in kernels + heads:
            if not isinstance(lane, dict):
                continue
            if str(lane.get("kind") or "").strip().lower() == "env":
                continue
            try:
                delta = float(lane.get("e2e_delta_pct") or 0.0)
            except (TypeError, ValueError):
                continue
            if delta <= 0.0:
                continue
            out.append(lane)
        return out


def _legacy_string_accepted_kernels(result: dict[str, Any]) -> list[str]:
    """Preserve pre-schema bare-string ``accepted_kernels`` lists."""
    raw = result.get("accepted_kernels") or []
    if not isinstance(raw, list) or not raw:
        return []
    if not all(isinstance(item, str) for item in raw):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _journey_paths(result: dict[str, Any]) -> list[Path]:
    """Every ``kernel_journey.json`` belonging to a GEAK session, pointer first.

    ``result.json`` is the *last* e2e cycle's return, so ``kernel_journey_path``
    (or ``eval_dir``) names that cycle alone. A run that accepts kernels in cycle
    0 and then opens a cycle 1 that accepts none leaves the pointer aimed at an
    empty journey. Enumerate the sibling cycles so attribution survives.

    Args:
        result (dict[str, Any]): The normalized ``result.json``.

    Returns:
        list[Path]: Existing journey files. The pointer cycle is first so it wins
            on any de-duplication; siblings follow in cycle order.
    """
    kj_path = str(result.get("kernel_journey_path") or "")
    if not kj_path:
        eval_dir = str(result.get("eval_dir") or "")
        if not eval_dir:
            return []
        kj_path = str(Path(eval_dir) / "kernel_journey.json")

    pointer = Path(kj_path)
    paths: list[Path] = [pointer] if pointer.is_file() else []
    try:
        siblings = sorted(pointer.parent.parent.glob("*/kernel_journey.json"), key=lambda p: p.parent.name)
    except OSError:
        siblings = []
    paths.extend(p for p in siblings if p.is_file() and p != pointer)
    return paths


def _geak_cand_tag_test() -> Any:
    """Return the one test for "this id is a slot tag, not a kernel symbol".

    The ledger
    (:func:`~hyperloom.orchestrator.loop.coordinator_helpers.geak_is_cand_tag`)
    owns the definition so the collector and the ledger cannot disagree about
    which half of a twin is the kernel. Collectors also run offline against a
    tarball with no orchestrator package importable; that case repeats the
    pattern rather than giving up and keeping an arbitrary id.

    Returns:
        Any: A callable taking one id and returning True for the slot-tag form.
    """
    try:
        from hyperloom.orchestrator.loop.coordinator_helpers import geak_is_cand_tag
    except Exception:  # pragma: no cover - offline replay without orchestrator
        import re as _re

        _tag = _re.compile(r"^(cand[_-])?c\d+([_-]|$)", _re.IGNORECASE)

        def geak_is_cand_tag(name: Any) -> bool:
            text = str(name or "").strip()
            return bool(text) and bool(_tag.match(text))

    return geak_is_cand_tag


def _kind_source_counts(rows: list[Any]) -> dict[str, int]:
    """Count how each admitted kernel got its ``kind``.

    Without this the report shows a kernel count and nothing else, and a reader
    cannot tell "7 kernels GEAK declared authored" from "7 rows nobody
    classified". A backfilled row's kind is recovered by a name join, and the
    join does not always land; the miss has to be countable or it is silent.

    The ``result`` path does not join, so its rows carry no ``kind_source``.
    They are still classified here rather than left uncounted: measured over the
    recorded campaign, 3 of 4 rows on that path declare no ``kind`` at all, so
    treating the path as "always declared" would empty the counter exactly where
    a reader needs it. Every admitted row is counted on every path.

    Args:
        rows (list[Any]): The admitted accepted-kernel descriptors.

    Returns:
        dict[str, int]: ``kind_source`` value to row count. One entry per
        admitted row, so the values sum to ``len(rows)``.
    """
    kind_of = _geak_kind_reader()
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("kind_source") or "").strip() if isinstance(row, dict) else ""
        if not source:
            # Straight off a ``result.json`` lane: declared, or declared empty.
            source = "result_json" if kind_of(row) else "result_json_undeclared"
        counts[source] = counts.get(source, 0) + 1
    return counts


def _geak_kind_reader() -> Any:
    """Return the canonical ``kind`` reader, with an offline fallback.

    Returns:
        Any: A callable taking one acceptance spec and returning its declared
        ``kind``, or ``None`` when it declares none.
    """
    try:
        from hyperloom.orchestrator.loop.coordinator_helpers import geak_spec_kind

        return geak_spec_kind
    except Exception:  # pragma: no cover - offline replay without orchestrator

        def geak_spec_kind(spec: Any) -> str | None:
            if not isinstance(spec, dict):
                return None
            raw = spec.get("kind")
            if raw is None:
                return None
            return str(raw).strip().lower() or None

        return geak_spec_kind


def _geak_kind_index(
    result: dict[str, Any],
    stack: Any = None,
) -> dict[str, tuple[str | None, str]]:
    """Map every acceptance name GEAK declared to its ``kind``, and to its source.

    Two artifacts declare a kind, in the same two lanes and the same spelling:
    ``result.json``, and the ``action == "geak_e2e"`` entries of
    ``state.optimization_stack``, which the KERNEL phase copies from the result
    of *that* cycle.

    Reading only the first loses runs. ``result.json`` is rewritten per cycle
    and the last write wins, so a later cycle that accepts nothing blanks the
    lanes an earlier cycle declared. The stack entry is append-only and keeps
    them. On ``Qwen3-14B-FP8/20260816T050457Z`` the flushed ``result.json``
    names 0 lanes while the run accepted three rows, so reading it alone leaves
    every recovered row ``kind_source: absent`` and the exclusion cannot run on
    them at all. (That ``state.json`` is mode 600 and was not read here; the
    stack is trusted because the KERNEL phase writes both lanes into the entry
    verbatim, not because this run's copy was inspected.)

    Args:
        result (dict[str, Any]): The normalized ``result.json``.
        stack (Any): ``state["optimization_stack"]``, if available. Non-list
            values and non-``geak_e2e`` entries are ignored.

    Returns:
        dict[str, tuple[str | None, str]]: Name to ``(kind, origin)``, where
        ``origin`` is ``"result_json"`` or ``"stack"``. A ``kind`` of ``None``
        means the lane exists but declared no kind. A declared kind always
        beats an undeclared one; between two declarations the earlier artifact
        (``result.json``) wins, so adding the stack can never change a kind the
        run itself published.
    """
    try:
        from hyperloom.orchestrator.loop.coordinator_helpers import (
            _geak_spec_name,
            geak_spec_kind,
        )
    except Exception:  # pragma: no cover - offline replay without orchestrator

        def _geak_spec_name(spec: Any) -> str:
            if isinstance(spec, str):
                return spec.strip()
            if not isinstance(spec, dict):
                return ""
            return str(spec.get("short_name") or spec.get("kernel_id") or spec.get("cand_tag") or "").strip()

        def geak_spec_kind(spec: Any) -> str | None:
            if not isinstance(spec, dict):
                return None
            raw = spec.get("kind")
            if raw is None:
                return None
            return str(raw).strip().lower() or None

    index: dict[str, tuple[str | None, str]] = {}

    def _absorb(specs: Any, origin: str) -> None:
        for spec in specs or []:
            name = _geak_spec_name(spec)
            if not name:
                continue
            kind = geak_spec_kind(spec)
            prior = index.get(name)
            if prior is not None and (kind is None or prior[0] is not None):
                continue
            index[name] = (kind, origin)

    if isinstance(result, dict):
        _absorb(result.get("accepted_kernels"), "result_json")
        _absorb(result.get("accepted_heads"), "result_json")
    for entry in stack if isinstance(stack, list) else []:
        if not isinstance(entry, dict) or entry.get("action") != "geak_e2e":
            continue
        _absorb(entry.get("accepted_kernels"), "stack")
        _absorb(entry.get("accepted_heads"), "stack")
    return index


def _stamp_journey_kind(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    stack: Any = None,
) -> list[dict[str, Any]]:
    """Stamp each backfilled row with a ``kind`` and say where it came from.

    A journey row carries no ``kind`` field — measured over
    ``/shared_nfs/hyperloom-claw``, 0 of 36 accepted journey rows have one. So
    the collector could not run the ``kind == "env"`` exclusion that
    :func:`~hyperloom.orchestrator.loop.coordinator_helpers._geak_accepted_kernel_specs`
    runs, and the same PR would have shipped two different admission tests: the
    ledger dropping library selections, the collector counting them as authored
    kernels.

    ``kind`` is recovered by joining the row's symbol to the same run's
    ``result.json`` lane, which does declare it. That join only became reliable
    once :func:`_collapse_journey_aliases` started keeping the resolved symbol
    rather than the slot tag — the two changes are one fix in two places.

    Where no lane names the row, the kind is genuinely unknown. Unknown is
    recorded as unknown and the row is admitted: guessing "authored" would
    inflate the kernel bucket with library picks, and guessing "env" would
    delete real kernels from dead runs, which is the loss this collector exists
    to recover. ``kind_source`` makes the residual countable instead of silent.

    Args:
        rows (list[dict[str, Any]]): Collapsed accepted-kernel descriptors.
        result (dict[str, Any]): The normalized ``result.json`` for the run.
        stack (Any): ``state["optimization_stack"]``, the second place a kind
            is declared. See :func:`_geak_kind_index`.

    Returns:
        list[dict[str, Any]]: The same rows, each with ``kind`` and
        ``kind_source``, and with known-env rows removed. ``kind_source`` names
        the artifact that supplied the kind (``result_json`` or ``stack``, each
        with an ``_undeclared`` form) or ``absent`` when neither did.
    """
    index = _geak_kind_index(result, stack)
    kept: list[dict[str, Any]] = []
    for row in rows:
        names = [str(row.get("name") or "").strip(), str(row.get("kernel_id") or "").strip()]
        names += [str(a).strip() for a in (row.get("aliases") or [])]
        kind: str | None = None
        source = "absent"
        for name in names:
            if name and name in index:
                kind, origin = index[name]
                source = origin if kind else f"{origin}_undeclared"
                break
        row["kind"] = kind
        row["kind_source"] = source
        # One admission test, shared with the ledger: exclude only what is
        # *known* to be an env selection.
        if kind == "env":
            continue
        kept.append(row)
    return kept


def _journey_row_symbol(row: dict[str, Any]) -> str:
    """Return the kernel symbol a journey row carries.

    ``kernel_id`` is a slug: GEAK strips the leading underscore when it builds
    one, so the row for ``_mxfp8_linear_kernel`` has
    ``kernel_id="mxfp8_linear_kernel"``. ``name`` keeps the symbol as written.
    Measured over ``/shared_nfs/hyperloom-claw``, joining on ``kernel_id``
    misses every underscore-prefixed kernel while joining on ``name`` matches
    the ``result.json`` lane exactly, so ``name`` is the id and ``kernel_id``
    is the fallback.

    Args:
        row (dict[str, Any]): One accepted-kernel descriptor.

    Returns:
        str: The symbol, or ``""`` when the row carries neither field.
    """
    return str(row.get("name") or row.get("kernel_id") or "").strip()


def _is_alias_twin_group(group: list[dict[str, Any]], is_cand_tag: Any) -> bool:
    """Return True only for a one-to-one slot-tag + symbol alias pair."""
    if len(group) != 2:
        return False
    measured = [r for r in group if r.get("gpu_pct") is not None]
    unmeasured = [r for r in group if r.get("gpu_pct") is None]
    if len(measured) != 1 or len(unmeasured) != 1:
        return False
    tag_id = _journey_row_symbol(measured[0])
    sym_id = _journey_row_symbol(unmeasured[0])
    return bool(tag_id) and bool(sym_id) and is_cand_tag(tag_id) and not is_cand_tag(sym_id)


def _collapse_journey_aliases(accepted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the twin rows GEAK writes for one accepted kernel.

    When the profiler resolves a dispatched candidate to a library symbol, the
    journey records the acceptance twice: once under the candidate id, carrying
    the measurement (``gpu_pct``), and once under the resolved symbol with
    ``gpu_pct: null``. Both rows repeat the same ``e2e_gain_pct``, so counting
    them separately doubles the kernel.

    Rows are grouped by gain (rounded — the twin is sometimes the rounded copy).
    A group holding both a measured and an unmeasured row is one kernel: the
    measured row survives and the other ids move to its ``aliases``. Groups
    that are all-measured or all-unmeasured are distinct kernels and are left
    alone, so a run that genuinely accepts two kernels of equal gain keeps both.

    Which row survives and which id it is named by are two separate questions,
    and answering them as one was a bug. The measurement only exists on the
    candidate row, so that row survives. The *name* has to be the resolved
    symbol, because that is the id
    :func:`~hyperloom.orchestrator.loop.coordinator_helpers._geak_accepted_kernel_specs`
    keeps for the same kernel — before this, ``collect_geak`` reported
    ``c0_triton`` while ``kernel_lifecycle.adopted`` reported
    ``dsa_sparse_attn_prefill_main_kernel``, one kernel under two names in two
    tables of the same report.

    The symbol is taken from the *unmeasured* twin, which is what the paragraph
    above defines it to be — not from "whichever id does not look like a slot
    tag". Both rules agree on ``c0_triton`` / ``dsa_sparse_attn_prefill_main_kernel``,
    and they disagree on ``decode_attention_grouped_mla`` /
    ``_fwd_grouped_kernel_stage1 (+_fwd_kernel_stage2)``, where neither id is a
    slot tag and only the second is what ``result.json`` names. Checked against
    all 10 twin groups in ``/shared_nfs/hyperloom-claw``: taking the unmeasured
    twin's symbol reproduces the ledger's name in every group that has one.

    Args:
        accepted (list[dict[str, Any]]): Accepted-kernel descriptors, in order.

    Returns:
        list[dict[str, Any]]: The descriptors with alias twins folded in.
    """
    is_cand_tag = _geak_cand_tag_test()

    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in accepted:
        gain = row.get("e2e_gain_pct")
        op_kind = str(row.get("op_kind") or "")
        if isinstance(gain, (int, float)):
            key: Any = (op_kind, round(float(gain), 3))
        else:
            key = (op_kind, "_", id(row))
        groups.setdefault(key, []).append(row)

    collapsed: list[dict[str, Any]] = []
    dropped: set[int] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        if not _is_alias_twin_group(group, is_cand_tag):
            continue
        measured = [r for r in group if r.get("gpu_pct") is not None]
        unmeasured = [r for r in group if r.get("gpu_pct") is None]
        primary = measured[0]

        # The unmeasured twin is the resolved symbol, by construction. When
        # several are unmeasured, prefer one that is not a slot tag; fall back
        # to the primary's own id so a group of slot tags still gets a name.
        candidates = [_journey_row_symbol(r) for r in unmeasured]
        candidates = [c for c in candidates if c]
        symbols = [c for c in candidates if not is_cand_tag(c)]
        chosen = (symbols or candidates or [_journey_row_symbol(primary)])[0]

        every_id: list[str] = []
        for r in group:
            for field in ("name", "kernel_id"):
                text = str(r.get(field) or "").strip()
                if text and text not in every_id:
                    every_id.append(text)
        if chosen:
            primary["kernel_id"] = chosen
            primary["name"] = chosen
        primary["aliases"] = sorted({i for i in every_id if i != chosen})
        dropped.update(id(r) for r in unmeasured)

    for row in accepted:
        if id(row) not in dropped:
            collapsed.append(row)
    return collapsed


def _geak_accepted_kernels_from_journey(
    result: dict[str, Any],
    warnings: list[str],
    stack: Any = None,
) -> list[dict[str, Any]]:
    """Derive the accepted (KEEP/integrated) kernels from ``kernel_journey.json``.

    Projects each kernel whose ``e2e`` sub-object was integrated (or decided
    ``KEEP``/``ADOPTED``) into a compact accepted-kernel descriptor. Used to
    back-fill an empty ``accepted_kernels`` in ``result.json``. Best-effort: a
    missing/partial file yields ``[]`` and never raises.

    Args:
        result (dict[str, Any]): The normalized ``result.json`` (carries
            ``kernel_journey_path`` / ``eval_dir`` used to locate the journey).
        warnings (list[str]): Shared warnings list (mutated in place).
        stack (Any): ``state["optimization_stack"]``, forwarded to
            :func:`_stamp_journey_kind` so the ``env`` exclusion can still run
            when ``result.json`` was rewritten empty by a later cycle.

    Returns:
        list[dict[str, Any]]: The accepted-kernel descriptors, or ``[]`` when the
            journey is absent, unreadable, or holds no integrated kernel.
    """
    journey_paths = _journey_paths(result)
    if not journey_paths:
        return []

    kernels: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for jp in journey_paths:
        journey = read_json(
            Path(jp),
            default=None,
            on_error=lambda exc: warnings.append(f"geak: kernel_journey read failed for backfill: {exc}"),
        )
        if not isinstance(journey, dict):
            continue
        for k in journey.get("kernels") or []:
            if not isinstance(k, dict):
                continue
            kid = str(k.get("kernel_id") or "")
            if kid and kid in seen_ids:
                continue
            if kid:
                seen_ids.add(kid)
            kernels.append(k)

    accepted: list[dict[str, Any]] = []
    for k in kernels:
        if not isinstance(k, dict):
            continue
        e2e = k.get("e2e")
        if not isinstance(e2e, dict):
            continue
        decision = str(e2e.get("decision") or "").strip().upper()
        integrated = e2e.get("integrated") is True
        if not (integrated or decision in ("KEEP", "ADOPTED")):
            continue
        kid = str(k.get("kernel_id") or "")
        if not kid:
            continue
        br = k.get("backend_result") if isinstance(k.get("backend_result"), dict) else {}
        verification = br.get("verification") if isinstance(br.get("verification"), dict) else {}
        dispatch = k.get("dispatch") if isinstance(k.get("dispatch"), dict) else {}
        backend = str(verification.get("best_backend") or (dispatch.get("backends") or [None])[0] or "")
        op_kind = str(k.get("op_kind") or dispatch.get("op_kind") or e2e.get("op_kind") or "")
        accepted.append(
            {
                "kernel_id": kid,
                "name": str(k.get("name") or kid),
                "op_kind": op_kind,
                "gpu_pct": _to_float(k.get("gpu_pct")),
                "micro_speedup": _to_float(k.get("micro_speedup") or verification.get("micro_speedup")),
                "e2e_gain_pct": _to_float(e2e.get("e2e_gain_pct")),
                "validated": e2e.get("validated") if isinstance(e2e.get("validated"), bool) else None,
                "decision": decision or "KEEP",
                "backend": backend,
                "target_file": e2e.get("target_file"),
                "extra_server_args": str(e2e.get("extra_server_args") or ""),
                "source": "kernel_journey_backfill",
            }
        )
    return _stamp_journey_kind(_collapse_journey_aliases(accepted), result, stack)


def _geak_accepted_kernels_from_integrate_results(
    exp_root: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Derive the accepted kernels from the per-candidate ``integrate_result.json``.

    ``kernel_journey.json`` is written once, at the end of the run, so a run
    that was killed never has one. Each candidate's ``integrate_result.json``
    is written as that candidate finishes, so it survives the kill. Both files
    record the same event; only the second one is on disk after a crash.

    Admission is the same test the journey backfill applies: ``gate ==
    "accepted"`` and a positive same-config ``e2e_delta_pct``. That delta is
    GEAK's own A/B, not an orchestrator rebench, so every row here is
    ``validated: False``.

    Args:
        exp_root (Path): The e2e experiment root holding ``overlay/``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        list[dict[str, Any]]: The accepted-kernel descriptors, or ``[]`` when
            no candidate qualifies. Never raises.
    """
    overlay_root = exp_root / "overlay"
    try:
        if not overlay_root.is_dir():
            return []
        cand_dirs = sorted(
            (d for d in overlay_root.iterdir() if d.is_dir()),
            key=lambda p: p.name,
        )
    except OSError as exc:
        warnings.append(f"geak: integrate_result scan failed: {exc}")
        return []

    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in cand_dirs:
        path = cand / "integrate_result.json"
        if not path.is_file():
            continue
        ir = read_json(
            path,
            default=None,
            on_error=lambda exc: warnings.append(f"geak: integrate_result read failed for {cand.name}: {exc}"),
        )
        if not isinstance(ir, dict):
            continue
        if str(ir.get("gate") or "").strip().lower() != "accepted":
            continue
        delta = _to_float(ir.get("e2e_delta_pct"))
        if delta is None or delta <= 0.0:
            continue
        # The same acceptance is filed under both spellings — the candidate
        # directory tag and the kernel's own short name. Keying on the short
        # name, when there is one, collapses that alias twin into one row.
        kid = str(ir.get("short_name") or "").strip() or cand.name
        if kid in seen:
            continue
        seen.add(kid)
        accepted.append(
            {
                "kernel_id": kid,
                "name": kid,
                "cand_tag": cand.name,
                "gpu_pct": _to_float(ir.get("pct_gpu_time")),
                "micro_speedup": _to_float(ir.get("isolated_speedup")),
                "e2e_gain_pct": delta,
                # GEAK's own same-config A/B, with no orchestrator rebench
                # behind it. Never present this as a validated e2e gain.
                "validated": False,
                "decision": "KEEP",
                "backend": "",
                "target_file": None,
                "extra_server_args": "",
                "output_parity": str(ir.get("output_parity") or ""),
                "engagement_hits": _to_int(ir.get("engagement_hits_cand")),
                "overlay": str(ir.get("accepted_overlay") or str(cand)),
                "source": "integrate_result_backfill",
            }
        )
    return accepted


def _geak_reconstruct_from_disk(
    session_dir: Path,
    warnings: list[str],
    stack: Any = None,
) -> dict[str, Any] | None:
    """Best-effort reconstruction of a GEAK run from on-disk survivors.

    Scans the runner's working tree under ``<session>/geak/`` to recover what
    actually ran when ``state.geak_result`` is empty/missing: the handoff, the
    e2e ``exp_root`` and the stages it reached (baseline / kernels / opbench /
    strategy), any flushed-but-unpromoted ``result.json`` status, and the
    per-kernel ``kernel_journey`` accepted kernels.

    Returns ``None`` when nothing usable is on disk (caller keeps the legacy
    ``missing`` section). Never raises — failures append to ``warnings``.

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place).
        stack (Any): ``state["optimization_stack"]``. The reconstruction runs
            because ``state.geak_result`` is empty, but the stack can still
            hold the ``geak_e2e`` entry an earlier cycle appended, so it is
            often the only surviving declaration of a recovered row's kind.

    Returns:
        dict[str, Any] | None: The recovered evidence, or ``None``.
    """
    pf = session_dir / "geak"
    try:
        if not pf.is_dir():
            return None
    except OSError:
        return None

    def _load_json(p: Path) -> dict[str, Any]:
        return read_json(
            p,
            default={},
            require_dict=True,
            on_error=lambda exc: warnings.append(f"geak: reconstruct read failed for {p.name}: {exc}"),
        )

    stages: list[str] = []
    recon: dict[str, Any] = {}

    # 1) handoff.json — proves HL built + handed the e2e contract off.
    handoff = _load_json(pf / "handoff.json")
    if handoff:
        stages.append("handoff")
        recon["handoff"] = {
            "model_path": handoff.get("model_path"),
            "framework": handoff.get("framework"),
            "gpu_type": handoff.get("gpu_type"),
            "tp": handoff.get("tp"),
            "workload": handoff.get("workload"),
            "accepted_flags": handoff.get("accepted_flags"),
            "raw_baseline_tput": _to_float(handoff.get("raw_baseline_tput")),
        }

    # 2) a flushed-but-unpromoted result.json (absent or non-ok status).
    flushed = _load_json(pf / "result.json")
    if flushed:
        stages.append("result_json")
        recon["flushed_result_status"] = flushed.get("status")

    # 3) the e2e exp_root (newest ``e2e_*`` dir) + the stages it reached.
    exp_root: Path | None = None
    try:
        e2e_dirs = sorted(
            (d for d in pf.iterdir() if d.is_dir() and d.name.startswith("e2e_")),
            key=lambda d: d.name,
        )
        if e2e_dirs:
            exp_root = e2e_dirs[-1]
    except OSError as exc:
        warnings.append(f"geak: reconstruct iterdir failed: {exc}")

    kernels_attempted: list[dict[str, Any]] = []
    if exp_root is not None:
        recon["exp_root"] = _rel(exp_root, session_dir)
        for name, label in (
            ("baseline", "baseline"),
            ("baseline_rerun", "baseline_rerun"),
            ("strategy.md", "strategy"),
            ("kernel_journey.json", "kernel_journey"),
        ):
            try:
                if (exp_root / name).exists():
                    stages.append(label)
            except OSError:
                continue
        kdir = exp_root / "kernels"
        try:
            if kdir.is_dir():
                stages.append("kernels")
                for d in sorted(kdir.iterdir(), key=lambda p: p.name):
                    if d.is_dir() and not d.name.startswith("_"):
                        kernels_attempted.append({"name": d.name})
                # ``_exp`` holds the per-team op-bench / recursive kernel work.
                if (kdir / "_exp").is_dir():
                    stages.append("opbench")
        except OSError as exc:
            warnings.append(f"geak: reconstruct kernels scan failed: {exc}")
    recon["kernels_attempted"] = kernels_attempted

    # 4) per-kernel accepted kernels from the journey (reuse the projection so
    #    the recovered section's shape matches the producer-populated one).
    #    The journey is written last, so a killed run never has one; fall back
    #    to the per-candidate ``integrate_result.json``, written as each
    #    candidate finishes. Absence of both stays an empty list — a dead run
    #    with no accepted candidate must not be given one.
    if exp_root is not None:
        kj = exp_root / "kernel_journey.json"
        source = ""
        try:
            if kj.is_file():
                # ``flushed`` is passed for its acceptance lanes, not its
                # status: a killed run often flushed a partial ``result.json``,
                # and that is the only thing on disk that declares a ``kind``.
                # When it is absent every recovered row is marked
                # ``kind_source: absent`` rather than assumed authored.
                recon["accepted_kernels"] = _geak_accepted_kernels_from_journey(
                    {
                        "kernel_journey_path": str(kj),
                        "accepted_kernels": flushed.get("accepted_kernels") or [],
                        "accepted_heads": flushed.get("accepted_heads") or [],
                    },
                    warnings,
                    stack,
                )
                source = "kernel_journey_backfill"
        except OSError as exc:
            warnings.append(f"geak: accepted kernels journey unreadable: {exc}")
        if not recon.get("accepted_kernels"):
            recovered = _geak_accepted_kernels_from_integrate_results(exp_root, warnings)
            if recovered:
                recon["accepted_kernels"] = _stamp_journey_kind(
                    recovered,
                    {
                        "accepted_kernels": flushed.get("accepted_kernels") or [],
                        "accepted_heads": flushed.get("accepted_heads") or [],
                    },
                    stack,
                )
                source = "integrate_result_backfill"
        if source:
            recon["accepted_kernels_source"] = source

    # 5) newest-artifact timestamp (how far the run got in wall-clock). Bounded
    #    to a handful of key paths to avoid a full rglob of the exp_root tree.
    candidates = [pf / "handoff.json", pf / "result.json"]
    if exp_root is not None:
        candidates += [
            exp_root,
            exp_root / "logs",
            exp_root / "kernels",
            exp_root / "strategy.md",
            exp_root / "kernel_journey.json",
        ]
    newest = 0.0
    for p in candidates:
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    if newest > 0:
        recon["last_artifact_ts"] = datetime.fromtimestamp(newest, tz=timezone.utc).isoformat()

    # 6) op-bench verdicts — each per-kernel ``opbench_result.json`` records
    #    whether the backend bake-off found a deployable winner
    #    (``winner_editable`` + ``isolated_speedup`` > 1). Bounded to the
    #    top-level per-task files (the deep ``_exp`` tree is skipped).
    opbench_results: list[dict[str, Any]] = []
    if exp_root is not None:
        try:
            task_dirs = (
                [d for d in (exp_root / "kernels").iterdir() if d.is_dir() and not d.name.startswith("_")]
                if (exp_root / "kernels").is_dir()
                else []
            )
            for task_dir in sorted(task_dirs, key=lambda p: p.name)[:12]:
                ob = _load_json(task_dir / "opbench_result.json")
                if ob:
                    opbench_results.append(
                        {
                            "task": ob.get("task") or task_dir.name,
                            "winner_backend": ob.get("winner_backend"),
                            "isolated_speedup": _to_float(ob.get("isolated_speedup")),
                            "winner_editable": bool(ob.get("winner_editable")),
                            "winner_kind": ob.get("winner_kind"),
                        }
                    )
        except OSError as exc:
            warnings.append(f"geak: reconstruct opbench scan failed: {exc}")
    if opbench_results:
        recon["opbench_results"] = opbench_results

    # 7) runner log tails — run_e2e stdout/stderr survivors under
    #    ``exp_root/logs/``, the recoverable proxy for how far / why the run
    #    got. Bounded to the newest handful, tail-only.
    log_tails: dict[str, str] = {}
    if exp_root is not None:
        logs_dir = exp_root / "logs"
        try:
            if logs_dir.is_dir():
                logs = sorted(
                    (p for p in logs_dir.iterdir() if p.is_file()),
                    key=lambda p: p.stat().st_mtime,
                )
                for p in logs[-4:]:
                    try:
                        log_tails[p.name] = p.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )[-1500:]
                    except OSError:
                        continue
        except OSError as exc:
            warnings.append(f"geak: reconstruct log-tail read failed: {exc}")
    if log_tails:
        recon["runner_log_tails"] = log_tails

    # 8) likely_cause — a conservative classification of WHY no result reached
    #    state, so a reader does not have to re-derive it from the raw survivors:
    #      * ``runner_reported_failure``  — a non-ok result.json was flushed.
    #      * ``ran_no_deployable_winner`` — op-bench ran but found no editable
    #        winner > 1.0x, so there was simply nothing to flush as a win.
    #      * ``killed_before_flush``      — stages were reached but neither a
    #        kernel_journey nor a result.json landed (the in-flight result died
    #        with the process — the incident pattern: SIGKILL / budget / hang).
    #      * ``indeterminate``            — not enough on-disk signal to classify.
    has_journey = "kernel_journey" in stages
    ran_opbench = "opbench" in stages or bool(opbench_results)
    any_deployable = any((r.get("isolated_speedup") or 0.0) > 1.0 and r.get("winner_editable") for r in opbench_results)
    if flushed and flushed.get("status") and flushed.get("status") != "ok":
        likely_cause = "runner_reported_failure"
    elif ran_opbench and not any_deployable and not has_journey:
        likely_cause = "ran_no_deployable_winner"
    elif stages and not has_journey and "result_json" not in stages:
        likely_cause = "killed_before_flush"
    else:
        likely_cause = "indeterminate"
    recon["likely_cause"] = likely_cause

    recon["stages_reached"] = stages
    # Nothing meaningful recovered → let the caller emit ``missing``.
    if not (handoff or flushed or exp_root):
        return None
    return recon


def collect_geak(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the GEAK/GEAK e2e KERNEL-phase section.

    Maps ``state.geak_result`` (the normalized ``result.json`` plus runner
    metadata) into the session-breakdown data contract: what the optimizer did
    (per-kernel / per-head), the accepted config, the validated regimes, the
    gain attribution, and — on a miss — the normalized failure reason.

    Returns an empty ``{}`` when GEAK was never engaged.

    Args:
        session_dir (Path): Absolute session root (used to relativize paths).
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: The GEAK section, or ``{}`` when not engaged.
    """
    optimizer = str(state.get("kernel_optimizer") or "").strip().lower()
    result = state.get("geak_result")
    # An empty ``geak_result`` dict must NOT count as engaged; engage only when
    # the optimizer flag selected geak or a non-empty result was recorded.
    has_result = isinstance(result, dict) and bool(result)
    engaged = optimizer == "geak" or has_result
    if not engaged:
        return {}
    if not has_result:
        # Engaged via the flag but no result recorded; reconstruct from the
        # on-disk ``geak/`` working tree before surfacing ``missing``.
        recon = _geak_reconstruct_from_disk(session_dir, warnings, state.get("optimization_stack"))
        if recon is None:
            return {
                "engaged": True,
                "status": "missing",
                "error_class": "no_result",
                "error": ("kernel_optimizer=geak but no geak_result recorded"),
                "accepted_kernels": [],
                "accepted_heads": [],
            }
        recovered_kernels = recon.get("accepted_kernels") or []
        return {
            "engaged": True,
            "status": "no_result_recovered_from_disk",
            "error_class": "no_result",
            "error": (
                "kernel_optimizer=geak but no geak_result was "
                "committed to state; reconstructed the run from on-disk "
                "geak/ artifacts. The runner handed off and produced "
                "intermediate output, but the normalized result.json was never "
                "folded into state — typically an external kill (SIGKILL / OOM "
                "/ budget) before the tick-boundary state.save, then a resume "
                "past KERNEL."
            ),
            "recovered_from_disk": True,
            "handoff": recon.get("handoff"),
            "exp_root": recon.get("exp_root"),
            "stages_reached": recon.get("stages_reached") or [],
            "kernels_attempted": recon.get("kernels_attempted") or [],
            "opbench_results": recon.get("opbench_results") or [],
            "runner_log_tails": recon.get("runner_log_tails") or {},
            "likely_cause": recon.get("likely_cause"),
            "flushed_result_status": recon.get("flushed_result_status"),
            "last_artifact_ts": recon.get("last_artifact_ts"),
            "accepted_kernels": recovered_kernels,
            # Which survivor the recovery actually read. A crashed run has no
            # journey, so this is usually ``integrate_result_backfill``.
            "accepted_kernels_source": (
                str(recon.get("accepted_kernels_source") or "") or None if recovered_kernels else None
            ),
            "accepted_kernels_kind_sources": _kind_source_counts(recovered_kernels),
            "accepted_heads": [],
            "kernels_optimized": len(recovered_kernels),
        }

    def _rel_if_under(p: Any) -> Any:
        """Relativize ``p`` against the session dir when it lives under it."""
        if not p:
            return p
        try:
            pp = Path(str(p))
            if pp.is_absolute() and is_path_within(pp, session_dir):
                return _rel(pp, session_dir)
        except (ValueError, OSError) as exc:
            # Relativizing is cosmetic: keep the absolute path on failure.
            warnings.append(f"geak: failed to relativize path {p!r}: {exc}")
        return p

    status = str(result.get("status") or "unknown")
    base = _to_float(result.get("baseline_throughput_tok_s"))
    final = _to_float(result.get("final_throughput_tok_s"))
    speedup = _to_float(result.get("throughput_speedup"))
    gain_pct: float | None = None
    if isinstance(base, (int, float)) and base > 0 and isinstance(final, (int, float)):
        gain_pct = (final - base) / base * 100.0
    elif isinstance(speedup, (int, float)) and speedup > 0:
        gain_pct = (speedup - 1.0) * 100.0

    accepted_kernels = result.get("accepted_kernels") or []
    accepted_heads = result.get("accepted_heads") or []
    if not isinstance(accepted_kernels, list):
        accepted_kernels = []
        warnings.append("geak: accepted_kernels was not a list")
    if not isinstance(accepted_heads, list):
        accepted_heads = []

    normalized_result = {
        **result,
        "accepted_kernels": accepted_kernels,
        "accepted_heads": accepted_heads,
    }

    # Normalize the direct result path through the same admission rules as the
    # orchestrator ledger: both lanes, env selections excluded, alias twins
    # collapsed. Journey backfill runs only when this yields nothing.
    accepted_kernels_source: str | None = None
    specs = _geak_result_kernel_specs(normalized_result)
    if specs:
        accepted_kernels = specs
        accepted_kernels_source = "result"
    else:
        legacy_strings = _legacy_string_accepted_kernels(normalized_result)
        if legacy_strings:
            accepted_kernels = legacy_strings
            accepted_kernels_source = "result"
        elif status in ("ok", "no_gain"):
            backfilled = _geak_accepted_kernels_from_journey(result, warnings, state.get("optimization_stack"))
            if backfilled:
                accepted_kernels = backfilled
                accepted_kernels_source = "kernel_journey_backfill"
            else:
                accepted_kernels = []
        else:
            accepted_kernels = []

    section: dict[str, Any] = {
        "engaged": True,
        "status": status,
        # Failure provenance (None on success).
        "error_class": result.get("error_class"),
        "error": result.get("error"),
        "returncode": result.get("returncode"),
        # Same-harness adjudication is terminal state, not pending work.  Keep
        # it with the GEAK result so clearing ``state.geak_pending`` does not
        # erase why a measured candidate was dropped.
        "revalidation_status": result.get("revalidation_status"),
        "revalidation_error_class": result.get("revalidation_error_class"),
        "revalidation_error": result.get("revalidation_error"),
        # Throughput / gain attribution (aggregate output tok/s).
        "baseline_throughput_tok_s": base,
        "final_throughput_tok_s": final,
        "throughput_speedup": speedup,
        "gain_pct": gain_pct,
        "metric_basis": result.get("metric_basis"),
        "bench_client": result.get("bench_client"),
        # Latency (median ms), field names aligned with the native sweep.
        "ttft_mean_ms": _to_float(result.get("ttft_ms")),
        "tpot_mean_ms": _to_float(result.get("tpot_ms")),
        "output_parity": result.get("output_parity"),
        # What the optimizer actually changed (per-kernel / head / config).
        "accepted_kernels": accepted_kernels,
        # Provenance of ``accepted_kernels``: ``result``,
        # ``kernel_journey_backfill``, or ``None``.
        "accepted_kernels_source": accepted_kernels_source,
        # How each admitted kernel's ``kind`` was resolved. Values sum to
        # ``kernels_optimized`` on every path.
        "accepted_kernels_kind_sources": _kind_source_counts(accepted_kernels),
        "accepted_heads": accepted_heads,
        "kernels_optimized": len(accepted_kernels),
        "accepted_config": dict(result.get("accepted_config") or {}),
        # Regimes the kernels were validated at (sweep points outside need reparity).
        "validated_regimes": list(result.get("validated_regimes") or []),
        # Reusable deliverables + human report (relativized when under the session).
        "eval_dir": _rel_if_under(result.get("eval_dir")),
        "report_path": _rel_if_under(result.get("report_path")),
        "final_launch_script": _rel_if_under(result.get("final_launch_script")),
        "bench_script": _rel_if_under(result.get("bench_script")),
        "final_patch": _rel_if_under(result.get("final_patch")),
        # Budget audit (present when the runner was budget-capped/skipped).
        "runner_timeout_s": _to_int(result.get("runner_timeout_s")),
        "kill_timeout_s": _to_int(result.get("kill_timeout_s")),
    }
    return section
