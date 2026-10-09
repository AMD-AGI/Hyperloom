# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read fleet session evidence from Pulse and project it into miner rows.

Pulse (``v1/session-breakdowns`` under the Pulse API base, ``$PULSE_URL``) is the fleet-wide index of
``session_breakdown.json``. Unlike the Recipe KB it carries the roofline
ceiling and both throughput arms per session, so the fraction of a run's
roofline gap that was actually closed -- the capture ratio -- is computable
here and is not computable from a replay record.

Two things Pulse does not carry, and one trap:

* No accepted server args, so a parallelism layout cannot be read out of a
  row. Rows are marked ``layout_unknown`` rather than defaulted, because
  labelling an unknown layout ``framework-default`` would invent evidence.
* ``roofline_baseline_geomean_within_pct``, ``roofline_optimized_*`` and
  ``roofline_optimized_trend`` are rolling fleet aggregates attached to every
  row, not per-session values. Per-session capture must come from the nested
  ``roofline`` object plus the row's two arms.
* Throughput is already per-GPU (``opt_tok_per_s_per_gpu``), whereas the KB
  path stores a total and divides by tp. The projector multiplies back up so
  one downstream division cannot silently halve a per-GPU figure.
"""

from __future__ import annotations

import json
import math
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any

LAYOUT_UNKNOWN = "layout_unknown"
_PAGE = 200


class PulseError(RuntimeError):
    """A Pulse request failed."""


class PulseClient:
    """Blocking Pulse reader over stdlib urllib.

    Takes an explicit CA bundle: the analytics host is signed by an internal
    CA, and relying on an ambient ``SSL_CERT_FILE`` makes the failure look
    like an auth problem on a host that happens to lack the root.
    """

    def __init__(self, base_url: str, token: str, *, ca_bundle: str | None = None, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._ctx = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else None
        #: Why the last :meth:`session_breakdowns` walk ended early or skipped rows; empty when it read cleanly.
        self.walk_notes: list[str] = []

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        query = {k: v for k, v in (params or {}).items() if v is not None}
        if query:
            url += "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {self._token}"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout, context=self._ctx) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:200]
            raise PulseError(f"GET {path} -> HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise PulseError(f"GET {path} transport error: {exc.reason!r}") from exc

    def summary(self, **filters: Any) -> dict[str, Any]:
        return self.get("/v1/session-breakdowns/summary", filters)

    def session_breakdowns(self, *, max_rows: int = 1000, **filters: Any) -> Iterator[dict[str, Any]]:
        """Page ``/v1/session-breakdowns`` with ``limit``/``offset``.

        ``page_size`` is silently ignored by the service, so paging must use
        ``limit``. An empty or short page ends the walk, and so does a page that adds no new row: one whose sessions
        were all read already (a server that ignores ``offset`` returns the same page forever) or that holds no row
        object at all. A row whose ``session_id`` was already yielded is skipped, so no session is counted twice.
        Every request either yields a new row or ends the walk, so it cannot loop. :attr:`walk_notes` says why a walk
        stopped early or what it skipped.
        """
        self.walk_notes = []
        offset = 0
        seen = 0
        session_ids: set[str] = set()
        repeated = malformed = 0
        while seen < max_rows:
            requested = min(_PAGE, max_rows - seen)
            payload = self.get("/v1/session-breakdowns", {**filters, "limit": requested, "offset": offset})
            rows = payload.get("results") if isinstance(payload, Mapping) else None
            if not isinstance(rows, list) or not rows:
                break
            added = 0
            for row in rows:
                if not isinstance(row, Mapping):
                    malformed += 1
                    continue
                session_id = str(row.get("session_id") or "")
                if session_id and session_id in session_ids:
                    repeated += 1
                    continue
                if session_id:
                    session_ids.add(session_id)
                yield dict(row)
                seen += 1
                added += 1
                if seen >= max_rows:
                    break
            if not added:
                self.walk_notes.append(
                    f"pulse: the page at offset {offset} added no new row, so the walk stopped at {seen} rows; "
                    "the service may be ignoring offset"
                )
                break
            offset += len(rows)
            if len(rows) < requested:
                break
        if repeated:
            self.walk_notes.append(f"pulse: skipped {repeated} row(s) repeating a session already read")
        if malformed:
            self.walk_notes.append(f"pulse: skipped {malformed} row(s) that were not objects")


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def roofline_ceiling(row: Mapping[str, Any]) -> tuple[float | None, str]:
    """Per-GPU ceiling for a row, chosen by the session's own bound kind.

    Hyperloom computes the snapshot's ``roofline_*_ceiling_tok_per_sec`` over the whole server -- HBM bandwidth and
    peak FLOPs times ``tp`` (``roofline_snapshot.py`` passes ``num_gpus=runtime.tp``) -- while Pulse's throughput arms
    are per GPU, so the ceiling is divided by the row's ``tp`` before the two are compared. A row with no ``tp`` is
    taken as one GPU.
    """
    roofline = row.get("roofline")
    if not isinstance(roofline, Mapping):
        return None, "no roofline snapshot"
    bound = str(roofline.get("roofline_bound_kind") or "").lower()
    gpus = _positive_int(row.get("tp")) or 1
    memory = _finite(roofline.get("roofline_mem_ceiling_tok_per_sec"))
    compute = _finite(roofline.get("roofline_cmp_ceiling_tok_per_sec"))
    memory = memory / gpus if memory else memory
    compute = compute / gpus if compute else compute
    if bound == "memory" and memory:
        return memory, "memory"
    if bound == "compute" and compute:
        return compute, "compute"
    # Unlabelled bound: the binding ceiling is the lower of the two.
    available = [c for c in (memory, compute) if c]
    return (min(available), "inferred min") if available else (None, "no ceiling in snapshot")


def capture_pct(row: Mapping[str, Any]) -> float | None:
    """Fraction of the roofline gap this session actually closed, in percent.

    ``None`` when unmeasured -- an unmeasured capture is not a zero capture,
    which is the bug that dragged the old forecast toward zero.
    """
    ceiling, _ = roofline_ceiling(row)
    baseline = _finite(row.get("baseline_tok_per_s_per_gpu"))
    optimized = _finite(row.get("opt_tok_per_s_per_gpu"))
    if ceiling is None or baseline is None or optimized is None:
        return None
    gap = ceiling - baseline
    if gap <= 0:
        return None
    return ((optimized - baseline) / gap) * 100.0


def project_pulse_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one Pulse row into the shape the miner's aggregates expect."""
    tp = _positive_int(row.get("tp"))
    per_gpu = _finite(row.get("opt_tok_per_s_per_gpu"))
    ceiling, ceiling_kind = roofline_ceiling(row)
    projected: dict[str, Any] = {
        "canonical_id": ":".join(
            str(row.get(key) or "?") for key in ("model_name", "gpu_type", "framework", "framework_version", "prec")
        ),
        "session_id": str(row.get("session_id") or ""),
        "model": str(row.get("model_name") or ""),
        "hardware": str(row.get("gpu_type") or ""),
        "framework_name": str(row.get("framework") or ""),
        "framework_version": str(row.get("framework_version") or ""),
        "precision": str(row.get("prec") or ""),
        "validated_e2e_gain": _finite(row.get("gain")),
        # Per-GPU already; scale back to a total so the shared per-GPU
        # division downstream lands on the same number Pulse reported.
        "optimized_throughput": (per_gpu * tp) if (per_gpu is not None and tp) else per_gpu,
        "tput_per_gpu": per_gpu,
        "baseline_tput_per_gpu": _finite(row.get("baseline_tok_per_s_per_gpu")),
        "ceiling_tput_per_gpu": ceiling,
        "ceiling_kind": ceiling_kind,
        "capture_pct": capture_pct(row),
        "token_spend": _finite(row.get("token_spend")),
        "elapsed_minutes": _finite(row.get("elapsed_minutes")),
        "usd": _finite(row.get("money")),
        # Pulse carries no accepted server args, so the layout is unknown
        # rather than default.
        "parallelism": {},
        "parallelism_label": LAYOUT_UNKNOWN,
        "status": str(row.get("status") or ""),
        "cluster": str(row.get("cluster_name") or ""),
    }
    for key in ("tp", "conc", "isl", "osl"):
        projected[key] = _positive_int(row.get(key))
    return projected
