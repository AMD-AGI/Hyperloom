# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check a generated tuner's output before anything downstream reads it.

A generated tuner is the one producer whose output was never reviewed by a
person, so the shape of what it writes has to be checked rather than assumed.
The checks are deliberately about form, not about performance: whether the
result is any *good* is settled later by :mod:`.referee`, and a file that passes
here has earned nothing except the right to be measured.

Every failure names the row, because the point of running this before the
expensive step is to say what to fix.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mandate import REQUIRED_OUTPUT_COLUMNS, TunerMandate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContractViolation:
    """One reason the output cannot be used."""

    where: str
    problem: str

    def __str__(self) -> str:
        return f"{self.where}: {self.problem}"


def _shape_key(row: dict[str, str], key_schema: list[str]) -> tuple:
    return tuple(str(row.get(k, "")).strip() for k in key_schema)


def validate_output_csv(
    csv_path: Path | str,
    mandate: TunerMandate,
) -> list[ContractViolation]:
    """Return every way ``csv_path`` fails the mandate; empty means usable."""
    path = Path(csv_path)
    bad: list[ContractViolation] = []
    if not path.is_file():
        return [ContractViolation(str(path), "no such file")]

    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]
    except (OSError, csv.Error) as exc:
        return [ContractViolation(str(path), f"unreadable: {exc}")]

    expected = mandate.output_columns
    if header != expected:
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        bad.append(
            ContractViolation(
                "header",
                f"expected {expected}, got {header}"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else ""),
            )
        )
        # Without the agreed columns the per-row checks would report noise.
        if missing:
            return bad

    if not rows:
        bad.append(ContractViolation(str(path), "no rows"))
        return bad

    wanted = {_shape_key(s, mandate.key_schema) for s in mandate.demand_shapes}
    seen: set[tuple] = set()

    for i, row in enumerate(rows, start=2):  # row 1 is the header
        where = f"row {i}"
        key = _shape_key(row, mandate.key_schema)
        if key in seen:
            bad.append(ContractViolation(where, f"duplicate shape {key}"))
        seen.add(key)

        for col in REQUIRED_OUTPUT_COLUMNS[:2]:
            raw = str(row.get(col, "")).strip()
            try:
                value = float(raw)
            except ValueError:
                bad.append(ContractViolation(where, f"{col}={raw!r} is not a number"))
                continue
            if value <= 0:
                bad.append(ContractViolation(where, f"{col}={value} is not a positive time"))

        improved = str(row.get("improved", "")).strip().lower()
        if improved not in ("true", "false"):
            bad.append(ContractViolation(where, f"improved={improved!r} is not a boolean"))
        else:
            # A row that claims an improvement its own numbers contradict is the
            # cheapest possible tell that the script is not measuring what it
            # reports, and it costs nothing to catch here.
            try:
                d, t = float(row["default_us"]), float(row["tuned_us"])
            except (KeyError, ValueError):
                # Missing or unparseable timings are already recorded by the
                # column checks above; there is nothing to cross-check here.
                pass
            else:
                if d > 0 and t > 0 and (improved == "true") != (t < d):
                    bad.append(
                        ContractViolation(
                            where,
                            f"improved={improved} contradicts default_us={d} tuned_us={t}",
                        )
                    )

        if "," in str(row.get("config", "")):
            bad.append(ContractViolation(where, "config contains a comma"))

    if wanted:
        unmet = wanted - seen
        if unmet:
            bad.append(
                ContractViolation(
                    str(path),
                    f"{len(unmet)} demanded shape(s) have no row: {sorted(unmet)[:5]}",
                )
            )

    if bad:
        log.warning(
            "generated tuner output %s failed the contract: %s",
            path,
            "; ".join(str(v) for v in bad[:5]),
        )
    return bad


def load_candidates(
    candidates_json: Path | str,
    mandate: TunerMandate,
) -> dict[str, list[dict[str, Any]]]:
    """Read the ranked candidate lists, dropping anything malformed.

    Never raises: a candidate file that cannot be read leaves nothing to
    re-time, which the caller already has to handle.
    """
    path = Path(candidates_json)
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("cannot read candidates from %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        log.warning("candidates in %s are not an object keyed by shape", path)
        return {}

    out: dict[str, list[dict[str, Any]]] = {}
    for shape, cands in data.items():
        if isinstance(cands, dict):
            cands = [cands]
        if not isinstance(cands, list):
            continue
        kept = [c for c in cands if isinstance(c, dict)]
        if kept:
            out[str(shape)] = kept[: mandate.max_candidates_per_shape]
    return out
