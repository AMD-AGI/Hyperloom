"""Knowledge ledger — typed, provenance-carrying environment facts.

``knowledge/*.md`` is human-owned and read-only to the agent. The ledger is the
agent-owned counterpart: append-only JSONL the agent writes when an iteration
establishes something durable about the environment.

Four record types:

``FACT``
    Measured and reproducible. Carries an instrument and a config fingerprint,
    so it can go stale when the environment moves out from under it.
``ASSUMPTION``
    Believed but unverified. Must carry a ``falsifier`` — the cheapest
    experiment that would prove it wrong.
``UNKNOWN``
    A named gap that blocks a decision. Also carries a falsifier.
``DEAD_END``
    Tried, did not work. Carries ``revisit_if`` so a later iteration knows the
    condition under which it is worth re-opening.

A record is superseded by a later record with the same ``id``. Nothing is ever
deleted — the file is the audit trail; ``current()`` is the view.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RECORD_TYPES = ("FACT", "ASSUMPTION", "UNKNOWN", "DEAD_END")

# An open ASSUMPTION/UNKNOWN is worth probing; a FACT/DEAD_END is settled.
PROBEABLE_TYPES = ("ASSUMPTION", "UNKNOWN")

LEDGER_FILENAME = "ledger.jsonl"

# Used when a record declares no cost estimate. A server run is ~30 min and an
# eval run ~2 min, so this sits between them: an unestimated probe is assumed
# non-trivial but not a server booking.
DEFAULT_COST_MIN = 15.0


@dataclass
class Finding:
    """One ledger record. ``id`` is the identity — later records supersede."""

    id: str
    type: str
    claim: str
    evidence: str = ""
    instrument: str = ""
    falsifier: str = ""
    revisit_if: str = ""
    status: str = "open"              # open | closed
    gates: list[str] = field(default_factory=list)
    config_keys: list[str] = field(default_factory=list)
    config_fingerprint: str = ""
    cost_est_min: float | None = None
    amdahl_share: float | None = None
    iteration: int = 0
    ts: float = field(default_factory=time.time)
    # The image and operating point it was measured at (arbor/measure/scope.py),
    # stamped by the harness on DEAD_END and FACT records.
    scope: dict | None = None

    def __post_init__(self) -> None:
        self.type = (self.type or "").upper()
        self.status = (self.status or "open").lower()

    @property
    def valid(self) -> bool:
        return bool(self.id) and self.type in RECORD_TYPES and bool(self.claim)

    def is_stale(self, current_fingerprints: dict[str, str]) -> bool:
        """True when the config this was measured under has since changed.

        A record with no ``config_keys`` makes no environmental claim and never
        goes stale — that is the right default for DEAD_END and for facts about
        upstream code rather than about this machine.
        """
        if not self.config_keys or not self.config_fingerprint:
            return False
        expected = current_fingerprints.get(self.id)
        if expected is None:
            return False
        return expected != self.config_fingerprint


def fingerprint(config: dict[str, Any], keys: list[str]) -> str:
    """Stable short hash of the subset of ``config`` named by ``keys``.

    Missing keys are recorded as null rather than skipped, so a key appearing or
    disappearing changes the fingerprint — that is a real environment change.
    """
    if not keys:
        return ""
    subset = {k: config.get(k) for k in sorted(keys)}
    blob = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def ledger_path(session_root: str | Path) -> Path:
    return Path(session_root) / LEDGER_FILENAME


def load(path: str | Path) -> list[Finding]:
    """Read every record, oldest first. Malformed lines are skipped, not raised.

    The ledger is written by an agent mid-campaign; one bad line must not be
    able to take down the loop that is trying to read it.
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[Finding] = []
    known = set(Finding.__dataclass_fields__)
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(raw, dict):
            continue
        try:
            f = Finding(**{k: v for k, v in raw.items() if k in known})
        except TypeError:
            continue
        if f.valid:
            out.append(f)
    return out


def append(path: str | Path, finding: Finding) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(asdict(finding), default=str) + "\n")


def current(findings: list[Finding]) -> list[Finding]:
    """Collapse to one record per id — the last one written wins."""
    by_id: dict[str, Finding] = {}
    for f in findings:
        by_id[f.id] = f
    return list(by_id.values())


def stale_fingerprints(findings: list[Finding], config: dict[str, Any]) -> dict[str, str]:
    """Recompute each finding's fingerprint against the live config."""
    return {f.id: fingerprint(config, f.config_keys) for f in findings if f.config_keys}


def probe_value(f: Finding, gate_counts: dict[str, int]) -> float:
    """Rank open uncertainties: (downstream work unblocked x share) / cost.

    ``gate_counts`` maps a finding id to the number of other findings that list
    it in their ``gates``. The Amdahl share defaults to a neutral 1.0 so that a
    finding nobody has profiled yet is not silently ranked to the bottom — an
    unprofiled component is exactly the kind of thing worth a cheap probe.
    """
    if f.type not in PROBEABLE_TYPES or f.status != "open":
        return 0.0
    blocked = 1 + gate_counts.get(f.id, 0)
    share = f.amdahl_share if f.amdahl_share is not None else 1.0
    cost = f.cost_est_min if f.cost_est_min else DEFAULT_COST_MIN
    return (blocked * share) / max(cost, 1.0)


def gate_counts(findings: list[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in findings:
        for g in f.gates:
            counts[g] = counts.get(g, 0) + 1
    return counts


def open_probes(findings: list[Finding]) -> list[Finding]:
    """Open ASSUMPTION/UNKNOWN records, most valuable probe first."""
    counts = gate_counts(findings)
    probeable = [f for f in findings if probe_value(f, counts) > 0]
    return sorted(probeable, key=lambda f: probe_value(f, counts), reverse=True)


def image_changed(f: Finding, scope: dict | None) -> bool:
    """True when ``f`` was recorded on an image other than the current one.

    Both images must be known: a record or an iteration without one says
    nothing about whether it moved.
    """
    then = f.scope.get("image") if isinstance(f.scope, dict) else None
    now = (scope or {}).get("image")
    return bool(then and now and then != now)


def regime_mark(f: Finding, op_knobs) -> str:
    """`[REGIME: C32 only]` on a dead end that names an operating-point knob.

    Most big settings are policies that flip sign with concurrency or length
    (serving-playbook amd-knobs.md §0 rule 7): a kill at one point is a row of
    a table, not a verdict on the knob.
    """
    from arbor.knobs import named_in
    keys = f.config_keys if isinstance(f.config_keys, list) else []
    if not named_in(f"{f.claim} {' '.join(map(str, keys))}", op_knobs):
        return ""
    conc = f.scope.get("concurrency") if isinstance(f.scope, dict) else None
    return f" **[REGIME: C{conc} only]**" if conc is not None else " **[REGIME: one operating point only]**"


def render(findings: list[Finding], config: dict[str, Any] | None = None,
           max_records: int = 40, scope: dict | None = None) -> str:
    """Markdown view of the ledger for the agent's program.

    Ordered so the agent reads what to do next before what is already settled:
    open probes (ranked), then facts, then dead ends. Stale facts are shown but
    flagged — a stale number is still a lead, it is just not evidence.

    ``scope`` is the current iteration's (arbor/measure/scope.py): a dead end
    recorded on another image is flagged for a re-test, since a kill on one
    image is stale on the next. A dead end that names a knob whose best value
    depends on the operating point (the serving playbook's knobs.json) is
    flagged as holding only where it was measured.
    """
    findings = current(findings)
    if not findings:
        return ""

    fps = stale_fingerprints(findings, config or {})
    counts = gate_counts(findings)

    probes = open_probes(findings)
    facts = [f for f in findings if f.type == "FACT"]
    dead = [f for f in findings if f.type == "DEAD_END"]

    facts.sort(key=lambda f: (f.is_stale(fps), -f.iteration))
    dead.sort(key=lambda f: -f.iteration)

    lines = [f"## Knowledge ledger ({len(findings)} records)\n"]

    if probes:
        lines.append("### Open uncertainties — ranked by (blocked x share) / cost\n")
        lines.append("Resolving one of these is a valid PROBE iteration.\n")
        for f in probes[:max_records]:
            v = probe_value(f, counts)
            cost = f.cost_est_min or DEFAULT_COST_MIN
            bits = [f"**{f.id}** [{f.type}] {f.claim}"]
            if f.falsifier:
                bits.append(f"falsifier: {f.falsifier}")
            bits.append(f"~{cost:.0f} min")
            if counts.get(f.id):
                bits.append(f"blocks {counts[f.id]}")
            lines.append(f"- (v={v:.3f}) " + " — ".join(bits))
        lines.append("")

    if facts:
        lines.append("### Established facts\n")
        for f in facts[:max_records]:
            mark = " **[STALE — config moved, re-measure before trusting]**" if f.is_stale(fps) else ""
            src = f" _(iter {f.iteration}"
            src += f", {f.instrument}" if f.instrument else ""
            src += f", {f.evidence}" if f.evidence else ""
            src += ")_"
            lines.append(f"- **{f.id}**: {f.claim}{mark}{src}")
        lines.append("")

    if dead:
        from arbor import knobs
        op_knobs = knobs.op_dependent()
        lines.append("### Dead ends — do not retry without the stated condition\n")
        for f in dead[:max_records]:
            cond = f" — revisit if: {f.revisit_if}" if f.revisit_if else ""
            mark = " **[STALE: image changed, re-test]**" if image_changed(f, scope) else ""
            mark += regime_mark(f, op_knobs)
            lines.append(f"- **{f.id}**: {f.claim}{mark}{cond}")
        lines.append("")

    return "\n".join(lines).strip()


def parse_agent_blocks(output: str) -> list[Finding]:
    """Extract ``LEDGER: {json}`` lines from agent output.

    One record per line keeps parsing trivial and partial output usable — a
    truncated run still yields every complete record written before the cut.
    """
    out: list[Finding] = []
    known = set(Finding.__dataclass_fields__)
    for line in output.splitlines():
        line = line.strip()
        if not line.upper().startswith("LEDGER:"):
            continue
        payload = line.split(":", 1)[1].strip()
        try:
            raw = json.loads(payload)
        except (ValueError, TypeError):
            continue
        if not isinstance(raw, dict):
            continue
        try:
            f = Finding(**{k: v for k, v in raw.items() if k in known})
        except TypeError:
            continue
        if f.valid:
            out.append(f)
    return out
