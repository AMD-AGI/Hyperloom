"""The campaign's objective, when it is not its pack's: ``--opt-objective "P50 interactivity"``.

A pack names one metric, and the agent, the bench and the gate all work toward it. A host that gates on another axis
(an AgentX host judges its KEEPs on P50 interactivity, while the agentx pack optimizes total throughput) would otherwise
get changes it then refuses, and miss the ones it would keep. So the objective can be named the way a person names it,
and it replaces the pack's for the whole campaign:

- ``arbor run --opt-objective TEXT``, or ``ARBOR_OPT_OBJECTIVE`` (the flag sets it for the run's processes);
- else the config's ``opt_objective``, which a host writes.

``resolve`` reads the text: a known metric by its name (``P50 interactivity``, an AgentX host's end-to-end axis
``e2e_norm_intvty_p50``; ``total throughput``, ...) or its key as written, or any metric the bench writes, by its key
and a direction (``tpot_p50:min``). Applied in ``load_campaign``, so every load sees it:
the program's Objective section, the bench's metric key, the gate and the bars. A pack's second metric that is the
asked one changes places with the first, so the gate keeps both axes.

**The structured form**, in the config only (the flag stays text), carries a host's whole rule::

    "opt_objective": {"metric": "e2e_norm_intvty_p50", "bar_pct": 2.0,
                      "guards": [{"metric": "e2e_norm_intvty_p90", "band_pct": 5.0},
                                 {"metric": "total_token_throughput", "band_pct": 5.0},
                                 {"metric": "request_error_rate", "max_pct": 10.0}]}

- ``metric`` (and ``direction``, else the known one): as the text form.
- ``bar_pct``: the gain a WIN must clear, in place of the pack's ``threshold_rel``.
- ``guards``: a band guard holds against the kept tree's reading; a cap ``max_pct`` is absolute
  (arbor/measure/guards.py). A broken guard is a REVERT. The guards replace the pack's second axis.

At each start the effective objective is recorded in ``<session>/objective.json``, for a host to check before it trusts
the run (a pin that does not know this key optimizes the pack's metric, and says so there by its absence). A text Arbor
cannot read, and a resume under another objective than the session's, are refused at the start: the bars of one metric
mean nothing for another.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

ENV = "ARBOR_OPT_OBJECTIVE"
CONFIG_KEY = "opt_objective"
FILE = "objective.json"

#: The metric keys Arbor knows: their direction and unit.
#: - ``e2e_norm_intvty_p50`` / ``_p90``: an AgentX host's axis and its tail guard, as InferenceX's harness publishes
#:   them (``request_metrics.latency.e2e_norm_intvty``): 1 / the p50 (p90) over requests of E2EL in s / OSL. TTFT is
#:   in it, so a prefill gain moves it.
#: - ``p50_interactivity`` / ``p90_interactivity``: 1000 / the p50 (p90) TPOT in ms, decode alone.
KNOWN = {
    "e2e_norm_intvty_p50": ("max", "tok/s/user"),
    "e2e_norm_intvty_p90": ("max", "tok/s/user"),
    "p50_interactivity": ("max", "tok/s/user"),
    "p90_interactivity": ("max", "tok/s/user"),
    "total_throughput": ("max", "tok/s/GPU"),
    "total_token_throughput": ("max", "tok/s/GPU"),   # an AgentX host's name for it (MB --agentx-objective)
    "output_throughput": ("max", "tok/s/GPU"),
    "request_error_rate": ("min", "%"),
}
#: The forms ``opt_objective`` takes, for ``version --json``.
FORMS = ("text", "structured")
#: The second-axis keys the guards replace.
_SECONDARY = ("secondary_metric", "secondary_direction", "secondary_noise_band_pct", "secondary_threshold_rel",
              "secondary_regression_cap_pct", "secondary_exchange_rate")
#: Names a person writes, as ``resolve`` normalizes them. "P50 interactivity" is the host's axis, the end-to-end one;
#: the decode-only one is "P50 TPOT interactivity", or its key written as it is.
NAMES = {
    "p50_interactivity": "e2e_norm_intvty_p50",
    "p90_interactivity": "e2e_norm_intvty_p90",
    "interactivity_p50": "e2e_norm_intvty_p50",
    "interactivity_p90": "e2e_norm_intvty_p90",
    "median_interactivity": "e2e_norm_intvty_p50",
    "p50_tpot_interactivity": "p50_interactivity",
    "p90_tpot_interactivity": "p90_interactivity",
    "throughput": "total_throughput",
    "tput": "total_throughput",
    "total_tput": "total_throughput",
    "tput_per_gpu": "total_throughput",
    "total_throughput_per_gpu": "total_throughput",
    "output_tput": "output_throughput",
}
_DIRECTED = re.compile(r"^(.*?)\s*(?::|\(|\s)\s*(max|min)\)?\s*$", re.I)
_KEY = re.compile(r"[a-z][a-z0-9_]*")


def asked(config: dict, environ=None) -> str | dict:
    """The objective asked for: the flag's (``ENV``), else the config's, text or structured; "" for the pack's own."""
    environ = os.environ if environ is None else environ
    if str(environ.get(ENV) or "").strip():
        return str(environ[ENV]).strip()
    given = (config or {}).get(CONFIG_KEY)
    return given if isinstance(given, dict) and given else str(given or "").strip()


def _number(spec: dict, key: str, what: str) -> float | None:
    value = spec.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"opt_objective: {what} {key} {value!r} is not a number of 0 or more")
    return float(value)


def structured(spec: dict) -> dict:
    """The structured form, checked and resolved: ``{metric, direction, unit, bar_pct, guards}``; ValueError saying
    what is wrong."""
    unknown = set(spec) - {"metric", "direction", "bar_pct", "guards"}
    if unknown:
        raise ValueError(f"opt_objective: unknown key(s) {sorted(unknown)}; it takes metric, direction, bar_pct, "
                         f"guards")
    if not isinstance(spec.get("metric"), str) or not spec["metric"].strip():
        raise ValueError("opt_objective: metric is missing")
    text = spec["metric"] + (f":{spec['direction']}" if spec.get("direction") else "")
    metric, direction, unit = resolve(text)
    if not isinstance(spec.get("guards", []), list):
        raise ValueError("opt_objective: guards is not a list")
    guards = []
    for g in spec.get("guards") or []:
        if not isinstance(g, dict) or not isinstance(g.get("metric"), str):
            raise ValueError(f"opt_objective: a guard {g!r} has no metric")
        if ("band_pct" in g) == ("max_pct" in g):
            raise ValueError(f"opt_objective: the guard on {g['metric']} takes band_pct or max_pct, one of them")
        key, gdir, _ = resolve(g["metric"] + (f":{g['direction']}" if g.get("direction") else ""))
        if "max_pct" in g:
            guards.append({"metric": key, "max_pct": _number(g, "max_pct", f"the guard on {key}")})
        else:
            guards.append({"metric": key, "direction": gdir, "band_pct": _number(g, "band_pct", f"the guard on {key}")})
    return {"metric": metric, "direction": direction, "unit": unit, "bar_pct": _number(spec, "bar_pct", "the"),
            "guards": guards}


def resolve(text: str) -> tuple[str, str, str]:
    """``(metric key, direction, unit)`` for ``text``; ValueError saying what it takes."""
    raw, direction = text.strip(), None
    m = _DIRECTED.match(raw)
    if m and m.group(1).strip():
        raw, direction = m.group(1).strip(), m.group(2).lower()
    # A key written as it is means that key; anything else is a name.
    key = raw if _KEY.fullmatch(raw) and raw in KNOWN else None
    if key is None:
        name = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
        key = NAMES.get(name, name)
    if key in KNOWN:
        return key, direction or KNOWN[key][0], KNOWN[key][1]
    if direction and _KEY.fullmatch(key):
        return key, direction, ""
    names = ", ".join(f'"{n}"' for n in ("P50 interactivity", "P90 interactivity", "P50 TPOT interactivity",
                                          "total throughput", "output throughput"))
    raise ValueError(f"opt_objective {text!r} is not a metric Arbor knows: name one of {names}, write a key "
                     f"({', '.join(KNOWN)}), or a key the bench writes with its direction (\"tpot_p50:min\")")


def apply(objective: dict, asked_for: str | dict) -> dict:
    """The pack's ``objective`` with the asked metric in place of its own; ``opt_objective`` records what was asked
    and what the pack had. Text: the pack's second metric, when it is the asked one, takes the pack's first.
    Structured: its bar replaces the pack's ``threshold_rel``, and its guards the pack's second axis."""
    out = copy.deepcopy(objective)
    was, was_dir, was_unit = out.get("metric"), out.get("direction"), out.get("unit", "")
    meas = out.setdefault("measurement", {})
    if isinstance(asked_for, dict):
        spec = structured(asked_for)
        metric, direction, unit = spec["metric"], spec["direction"], spec["unit"]
        if spec["bar_pct"] is not None:
            meas["threshold_rel"] = round(1 + spec["bar_pct"] / 100, 6)
        if spec["guards"]:
            for key in _SECONDARY:
                meas.pop(key, None)
            meas["guards"] = spec["guards"]
    else:
        metric, direction, unit = resolve(asked_for)
        if metric != was and meas.get("secondary_metric") == metric:
            meas["secondary_metric"] = was
            meas["secondary_direction"] = was_dir
    out.update(metric=metric, direction=direction, unit=unit or (was_unit if metric == was else ""))
    out["opt_objective"] = {"asked": asked_for, "pack_metric": was, "pack_direction": was_dir}
    return out


def effective(campaign) -> dict:
    """What this campaign optimizes, for the record."""
    obj = campaign.pack.objective
    over = obj.get("opt_objective") or {}
    meas = obj.get("measurement") or {}
    return {"metric": obj.get("metric"), "direction": obj.get("direction"), "unit": obj.get("unit", ""),
            "secondary_metric": meas.get("secondary_metric"),
            "threshold_rel": meas.get("threshold_rel"), "guards": meas.get("guards") or [],
            "source": CONFIG_KEY if over else "pack", "asked": over.get("asked", ""),
            "pack_metric": over.get("pack_metric", obj.get("metric"))}


def load(session_dir) -> dict | None:
    try:
        doc = json.loads((Path(session_dir) / FILE).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def problem(campaign, session_dir) -> str | None:
    """Why this start must be refused: an objective Arbor could not read, or a session that started on another."""
    if getattr(campaign, "objective_error", ""):
        return campaign.objective_error
    old = load(session_dir) if session_dir else None
    now = effective(campaign)
    if old and (old.get("metric"), old.get("direction")) != (now["metric"], now["direction"]):
        asked_then = old.get("asked")
        how = (f"the config's opt_objective {json.dumps(asked_then)}" if isinstance(asked_then, dict)
               else f'--opt-objective "{asked_then}"' if asked_then else "no --opt-objective")
        return (f"this session optimizes {old.get('metric')} ({old.get('direction')}), not {now['metric']} "
                f"({now['direction']}): its bars mean nothing for another metric. Resume it with {how}, or start a "
                f"new session")
    return None


def record(campaign, session_dir) -> dict:
    """Write ``<session>/objective.json``: what this campaign optimizes, and where that came from."""
    doc = effective(campaign)
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    part = Path(session_dir) / f".{FILE}.tmp"
    part.write_text(json.dumps(doc, indent=1) + "\n")
    part.replace(Path(session_dir) / FILE)
    return doc
