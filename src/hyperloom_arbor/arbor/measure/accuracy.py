"""Accuracy-changing knobs are gated on accuracy.

A throughput gate cannot see accuracy: an FP8 MoE combine was merged on its
throughput and then failed GSM8K far below its golden score, and a decode-only
MTP setting cut GSM8K from 0.96 to 0.88 with no error and no slowdown
(serving-playbook numerics-knobs.md §0). The packs' per-iteration correctness
check is a floor for a broken model (Model M: 400 questions, pass at 0.92), not
a comparison with the baseline.

So, when the pack turns ``policy.accuracy_gate`` on and the iteration's change
moves a knob that changes numerics (knobs.json) or the agent writes
``NUMERICS: yes`` for a code change, the harness runs the pack's correctness
script again on the full GSM8K set and the gate compares its score with the
baseline's full-set score (the config's ``baseline_record``,
``correctness_full.score``, as scripts/model-m-campaign/baseline_model_m.py stores
it) minus ``policy.accuracy_tolerance``. Without a golden score it compares
with the pack's ``GSM8K_MIN`` and says so.

Under a host there is no baseline record, and the host's own eval of the root
reads on another scale (MB: 0.939 against this pack's 0.870 on Qwen3-8B, Cluster A,
2026-11-01), so it cannot stand in. So in component mode, before iteration 1,
the driver serves the root as initialized and runs the full set once
(``measure_root``), into ``<session>/baseline_accuracy.json``; the gate then
holds a numerics change to that score less the tolerance, on the same eval.
When the root could not be measured, ``GSM8K_MIN`` stays the floor.

In component mode every KEEP is held to the full set, not only a numerics one
(operator, 2026-11-03): a host imports what Arbor keeps, and a change the agent
did not call numerics can still break the served model (Hyperloom MB's T24: a
kept decode GEMM, its switch turned on, read 0.000 against 0.945,
boards/hyperloom-multibranch#155 c5075). The gate decides first without the
eval, and only an arm it would keep pays for the set. Turning on a switch that
an earlier KEEP landed off (arbor/measure/switches.py) calls for it in every
mode.

All of it applies to a GSM8K workload only (``gsm8k_workload``): the pack's
accuracy gate is on, which the packs whose correctness script runs GSM8K set
(the config's ``policy`` may turn it off), and the launch serves the model's
own text. An AgentX launch simulates speculative acceptance
(``SGLANG_SIMULATE_ACC_*``, vLLM's ``rejection_sample_method: synthetic``), so
the text it returns is not the model's and GSM8K would score the simulation;
its accuracy is the host's request error rate and the pack's acceptance check.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path

from arbor import knobs

FULL_GSM8K = 1319
# The pack's eval size and floor: passed to every correctness run from the
# campaign config (arbor/execute/bench.py).
EVAL_KEYS = ("GSM8K_LIMIT", "GSM8K_MIN")
# The root's full set, as this session measured it (measure_root), and the sources golden() names.
ROOT_FILE = "baseline_accuracy.json"
BASELINE_SOURCE = "the baseline's full GSM8K (baseline_record)"
ROOT_SOURCE = "the root's full GSM8K, measured by this session"
FLOOR_SOURCE = "GSM8K_MIN: no golden score on record"
# What measuring the root needs before the deadline: one server boot and the full set (an 8B
# model at TP1 on one MI350X takes a few minutes for each). A run with less left keeps the floor.
ROOT_NEED_S = 30 * 60
# The full set's result beside its output (measure), which a KEEP's snapshot carries.
ACCURACY_FILE = "accuracy.json"
# What the gate names when component mode holds a KEEP to the full set, whatever it changed.
EVERY_KEEP = "a KEEP in component mode"
# A launch whose speculative acceptance is simulated: SGLang's variables, vLLM's speculative config.
SIMULATED_ACCEPTANCE_ENV = "SGLANG_SIMULATE_ACC_"
SYNTHETIC_REJECTION = re.compile(r"""rejection_sample_method["']?\s*[:=]\s*["']?synthetic""")


def declares_numerics(output: str) -> bool:
    """A ``NUMERICS: yes`` line, decorated or not."""
    return any(re.match(r"[*_`>#\-\s]*NUMERICS\s*[*_]*\s*:[\s*_`]*yes\b", line, re.IGNORECASE)
               for line in (output or "").splitlines())


def trigger(changes: dict, output: str, switched_on: list[str] | tuple = ()) -> list[str]:
    """What calls for the gate: the numerics knobs the change moved, ``NUMERICS``, and the
    switches an earlier KEEP landed off that the change turns on (``switches.turned_on``)."""
    numerics = knobs.numerics()
    by = sorted(name for name in changes if name in numerics)
    by += [f"switch {name} on" for name in sorted(switched_on) if name not in by]
    return by + (["NUMERICS"] if declares_numerics(output) else [])


def simulated_acceptance(env: dict | None, flags: str | None) -> str:
    """The setting that makes the launch's speculative acceptance synthetic; "" when none does."""
    for name, value in (env or {}).items():
        if str(name).startswith(SIMULATED_ACCEPTANCE_ENV) and str(value).strip():
            return str(name)
    if SYNTHETIC_REJECTION.search(flags or ""):
        return "rejection_sample_method: synthetic"
    return ""


def gsm8k_workload(config: dict, policy) -> tuple[bool, str]:
    """Is the campaign's workload one GSM8K judges; why not when it is not. Read from the launch
    the campaign was given (the config's ``server_env`` and ``server_flags``), never from what an
    arm changed: an arm cannot take its change out of the gate."""
    if not getattr(policy, "accuracy_gate", False):
        return False, "the accuracy gate is off: the pack's correctness check is not GSM8K"
    simulated = simulated_acceptance(config.get("server_env"), str(config.get("server_flags") or ""))
    if simulated:
        return False, (f"the launch simulates speculative acceptance ({simulated}), so GSM8K would "
                       f"score the simulation, not the model (an AgentX recipe)")
    return True, ""


def every_keep(config: dict, policy) -> bool:
    """Is every KEEP held to the full set: component mode, on a GSM8K workload."""
    from arbor import component
    return component.is_component(config) and gsm8k_workload(config, policy)[0]


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def root_record(config: dict) -> dict | None:
    """``<session>/baseline_accuracy.json``, the root's full set as this session measured it; None without one."""
    root = config.get("session_root")
    if not root:
        return None
    try:
        rec = json.loads((Path(root) / ROOT_FILE).read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def golden(config: dict) -> tuple[float | None, str]:
    """The score to hold the arm to, and where it came from: the baseline record's full set,
    else the root's as this session measured it, else the pack's ``GSM8K_MIN``."""
    rec = config.get("baseline_record")
    if isinstance(rec, str) and rec:
        try:
            rec = json.loads(Path(rec).read_text())
        except (OSError, ValueError):
            rec = None
    full = rec.get("correctness_full") if isinstance(rec, dict) else None
    score = _number(full.get("score") if isinstance(full, dict) else None)
    if score is not None:
        return score, BASELINE_SOURCE
    root = _number((root_record(config) or {}).get("score"))
    if root is not None:
        return root, ROOT_SOURCE
    try:
        return float((config.get("server_env") or {})["GSM8K_MIN"]), FLOOR_SOURCE
    except (KeyError, TypeError, ValueError):
        return None, "no golden score and no GSM8K_MIN"


def root_skip(config: dict, policy, iteration: int, now: float, deadline: float | None) -> str:
    """Why the root's full set is not measured now; "" when it is to be: the workload is a GSM8K
    one (``gsm8k_workload``), this is not a dossier pass, nothing measured it already, no iteration has moved the workspace off the
    root, and the deadline leaves ``ROOT_NEED_S``."""
    from arbor import component
    gsm8k, why = gsm8k_workload(config, policy)
    if not gsm8k:
        return why
    if component.stops_at_map(config):
        return "a dossier pass measures nothing"
    if root_record(config) is not None:
        return f"measured already ({ROOT_FILE})"
    if golden(config)[1] == BASELINE_SOURCE:
        return "the baseline record holds the full set's score"
    if iteration > 0:
        return "an iteration has run, so the workspace is no longer the root"
    if deadline is not None and deadline - now < ROOT_NEED_S + component.DEADLINE_MARGIN_S:
        return (f"{max(0.0, deadline - now) / 60:.0f} min left before the deadline, under the "
                f"{ROOT_NEED_S // 60} min it needs")
    return ""


def measure_root(config: dict, *, start: Callable[[dict], int], healthy: Callable[[dict, int], bool],
                 stop: Callable[[dict], None], run: Callable | None = None,
                 clock: Callable[[], float] = time.time) -> dict:
    """Serve the workspace as initialized (the root), run the pack's correctness script on the
    full set, stop the server, and record ``<session>/baseline_accuracy.json``: ``{score, n,
    commit, ts, seconds, source}``, or a null score and ``error`` when it could not be measured,
    so a resumed session does not try again and the gate keeps ``GSM8K_MIN``.

    ``start``, ``healthy`` and ``stop`` are the harness's own server helpers
    (arbor/policy/claude_loop.py): in component mode they start and stop only the server Arbor
    launched, by its PID file, and set no GPU masks. Each phase keeps its cap and the deadline."""
    from arbor import component
    from arbor.execute.bench import run_correctness
    session = Path(config["session_root"])
    began = clock()
    record = {"score": None, "n": FULL_GSM8K, "commit": component.block(config).get("root", ""),
              "source": ROOT_SOURCE}
    try:
        error = _serve_and_measure(config, session / "root_accuracy", start, healthy, run or run_correctness, record)
    finally:
        stop(config)
    record.update(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seconds=round(clock() - began, 1))
    if error:
        record["error"] = error
    tmp = session / f".{ROOT_FILE}.tmp"
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    tmp.replace(session / ROOT_FILE)
    return record


def _serve_and_measure(config: dict, out_dir: Path, start, healthy, run, record: dict) -> str | None:
    """Fill ``record["score"]``; the reason it could not, else None."""
    from arbor import component
    boot = component.phase_seconds(config, "server")
    if boot is not None and boot <= 0:
        return "no time is left to start a server"
    component.phase("server")
    if not start(config):
        return "the server did not start"
    if not healthy(config, int(boot if boot is not None else 600)):
        return "the server did not become healthy"
    cap = component.phase_seconds(config, "correctness")
    if cap is not None and cap <= 0:
        return "no time is left for the full set"
    component.phase("correctness")
    result = run(config, out_dir, extra_env={"GSM8K_LIMIT": str(FULL_GSM8K)}, cap=cap)
    if result.get("capped"):
        return f"the full set timed out after {result['capped']} s (its cap or the deadline)"
    try:
        raw = json.loads((out_dir / "correctness_output.json").read_text())
    except (OSError, ValueError):
        raw = {}
    score = _number(raw.get("score") if isinstance(raw, dict) else None)
    if score is None or (result.get("error") and not score):
        return result.get("error") or "the correctness script wrote no score"
    record["score"] = score
    return None


def measure(config: dict, iter_dir, by: list[str], tolerance: float, out_dir=None,
            run: Callable | None = None, cap: float | None = None) -> dict:
    """Run the full set into ``out_dir`` (``iter_dir/accuracy/``) against the live server and
    return the gate's input, which is also written there as ``ACCURACY_FILE`` for the KEEP's
    snapshot (arbor/component.py). ``run`` and ``cap`` are ``measure_root``'s."""
    from arbor.execute.bench import run_correctness
    out_dir = Path(out_dir) if out_dir is not None else Path(iter_dir) / "accuracy"
    result = (run or run_correctness)(config, out_dir, extra_env={"GSM8K_LIMIT": str(FULL_GSM8K)},
                                      **({"cap": cap} if cap is not None else {}))
    if result.get("capped"):
        result = {**result, "error": f"the full set timed out after {result['capped']} s"}
    try:
        raw = json.loads((out_dir / "correctness_output.json").read_text())
    except (OSError, ValueError):
        raw = {}
    gold, source = golden(config)
    acc = {
        "triggered": True, "by": by,
        "score": raw.get("score") if isinstance(raw, dict) else None,
        "measurable": isinstance(raw, dict) and "score" in raw,
        "error": result.get("error"),
        "golden": gold, "source": source,
        # The tolerance is a margin under a measured score, the baseline's or the root's; a pack
        # floor is one already.
        "tolerance": 0.0 if source == FLOOR_SOURCE else tolerance,
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ACCURACY_FILE).write_text(json.dumps(acc, indent=1, default=str) + "\n")
    except OSError:
        pass   # the record is bookkeeping; the gate reads the return value
    return acc


def reading(acc: dict) -> tuple[str, str]:
    """``measure``'s result as the gate reads it: ("pass" | "fail" | "not_run", why). A score
    under the golden one less the tolerance fails; an eval with no score, or nothing to compare
    it with, measured nothing."""
    score, gold = acc.get("score"), acc.get("golden")
    if acc.get("error") and not score:
        return "not_run", f"the eval did not run ({acc['error']})"
    if not acc.get("measurable", True) or _number(score) is None:
        return "not_run", "the correctness output has no score"
    if gold is None:
        return "not_run", f"{acc.get('source', 'no golden score')} to compare {score} with"
    tol = float(acc.get("tolerance") or 0.0)
    if score < gold - tol:
        return "fail", f"{score} < {gold} − {tol} ({acc.get('source', '')})"
    return "pass", f"{score} against {gold} − {tol} ({acc.get('source', '')})"


def brief(acc: dict | None) -> dict | None:
    """The row's record."""
    if not acc:
        return None
    return {k: acc.get(k) for k in ("by", "score", "golden", "tolerance", "source")}
