"""Load a workload schema (schemas/<name>.json) and check a run against it.

Applies to processes with engine steps: every engine step must carry every
required step field, with the declared type, and the process's context must
carry every required run field. The result names what is missing, and where.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "schemas"))


def load(name: str = "serving") -> dict:
    path = name if name.endswith(".json") else os.path.join(SCHEMA_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def _type_ok(value, t: str) -> bool:
    if t == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "str":
        return isinstance(value, str)
    if t == "list[int]":
        return isinstance(value, list) and all(isinstance(x, int) for x in value)
    if t.startswith("enum["):
        return value in t[5:-1].split(",")
    if t == "object":
        return isinstance(value, dict)
    return True


def validate_run(analyses, name: str = None) -> Dict:
    """analyses: engine.Analysis objects of one run. The schema is ``name``, or
    the one the run declared (ruppell.init(schema=...) records it in the
    context: serving for vLLM, training for the DLRMv4 adapter), or serving."""
    if name is None:
        declared = [a.proc.context.get("schema") for a in analyses if a.proc.context.get("schema")]
        name = declared[0] if declared else "serving"
    try:
        schema = load(name)
    except FileNotFoundError:
        return {"schema": name, "checked": False, "reason": "schema file not found"}
    req_step = {k: v for k, v in schema["step"].items() if v.get("required")}
    req_run = {k: v for k, v in schema["run"].items() if v.get("required")}
    out: Dict = {"schema": f"{schema['name']} v{schema['version']}", "checked": True, "processes": {}}
    ok = True
    for a in analyses:
        engine = [s for s in a.step_list if s["fields"].get("phase") == "engine"]
        if not engine:
            continue
        missing_run = [k for k, v in req_run.items() if k not in a.proc.context or not _type_ok(a.proc.context[k], v["type"])]
        bad: Dict[str, int] = {}
        first_bad: Dict[str, int] = {}
        for s in engine:
            for k, v in req_step.items():
                if k not in s["fields"] or not _type_ok(s["fields"][k], v["type"]):
                    bad[k] = bad.get(k, 0) + 1
                    first_bad.setdefault(k, s["step"])
        good = not missing_run and not bad
        ok = ok and good
        out["processes"][a.proc.label] = {
            "engine_steps": len(engine),
            "missing_or_wrong_run_fields": missing_run,
            "steps_missing_or_wrong": {k: {"steps": n, "first_step": first_bad[k]} for k, n in bad.items()},
            "ok": good,
        }
    out["ok"] = ok
    out["processes_checked"] = len(out["processes"])
    return out


def required_step_fields(name: str = "serving") -> List[str]:
    return [k for k, v in load(name)["step"].items() if v.get("required")]
