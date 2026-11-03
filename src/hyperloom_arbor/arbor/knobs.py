"""The serving playbook's knob catalog, for the harness.

``.claude/skills/serving-playbook/knobs.json`` is the machine-readable index of
the skill's ``amd-knobs.md`` and ``numerics-knobs.md``: every knob with its
kind (env or flag), whether it changes numerics, and whether its best value
depends on the operating point; and the documented bundles, the sets whose
members only pay together. The md files are for the agent; this is what the
gate reads: a bundle is one claim (arbor/measure/bundles.py), a numerics knob
calls for the accuracy gate, and a dead end on an operating-point knob holds
only where it was measured (ledger.render).

An absent or unreadable catalog is an empty one: every check that reads it
then has nothing to say, which is the behaviour before it existed.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

CATALOG = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "serving-playbook" / "knobs.json"


@lru_cache(maxsize=4)
def load(path: str | None = None) -> dict:
    p = Path(path) if path else CATALOG
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return {"knobs": [], "bundles": []}
    return {"knobs": list(doc.get("knobs") or []), "bundles": list(doc.get("bundles") or [])}


def by_name(path: str | None = None) -> dict[str, dict]:
    return {k["name"]: k for k in load(path)["knobs"]}


def numerics(path: str | None = None) -> set[str]:
    return {k["name"] for k in load(path)["knobs"] if k.get("numerics")}


def op_dependent(path: str | None = None) -> set[str]:
    return {k["name"] for k in load(path)["knobs"] if k.get("op_dependent")}


def bundle(bundle_id: str, path: str | None = None) -> dict | None:
    return next((b for b in load(path)["bundles"] if b["id"] == bundle_id), None)


def bundles(framework: str | None = None, path: str | None = None) -> list[dict]:
    """The catalog's bundles, those for ``framework`` when one is given."""
    return [b for b in load(path)["bundles"]
            if framework is None or framework in (b.get("frameworks") or [framework])]


def named_in(text: str, names) -> list[str]:
    """The names that appear in ``text`` as whole tokens, so that
    VLLM_ROCM_USE_AITER does not match inside VLLM_ROCM_USE_AITER_MOE."""
    return sorted(n for n in names if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text or ""))


def settings(config: dict, ws_cfg: dict | None) -> dict[str, object]:
    """One arm's settings by name, the way the server gets them (arbor/execute/bench.py):
    the workspace's server_env over the campaign's, and its server_flags, when it has
    the key, instead of the campaign's. Env names and flag names do not collide."""
    from arbor.measure.armcheck import flags_of
    ws_cfg = ws_cfg or {}
    env = {k: str(v) for k, v in (config.get("server_env") or {}).items()}
    env.update({k: str(v) for k, v in (ws_cfg.get("server_env") or {}).items()})
    flags = ws_cfg.get("server_flags")
    if flags is None:
        flags = config.get("server_flags", "")
    return {**env, **flags_of(str(flags or ""))}


def changes(before: dict, after: dict) -> dict[str, tuple]:
    """``{name: (before, after)}`` for every setting the change added, removed or moved."""
    from arbor.measure.armcheck import same_value
    out = {}
    for name in sorted(set(before) | set(after)):
        a, b = before.get(name), after.get(name)
        if (a is None) != (b is None) or (a is not None and not same_value(a, b)):
            out[name] = (a, b)
    return out
