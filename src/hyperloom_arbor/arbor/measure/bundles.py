"""A bundle is one claim: a documented set of settings that only pays together.

Packs say "one change at a time", and for most changes that is right. But some
wins are a set whose members are each a no-op, a crash or under the gate alone
(serving-playbook amd-knobs.md §0 rule 6, big-directions.md §0): INT4
quick-reduce needs its minimum size or it never fires, two-batch overlap needs
the hardware queues or its streams serialize. An agent that tries them one at a
time measures nothing, and records a dead end for a direction that works.

So the agent may apply a catalogued bundle (knobs.json) in one iteration and
write ``BUNDLE: <id>``. The gate then takes it as one change, provided the
bundle really is what ran:

- every member is set in the arm's config as the catalog gives it, and the
  change moved at least one of them (one already in the base counts as set);
- the live server shows every member (arbor/measure/armcheck.py).

A member that is not set, or not live, or a bundle the catalog does not know,
is NOT_MEASURED: the arm is not the bundle it names. A bundle whose members
cannot be read off the live server (no capture) is NOT_MEASURED too, because
"each member took effect" is the claim.
"""

from __future__ import annotations

from arbor import knobs
from arbor.measure.armcheck import same_value


def _value(m: dict) -> str:
    return f"{m['name']}={m['value']}" if "value" in m else m["name"]


def members(b: dict) -> list[str]:
    return [_value(m) for m in b.get("members", [])]


def _matches(m: dict, got) -> bool:
    if got is None:
        return False
    return "value" not in m or same_value(m["value"], got)


def check(bundle_id: str, before: dict, after: dict, arm: dict | None) -> dict:
    """``{"id", "members", "ok", "reason"}`` for one declared bundle.

    ``before`` and ``after`` are the arm's settings at HEAD~1 and HEAD
    (knobs.settings); ``arm`` is the arm check, carrying the live process's
    ``effective`` env and flags.
    """
    b = knobs.bundle(bundle_id)
    if b is None:
        known = ", ".join(x["id"] for x in knobs.bundles()) or "none"
        return {"id": bundle_id, "members": [], "ok": False,
                "reason": f"unknown bundle '{bundle_id}' — the catalog's bundles are: {known}"}
    out = {"id": bundle_id, "members": members(b), "ok": False}
    unset = [_value(m) for m in b["members"] if not _matches(m, after.get(m["name"]))]
    if unset:
        out["reason"] = f"bundle {bundle_id}: member {', '.join(unset)} not applied in arbor/config.json"
        return out
    moved = knobs.changes(before, after)
    if not any(m["name"] in moved for m in b["members"]):
        out["reason"] = (f"bundle {bundle_id}: the change moved none of its members; every one "
                         f"was already set before it")
        return out
    eff = (arm or {}).get("effective")
    if not eff:
        why = (arm or {}).get("reason") or "the arm check is off"
        out["reason"] = (f"bundle {bundle_id}: its members cannot be proven live on the server "
                         f"({why})")
        return out
    live = {**eff.get("env", {}), **eff.get("flags", {})}
    dead = [_value(m) for m in b["members"] if not _matches(m, live.get(m["name"]))]
    if dead:
        out["reason"] = f"bundle {bundle_id}: member {', '.join(dead)} not live on the server"
        return out
    out["ok"] = True
    out["reason"] = f"bundle {bundle_id}: every member applied and live"
    return out
