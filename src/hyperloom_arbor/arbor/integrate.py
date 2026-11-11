"""The master's integration: a drained round's additions stacked onto the master's tree, each with a same-node A/B
against the tree integrated so far. From arbor/parallel/integrate.py, without its Slurm jobs and its launcher's A/B:
the master's own workspace on this node, and arbor/ab.py.

For each addition, in the order given (the caller's: by rank):
1. **Merge** it onto the workspace's HEAD: its patch by ``git apply --3way`` (the image's blobs are in every workspace,
   so the base it was made on is the ancestor), its server flags and env per key (``merge_flag_sets``). A conflict
   drops it, with its paths or keys.
2. **Commit** the merge (``arbor integrate: ...``).
3. **Measure:** ``arbor.ab`` serves the tree before and after on this node. Confirmed: kept. Anything else: the
   workspace goes back to the tree before, and the addition is dropped with the A/B's verdict; the next map may bring
   it back.
4. **Book** it in the master's session as an iteration: a KEEP with the A/B's current mean, or a REVERT saying why,
   citing the order's directions, so the round's record counts what landed (``directions.reconcile``, ``_close_round``).

``<session>/integrate/round-<n>/integration.json`` holds each addition's outcome, and each A/B has its own directory.
Run again, it skips the additions it decided.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from arbor.execute import workspace as ws
from arbor.parallel.trees import FlagConflict, flags_from_workspace, flags_to_workspace, merge_flag_sets
from arbor.state import CampaignState

KEPT, DROPPED, CONFLICT = "kept", "dropped", "conflict"
PREFIX = "arbor integrate:"


def where(session_dir, round_no) -> Path:
    return Path(session_dir) / "integrate" / f"round-{round_no}"


def load(session_dir, round_no) -> dict | None:
    try:
        return json.loads((where(session_dir, round_no) / "integration.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save(session_dir, round_no, doc: dict) -> None:
    d = where(session_dir, round_no)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "integration.json.tmp"
    tmp.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, d / "integration.json")


def _back(container: str, root: str, sha: str) -> None:
    ws.git(container, root, "reset", "-q", "--hard", sha)
    ws.git(container, root, "clean", "-fdq")


def merge(container: str, root: str, add: dict, message: str) -> str | dict:
    """``add`` onto the workspace's HEAD and committed: the commit; or ``{"paths"|"keys", "detail"}`` on a conflict,
    the workspace left as it was."""

    before = ws.git(container, root, "rev-parse", "HEAD")
    patch = Path(add["patch"]).read_bytes() if add.get("patch") else b""
    if patch.strip():
        r = ws.run_in(container, ["git", "apply", "--3way", "--binary", "-"], workdir=root, input=patch,
                      capture_output=True)
        if r.returncode != 0:
            unmerged = ws.run_in(container, ["git", "diff", "--name-only", "--diff-filter=U"], workdir=root,
                                 capture_output=True, text=True).stdout.split()
            _back(container, root, before)
            detail = (r.stderr or r.stdout or b"").decode(errors="replace").strip().splitlines()
            return {"paths": unmerged, "detail": " | ".join(detail[-3:])[:400]}
    flags = merge_flag_sets(flags_from_workspace(add.get("base_config") or {}),
                            flags_from_workspace(ws.read_workspace_config(container, root)),
                            flags_from_workspace(add.get("config") or {}))
    if isinstance(flags, FlagConflict):
        _back(container, root, before)
        return {"keys": list(flags.keys), "detail": "the server flags or env clash"}
    merged = flags_to_workspace(flags)
    ws.write_workspace_config(container, root, merged["server_env"], merged["server_flags"])
    return ws.commit(container, root, message)


def _book(session_dir, entry: dict, add: dict) -> None:
    """The addition in the master's session: a measured one as an iteration (a KEEP, or a REVERT with its numbers), one
    that never merged as a NOTE."""

    path = Path(session_dir) / "session.json"
    state = CampaignState.load(path)
    n = state.iteration + 1
    cites = ", ".join(add.get("directions") or [])
    change = (f"integrate order {add['order']} ({cites})"
              + (f": {'; '.join(add['keeps'])}" if add.get("keeps") else ""))[:400]
    note = {k: entry.get(k) for k in ("order", "outcome", "commit", "verdict", "ab") if entry.get(k) is not None}
    if entry["outcome"] == KEPT:
        state.iteration_notes = {"integration": note}
        state.record_keep(n, entry["metric"], change, state.tier, findings=[f"same-node A/B: {entry['reason']}"])
        entry["iteration"] = n
    elif entry.get("metric") is not None:
        state.iteration_notes = {"integration": note}
        state.record_revert(n, entry["metric"], change, state.tier, f"{entry['outcome']}: {entry['reason']}")
        entry["iteration"] = n
    else:
        state.lessons.append({"iteration": state.iteration, "decision": "NOTE", "change": change,
                              "reason": f"{entry['outcome']}: {entry['reason']}", "integration": note,
                              "ts": time.time()})
    state.save(path)


def run(config_path, session_dir, round_no: int, additions: list[dict], deadline: float, ab_run=None,
        rounds: int | None = None) -> dict:
    """Integrate ``additions`` (``slots.addition``'s, each with its ``order``, ``directions`` and ``keeps``) into the
    master's workspace, in order; the record. ``ab_run`` stands in for ``arbor.ab.run`` (its signature); ``rounds``,
    the measured pairs of each A/B, its default when None."""

    from arbor import ab

    ab_run = ab_run or ab.run
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    container, root = config.get("docker_container", ""), config.get("workspace_root", "")
    doc = load(session_dir, round_no) or {"round": round_no, "base": ws.git(container, root, "rev-parse", "HEAD"),
                                          "additions": []}
    done = {e["order"]: e for e in doc["additions"] if e.get("outcome")}
    for k, add in enumerate(additions, 1):
        if add["order"] in done:
            continue
        before = ws.git(container, root, "rev-parse", "HEAD")
        entry = {"order": add["order"], "directions": add.get("directions") or [], "before": before}
        stale = next((e for e in doc["additions"] if e["order"] == add["order"]), None)
        if stale and stale.get("commit") == before:          # a run that stopped mid-A/B: from the tree before it
            _back(container, root, stale["before"])
            entry["before"] = before = stale["before"]
        doc["additions"] = [e for e in doc["additions"] if e["order"] != add["order"]] + [entry]
        made = merge(container, root, add, f"{PREFIX} order {add['order']} ({', '.join(entry['directions'])})")
        if isinstance(made, dict):
            entry.update(outcome=CONFLICT, conflict=made, metric=None,
                         reason=f"does not merge: {', '.join(made.get('paths') or made.get('keys') or [])} "
                                f"({made['detail']})")
        else:
            entry["commit"] = made
            _save(session_dir, round_no, doc)
            out = where(session_dir, round_no) / f"{k:02d}-{add['order']}"
            summary, rc = ab_run(str(config_path), out, deadline, base=before, current=made,
                                 **({"rounds": int(rounds)} if rounds else {}))
            entry.update(ab=str(out), verdict=summary.get("verdict"), metric=summary.get("current_mean"),
                         reason=f"{summary.get('verdict')}, gain {summary.get('gain')} "
                                f"({summary.get('reason', '')})".replace(" ()", ""))
            if rc == ab.EXIT_RESTORE:
                entry.update(outcome=None, reason=f"the A/B could not put the tree back: {summary.get('reason')}")
                doc["restore_failed"] = True
                _save(session_dir, round_no, doc)
                return doc
            if summary.get("verdict") == "confirmed":
                entry["outcome"] = KEPT
            else:
                entry["outcome"] = DROPPED
                _back(container, root, before)
        _book(session_dir, entry, add)
        print(f"[arbor] Integrate: order {add['order']} {entry['outcome']} ({entry['reason']})")
        _save(session_dir, round_no, doc)
    doc["head"] = ws.git(container, root, "rev-parse", "HEAD")
    doc["kept"] = [e["order"] for e in doc["additions"] if e.get("outcome") == KEPT]
    doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if doc["kept"]:
        ws.save_landed_patch(container, root, Path(session_dir) / "landed.patch", config)
    _save(session_dir, round_no, doc)
    return doc
