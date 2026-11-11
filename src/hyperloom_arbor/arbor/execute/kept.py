"""The tree the campaign last kept, and how a REVERT or a NOT MEASURED iteration goes back to it.

A REVERT used to undo only its own commit (``git revert HEAD``), and a NOT MEASURED iteration kept its tree, so the tree
could drift from what the session records:
- an INVALID iteration's change stayed in, unmeasured, under every later change;
- a REVERT of a later commit that only re-measured it (an empty marker) left the change in;
- reverting a control that had removed an earlier change put that change back.

Later KEEPs were measured with these leftovers and credited with their effect, and an unmeasured change could ship in
``landed.patch``.

So ``<session>/kept.json`` records the commit of the tree the campaign last kept: the start (``ensure``), then every
KEEP and PROBE (``record``), and any commit the harness made between passes (``refresh``, at each pass's start), with
each editable tree's commit beside the workspace's (arbor/execute/trees.py). After every REVERT and
every NOT MEASURED iteration, the driver restores that tree (``restore``): one commit per repository, made only when the
tree differs, that sets it to the kept commit's tree. History stays. An INVALID iteration's committed change is noted
in ``<session>/invalids.json`` first (``note_invalid``), for the revisit phase (arbor/revisit.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from arbor.execute import trees as _trees
from arbor.execute.trees import _git

FILE = "kept.json"
INVALIDS = "invalids.json"


def _head(container: str, repo: str) -> str | None:
    try:
        return _git(container, repo, "rev-parse", "HEAD") or None
    except RuntimeError:
        return None


def load(session_dir: str | Path) -> dict | None:
    """The kept point, or None."""
    try:
        doc = json.loads((Path(session_dir) / FILE).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("workspace") else None


def _heads(container: str, config: dict) -> tuple[str | None, dict[str, str]]:
    workspace = config.get("workspace_root", "")
    head = _head(container, workspace) if container and workspace else None
    return head, {t: h for t in _trees.paths(config) if (h := _head(container, t))}


def record(container: str, config: dict, session_dir: str | Path, iteration: int = 0,
           verdict: str = "start", label: str = "") -> dict | None:
    """Record the tree as it is now as the kept point: the workspace's HEAD and each editable tree's."""
    head, tree_heads = _heads(container, config)
    if not head:
        return None
    doc = {"workspace": head, "trees": tree_heads, "iteration": iteration, "verdict": verdict,
           "label": label or (f"iter-{iteration}" if iteration else "the start"),
           "container": _trees._container_id(container), "ts": time.time()}
    part = Path(session_dir) / f".{FILE}.tmp"
    part.write_text(json.dumps(doc, indent=1) + "\n")
    part.replace(Path(session_dir) / FILE)
    return doc


def refresh(container: str, config: dict, session_dir: str | Path, iteration: int) -> dict | None:
    """At a pass's start, before the agent's turn: the tree is the kept one then (the last verdict applied, orphans
    reverted), so a commit the harness made between passes (the master's integration of its workers' KEEPs, a
    retraction) moves the point too. Only a record that exists moves (``ensure`` makes it); the new one, or None when
    nothing moved."""
    doc = load(session_dir)
    if not doc:
        return None
    head, tree_heads = _heads(container, config)
    if not head or (head == doc["workspace"] and tree_heads == (doc.get("trees") or {})):
        return None
    return record(container, config, session_dir, iteration - 1, "carried", f"iteration {iteration}'s start")


START = "arbor: the tree at this start"


def ensure(container: str, config: dict, session_dir: str | Path) -> dict | None:
    """At a start: keep the record when it is this container's, else record the tree as it stands (a fresh session's
    start, or a resume in a new container, whose launcher restored the last KEEP's tree).

    What the workspace holds uncommitted then (a launcher's restore) is committed first, so the kept point has it and a
    restore cannot clean it away. Under a host the workspace is the host's to commit: a dirty one records nothing, and
    a REVERT there undoes its own commit only, as before."""
    from arbor import component
    from arbor.execute.workspace import has_changes
    doc = load(session_dir)
    if doc and doc.get("container") == _trees._container_id(container):
        return doc
    workspace = config.get("workspace_root", "")
    if not container or not workspace or not _head(container, workspace):
        return None
    try:
        dirty = has_changes(container, workspace)
    except RuntimeError:
        return None
    if dirty:
        if component.is_component(config):
            print("[arbor] kept tree: the host's workspace has uncommitted changes at the start; none recorded")
            return None
        _git(container, workspace, "add", "-A")
        _git(container, workspace, "commit", "-q", "-m", START)
    return record(container, config, session_dir, verdict="start")


def _exists(container: str, repo: str, sha: str) -> bool:
    try:
        _git(container, repo, "cat-file", "-e", f"{sha}^{{commit}}")
        return True
    except RuntimeError:
        return False


def _restore_repo(container: str, repo: str, sha: str, message: str) -> bool:
    """Set ``repo``'s tree to ``sha``'s in one commit; False when it already is. Untracked files go too, ignored ones
    stay: what ``changed`` would count as a change (a table the turn left uncommitted) is not part of the kept tree,
    while bytecode, builds and JIT outputs (``.gitignore``, ``trees.EXCLUDES``) are not changes."""
    _git(container, repo, "checkout", "--", ".", check=False)
    _git(container, repo, "clean", "-fdq", check=False)
    _git(container, repo, "restore", f"--source={sha}", "--staged", "--worktree", "--", ".")
    if not _git(container, repo, "diff", "--cached", "--name-only"):
        return False
    _git(container, repo, "commit", "-q", "-m", message)
    return True


def restore(container: str, config: dict, session_dir: str | Path, iteration: int, why: str) -> list[str]:
    """Set the workspace and each editable tree back to the kept point, after a REVERT or a NOT MEASURED iteration;
    the repositories that changed. Nothing when there is no kept point (an older session), or its commit is gone."""
    doc = load(session_dir)
    workspace = config.get("workspace_root", "")
    if not doc or not container or not workspace:
        return []
    label = doc.get("label") or (f"iter-{doc['iteration']}" if doc.get("iteration") else "the start")
    message = f"Restore the kept tree of {label} after iteration {iteration} ({why})"
    changed = []
    pairs = [(workspace, doc["workspace"])]
    pairs += [(t, sha) for t, sha in (doc.get("trees") or {}).items() if t in _trees.paths(config)]
    for repo, sha in pairs:
        if not _exists(container, repo, sha):
            print(f"[arbor] kept tree: {sha[:12]} is not a commit of {repo}; left as it is")
            continue
        try:
            if _restore_repo(container, repo, sha, message):
                changed.append(repo)
        except RuntimeError as exc:
            print(f"[arbor] kept tree: could not restore {repo}: {exc}")
    return changed


def note_invalid(container: str, config: dict, session_dir: str | Path, iteration: int, result: dict) -> dict | None:
    """Note a NOT MEASURED iteration's committed change in ``<session>/invalids.json``, before the restore takes it out
    of the tree: its commits (the workspace's ``iter-N:`` HEAD and the editable trees' from ``iter_N/trees.json``), its
    change, intent and reason. None when it committed nothing (the agent did not run, a phase timed out first)."""
    workspace = config.get("workspace_root", "")
    sha = None
    if container and workspace:
        try:
            head = _git(container, workspace, "log", "-1", "--format=%H%x09%s")
            h, _, subject = head.partition("\t")
            if subject.startswith(f"iter-{iteration}:"):
                sha = h
        except RuntimeError:
            pass
    tree_shas = _trees.committed(Path(session_dir) / f"iter_{iteration}")
    if not sha and not tree_shas:
        return None
    entry = {"iteration": iteration, "change": result.get("change", ""),
             "intent": result.get("claim_class", "B"), "reason": result.get("reason", ""),
             "workspace": sha, "trees": tree_shas, "revisit": None}
    path = Path(session_dir) / INVALIDS
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        doc = []
    doc = [e for e in doc if e.get("iteration") != iteration] + [entry]
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return entry


def after_pass(container: str, config: dict, session_dir: str | Path, iteration: int, result: dict) -> str:
    """What the driver does to the tree after an iteration, by its decision; a line for the log, or ""."""
    decision = str(result.get("decision") or "").upper()
    if not container or not config.get("workspace_root"):
        return ""
    if decision in ("KEEP", "PROBE") or result.get("retracted"):
        # A KEEP or PROBE keeps its tree; a retraction rewrote the tree after the verdict, so the tree it left is
        # the one kept now, whatever the iteration's own verdict.
        record(container, config, session_dir, iteration, decision.lower() or "retraction")
        return ""
    if decision not in ("REVERT", "INVALID"):
        return ""
    noted = note_invalid(container, config, session_dir, iteration, result) if decision == "INVALID" else None
    changed = restore(container, config, session_dir, iteration,
                      "not measured" if decision == "INVALID" else "reverted")
    parts = []
    if noted:
        parts.append(f"iteration {iteration}'s change noted for a revisit")
    if changed:
        parts.append(f"restored the kept tree in {', '.join(changed)}")
    return "; ".join(parts)
