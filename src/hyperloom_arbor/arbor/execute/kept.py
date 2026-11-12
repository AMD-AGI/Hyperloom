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
import re
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
    retraction) moves the point too. Only a record that exists moves (``ensure`` makes it).

    An iteration the session never kept never moves it. Its commits are still in the tree when its restore never ran (a
    SIGTERM after its commit, a crash between its verdict and the restore, a restore that failed), and an editable
    tree keeps the commit of an iteration never booked, which the orphan walk (the workspace's only) leaves. Adopting
    them put an unmeasured or reverted change under every later KEEP. So they are reverted (``_live``, ``_undo``), a NOT
    MEASURED one noted for the revisit first, and what the harness committed between passes stays.

    The new record, with ``line`` for the log; or None when nothing moved."""
    doc = load(session_dir)
    if not doc:
        return None
    head, tree_heads = _heads(container, config)
    if not head or (head == doc["workspace"] and tree_heads == (doc.get("trees") or {})):
        return None
    live = _live(container, config, session_dir, doc)
    if not live:
        new = record(container, config, session_dir, iteration - 1, "carried", f"iteration {iteration}'s start")
        return {**new, "line": "moved to the tree at this pass's start"} if new else None
    names = ", ".join(str(n) for n in sorted({n for found in live.values() for n, _, _ in found}))
    noted = _note_live(config, session_dir, live)
    label = doc.get("label") or (f"iter-{doc['iteration']}" if doc.get("iteration") else "the start")
    message = f"Restore the kept tree of {label} after iteration {iteration - 1} (iteration {names} never kept)"
    kept_shas = dict(_pairs(config, doc))
    done = [f"{_undo(container, repo, found, kept_shas[repo], message)} in {repo}" for repo, found in live.items()]
    # The same tree as the kept point's: the point keeps its name. A tree the harness also committed to since: carried.
    new = (record(container, config, session_dir, doc.get("iteration", 0), doc.get("verdict", "start"), label)
           if not _moved(container, config, doc) else
           record(container, config, session_dir, iteration - 1, "carried", f"iteration {iteration}'s start"))
    line = (f"iteration {names} was never kept, yet its commits were still in the tree (its restore never ran): "
            f"{'; '.join(done)}" + (f"; noted for the revisit: iteration {', '.join(map(str, noted))}" if noted else ""))
    return {**(new or {}), "line": line}


#: A gate row's decisions (arbor/measure/intents.py, ``legacy_decision``), and the two that keep a tree. Every other
#: decision is an annotation (arbor/directions.py, ``ANNOTATIONS``).
GATE_DECISIONS = ("KEEP", "PROBE", "REVERT", "INVALID")
KEPT_DECISIONS = ("KEEP", "PROBE")

_ITER = re.compile(r"iter-(\d+):")
_REVERT_OF = re.compile(r'Revert "iter-(\d+):')
#: How ``restore``'s commit begins: it set the tree back to the kept point, undoing every commit before it.
_RESTORE = "Restore the kept tree of "


def _pairs(config: dict, doc: dict) -> list[tuple[str, str]]:
    """``(repository, kept commit)``: the workspace's, then each editable tree's the config still names."""
    pairs = [(config.get("workspace_root", ""), doc["workspace"])]
    return pairs + [(t, sha) for t, sha in (doc.get("trees") or {}).items() if t in _trees.paths(config)]


def _verdicts(session_dir: str | Path) -> dict[int, dict]:
    """Each iteration's last gate row (``directions.verdict_row``'s rule), from ``session.json``."""
    try:
        lessons = json.loads((Path(session_dir) / "session.json").read_text()).get("lessons") or []
    except (OSError, ValueError, AttributeError):
        return {}
    return {row["iteration"]: row for row in lessons
            if isinstance(row, dict) and isinstance(row.get("iteration"), int)
            and row.get("decision") in GATE_DECISIONS}


def _same_tree(container: str, repo: str, sha: str) -> bool:
    try:
        return _git(container, repo, "rev-parse", f"{sha}^{{tree}}") == _git(container, repo, "rev-parse", "HEAD^{tree}")
    except RuntimeError:
        return False


def _moved(container: str, config: dict, doc: dict) -> bool:
    """Whether any repository's tree differs from the kept point's."""
    return not all(_same_tree(container, repo, sha) for repo, sha in _pairs(config, doc))


def _live(container: str, config: dict, session_dir: str | Path, doc: dict) -> dict[str, list[tuple[int, str, str]]]:
    """Per repository whose tree differs from the kept point's, the commits since that point of iterations the session
    never kept that nothing has undone: ``{repo: [(iteration, sha, subject), ...]}``, oldest first.

    The walk follows the repository's own line (``--first-parent``): a merged branch's commits are another session's.
    A ``Revert "iter-N: ..."`` undoes N's commits, and ``restore``'s commit undoes every one before it. An iteration is
    kept when its last gate row is a KEEP or a PROBE; one with no row was never booked."""
    verdicts = _verdicts(session_dir)
    found: dict[str, list[tuple[int, str, str]]] = {}
    for repo, sha in _pairs(config, doc):
        if _same_tree(container, repo, sha):
            continue
        try:
            log = _git(container, repo, "log", "--reverse", "--first-parent", "--format=%H%x09%s", f"{sha}..HEAD")
        except RuntimeError:
            continue
        live: list[tuple[int, str, str]] = []
        for line in log.splitlines():
            commit, _, subject = line.partition("\t")
            if subject.startswith(_RESTORE):
                live = []
            elif m := _REVERT_OF.match(subject):
                live = [c for c in live if c[0] != int(m.group(1))]
            elif (m := _ITER.match(subject)) and \
                    (verdicts.get(int(m.group(1))) or {}).get("decision") not in KEPT_DECISIONS:
                live.append((int(m.group(1)), commit, subject))
        if live:
            found[repo] = live
    return found


def _undo(container: str, repo: str, live: list[tuple[int, str, str]], kept_sha: str, message: str) -> str:
    """Revert ``live``'s commits in ``repo``, newest first, as the orphan walk does; what was done. A revert that does
    not apply (a later commit touched the same lines) sets the whole tree back to the kept point instead, as
    ``restore`` would: what the harness committed since goes too, and the log says so."""
    _git(container, repo, "checkout", "--", ".", check=False)
    _git(container, repo, "clean", "-fdq", check=False)
    try:
        for _, sha, subject in reversed(live):
            if _git(container, repo, "diff", "--name-only", f"{sha}~1", sha):
                _git(container, repo, "revert", "--no-edit", sha)
            else:
                _git(container, repo, "commit", "-q", "--allow-empty", "-m", f'Revert "{subject}"')
        return f"reverted {', '.join(sha[:12] for _, sha, _ in live)}"
    except RuntimeError as exc:
        _git(container, repo, "revert", "--abort", check=False)
        _restore_repo(container, repo, kept_sha, message)
        why = (str(exc).strip().splitlines() or [repr(exc)])[-1][:160]
        return f"a revert did not apply ({why}): the kept tree is back, and what the harness committed since went too"


def _note_live(config: dict, session_dir: str | Path, live: dict[str, list[tuple[int, str, str]]]) -> list[int]:
    """Note each NOT MEASURED iteration among ``live`` that ``invalids.json`` lacks, for the revisit, as
    ``note_invalid`` would have; the iterations noted."""
    verdicts = _verdicts(session_dir)
    path = Path(session_dir) / INVALIDS
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        doc = []
    have = {e.get("iteration") for e in doc if isinstance(e, dict)}
    workspace = config.get("workspace_root", "")
    commits: dict[int, dict[str, str]] = {}
    for repo, found in live.items():
        for n, sha, _ in found:
            commits.setdefault(n, {})[repo] = sha        # the newest: a retried iteration's last attempt
    noted = []
    for n, by_repo in sorted(commits.items()):
        row = verdicts.get(n)
        if not row or row.get("decision") != "INVALID" or n in have:
            continue
        doc.append({"iteration": n, "change": row.get("change", ""), "intent": row.get("claim_class", "B"),
                    "reason": row.get("reason", ""), "workspace": by_repo.get(workspace),
                    "trees": {r: s for r, s in by_repo.items() if r != workspace}, "revisit": None})
        noted.append(n)
    if noted:
        path.write_text(json.dumps(doc, indent=1) + "\n")
    return noted


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
