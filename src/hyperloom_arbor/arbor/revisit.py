"""The revisit phase: the NOT MEASURED iterations' changes, measured once more after the map has closed every direction,
while the time left still fits an iteration.

A NOT MEASURED iteration's change leaves the tree at once (arbor/execute/kept.py), noted in ``<session>/invalids.json``:
no GPU time goes to measuring it again while the map still has directions open. When every direction is closed:

1. **Arm** (``arm``, from the master's ``before_pass``). The oldest noted change not yet revisited, when the time left
   fits one more iteration (``time_fits``: the time budget less what is spent, against ``component.estimate``; under a
   host or a rolling job the master has already checked its own deadline). A PROBE's change, one that holds nothing,
   and one whose diff no longer applies to the kept tree (``git apply --check``) are closed with why and passed over.
   ``<session>/revisit.json`` names the change for the next iteration.
2. **Hold the round** (``hold``, from ``directions.advance``). An exhausted map with a revisit armed does not close its
   round: the iteration runs.
3. **Apply** (``apply``, in the iteration before the agent's turn). The change goes back into the workspace and the
   editable trees, uncommitted, and the agent's prompt starts with ``brief``: measure this change as it is.
4. **Close** (``close``, the driver after the pass). The entry's ``revisit`` records the iteration and its verdict. A
   revisit that is not measured either is not revisited again.

Each change is revisited at most once. With none left, or no time for one, the round ends as it did before.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from arbor.execute import trees as _trees
from arbor.execute.kept import INVALIDS
from arbor.execute.workspace import run_in

FILE = "revisit.json"
ATTEMPTS = 2          # passes that may start on one armed revisit before it is closed unmeasured


def _invalids(session_dir) -> list[dict]:
    try:
        doc = json.loads((Path(session_dir) / INVALIDS).read_text())
    except (OSError, ValueError):
        return []
    return [e for e in doc if isinstance(e, dict)] if isinstance(doc, list) else []


def _save_invalids(session_dir, doc: list[dict]) -> None:
    part = Path(session_dir) / f".{INVALIDS}.tmp"
    part.write_text(json.dumps(doc, indent=1) + "\n")
    part.replace(Path(session_dir) / INVALIDS)


def pending(session_dir) -> list[dict]:
    """The noted changes not yet revisited, oldest first."""
    return sorted((e for e in _invalids(session_dir) if e.get("revisit") is None),
                  key=lambda e: e.get("iteration") or 0)


def armed(session_dir, iteration: int | None = None) -> dict | None:
    """The armed revisit, or only the one armed for ``iteration``."""
    try:
        doc = json.loads((Path(session_dir) / FILE).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or (iteration is not None and doc.get("iteration") != iteration):
        return None
    return doc


def _write(session_dir, doc: dict) -> None:
    part = Path(session_dir) / f".{FILE}.tmp"
    part.write_text(json.dumps(doc, indent=1) + "\n")
    part.replace(Path(session_dir) / FILE)


def _close_entry(session_dir, of: int, outcome: dict, also: int | None = None) -> None:
    """Record ``outcome`` as the revisit of iteration ``of``; ``also``, a revisit's own iteration noted NOT MEASURED,
    is closed as not to be revisited."""
    doc = _invalids(session_dir)
    for e in doc:
        if e.get("iteration") == of:
            e["revisit"] = outcome
        elif also is not None and e.get("iteration") == also and e.get("revisit") is None:
            e["revisit"] = {"decision": "not revisited", "reason": f"the revisit of iteration {of}, not measured"}
    _save_invalids(session_dir, doc)


def time_fits(state, time_budget_min: float | None, start_time: float, now: float | None = None) -> tuple[bool, str]:
    """Whether one more iteration fits the time budget: what is left against the next iteration's estimate."""
    from arbor import component
    if not time_budget_min:
        return True, "no time budget"
    now = time.time() if now is None else now
    est, basis = component.estimate(state.lessons)
    left = time_budget_min * 60.0 - (now - start_time)
    line = f"{left / 60:.0f} min left, the next iteration est {est / 60:.0f} min ({basis})"
    return left >= est, line


def _patch(container: str, repo: str, sha: str) -> str:
    """The commit's own change, binary files included. Never through a helper that strips: ``git apply`` needs the
    last newline."""
    r = run_in(container, ["git", "diff", "--binary", f"{sha}~1", sha], workdir=repo, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{sha[:12]} in {repo}: {(r.stderr or r.stdout).strip()[-200:]}")
    out = r.stdout
    return out if not out or out.endswith("\n") else out + "\n"


def _patches(container: str, config: dict, entry: dict) -> list[tuple[str, str]]:
    """``[(repo, patch)]`` for the entry's commits that still change something: the workspace's and each tree's."""
    out = []
    if entry.get("workspace"):
        out.append((config.get("workspace_root", ""), entry["workspace"]))
    known = set(_trees.paths(config))
    out += [(t, sha) for t, sha in (entry.get("trees") or {}).items() if t in known]
    return [(repo, p) for repo, sha in out if (p := _patch(container, repo, sha)).strip()]


def _apply(container: str, repo: str, patch: str, check: bool) -> str | None:
    """None when ``patch`` applies to ``repo``'s working tree (and was applied unless ``check``), else why not."""
    r = run_in(container, ["git", "apply", *(["--check"] if check else []), "-"], workdir=repo, input=patch,
               capture_output=True, text=True)
    return None if r.returncode == 0 else (r.stderr or r.stdout).strip()[-300:] or f"exit {r.returncode}"


def _map_exhausted(session_dir) -> bool:
    from arbor import directions
    doc, err = directions.load(session_dir)
    return bool(doc) and not err and not directions.is_lane(doc) and doc.get("phase") == "active" \
        and directions.exhausted(doc)


def arm(container: str, config: dict, session_dir, iteration: int, state, time_ok: bool) -> dict | None:
    """Arm a revisit for ``iteration`` when the map has closed every direction and ``time_ok``; the armed revisit, or
    None. An armed one the last pass did not close is closed first, from what the session recorded for it, or kept
    for another pass up to ``ATTEMPTS``."""
    session_dir = Path(session_dir)
    old = armed(session_dir)
    if old:
        if old.get("iteration") == iteration and old.get("attempts", 1) < ATTEMPTS:
            old["attempts"] = old.get("attempts", 1) + 1
            _write(session_dir, old)
            return old
        row = next((l for l in reversed(state.lessons or []) if l.get("iteration") == old.get("iteration")), None)
        _close_entry(session_dir, old["of"], {"iteration": old.get("iteration"),
                                              "decision": (row or {}).get("decision") or "not finished",
                                              "reason": (row or {}).get("reason") or "the pass did not finish"},
                     also=old.get("iteration"))
        (session_dir / FILE).unlink(missing_ok=True)
    if not time_ok or not container or not config.get("workspace_root") or not _map_exhausted(session_dir):
        return None
    for entry in pending(session_dir):
        why = None
        if entry.get("intent") == "PROBE":
            why = "a probe's change, not measured against a bar"
        else:
            try:
                patches = _patches(container, config, entry)
            except RuntimeError as exc:
                patches, why = [], f"its commit is not readable: {exc}"
            if not why and not patches:
                why = "its commits hold no change"
            for repo, patch in patches if not why else []:
                problem = _apply(container, repo, patch, check=True)
                if problem:
                    why = f"it does not apply to the kept tree in {repo}: {problem}"
                    break
        if why:
            _close_entry(session_dir, entry["iteration"], {"decision": "not revisited", "reason": why})
            print(f"[arbor] revisit: iteration {entry['iteration']}'s change passed over: {why}")
            continue
        doc = {"iteration": iteration, "of": entry["iteration"], "change": entry.get("change", ""),
               "intent": entry.get("intent", "B"), "reason": entry.get("reason", ""), "attempts": 1,
               "accepted": False, "ts": time.time()}
        _write(session_dir, doc)
        return doc
    return None


def hold(session_dir, iteration: int) -> bool:
    """For ``directions.advance``: True when a revisit is armed for ``iteration``, which then runs before the round
    closes; marks it accepted."""
    doc = armed(session_dir, iteration)
    if not doc:
        return False
    if not doc.get("accepted"):
        doc["accepted"] = True
        _write(session_dir, doc)
    return True


def brief(doc: dict, trees: list[str]) -> str:
    """The text the agent's prompt starts with on a revisit."""
    from arbor.measure import intents
    intent = intents.resolve(doc.get("intent")).name
    where = "the workspace" + (f" and in {', '.join(trees)}" if trees else "")
    return (
        "## This iteration is a revisit\n\n"
        f"Iteration {doc['of']} was NOT MEASURED ({doc.get('reason') or 'no reason recorded'}). Its change, "
        f"\"{doc.get('change', '')}\", left the tree then, and the map has since closed every direction. The change "
        f"is applied again, uncommitted, in {where}.\n\n"
        f"- Measure it as it is, as iteration {doc['of']} would have been measured. Write no other change.\n"
        "- If it was not measured because of the change itself (it does not build, or the server does not start "
        "with it), fix only that, and say so.\n"
        f"- End with `DONE: revisit of iteration {doc['of']}: {doc.get('change', '')}` and `INTENT: {intent}`, "
        f"the intent iteration {doc['of']} declared.\n"
    )


def apply(container: str, config: dict, session_dir, iteration: int) -> str | None:
    """Put the armed change back into the trees for ``iteration``; its brief, or None when none is armed and held.
    RuntimeError when it no longer applies: the entry is closed, the trees cleaned."""
    doc = armed(session_dir, iteration)
    if not doc or not doc.get("accepted"):
        return None
    entry = next((e for e in _invalids(session_dir) if e.get("iteration") == doc["of"]), None)
    if entry is None:
        (Path(session_dir) / FILE).unlink(missing_ok=True)
        return None
    patches = _patches(container, config, entry)
    for repo, patch in patches:
        problem = _apply(container, repo, patch, check=False)
        if problem:
            for r, _ in patches:
                _trees._git(container, r, "checkout", "--", ".", check=False)
                _trees._git(container, r, "clean", "-fdq", check=False)
            _close_entry(session_dir, doc["of"], {"iteration": iteration, "decision": "not revisited",
                                                  "reason": f"it did not apply in {repo}: {problem}"})
            (Path(session_dir) / FILE).unlink(missing_ok=True)
            raise RuntimeError(f"iteration {doc['of']}'s change did not apply in {repo}: {problem}")
    workspace = config.get("workspace_root", "")
    return brief(doc, [r for r, _ in patches if r != workspace])


def close(session_dir, iteration: int, result: dict) -> dict | None:
    """After the pass: the revisit's verdict on its entry; the outcome, or None when this was no revisit."""
    doc = armed(session_dir, iteration)
    if not doc or not doc.get("accepted"):
        return None
    outcome = {"iteration": iteration, "decision": result.get("decision"), "metric": result.get("metric"),
               "reason": str(result.get("reason") or "")[:300]}
    _close_entry(session_dir, doc["of"], outcome, also=iteration)
    (Path(session_dir) / FILE).unlink(missing_ok=True)
    return outcome
