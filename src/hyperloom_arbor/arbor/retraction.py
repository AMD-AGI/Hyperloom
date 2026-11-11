"""A KEEP the campaign's own agent later finds invalid, taken back.

The agent says so in its ledger: a ``FACT`` or ``DEAD_END`` whose ``retracts`` names the KEEP's
iteration (arbor/ledger.py). After the iteration that wrote it, :func:`after_iteration`:

1. reverts that iteration's ``iter-N:`` commits in the workspace, as a REVERT verdict reverts its
   own, so the tree's history says ``Revert "iter-N: ..."`` (:func:`revert`);
2. in component mode, writes the reverted tree's KEEP snapshot, with ``retracts: {iteration, keep,
   finding}`` in its keep.json (arbor/component.py), before the row moves the KEEP count, as every
   KEEP's snapshot is;
3. records the row (``CampaignState.record_retraction``): a KEEP a host counts, claiming no metric,
   with the bars put back where they stood before the retracted KEEP (:func:`bar_before`), since the
   tree no longer holds what that KEEP measured.

No gate measures the reverted tree: a retraction is slower by construction. A host tests it for
errors and consistency, not for speed. A revert that does not apply, such as
when a later KEEP builds on the change, leaves the tree as it was and writes :data:`CONFLICT_FILE`.
The loop then ends the run with :data:`CONFLICT_REASON`, the file's first line as its detail
(arbor/master.py). Remove the file to run the campaign on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from arbor import component, ledger, rebaseline

CONFLICT_FILE = "RETRACTION_CONFLICT.md"
CONFLICT_REASON = "retraction_conflict"
# The ledger types that can retract: settled ones, not beliefs (ASSUMPTION) or gaps (UNKNOWN).
RETRACTING_TYPES = ("FACT", "DEAD_END")
# Rows that set the bar outright rather than ratchet it (arbor/state.py, arbor/rebaseline.py).
_SETS_BAR = ("BASELINE", "REBASELINE", "ROUND_AB")


class Conflict(RuntimeError):
    """A retraction that cannot be made: the tree is as it was before the attempt."""


def _iteration(value) -> int | None:
    """``retracts`` as an iteration number: an int, or a string of digits an agent may write."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _standing(lessons) -> tuple[set[int], set[int]]:
    """The iterations of the KEEPs on record, and of those already retracted."""
    rows = [r for r in lessons or [] if isinstance(r, dict)]
    kept = {r["iteration"] for r in rows
            if r.get("decision") == "KEEP" and isinstance(r.get("iteration"), int) and not r.get("retracts")}
    retracted = {r["retracts"].get("iteration") for r in rows if isinstance(r.get("retracts"), dict)}
    return kept, retracted


def wanted(findings, lessons) -> list[tuple[ledger.Finding, int]]:
    """The records among ``findings`` that retract a KEEP still standing, each with its iteration, once."""
    kept, retracted = _standing(lessons)
    out: list[tuple[ledger.Finding, int]] = []
    for f in findings:
        if f.retracts is None:
            continue
        n = _iteration(f.retracts)
        if f.type not in RETRACTING_TYPES or n is None:
            print(f"[arbor] Retraction: {f.id} names {f.retracts!r} in retracts, which a {f.type} cannot "
                  f"retract — only a FACT or DEAD_END naming a KEEP's iteration does", file=sys.stderr, flush=True)
            continue
        if n in retracted or any(n == m for _, m in out):
            continue
        if n not in kept:
            print(f"[arbor] Retraction: {f.id} retracts iteration {n}, which is no KEEP on record — nothing "
                  f"to take back", file=sys.stderr, flush=True)
            continue
        out.append((f, n))
    return out


def keep_number(lessons, iteration: int) -> int:
    """The retracted KEEP's number in the KEEP count a host reads: the KEEP rows up to its own."""
    n = 0
    for row in lessons or []:
        if isinstance(row, dict) and row.get("decision") == "KEEP":
            n += 1
            if row.get("iteration") == iteration and not row.get("retracts"):
                return n
    return n


def _better(best, value, direction: str):
    if value is None:
        return best
    if best is None:
        return value
    return min(best, value) if direction == "min" else max(best, value)


def bar_before(lessons, iteration: int, direction: str = "max",
               secondary_direction: str = "max") -> tuple[float | None, float | None]:
    """The (primary, secondary) bars as they stood before the KEEP of ``iteration``, on the scale in force now.

    The rows before that KEEP are replayed as CampaignState ratchets them: a BASELINE, REBASELINE or
    ROUND_AB sets the bar, a KEEP that is no frontier trade and no retraction moves it, and a KEEP
    retracted since does not count. Each reading goes onto the campaign's first node's scale first
    (arbor/rebaseline.py), so the nodes a rolling campaign moved across compare. ``(None, None)``
    when the KEEP is not on record, or a round end's A/B came after it: that A/B measured a tree
    holding the retracted change, and started the scale again.
    """
    rows = [r for r in lessons or [] if isinstance(r, dict)]
    factors = rebaseline.scales(rows)
    at = next((i for i, r in enumerate(rows) if r.get("decision") == "KEEP"
               and r.get("iteration") == iteration and not r.get("retracts")), None)
    if at is None or any(r.get("decision") == "ROUND_AB" for r in rows[at:]):
        return None, None
    _, retracted = _standing(rows)
    bar = secondary = None
    for row, (f, f2) in zip(rows[:at], factors[:at]):
        value = rebaseline.on_first_scale(row.get("metric"), f)
        value2 = rebaseline.on_first_scale(row.get("secondary"), f2)
        decision = row.get("decision")
        if decision in _SETS_BAR:
            bar = value if value is not None else bar
            secondary = value2 if value2 is not None else secondary
        elif (decision == "KEEP" and not row.get("frontier") and not row.get("retracts")
              and row.get("iteration") not in retracted):
            bar = _better(bar, value, direction)
            secondary = _better(secondary, value2, secondary_direction)
    now, now2 = rebaseline.current(rows)
    return (None if bar is None else bar / now, None if secondary is None else secondary / now2)


def revert(container: str, workspace: str, iteration: int, root: str) -> list[str]:
    """Revert the ``iter-N:`` commits of ``iteration`` that ``root..HEAD`` holds and nothing reverted yet,
    newest first, one revert commit each; their shas. Raises :class:`Conflict`, with the tree as it was,
    when there is none or one does not apply."""
    from arbor.execute.workspace import git
    try:
        subjects = [line.split("\t", 1) for line in git(
            container, workspace, "log", "--format=%H%x09%s", f"{root}..HEAD").splitlines() if "\t" in line]
    except RuntimeError as e:
        raise Conflict(f"the workspace's history could not be read: {e}") from e
    reverted = {s[len('Revert "'):-1] for _, s in subjects if s.startswith('Revert "') and s.endswith('"')}
    shas = [sha for sha, s in subjects
            if (m := component._ITER_SUBJECT.match(s)) and int(m.group(1)) == iteration and s not in reverted]
    if not shas:
        raise Conflict(f"the tree holds no commit of iteration {iteration} to revert")
    try:
        git(container, workspace, "revert", "--no-edit", *shas)
    except RuntimeError as e:
        try:
            git(container, workspace, "revert", "--abort")
        except RuntimeError:
            pass
        raise Conflict(f"git revert of iteration {iteration} does not apply: {e}") from e
    return shas


def write_conflict(session_dir, retracts: dict, why: str) -> Path:
    """CONFLICT_FILE: what could not be retracted and why; its first line is the exit's detail."""
    path = Path(session_dir) / CONFLICT_FILE
    path.write_text(
        f"{retracts['finding']}: KEEP {retracts['keep']} (iteration {retracts['iteration']}) could not be "
        f"retracted: {why}\n\n"
        f"The tree is as it was. The run ends with {CONFLICT_REASON}; remove this file to run it on.\n")
    print(f"[arbor] Retraction: CONFLICT — {path.read_text().splitlines()[0]}", file=sys.stderr, flush=True)
    return path


def after_iteration(state, campaign, *, container: str, workspace: str, session_dir,
                    findings, iteration: int, save: Callable[[], None]) -> list[dict]:
    """Take back each KEEP that ``findings``, the iteration's ledger records, retract (the module's steps);
    the ``retracts`` of each one taken back. A conflict writes CONFLICT_FILE and stops there."""
    from arbor.execute.workspace import base_commit, save_landed_patch
    done: list[dict] = []
    for finding, n in wanted(findings, state.lessons):
        retracts = {"iteration": n, "keep": keep_number(state.lessons, n), "finding": finding.id}
        try:
            revert(container, workspace, n, base_commit(container, workspace, campaign.config))
        except (Conflict, RuntimeError) as e:
            write_conflict(session_dir, retracts, str(e))
            break
        # The same iteration's commits in the other editable trees (arbor/execute/trees.py), from each one's baseline.
        from arbor.execute import trees as _trees
        for tree, base in _trees.baselines(session_dir).items():
            if tree in _trees.paths(campaign.config):
                try:
                    revert(container, tree, n, base)
                except (Conflict, RuntimeError) as e:
                    print(f"[arbor] WARNING: retraction of iteration {n} in {tree}: {e}", file=sys.stderr)
        change = f"retract iteration {n} (KEEP {retracts['keep']}): {finding.id}"
        try:
            save_landed_patch(container, workspace, str(Path(session_dir) / "landed.patch"), campaign.config)
        except Exception as e:   # bookkeeping: the revert stands
            print(f"[arbor] WARNING: landed.patch not written after the retraction: {e}", file=sys.stderr)
        if component.is_component(campaign.config):
            try:
                component.snapshot_keep(
                    campaign.config, session_dir, container=container, workspace=workspace,
                    number=component.keep_count(state.lessons) + 1, iteration=iteration, metric=None,
                    change=change, intent="FIX-CORRECTNESS", retracts=retracts)
            except Exception as e:   # as for any KEEP: the next snapshot's `previous` shows the gap
                print(f"[arbor] WARNING: the retraction's snapshot not written: {e}", file=sys.stderr, flush=True)
        bar, secondary = bar_before(state.lessons, n, state.metric_direction or "max",
                                    state.secondary_direction or "max")
        state.record_retraction(iteration, retracts, change, bar, secondary)
        save()
        print(f"[arbor] Retraction: {change} — the bar is back to {bar if bar is not None else 'where it was'}",
              file=sys.stderr, flush=True)
        done.append(retracts)
    return done
