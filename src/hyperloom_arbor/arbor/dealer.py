"""``arbor run --split workers``: one master, several workers per round,
standalone, in one job on one node.

**A round:**
1. **The master maps** (``arbor master --deal``): the setup, the dossier, the map, which it starts itself. It exits
   ``ordered``, its work order listing the map's open directions.
2. **Dealing:** each free slot (arbor/slots.py) gets an order of a few open directions (``roles.deal``: by rank, as
   their dependencies allow, none a running worker holds). The slot's workspace is reset to the round's base first.
3. **The workers run at once,** each ``arbor worker --order W --config C`` in a session of its own, its output in
   ``<session>/workers/<order>/worker.log``. Its budget ends the integration's reserve before the run's.
4. **A slot frees when its worker exits.** Its report folds into the master's map (``roles.fold``), what it kept is
   taken from its workspace (``slots.addition``), the slot's server is stopped, and the slot is dealt again.
5. **Drain:** nothing is left to deal, and no worker runs. A direction still open was handed back too often, or waits
   on one that was: it is closed ``killed``, saying so, and the next map may bring it back. When the time left is the
   reserve, they stay open: the master ends the campaign instead of a round.
6. **Integration** (arbor/integrate.py): the additions by rank, each A/B'd on this node against the tree so far.
7. **The round's report** answers the master's order (``worker_report.json`` in its session), so its next command takes
   the map on: the update round, then the next round.

Each worker's stretch is a NOTE in the master's session, at an iteration of its own, and each measured addition a KEEP
or a REVERT (arbor/integrate.py): the round's record (``directions._close_round``) counts both.

A SIGTERM goes on to every running command; nothing starts after it, and the round is not integrated.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from arbor import integrate, roles, slots
from arbor.state import CampaignState

POLL_S = 10.0


def _spawn(argv: list[str], log, env: dict):
    return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)


def run(argv: list[str] | None, args, spawn=None) -> int:
    """The rounds, until the master ends the campaign; the exit code. ``spawn(argv, log, env)`` starts a worker's
    command and returns its process (``subprocess.Popen``'s ``poll`` and ``send_signal``)."""
    from arbor import driver, master
    from arbor.cli.iterate import _find_arbor_root

    config_path = str(Path(args.config).resolve())
    try:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        pool = slots.plan(raw)
        session_dir = master.session_of(config_path)
    except (OSError, ValueError) as exc:
        print(f"[arbor] Run: --split workers: {exc}", file=sys.stderr)
        return 2
    started = time.time()
    budget = raw.get("time_budget_minutes")
    deadline = started + 60 * float(budget) if budget else None
    rest = driver._without(list(argv or []), "--split")
    for_worker = driver._without(driver._without(driver._without(rest, "--config"), "--review-gate", False),
                                 "--no-review-gate", False)
    rest += ["--started", repr(started), "--deal"]
    env = {**os.environ, "ARBOR_ROOT": str(_find_arbor_root(Path(config_path)))}
    print(f"[arbor] Run: {len(pool)} worker slots: " + "; ".join(
        f"{s.name} {s.container} cards {','.join(map(str, s.gpu_ids))} port {s.port}" for s in pool))
    while True:
        (session_dir / roles.EXIT_FILE).unlink(missing_ok=True)
        rc = driver._call([sys.executable, "-m", "arbor", "master", *rest])
        done = roles.load(session_dir, roles.EXIT_FILE) or {}
        if rc != 0 or driver._stopping or done.get("outcome") != roles.ORDERED:
            return rc
        order = roles.read_order(done["path"])
        rc = Round(config_path, raw, session_dir, order, pool, deadline, for_worker, env, spawn or _spawn).run()
        if rc != 0 or driver._stopping:
            return rc


class Round:
    """One round's dealing, from the master's order to the round's report."""

    def __init__(self, config_path, raw, session_dir, order, pool, deadline, worker_args, env, spawn):
        self.config_path, self.raw, self.session_dir, self.order = config_path, raw, Path(session_dir), order
        self.pool, self.deadline, self.worker_args, self.env, self.spawn = pool, deadline, worker_args, env, spawn
        self.n = order.get("round")
        self.reserve = slots.reserve_s(raw)
        self.reports: list[dict] = []
        self.additions: list[dict] = []
        self.running: dict[str, dict] = {}

    def left(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.time()

    def held(self) -> list[str]:
        return [did for w in self.running.values() for did in w["order"]["map"]["open"]]

    def run(self) -> int:
        from arbor import driver

        base = slots.export_base(self.raw, integrate.where(self.session_dir, self.n) / "base")
        slots.stop_master(self.raw)                  # its cards are the slots' until the integration
        print(f"[arbor] Dealer: round {self.n}, base {base['commit'][:10]}, "
              f"{len(self.order['map']['open'])} open directions: {', '.join(self.order['map']['open'])}")
        free, timed_out = list(self.pool), False
        previous = signal.signal(signal.SIGTERM, self._forward)
        try:
            while True:
                left = self.left()
                timed_out = left is not None and left <= self.reserve
                while free and not driver._stopping and not timed_out:
                    dealt = roles.deal(self.order, self.reports, self.held(), slots.size(self.raw))
                    if dealt is None:
                        break
                    slot = free.pop(0)
                    try:
                        self.start(slot, dealt, base)
                    except Exception as exc:  # noqa: BLE001 - the slot sits the round out; its order goes back
                        print(f"[arbor] Dealer: {slot.name} could not start order {dealt['id']}: {exc}",
                              file=sys.stderr)
                        self.reports.append(self._unreported(dealt, f"not started: {exc}"))
                if not self.running:
                    break
                time.sleep(POLL_S)
                for name, w in list(self.running.items()):
                    rc = w["proc"].poll()
                    if rc is not None:
                        del self.running[name]
                        self.finish(w, rc)
                        free.append(w["slot"])
        finally:
            signal.signal(signal.SIGTERM, previous)
        if driver._stopping:
            print("[arbor] Dealer: stopped; the round is not integrated")
            return 128 + driver._stopping[0]
        self.close_left(timed_out)
        doc = integrate.run(self.config_path, self.session_dir, self.n,
                            sorted(self.additions, key=lambda a: (a["rank"], a["order"])),
                            self.deadline if self.deadline is not None else time.time() + 365 * 86400,
                            rounds=slots.block(self.raw).get("ab_rounds"))
        if doc.get("restore_failed"):
            print("[arbor] Dealer: the integration could not put the master's tree back; stopping", file=sys.stderr)
            return 1
        self.report(doc, "time" if timed_out else "drained")
        return 0

    def _forward(self, signum, _frame) -> None:
        from arbor import driver

        driver._stopping.append(signum)
        for w in self.running.values():
            w["proc"].send_signal(signum)

    def start(self, slot: slots.Slot, dealt: dict, base: dict) -> None:
        """Deal ``dealt`` to ``slot``: its workspace at the base, its config and order written, its worker started."""
        where = slots.order_dir(self.session_dir, dealt["id"])
        at = slots.prepare(slot, base, f"round {self.n}, order {dealt['id']}, master {base['commit'][:10]}")
        doc = slots.worker_config(self.raw, slot, where, base.get("config") or {})
        limits = dict(dealt.get("limits") or {})
        if limits.get("time_budget_minutes"):
            # The worker's budget is its config's, from the run's start (the order's): it ends the reserve early.
            limits["time_budget_minutes"] = max(1.0, float(limits["time_budget_minutes"]) - self.reserve / 60)
            doc["time_budget_minutes"] = limits["time_budget_minutes"]
        cfg = slots.write_config(where, doc)
        dealt.update(limits=limits, slot={"name": slot.name, "container": slot.container,
                                          "gpu_ids": list(slot.gpu_ids), "port": slot.port, "base": at})
        path = where / roles.ORDER_FILE
        path.write_text(json.dumps(dealt, indent=2) + "\n", encoding="utf-8")
        log = open(where / "worker.log", "ab")
        try:
            proc = self.spawn([sys.executable, "-m", "arbor", "worker", "--order", str(path), "--config", str(cfg),
                               *self.worker_args], log, self.env)
        finally:
            log.close()                 # the child holds its own descriptor
        self.running[slot.name] = {"slot": slot, "order": dealt, "dir": where, "at": at, "proc": proc}
        print(f"[arbor] Dealer: {slot.name} (cards {','.join(map(str, slot.gpu_ids))}) works "
              f"{', '.join(dealt['map']['open'])} (order {dealt['id']})")

    def _unreported(self, dealt: dict, why: str) -> dict:
        """A report for an order whose worker wrote none: its directions come back open."""
        return {"schema": roles.REPORT_SCHEMA, "order": dealt["id"], "ended": why, "iterations": [None, None],
                "keeps": [], "summary": f"no report: {why}",
                "directions": {"closed": [], "open": list(dealt["map"]["open"]), "added": [], "proposals": []}}

    def finish(self, w: dict, rc: int) -> None:
        """A worker exited: its report folded into the master's map, its stretch booked, what it kept taken."""
        slot, dealt = w["slot"], w["order"]
        report = roles.load(w["dir"] / "session", roles.REPORT_FILE)
        if not report or report.get("order") != dealt["id"]:
            report = self._unreported(dealt, f"the worker exited {rc}")
        self.reports.append(report)
        folded = roles.fold(self.session_dir, report, at=self.order["start_iteration"])
        try:
            add = slots.addition(slot, w["at"], self.raw, w["dir"] / "addition.patch")
        except Exception as exc:  # noqa: BLE001 - its KEEPs stay in its session; the master's tree goes on without
            print(f"[arbor] Dealer: {slot.name}: what order {dealt['id']} kept could not be taken: {exc}",
                  file=sys.stderr)
            add = None
        if add is not None:
            ranks = [d.get("rank") for d in dealt["map"]["directions"] if isinstance(d.get("rank"), int)]
            add.update(order=dealt["id"], directions=list(dealt["map"]["open"]), rank=min(ranks or [999]),
                       keeps=[str(k.get("change") or "")[:120] for k in report.get("keeps") or []])
            self.additions.append(add)
        try:
            slots.stop(slot)
        except Exception as exc:  # noqa: BLE001 - the master's A/B stops every server it finds anyway
            print(f"[arbor] Dealer: {slot.name}: its server did not stop cleanly: {exc}", file=sys.stderr)
        self._book(report, dealt, add)
        print(f"[arbor] Dealer: {slot.name} reported order {dealt['id']} ({report.get('ended')}): "
              f"{report.get('summary', '')}" + (f"; closed {', '.join(folded['closed'])}" if folded["closed"] else "")
              + ("; an addition for the integration" if add else ""))

    def _book(self, report: dict, dealt: dict, add: dict | None) -> None:
        """The worker's stretch in the master's session: a NOTE at an iteration of its own, citing its directions."""
        path = self.session_dir / "session.json"
        state = CampaignState.load(path)
        state.iteration += 1
        cites = ", ".join(dealt["map"]["open"])
        state.lessons.append({
            "iteration": state.iteration, "decision": "NOTE", "ts": time.time(),
            "change": f"worker order {dealt['id']} ({cites}): {report.get('summary', '')}"[:600],
            "worker": {"order": dealt["id"], "ended": report.get("ended"), "session": report.get("session"),
                       "keeps": len(report.get("keeps") or []), "addition": bool(add)}})
        state.save(path)

    def close_left(self, timed_out: bool) -> list[str]:
        """The directions the round leaves open in the master's map, closed ``killed`` saying why, unless the time is
        out (the master then ends the campaign)."""
        from arbor import directions

        doc, _ = directions.load(self.session_dir)
        if not doc or timed_out:
            return []
        state = roles.pool(self.order, self.reports)
        closed = []
        for d in doc.get("directions") or []:
            if d.get("status") != directions.OPEN:
                continue
            n = state["returned"].get(d["id"], 0)
            why = (f"handed back open by {n} workers: not settled in round {self.n}" if n else
                   f"waits on {', '.join(d.get('depends_on') or [])}, which no worker settled in round {self.n}"
                   if d.get("depends_on") else f"never dealt in round {self.n}")
            d.update(status="killed", why=why, closed_at=self.order["start_iteration"], unsettled=True)
            closed.append(d["id"])
        if closed:
            directions.save(self.session_dir, doc)
            print(f"[arbor] Dealer: closed what the round left open: {', '.join(closed)}")
        return closed

    def report(self, doc: dict, ended: str) -> dict:
        """The round's report, answering the master's order."""
        state = CampaignState.load(self.session_dir / "session.json")
        kept = len(doc.get("kept") or [])
        report = roles.worker_report(None, state, self.session_dir, self.order, ended,
                                     f"{len(self.reports)} worker stretches, {len(self.additions)} additions, "
                                     f"{kept} kept by the integration")
        report["workers"] = [{"order": r.get("order"), "ended": r.get("ended"), "session": r.get("session"),
                              "keeps": len(r.get("keeps") or []), "summary": r.get("summary")} for r in self.reports]
        report["integration"] = str(integrate.where(self.session_dir, self.n) / "integration.json")
        report["summary"] = roles.summary(report)
        roles.write_report(self.session_dir, report)
        print(f"[arbor] Dealer: round {self.n}: {report['summary']}")
        return report
