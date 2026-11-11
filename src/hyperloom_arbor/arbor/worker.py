"""The worker's side of a session: a stretch of passes that starts with a work order and ends with a report.

A worker runs the rounds: the passes in which the map is active (``roles.role_of``). Its stretch:

- **Starts** when the next pass is the worker's: the work order goes into the session (``arbor/roles.py``).
- **Ends** when the next pass is the master's (a round end, or the map leaving ``active``) or when the run ends: the
  report goes beside the order, with a summary.

**Where it runs** (``arbor/driver.py``):
- **``arbor run``:** opens and closes the stretches in one process.
- **``python3 -m arbor worker --order``:** one stretch, from an order ``arbor master`` wrote (``resume``).
- **Under a host:** one per share already runs as component mode with ``component.plan``.
"""

from __future__ import annotations

import calendar
import time

from arbor import directions, lifecycle, roles, rounds
from arbor.state import CampaignState


class Stretch:
    """The worker's stretch in progress in a session (``master.Session``), if any."""

    def __init__(self, session, spends: bool = False):
        self.session = session
        self.order: dict | None = None
        self.spends = spends            # a worker of its own session hands a spent share back (``spend``)
        self.spent: list[str] = []

    @property
    def open(self) -> bool:
        return self.order is not None

    def switch(self, role: str, state: CampaignState) -> None:
        """Start a stretch when the next pass is the worker's, and end one when it is the master's."""

        if role == lifecycle.WORKER and not self.open:
            self.start(state)
        elif role == lifecycle.MASTER and self.open:
            self.end(*self.handover(state))

    def start(self, state: CampaignState) -> None:
        s = self.session
        self.order = roles.work_order(s.campaign, state, s.session_dir, s.arbor_root, s.config_path,
                                      limits=s.limits(), target=s.target())
        roles.write_order(s.session_dir, self.order)
        opened = self.order["map"]["open"]
        print(f"[arbor] Worker: round {self.order.get('round')} from iteration {self.order['start_iteration']}"
              + (f", {len(opened)} open directions" if opened else "") + f" ({roles.ORDER_FILE})")

    def resume(self, order: dict) -> None:
        """Take up a stretch from the order the master wrote (``arbor worker --order``)."""

        self.order = order
        opened = order.get("map", {}).get("open") or []
        print(f"[arbor] Worker: round {order.get('round')} from iteration {order['start_iteration']}"
              + (f", {len(opened)} open directions" if opened else "") + f" (order {order.get('id')})")

    def end(self, ended: str, detail: str = "") -> None:
        """Write the stretch's report, if one is open: why it ended, and what it did."""

        if not self.open:
            return
        s = self.session
        report = roles.worker_report(s.campaign, CampaignState.load(s.state_path), s.session_dir, self.order,
                                     ended, detail)
        path = roles.write_report(s.session_dir, report)
        print(f"[arbor] Worker: {report['summary']} ({roles.REPORT_DOC})")
        # The worker's own account: a pack's stretch_end skill (worker-summary) writes it, blocking, within the time
        # the run has left; the report names it.
        before = roles.account_mark(s.session_dir)
        s.fire("stretch_end", report["iterations"][1], order=report.get("order"), ended=ended,
               report=str(path), first=report["iterations"][0])
        if roles.attach_account(s.session_dir, report, before):
            print(f"[arbor] Worker: its account is {roles.ACCOUNT_DOC}")
        self.order = None

    def spend(self, state: CampaignState) -> list[str]:
        """Hand a spent share back (plan 3): in a worker of its own session, when every open direction of its map was
        worked ``map_worked_flag`` times since its last KEEP. Each is closed ``killed``, marked ``spent``, so the report
        settles it and the master deals it to no one else this round; its next map may bring it back. The directions
        closed; none otherwise."""

        if not self.spends or not self.open:
            return []
        s = self.session
        doc = directions.current(s.campaign, state)
        if not doc or doc.get("phase") != "active":
            return []
        flag_at = directions.knob("map_worked_flag", s.campaign, state)
        directions.reconcile(doc, state.lessons, flag_at)
        opened = [d["id"] for d in doc["directions"] if d.get("status") == directions.OPEN]
        gone = directions.spent(doc, flag_at)
        if not opened or gone != opened:
            return []
        for d in doc["directions"]:
            if d["id"] in gone:
                d.update(status="killed", spent=True, closed_at=state.iteration, evidence={"worked": d["worked"]},
                         why=f"spent: worked {len(d['worked'])} times, {flag_at} of them since its last KEEP "
                             f"(iterations {', '.join(map(str, d['worked']))})")
        directions.save(s.campaign.session_root, doc)
        self.spent = gone
        print(f"[arbor] Worker: spent: {', '.join(gone)} worked {flag_at} times each since a KEEP; the share goes back")
        return gone

    def handover(self, state: CampaignState) -> tuple[str, str]:
        """Why the master takes the next pass: a round end, a spent share, or the map leaving ``active``."""

        s = self.session
        pend = rounds.pending(s.session_dir)
        if pend:
            return "round_end", f"round end {pend.get('end')} ({pend.get('trigger')})"
        if self.spent:
            return roles.SPENT, f"spent: {', '.join(self.spent)}"
        doc = directions.current(s.campaign, state) or {}
        if doc.get("phase") == "active" and directions.exhausted(doc):
            return "map", "every direction of the map is closed"
        return "map", f"the map is {doc.get('phase')}"


def iso(epoch: float | None) -> str | None:
    """An epoch time as UTC ISO 8601, for a record."""

    return None if epoch is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def epoch(stamp: str | None) -> float | None:
    """``iso``'s inverse; None for no time or one it did not write."""

    try:
        return float(calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))) if stamp else None
    except (TypeError, ValueError):
        return None
