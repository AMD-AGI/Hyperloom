"""Arbor's parallel mode: a run of rounds, each a map, a fan-out of lanes, a drain and an integration.

Runner decision 0003 (2026-10-20). A round maps the tree (a profile on a node
and the dossier's update round on the host, in parallel), runs every open
direction as a lane (an Arbor campaign on its own node, its map the direction's
plan), as far as `max_lanes` nodes and the dependencies allow, waits until every
lane has closed, and stacks the lanes' KEEPs onto the base with a same-node A/B
each. The integrated tree is the next round's base. Nothing closed reopens
inside a run; only the next round's dossier may bring it back, on evidence.
It is a campaign's round end (arbor/rounds.py) run faster: there a reached target
or an explored map triggers the A/B and the next round; here every lane closed,
with no dependent work left to schedule, does.

    python3 -m arbor.parallel --help

The controller (`controller.tick`) is stateless between ticks: everything is in
the run's store (`store.py`). Nothing here imports `arbor.mesh` or
`arbor.mesh2`; `trees`, `cluster` and `gatecheck` are ported from mesh3.
"""

SCHEMA_VERSION = 1
