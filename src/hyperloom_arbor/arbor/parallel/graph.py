"""The round's dependency graph, pure: which direction waits for which, and in what order.

A direction's `depends_on` names map IDs on the same list (arbor/directions.py
refuses cycles and unknown IDs when it parses the map). Here the graph is
restricted to the directions that get a lane this round: a dependency with no
lane (closed already, or a rewrite nobody approved) is met from the start, and
the lane's directives say so.

A direction's level is its depth in the graph (0: no dependency with a lane),
and the order lanes are started in is (level, rank, id).
"""

from __future__ import annotations

from arbor import directions


def deps(entries: list[dict], ids: set[str] | None = None) -> dict[str, list[str]]:
    """Each direction's dependencies among ``ids`` (default: every direction listed)."""
    keep = {e["id"] for e in entries} if ids is None else set(ids)
    return {e["id"]: [x for x in e.get("depends_on") or [] if x in keep] for e in entries if e["id"] in keep}


def errors(entries: list[dict]) -> list[str]:
    """Unknown dependencies and cycles, as arbor/directions.py words them."""
    return directions._graph_errors(entries)


def levels(graph: dict[str, list[str]]) -> dict[str, int]:
    """Depth of each node: 0 with no dependency, else one more than its deepest dependency."""
    out: dict[str, int] = {}

    def depth(n: str, seen: tuple = ()) -> int:
        if n in out:
            return out[n]
        if n in seen:
            raise ValueError(f"a dependency cycle through {n}")
        out[n] = 1 + max((depth(m, seen + (n,)) for m in graph.get(n, [])), default=-1)
        return out[n]

    for n in graph:
        depth(n)
    return out


def _rank(e: dict) -> int:
    return e["rank"] if isinstance(e.get("rank"), int) else 999


def order(entries: list[dict], graph: dict[str, list[str]]) -> list[str]:
    """The IDs in ``graph`` by (level, rank, id): the order lanes are queued and integrated in."""
    lv = levels(graph)
    by = {e["id"]: e for e in entries}
    return sorted(graph, key=lambda i: (lv[i], _rank(by.get(i, {})), i))


def met(node: str, graph: dict[str, list[str]], closed: set[str]) -> bool:
    """Every dependency of ``node`` has closed."""
    return all(d in closed for d in graph.get(node, []))
