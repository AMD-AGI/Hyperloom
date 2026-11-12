"""The raw files a record links, and the screen every one of them passes before anything is published.

Raw files go as Arbor and its tools wrote them, byte for byte, with two reductions the manifest names (with the
original's size and SHA-256): aiperf's server metrics, cut to the series Overseer reads, and, for a measured iteration
that is not detailed, its per-request records cut to the fields Overseer reads. A file with a credential-shaped value
(``arbor.checkpoints.secrets_in``) is never published: it is left out, and the manifest's ``omitted`` names it with the
kinds and lines found, never the value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arbor.collect.kb.layout import (
    ARTIFACTS,
    BENCH_FILES,
    DETAIL_ARTIFACTS,
    DETAIL_FILES,
    ITERATION_FILES,
    REQUEST_METADATA,
    REQUEST_METRICS,
    REQUESTS,
    REQUESTS_REDUCED,
    SERVER_METRIC_NAMES,
    SERVER_METRICS,
    SERVER_METRICS_REDUCED,
    Link,
    Stage,
    as_dict,
    sha256_file,
)
from arbor.collect.kb.session import Session

#: The most hits a manifest names for one omitted file.
MAX_HITS = 20


# ---------------------------------------------------------------- reductions
def reduced_server_metrics(data: dict) -> bytes | None:
    """The server metrics Overseer reads: every series of each kept metric, the phases and the warmup section."""

    if not data:
        return None
    out: dict[str, Any] = {
        "reduced": SERVER_METRICS_REDUCED,
        "summary": {"phase_time_ranges": as_dict(data.get("summary")).get("phase_time_ranges")},
    }
    for part in ("warmup_metrics", "metrics"):
        if isinstance(data.get(part), dict):
            out[part] = {k: v for k, v in data[part].items() if k in SERVER_METRIC_NAMES}
    return (json.dumps(out, sort_keys=True, ensure_ascii=False) + "\n").encode()


def reduced_requests(path: Path) -> bytes:
    """Each request's fields Overseer reads, one JSON object per line as aiperf writes them."""

    lines = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            meta, metrics = as_dict(as_dict(row).get("metadata")), as_dict(as_dict(row).get("metrics"))
            lines.append(json.dumps({
                "metadata": {k: meta[k] for k in REQUEST_METADATA if k in meta},
                "metrics": {k: {"value": as_dict(metrics[k]).get("value")} for k in REQUEST_METRICS if k in metrics},
            }, separators=(",", ":")))
    return "".join(line + "\n" for line in lines).encode()


def _reduced_link(stage: Stage, rel: str, original: Path, data: bytes, how: str) -> Link:
    return Link(stage.write(rel, data), rel, extra={
        "reduced": how, "original_bytes": original.stat().st_size, "original_sha256": sha256_file(original)})


# ---------------------------------------------------------------- one bench root
def _artifact_links(s: Session, it: int, root: str, stage: Stage, detailed: bool) -> list[Link]:
    artifacts = f"{root}/{ARTIFACTS.get(root.split('/')[0], 'aiperf_artifacts')}"
    links = s.iteration_links(it, tuple(f"{artifacts}/{name}" for name in DETAIL_ARTIFACTS)) if detailed else []
    requests = s.path(it, f"{artifacts}/{REQUESTS}")
    if not detailed and requests.is_file():
        links.append(_reduced_link(stage, f"iter_{it}/{artifacts}/{REQUESTS}", requests, reduced_requests(requests),
                                   REQUESTS_REDUCED))
    metrics = s.path(it, f"{artifacts}/{SERVER_METRICS}")
    reduced = reduced_server_metrics(s.iteration_json(it, f"{artifacts}/{SERVER_METRICS}")) if detailed else None
    if reduced is not None:
        links.append(_reduced_link(stage, f"iter_{it}/{artifacts}/{SERVER_METRICS}", metrics, reduced,
                                   SERVER_METRICS_REDUCED))
    return links


def bench_links(s: Session, it: int, stage: Stage, detailed: bool) -> list[Link]:
    """Each bench root's files: its own on every iteration, aiperf's artifacts as ``detailed`` says."""

    links: list[Link] = []
    for root in s.bench_roots(it):
        links += s.iteration_links(it, tuple(f"{root}/{name}" for name in BENCH_FILES))
        if detailed:
            links += s.iteration_links(it, tuple(f"{root}/{name}" for name in DETAIL_FILES))
        links += _artifact_links(s, it, root, stage, detailed)
    return links


# ---------------------------------------------------------------- files that live on after their iteration
def _eval_links(s: Session, it: int) -> list[Link]:
    """Files under eval/ that the iteration's ledger records name, modified inside its window."""

    text = " ".join(json.dumps(item["row"], ensure_ascii=False) for item in s.own_ledger.get(it, []))
    at = s.owners.gated.index(it)
    start = s.owners.bounds[at - 1] if at > 0 else float("-inf")
    return s.eval_files(text, (start, s.owners.bounds[at]))


def frozen(s: Session, it: int, stage: Stage, kind: str, live: list[Link]) -> list[Link]:
    """``live`` copied into the export, and, once the iteration is final, the same copies on every later export.

    Eval files and transcripts live on after their iteration: a resumed campaign can rewrite an eval file or append
    to a transcript. A final record must not change, so the first export of a final iteration copies them and lists
    the copies (``iter_N/<kind>.json``), and every later export to the same directory links that list as it is.
    """

    listing = stage.root / "files" / f"iter_{it}" / f"{kind}.json"
    final = s.owners.final(it)
    if final and listing.is_file():
        kept = json.loads(listing.read_text(encoding="utf-8"))
        links = [Link(stage.root / "files" / e["staged"], e["path"], e["as"]) for e in kept]
        if all(link.source.is_file() for link in links):
            return links
    links, entries = [], []
    for link in live:
        staged = f"iter_{it}/{kind}/{link.path}"
        links.append(Link(stage.write(staged, link.source.read_bytes()), link.path, link.kind))
        entries.append({"path": link.path, "as": link.kind, "staged": staged})
    if final:
        stage.write_json(f"iter_{it}/{kind}.json", entries)
    return links


def iteration_links(s: Session, it: int, stage: Stage, detailed: bool) -> list[Link]:
    """Every file the iteration's record links, its manifest aside, before the screen."""

    rel = f"iter_{it}"
    links = s.iteration_links(it, ITERATION_FILES)
    agg = s.agg(it)
    if agg is not None:
        links.append(Link(agg, f"{rel}/{agg.name}"))
    links += bench_links(s, it, stage, detailed)
    events = sorted(s.own_events.get(it, []))
    if events:
        staged = stage.write(f"{rel}/events.jsonl", "".join(line + "\n" for _, line in events).encode())
        links.append(Link(staged, "reports/trace/events.jsonl", "trace-slice"))
    segment = s.log.get(it)
    if segment is not None:
        links.append(Link(stage.write(f"{rel}/campaign-log.txt", segment.text.encode()), f"logs/{rel}.log",
                          "campaign-log-segment"))
    links += s.keeps(it)
    links += frozen(s, it, stage, "transcripts", s.transcripts(it))
    return links + frozen(s, it, stage, "eval", _eval_links(s, it))


# ---------------------------------------------------------------- the screen
class Screen:
    """Arbor's credential detector over every text file a record links; each file is read once per export."""

    def __init__(self) -> None:
        from arbor.checkpoints import secrets_in

        self._detect = secrets_in
        self._seen: dict[Path, list[str]] = {}
        self.flagged: dict[str, list[str]] = {}  # every file left out, by its place in the session

    def hits(self, path: Path) -> list[str]:
        """The kinds and lines of the credential-shaped values ``path`` holds; ``[]`` for a binary file."""

        if path not in self._seen:
            data = path.read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = None if b"\0" in data[:8192] else data.decode("utf-8", errors="replace")
            self._seen[path] = self._detect(text) if text is not None else []
        return self._seen[path]

    def keep(self, links: list[Link], omitted: list[dict]) -> list[Link]:
        """The links whose file holds no credential-shaped value; each other goes to ``omitted``, never published."""

        kept = []
        for link in links:
            found = self.hits(link.source)
            if found:
                self.flagged[link.path] = found[:MAX_HITS]
                omitted.append({"path": link.path, "as": link.kind, "why": "credential-shaped values",
                                "found": found[:MAX_HITS], "found_total": len(found)})
            else:
                kept.append(link)
        return kept
