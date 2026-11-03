"""A run's directory on shared storage, and the only code that knows its layout.

    <store>/<run_id>/
      run.json            the config as initialized, its sha256, the gates' sha256, the Arbor sha
      gates.json          the operator's gates, pinned (ab_model_m.sbatch re-checks the sha on the node)
      approvals.json      rewrites the operator let into a round (`arbor.parallel approve`)
      HOST                {host, pid, heartbeat}: the one host whose ticks may write
      state.json          the run's phase and round (controller.py)
      events.jsonl        everything that happened
      metrics.jsonl       one row per tick
      STOP, PAUSE         operator switches; ATTENTION.md what a person should look at; STOP.md why it stopped
      trees/              the git store behind the bases (trees.py)
      bases/base_NNN/     {base.diff, base.blobs, flags.json, workspace_config.json, base.json}; current.json
      rounds/rNN/
        map/profile/attempt-K/{bar.json, profile/, DONE}
        map/dossier/{ws/, session/, config.json, jobs/<A|B>/attempt-K/{request.json, DONE}}
        directions.json   the round's map (scope "run"), each direction's lane and outcome
        plans/<file>.md   pinned plans
        lanes/<lane>/{lane.json, config.json, arbor/ (worktree), start/, final/}
        integrate/{integration.json, ab/<k>/attempt-K/{runs.jsonl, DONE}}
        ROUND.md, round.json

Every write is atomic: a temp file in the same directory, then os.replace. A
reader therefore sees the old file or the new one, never half of either, and
a tick killed between two writes leaves a state the next tick can continue
from (each step is recorded before it is taken, and re-done when found
unfinished).
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

HOST_STALE_S = 15 * 60


class HostLocked(RuntimeError):
    pass


# -- atomic files ----------------------------------------------------------------------


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=1, sort_keys=False, default=str) + "\n")


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fh:
        cut = fh.tell() > 0 and _last_byte(path) != b"\n"      # a crash mid-append: start a new line
        fh.write((("\n" if cut else "") + json.dumps(row, default=str) + "\n").encode("utf-8"))


def _last_byte(path: Path) -> bytes:
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1)


def read_jsonl(path: str | Path) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue        # a line cut by a crash mid-append
    return out


# -- the run's directory ---------------------------------------------------------------


class Store:
    def __init__(self, root: str | Path, clock=time.time, host: str | None = None) -> None:
        self.root = Path(root)
        self.clock = clock
        self.host = host or socket.gethostname()

    # files of the run
    @property
    def run_json(self) -> Path: return self.root / "run.json"
    @property
    def gates_json(self) -> Path: return self.root / "gates.json"
    @property
    def approvals_json(self) -> Path: return self.root / "approvals.json"
    @property
    def host_file(self) -> Path: return self.root / "HOST"
    @property
    def state_json(self) -> Path: return self.root / "state.json"
    @property
    def events(self) -> Path: return self.root / "events.jsonl"
    @property
    def metrics(self) -> Path: return self.root / "metrics.jsonl"
    @property
    def stop_file(self) -> Path: return self.root / "STOP"
    @property
    def pause_file(self) -> Path: return self.root / "PAUSE"
    @property
    def attention(self) -> Path: return self.root / "ATTENTION.md"
    @property
    def trees(self) -> Path: return self.root / "trees"
    @property
    def bases(self) -> Path: return self.root / "bases"

    def base_dir(self, index: int) -> Path:
        return self.bases / f"base_{index:03d}"

    def round_dir(self, n: int) -> Path:
        return self.root / "rounds" / f"r{n:02d}"

    def lanes_dir(self, n: int) -> Path:
        return self.round_dir(n) / "lanes"

    def lane_dir(self, n: int, lane: str) -> Path:
        return self.lanes_dir(n) / lane

    # reading and writing
    def read(self, path: Path, default: Any = None) -> Any:
        return read_json(path, default)

    def write(self, path: Path, value: Any) -> None:
        atomic_write_json(path, value)

    def event(self, kind: str, **fields: Any) -> dict:
        row = {"ts": self.clock(), "kind": kind, **fields}
        append_jsonl(self.events, row)
        return row

    def read_events(self, prefix: str = "") -> list[dict]:
        return [r for r in read_jsonl(self.events) if str(r.get("kind", "")).startswith(prefix)]

    # the run
    def meta(self) -> dict:
        return read_json(self.run_json, {}) or {}

    def state(self) -> dict:
        return read_json(self.state_json) or {"phase": "INIT", "round": 0, "since": self.clock()}

    def set_state(self, st: dict) -> None:
        atomic_write_json(self.state_json, st)

    def approvals(self) -> dict:
        return read_json(self.approvals_json, {}) or {}

    def current_base(self) -> dict | None:
        return read_json(self.bases / "current.json")

    def gates(self) -> dict:
        """The pinned gates, after checking their sha256 against run.json."""
        from arbor.parallel.config import ConfigError, gates_sha
        gates = read_json(self.gates_json)
        if gates is None or gates_sha(gates) != self.meta().get("gates_sha256"):
            raise ConfigError("the pinned gates.json no longer matches its sha256; restore it "
                              "(a run's gates are not edited)")
        return gates

    # lanes
    def lanes(self, n: int) -> list[str]:
        d = self.lanes_dir(n)
        if not d.exists():
            return []
        return sorted((p.name for p in d.iterdir() if (p / "lane.json").exists()), key=_lane_key)

    def lane(self, n: int, lane: str) -> dict | None:
        return read_json(self.lane_dir(n, lane) / "lane.json")

    def set_lane(self, n: int, lane: str, rec: dict) -> None:
        atomic_write_json(self.lane_dir(n, lane) / "lane.json", rec)

    # switches
    def stopped(self) -> bool: return self.stop_file.exists()
    def paused(self) -> bool: return self.pause_file.exists()

    # the host lock
    def take_host(self, *, takeover: bool = False) -> dict:
        """Claim the run for this host and beat its heart; HostLocked when another host holds it.

        Another host's claim is taken over only when its heartbeat is over 15
        minutes old and the operator asked (`tick --takeover`): two controllers
        on one store corrupt it, and a stale heartbeat alone may be a slow NFS.
        """
        now = self.clock()
        cur = read_json(self.host_file) or {}
        if cur and cur.get("host") != self.host:
            age = now - float(cur.get("heartbeat") or 0)
            if not (takeover and age > HOST_STALE_S):
                raise HostLocked(
                    f"{cur.get('host')} holds this run (heartbeat {age / 60:.0f} min ago); "
                    + ("ticks from here are refused" if age <= HOST_STALE_S
                       else "take it over with --takeover once that host is known to be dead"))
            self.event("host.takeover", was=cur.get("host"), heartbeat_age_s=round(age))
        rec = {"host": self.host, "pid": os.getpid(), "heartbeat": now}
        atomic_write_json(self.host_file, rec)
        return rec


def _lane_key(name: str) -> tuple:
    head, _, tail = name.rpartition("-l")
    return (head, int(tail) if tail.isdigit() else 0, name)


def init_run(cfg, *, clock=time.time, host: str | None = None) -> Store:
    """Create the run's store, pinning the config, the gates and the Arbor commit. Idempotent for the
    same config; a different config under the same run_id is refused."""
    import subprocess
    from arbor.parallel.config import ConfigError, read_gates, ref_sha, sha256_text
    store = Store(cfg.root, clock, host)
    text = json.dumps(cfg.raw, sort_keys=True)
    meta = store.meta()
    if meta:
        if meta.get("config_sha256") != sha256_text(text):
            raise ConfigError(f"run {cfg.run_id} exists with a different config; a run's config is not "
                              "edited: use a new run_id")
        return store
    gates, gsha = read_gates(cfg.gates)
    p = subprocess.run(["git", "-C", str(cfg.arbor_root), "rev-parse", "--verify",
                        f"{ref_sha(cfg.arbor_ref)}^{{commit}}"], capture_output=True, text=True)
    if p.returncode != 0:
        raise ConfigError(f"arbor_ref {cfg.arbor_ref!r} is not a commit of {cfg.arbor_root}: "
                          f"{p.stderr.strip()[:200]}")
    for sub in ("rounds", "bases", "trees"):
        (store.root / sub).mkdir(parents=True, exist_ok=True)
    atomic_write_text(store.gates_json, cfg.gates.read_text(encoding="utf-8"))
    if not store.approvals_json.exists():
        atomic_write_json(store.approvals_json, {})
    atomic_write_json(store.run_json, {
        "schema": 1, "run_id": cfg.run_id, "config": cfg.raw, "config_sha256": sha256_text(text),
        "gates_sha256": gsha, "arbor_sha": p.stdout.strip(), "created": clock()})
    store.set_state({"phase": "INIT", "round": 0, "since": clock()})
    store.event("run.init", gates_sha256=gsha, arbor_sha=p.stdout.strip(), max_lanes=cfg.max_lanes)
    return store
