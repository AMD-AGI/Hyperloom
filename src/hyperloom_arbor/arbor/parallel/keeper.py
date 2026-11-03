"""Keep the controller's run loop alive, and write down why it died.

Ported from arbor/mesh2/keeper.py (<fork>/mesh3-confirm a691edc), with the
mesh's storage helpers replaced by store.py's and the supervisor being
`python3 -m arbor.parallel run`. Its history, as mesh2 wrote it:

The v9 supervisor died on a Cluster B login node at 14:45:22 UTC and nothing
noticed for six hours. Its log held a startup warning and no traceback, which
is what an external kill looks like, and there were two suspects -- the Login
Node Guardian's per-process memory limit and the per-user cgroup, which had
throttled 528,119 times -- with no record that could tell them apart.

So the keeper does two things. It restarts the supervisor, with backoff,
whenever it exits. And before restarting it records what distinguishes one
cause from another: the exit status and signal, how long it ran, the memory of
its whole process tree at the last sample and at peak, the cgroup's memory
counters before and after, and the tail of its log. A second silent death then
arrives with its own diagnosis.

It also kills a supervisor whose heartbeat has gone stale. Every tick rewrites
the run's HOST file, paused or not, so a pause is not a hang; only a loop that
has stopped ticking stops writing it. A run that has STOPPED writes the
keeper's STOP file as its loop exits, so nothing restarts it.

    python3 -m arbor.parallel.keeper run    --config <run.json>
    python3 -m arbor.parallel.keeper status --config <run.json>
    touch <store>/<run_id>/keeper/STOP      # stop cleanly; nothing is restarted
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from arbor.parallel.store import append_jsonl, atomic_write_text

DEFAULT_BACKOFF = (10.0, 30.0, 120.0, 300.0, 600.0)
RSS_LOG_LIMIT_BYTES = 4 * 1024 * 1024


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Settings:
    command: list[str]
    state_dir: Path
    log_path: Path
    heartbeat_path: Path | None = None
    heartbeat_seconds: float = 900.0
    sample_seconds: float = 30.0
    tick_seconds: float = 1.0
    backoff: tuple[float, ...] = DEFAULT_BACKOFF
    healthy_seconds: float = 3600.0
    term_grace_seconds: float = 60.0
    cwd: str | None = None
    env: dict[str, str] | None = None
    guard_pattern: tuple[str, ...] = field(default_factory=tuple)


# ---- process and cgroup facts --------------------------------------------


def _stat(pid: int) -> tuple[int, str] | None:
    """(ppid, comm) from /proc/<pid>/stat. comm can contain spaces and ')'."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    head, _, tail = raw.rpartition(")")
    comm = head.partition("(")[2]
    fields = tail.split()
    try:
        return int(fields[1]), comm
    except (IndexError, ValueError):
        return None


def _rss_kb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def process_tree(root_pid: int) -> list[dict[str, Any]]:
    """The process and all its descendants, each with its resident memory."""

    proc = Path("/proc")
    if not proc.exists():
        return []
    parents: dict[int, list[int]] = {}
    names: dict[int, str] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        info = _stat(int(entry.name))
        if info is None:
            continue
        parents.setdefault(info[0], []).append(int(entry.name))
        names[int(entry.name)] = info[1]
    found = []
    frontier = [root_pid]
    while frontier:
        pid = frontier.pop()
        if pid in names or pid == root_pid:
            found.append({"pid": pid, "comm": names.get(pid, "?"), "rss_kb": _rss_kb(pid)})
        frontier.extend(parents.get(pid, []))
    return found


def cgroup_memory() -> dict[str, dict[str, Any]]:
    """Memory counters for this process's cgroup and its ancestors (cgroup v2)."""

    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return {}
    relative = next((line.split("::", 1)[1] for line in lines if line.startswith("0::")), None)
    if relative is None:
        return {}
    found: dict[str, dict[str, Any]] = {}
    path = Path("/sys/fs/cgroup") / relative.lstrip("/")
    for _ in range(4):
        events = path / "memory.events"
        if events.exists():
            record: dict[str, Any] = {}
            try:
                for line in events.read_text().splitlines():
                    key, _, value = line.partition(" ")
                    record[key] = int(value)
                for name in ("memory.current", "memory.max", "memory.high"):
                    target = path / name
                    if target.exists():
                        record[name] = target.read_text().strip()
            except (OSError, ValueError):
                pass
            found[str(path)] = record
        if path == Path("/sys/fs/cgroup"):
            break
        path = path.parent
    return found


def _delta(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, Any]:
    changed: dict[str, Any] = {}
    for path, counters in after.items():
        old = before.get(path, {})
        diff = {
            k: v - old.get(k, 0)
            for k, v in counters.items()
            if isinstance(v, int) and v != old.get(k, 0)
        }
        if diff:
            changed[path] = diff
    return changed


def _tail(path: Path, lines: int = 40) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 16384))
            data = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return data.splitlines()[-lines:]


def already_running(patterns: tuple[str, ...], *, exclude: int) -> int | None:
    """A process whose command line contains every pattern, other than ours."""

    if not patterns or not Path("/proc").exists():
        return None
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == exclude:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if all(p in cmdline for p in patterns):
            return int(entry.name)
    return None


# ---- the keeper ------------------------------------------------------------


class Keeper:
    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.s = settings
        self.clock = clock
        self.sleep = sleep
        self.s.state_dir.mkdir(parents=True, exist_ok=True)
        self.exits_path = self.s.state_dir / "exits.jsonl"
        self.rss_path = self.s.state_dir / "rss.jsonl"
        self.events_path = self.s.state_dir / "events.jsonl"
        self.stop_file = self.s.state_dir / "STOP"
        self._signalled = False
        self.child: subprocess.Popen | None = None

    def request_stop(self, *_: Any) -> None:
        self._signalled = True

    def _stopping(self) -> bool:
        return self._signalled or self.stop_file.exists()

    def _event(self, kind: str, **fields: Any) -> None:
        append_jsonl(self.events_path, {"at": utc_now(), "kind": kind, **fields})

    def _sample(self, pid: int) -> dict[str, Any]:
        tree = process_tree(pid)
        total = sum(p["rss_kb"] for p in tree)
        biggest = max(tree, key=lambda p: p["rss_kb"], default=None)
        sample = {
            "at": utc_now(),
            "pid": pid,
            "processes": len(tree),
            "tree_rss_mb": round(total / 1024, 1),
            "largest": biggest,
        }
        if self.rss_path.exists() and self.rss_path.stat().st_size > RSS_LOG_LIMIT_BYTES:
            kept = self.rss_path.read_text(encoding="utf-8").splitlines()[-2000:]
            atomic_write_text(self.rss_path, "\n".join(kept) + "\n")
        append_jsonl(self.rss_path, sample)
        return sample

    def _heartbeat_age(self) -> float | None:
        if self.s.heartbeat_path is None:
            return None
        try:
            return time.time() - self.s.heartbeat_path.stat().st_mtime
        except OSError:
            return None

    def _terminate(self, child: subprocess.Popen) -> None:
        try:
            child.terminate()
        except OSError:
            return
        deadline = self.clock() + self.s.term_grace_seconds
        while child.poll() is None and self.clock() < deadline:
            self.sleep(min(self.s.tick_seconds, 1.0))
        if child.poll() is None:
            try:
                child.kill()
            except OSError:
                pass
            child.wait()

    def run_once(self) -> dict[str, Any]:
        """Start the supervisor, watch it until it exits, and record why."""

        env = dict(os.environ if self.s.env is None else self.s.env)
        with self.s.log_path.open("ab") as log:
            log.write(f"\n[keeper] {utc_now()} starting: {' '.join(self.s.command)}\n".encode())
            log.flush()
            child = subprocess.Popen(
                self.s.command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=self.s.cwd,
                env=env,
            )
        self.child = child
        atomic_write_text(self.s.state_dir / "supervisor.pid", f"{child.pid}\n")
        self._event("started", pid=child.pid)
        started = self.clock()
        cgroup_before = cgroup_memory()
        last: dict[str, Any] | None = None
        peak: dict[str, Any] | None = None
        next_sample = started
        hung = False
        stopped = False
        while child.poll() is None:
            now = self.clock()
            if now >= next_sample:
                last = self._sample(child.pid)
                if peak is None or last["tree_rss_mb"] > peak["tree_rss_mb"]:
                    peak = last
                next_sample = now + self.s.sample_seconds
            if self._stopping():
                stopped = True
                self._terminate(child)
                break
            age = self._heartbeat_age()
            if (
                age is not None
                and age > self.s.heartbeat_seconds
                and now - started > self.s.heartbeat_seconds
            ):
                hung = True
                self._event("hung", pid=child.pid, heartbeat_age_seconds=round(age))
                self._terminate(child)
                break
            self.sleep(self.s.tick_seconds)

        code = child.wait()
        record = {
            "at": utc_now(),
            "pid": child.pid,
            "returncode": code,
            "signal": signal.Signals(-code).name if code < 0 and -code in signal.valid_signals() else None,
            "runtime_seconds": round(self.clock() - started, 1),
            "hung": hung,
            "stopped": stopped,
            "last_sample": last,
            "peak_sample": peak,
            "cgroup_delta": _delta(cgroup_before, cgroup_memory()),
            "log_tail": _tail(self.s.log_path),
        }
        append_jsonl(self.exits_path, record)
        return record

    def run(self, *, max_starts: int | None = None) -> int:
        failures_in_a_row = 0
        starts = 0
        while not self._stopping():
            if max_starts is not None and starts >= max_starts:
                break
            record = self.run_once()
            starts += 1
            if record["stopped"] or self._stopping():
                self._event("stopped")
                break
            if record["runtime_seconds"] >= self.s.healthy_seconds:
                failures_in_a_row = 0
            delay = self.s.backoff[min(failures_in_a_row, len(self.s.backoff) - 1)]
            failures_in_a_row += 1
            self._event(
                "restarting",
                after_seconds=delay,
                returncode=record["returncode"],
                signal=record["signal"],
                hung=record["hung"],
            )
            deadline = self.clock() + delay
            while self.clock() < deadline and not self._stopping():
                self.sleep(min(self.s.tick_seconds, max(0.0, deadline - self.clock())))
        return 0


# ---- command line --------------------------------------------------------


def settings_from_config(config_path: Path, config: dict[str, Any]) -> Settings:
    """The run config's keeper: the run loop's command, its log and heartbeat in the run's store."""
    run_dir = Path(config["store"]) / config["run_id"]
    options = dict(config.get("keeper", {}))
    env = dict(os.environ)
    arbor_root = str(Path(config.get("arbor_root", ".")).resolve())
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [arbor_root, env.get("PYTHONPATH", "")]))
    config_arg = str(config_path.resolve())
    interval = str(int(options.get("interval_seconds", 60)))
    return Settings(
        command=[sys.executable, "-m", "arbor.parallel", "run", "--config", config_arg, "--interval", interval],
        state_dir=run_dir / "keeper",
        log_path=run_dir / "controller.log",
        heartbeat_path=run_dir / "HOST",
        heartbeat_seconds=float(options.get("heartbeat_seconds", 900)),
        sample_seconds=float(options.get("sample_seconds", 30)),
        backoff=tuple(float(x) for x in options.get("backoff_seconds", DEFAULT_BACKOFF)),
        healthy_seconds=float(options.get("healthy_seconds", 3600)),
        cwd=arbor_root,
        env=env,
        guard_pattern=("arbor.parallel", " run ", config_arg),
    )


def _status(settings: Settings, count: int) -> int:
    state = settings.state_dir
    for name in ("keeper.pid", "supervisor.pid"):
        path = state / name
        pid = path.read_text().strip() if path.exists() else "-"
        alive = pid.isdigit() and Path(f"/proc/{pid}").exists()
        print(f"{name:<15} {pid:<10} {'alive' if alive else 'not running'}")
    exits = state / "exits.jsonl"
    if not exits.exists():
        print("no exits recorded")
        return 0
    records = [json.loads(line) for line in exits.read_text().splitlines() if line.strip()]
    for record in records[-count:]:
        peak = (record.get("peak_sample") or {}).get("tree_rss_mb")
        print(
            f"{record['at']}  rc={record['returncode']} signal={record.get('signal')} "
            f"ran {record['runtime_seconds']}s hung={record['hung']} "
            f"stopped={record['stopped']} peak_tree_rss={peak}MB "
            f"cgroup_delta={json.dumps(record.get('cgroup_delta') or {})}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m arbor.parallel.keeper", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["run", "status"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--last", type=int, default=10, help="status: exits to show")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = settings_from_config(config_path, config)
    if args.action == "status":
        return _status(settings, args.last)

    settings.state_dir.mkdir(parents=True, exist_ok=True)
    pid_file = settings.state_dir / "keeper.pid"
    if pid_file.exists():
        old = pid_file.read_text().strip()
        if old.isdigit() and old != str(os.getpid()) and Path(f"/proc/{old}").exists():
            print(f"a keeper is already running as pid {old}; not starting a second")
            return 1
    other = already_running(settings.guard_pattern, exclude=os.getpid())
    if other is not None:
        print(
            f"a controller for this config is already running as pid {other}. Two "
            "controllers against one run store corrupt it; stop that one first."
        )
        return 1
    stop_file = settings.state_dir / "STOP"
    if stop_file.exists():
        stop_file.unlink()
    atomic_write_text(pid_file, f"{os.getpid()}\n")
    keeper = Keeper(settings)
    signal.signal(signal.SIGTERM, keeper.request_stop)
    signal.signal(signal.SIGINT, keeper.request_stop)
    try:
        return keeper.run()
    finally:
        if pid_file.exists() and pid_file.read_text().strip() == str(os.getpid()):
            pid_file.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
