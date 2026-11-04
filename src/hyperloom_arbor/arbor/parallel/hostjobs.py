"""Jobs on the controller's own host: the round's document campaigns, as systemd user units.

The dossier (map step A and B) is a document campaign: no node, no GPU, an
agent on the LLM gateway editing files. It runs where the controller runs, as
a user unit, because a login node kills whatever an ssh session leaves behind
(runner AGENTS.md section 3.1), with `MemoryMax` under the login node's
per-user throttle and `OOMPolicy=continue`.

The contract is cluster.py's: `submit` returns the unit's name, Failed or
Unknown; `status` maps names to Known (RUNNING, COMPLETED), Failed or Unknown;
`find_by_name` resolves an Unknown submit. A unit that exits 0 is unloaded by
systemd, so a unit that is not loaded reads COMPLETED; the controller decides
success by the job's DONE marker, never by that. A unit that failed stays
loaded as failed until `reset` clears it.

`FakeHost` runs units in-process for the tests, as FakeCluster runs jobs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Callable

from arbor.parallel.cluster import Failed, FakeJob, Known, Status, Unknown

LIVE = ("active", "activating", "deactivating", "reloading")


def _unit(name: str) -> str:
    return name if name.endswith(".service") else f"{name}.service"


def parse_show(out: str) -> dict[str, dict]:
    """`systemctl show -p Id,LoadState,ActiveState,Result` blocks, by unit name (no .service)."""
    units, cur = {}, {}
    for line in out.splitlines() + [""]:
        if not line.strip():
            if cur.get("Id"):
                units[cur["Id"].removesuffix(".service")] = cur
            cur = {}
            continue
        k, _, v = line.partition("=")
        cur[k.strip()] = v.strip()
    return units


def status_of(props: dict | None) -> Status:
    if props is None:
        return Unknown("systemctl printed nothing for it")
    if props.get("LoadState") == "not-found":
        return Known("COMPLETED")
    active = props.get("ActiveState", "")
    if active in LIVE:
        return Known("RUNNING")
    if active == "failed":
        return Failed("FAILED", props.get("Result", ""))
    if active == "inactive":
        return Known("COMPLETED")
    return Unknown(f"ActiveState {active!r}")


class SystemdHost:
    def __init__(self, run: Callable = subprocess.run, *, timeout_s: float = 30.0,
                 memory_max: str = "2200M", path: str | None = None) -> None:
        self.run, self.timeout_s, self.memory_max = run, timeout_s, memory_max
        self.path = path or os.environ.get("PATH", "")

    def _cmd(self, cmd: list[str]) -> tuple[int, str, str] | None:
        try:
            p = self.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return p.returncode, p.stdout or "", p.stderr or ""

    def submit(self, script: str, *, name: str, args: list[str] = (), env: dict | None = None,
               output: str | None = None, **_ignored) -> str | Failed | Unknown:
        inner = "exec bash " + " ".join(shlex.quote(str(x)) for x in [script, *args])
        if output:
            inner += f" >> {shlex.quote(output)} 2>&1"
        cmd = ["systemd-run", "--user", f"--unit={name}", "-p", f"MemoryMax={self.memory_max}",
               "-p", "OOMPolicy=continue", f"--setenv=PATH={self.path}"]
        cmd += [f"--setenv={k}={v}" for k, v in (env or {}).items()]
        cmd += ["/bin/bash", "-lc", inner]
        r = self._cmd(cmd)
        if r is None:
            return Unknown("systemd-run timed out; the unit may exist")
        rc, out, err = r
        if rc == 0:
            return name
        if "already" in err or "exists" in err:
            return Unknown(f"a unit {name} exists already")
        return Failed("REJECTED", (err or out).strip()[:300])

    def status(self, names: list[str]) -> dict[str, Status]:
        names = [n for n in names if n]
        if not names:
            return {}
        r = self._cmd(["systemctl", "--user", "show", "-p", "Id,LoadState,ActiveState,Result",
                       *map(_unit, names)])
        if r is None or r[0] != 0:
            return {n: Unknown("systemctl could not be read") for n in names}
        units = parse_show(r[1])
        return {n: status_of(units.get(n.removesuffix(".service"))) for n in names}

    def find_by_name(self, name: str) -> list[str] | None:
        st = self.status([name]).get(name)
        if isinstance(st, Unknown):
            return None
        return [name] if isinstance(st, Known) and st.state == "RUNNING" else []

    def cancel(self, name: str) -> bool | None:
        r = self._cmd(["systemctl", "--user", "stop", _unit(name)])
        return None if r is None else r[0] == 0

    def reset(self, name: str) -> None:
        self._cmd(["systemctl", "--user", "reset-failed", _unit(name)])


@dataclass
class FakeHost:
    """Units run in-process at `advance()`; `handlers` as FakeCluster's (by the script's basename)."""
    handlers: dict[str, Callable[[FakeJob], object]] = field(default_factory=dict)
    units: dict[str, FakeJob] = field(default_factory=dict)
    unknown: int = 0

    def submit(self, script: str, *, name: str, args: list[str] = (), env: dict | None = None,
               output: str | None = None, **_ignored) -> str | Failed | Unknown:
        cur = self.units.get(name)
        if cur is not None and cur.state in ("RUNNING", "FAILED"):
            return Unknown(f"a unit {name} exists already")
        self.units[name] = FakeJob(name, script, name, list(map(str, args)), dict(env or {}), [], "RUNNING", "host")
        return name

    def status(self, names: list[str]) -> dict[str, Status]:
        if self.unknown:
            self.unknown -= 1
            return {n: Unknown("fake: unreadable") for n in names}
        out = {}
        for n in names:
            u = self.units.get(n)
            out[n] = (Known("COMPLETED") if u is None or u.state == "COMPLETED"
                      else Known("RUNNING") if u.state == "RUNNING" else Failed(u.state))
        return out

    def find_by_name(self, name: str) -> list[str] | None:
        u = self.units.get(name)
        return [name] if u is not None and u.state == "RUNNING" else []

    def cancel(self, name: str) -> bool | None:
        u = self.units.get(name)
        if u is not None and u.state == "RUNNING":
            u.state = "FAILED"
        return True

    def reset(self, name: str) -> None:
        u = self.units.get(name)
        if u is not None and u.state == "FAILED":
            del self.units[name]

    def advance(self) -> list[FakeJob]:
        ran = [u for u in self.units.values() if u.state == "RUNNING"]
        for u in ran:
            h = self.handlers.get(os.path.basename(u.script))
            r = True if h is None else h(u)
            if r == "RUNNING":
                continue
            u.state = "COMPLETED" if r is True or r is None or r == "COMPLETED" else "FAILED"
        return ran
