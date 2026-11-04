"""Prove each arm's configuration: what the served process actually ran with.

A knob in arbor/config.json is a claim about the server, not a fact about it.
The recipes behind the serving-playbook skill lost settings on the way to the
process several times (amd-knobs.md §0, rule 1), and this harness did too: the
server sub-agent's suggested launch passed no env, while the baseline
(scripts/model-m-campaign/baseline_model_m.py) passed ``server_env`` with ``-e``, so
an iteration and its bar could run under different env.

``capture`` reads the live server process inside the container: its env,
filtered to the serving prefixes below, its argv, and the lines of its log
that say what it was given and what it selected. It writes them to
``iter_dir/effective_config.json``, with the log's tail beside it as
``iter_dir/server.log`` (the log itself lives in the container's /tmp and is
gone with it). ``expected`` is what the arm claims: the live env and flags
(arbor/execute/bench.py). ``compare`` says what the process lacks.

A capture that cannot read the process (no container, no matching process, a
docker error) is UNKNOWN, never a mismatch: no evidence about the arm is not
evidence that it differed.

The gate enforces a mismatch only under ``policy.arm_check: enforce``. The
default, ``record``, shows it to the agent and the critic and leaves the
verdict alone; ``off`` skips the capture.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Callable

from arbor import frameworks
from arbor.frameworks.base import SERVER_LOG

MODES = ("off", "record", "enforce")

# The env the comparison can see: each framework's own prefix and the ROCm
# stack's (arbor/frameworks/; amd-knobs.md §0 rule 1 greps the same families).
# A key outside them is not captured, so it is never reported missing. Every
# framework's at once: ``tracked`` is not told which one ran.
ENV_PREFIXES = frameworks.union("env_prefixes")

# The server process, by framework (arbor/frameworks/). The bracket keeps
# pgrep's own command line from matching.
SERVER_PATTERNS = {name: frameworks.get(name).server_pattern for name in frameworks.names()}
# A wrapper that names the server in its argv but is not it (`bash -c '... vllm
# serve ...'` while it waits): its env is not the server's.
_WRAPPERS = {"bash", "sh", "nohup", "env", "timeout", "setsid"}

LOG_TAIL_BYTES = 2 * 1024 * 1024

# Log lines worth keeping, every framework's (arbor/frameworks/vllm.py has the
# list and why it is short). The whole tail is archived beside them.
LOG_PATTERNS = frameworks.union("log_patterns")
_MAX_LOG_LINES = 100

# Runs argv where the server lives; returns (returncode, stdout).
Runner = Callable[[list], tuple]


def _container_runner(container: str) -> Runner:
    from arbor.execute.workspace import run_in

    def run(argv: list) -> tuple:
        r = run_in(container, argv, capture_output=True, timeout=60)
        return r.returncode, (r.stdout or b"").decode(errors="replace")
    return run


def tracked(key: str) -> bool:
    return key.startswith(ENV_PREFIXES)


def parse_environ(raw: str) -> dict[str, str]:
    """/proc/<pid>/environ (NUL-separated) → the tracked keys."""
    env = {}
    for item in raw.split("\0"):
        k, sep, v = item.partition("=")
        if sep and tracked(k):
            env[k] = v
    return env


def parse_cmdline(raw: str) -> list[str]:
    """/proc/<pid>/cmdline (NUL-separated) → argv."""
    return [a for a in raw.split("\0") if a]


def flags_of(args) -> dict[str, object]:
    """Flags from an argv list or a flags string: ``{"--flag": value or True}``.

    ``--flag=value`` and ``--flag value`` read the same. A token is a flag when
    it starts with ``--``, or with ``-`` and a letter; a flag's value is the next
    token unless that is a flag. Positionals (the model path) are dropped.
    """
    if isinstance(args, str):
        try:
            args = shlex.split(args)
        except ValueError:
            args = args.split()
    out: dict[str, object] = {}
    i = 0
    while i < len(args):
        tok = args[i]
        if _is_flag(tok):
            name, eq, val = tok.partition("=")
            if eq:
                out[name] = val
            elif i + 1 < len(args) and not _is_flag(args[i + 1]):
                out[name] = args[i + 1]
                i += 1
            else:
                out[name] = True
        i += 1
    return out


def _is_flag(tok: str) -> bool:
    return tok.startswith("--") or (len(tok) > 1 and tok[0] == "-" and tok[1].isalpha())


def same_value(a, b) -> bool:
    """Equal as JSON when both parse (a compilation config), else as strings."""
    if a is True or b is True:
        return a is b
    try:
        return json.loads(a) == json.loads(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def log_lines(text: str) -> list[str]:
    seen, out = set(), []
    for line in text.splitlines():
        line = line.strip()
        if line and line not in seen and any(p.search(line) for p in LOG_PATTERNS):
            seen.add(line)
            out.append(line[:400])
    return out[-_MAX_LOG_LINES:]


def _pick_pid(pgrep_out: str) -> int | None:
    """The server among `pgrep -af` matches: the first that is not a wrapper."""
    rows = []
    for line in pgrep_out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit():
            rows.append((int(pid), cmd))
    for pid, cmd in rows:
        first = cmd.split(" ", 1)[0].rsplit("/", 1)[-1]
        if first not in _WRAPPERS:
            return pid
    return rows[0][0] if rows else None


def capture(config: dict, iter_dir, runner: Runner | None = None,
            pid_file=None, server_log=None) -> dict:
    """Read the live server process. Never raises: a failure is ``{"error": ...}``.

    Writes ``effective_config.json`` and ``server.log`` into ``iter_dir`` on
    success. A failed capture writes its error only when no earlier capture of
    this iteration is there, so a server that is gone by the second capture does
    not erase what the first one read.

    ``pid_file`` and ``server_log`` are component mode's (arbor/component.py): the
    server Arbor launched on this host, read by the PID its launch wrote and from
    its log in the session. Never a ``pgrep`` first match there: on the host's node
    that may be the host's own server.
    """
    from arbor.execute.workspace import HOST

    iter_dir = Path(iter_dir)
    container = config.get("docker_container", "")
    log_path = SERVER_LOG
    if pid_file is not None:
        pattern = None
        run = runner or _container_runner(HOST)
        log_path = str(server_log or SERVER_LOG)
    else:
        # An unknown framework is looked for as vLLM, though the launch takes it for
        # SGLang (arbor/frameworks/__init__.py).
        pattern = SERVER_PATTERNS.get(frameworks.name_of(config), SERVER_PATTERNS["vllm"])
        if not container or container == HOST:
            return _failed(iter_dir, "no container to read the server from")
        run = runner or _container_runner(container)
    try:
        if pattern is None:
            try:
                pid = int(Path(pid_file).read_text().split()[0])
            except (OSError, ValueError, IndexError):
                return _failed(iter_dir, f"no PID in {pid_file}: the server was not started "
                                         "with the component-mode launch command")
        else:
            _, listing = run(["pgrep", "-af", pattern])
            pid = _pick_pid(listing)
            if pid is None:
                return _failed(iter_dir, f"no process matches '{pattern}'")
        rc, environ = run(["cat", f"/proc/{pid}/environ"])
        rc2, cmdline = run(["cat", f"/proc/{pid}/cmdline"])
        if rc != 0 or rc2 != 0:
            return _failed(iter_dir, f"process {pid} could not be read (gone?)")
        _, log = run(["tail", "-c", str(LOG_TAIL_BYTES), log_path])
    except Exception as e:  # noqa: BLE001 - a capture is evidence, never a verdict
        return _failed(iter_dir, f"capture failed: {e}")
    argv = parse_cmdline(cmdline)
    eff = {"pid": pid, "pattern": pattern, "env": parse_environ(environ),
           "argv": argv, "flags": flags_of(argv), "log_lines": log_lines(log)}
    if pid_file is not None:
        eff["pid_file"] = str(pid_file)
    try:
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "server.log").write_text(log)
        (iter_dir / "effective_config.json").write_text(json.dumps(eff, indent=1) + "\n")
    except OSError:
        pass
    return eff


def _failed(iter_dir: Path, error: str) -> dict:
    path = iter_dir / "effective_config.json"
    try:
        if not path.exists():
            iter_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"error": error}) + "\n")
    except OSError:
        pass
    return {"error": error}


def expected(config: dict) -> dict:
    """What the arm claims: the live env (tracked keys) and flags.

    In component mode without the GPU masks: the launch drops them (the host sets
    them), so a mask left in arbor/config.json is not a claim the server carries.
    """
    from arbor import component
    from arbor.execute.bench import _live_env, _live_flags
    live = _live_env(config)
    if component.is_component(config):
        live = component.strip_masks(live)
    env = {k: v for k, v in live.items() if tracked(k)}
    return {"env": env, "flags": _live_flags(config)}


def compare(want: dict, eff: dict) -> dict:
    """What the live process lacks of the claim.

    ``status`` is ``ok``, ``mismatch``, or ``unknown`` (no capture). Extra env
    or flags on the process are not reported: a profiling flag the agent added
    to the launch is not a claim the arm failed.
    """
    if eff.get("error"):
        return {"status": "unknown", "reason": eff["error"]}
    live_env, live_flags = eff.get("env", {}), eff.get("flags", {})
    missing_env = sorted(k for k in want.get("env", {}) if k not in live_env)
    different_env = {k: [v, live_env[k]] for k, v in sorted(want.get("env", {}).items())
                     if k in live_env and live_env[k] != v}
    claimed = flags_of(want.get("flags", ""))
    missing_flags = sorted(f for f in claimed if f not in live_flags)
    different_flags = {f: [v, live_flags[f]] for f, v in claimed.items()
                       if f in live_flags and v is not True and not same_value(v, live_flags[f])}
    out = {"status": "ok", "pid": eff.get("pid"),
           "missing_env": missing_env, "different_env": different_env,
           "missing_flags": missing_flags, "different_flags": different_flags}
    items = items_of(out)
    if items:
        out["status"] = "mismatch"
        out["reason"] = "; ".join(items[:8]) + (f"; and {len(items) - 8} more" if len(items) > 8 else "")
    else:
        out["reason"] = "the live process carries every claimed env key and flag"
    return out


def items_of(cmp: dict) -> list[str]:
    """One short string per difference, env first."""
    return ([f"env {k} absent" for k in cmp.get("missing_env", [])]
            + [f"env {k}={w} (live {g})" for k, (w, g) in cmp.get("different_env", {}).items()]
            + [f"flag {f} absent" for f in cmp.get("missing_flags", [])]
            + [f"flag {f} {w} (live {g})" for f, (w, g) in cmp.get("different_flags", {}).items()])


def check(config: dict, iter_dir, runner: Runner | None = None,
          pid_file=None, server_log=None) -> dict:
    """capture → expected → compare, for one arm. A read carries the live
    process's env and flags as ``effective``, for checks that name their own
    settings (arbor/measure/bundles.py). ``pid_file`` and ``server_log``: capture's."""
    eff = capture(config, iter_dir, runner, pid_file=pid_file, server_log=server_log)
    if eff.get("error"):
        return compare({}, eff)
    cmp = compare(expected(config), eff)
    cmp["effective"] = {"env": eff["env"], "flags": eff["flags"]}
    return cmp


def brief(cmp: dict | None) -> dict | None:
    """The lesson's record: ``{"ok": True|False|None, "missing": [...]}``."""
    if not cmp:
        return None
    if cmp.get("status") == "unknown":
        return {"ok": None, "missing": [], "note": cmp.get("reason", "")[:200]}
    return {"ok": cmp.get("status") == "ok", "missing": items_of(cmp)[:12]}


def critic_block(cmp: dict | None, mode: str) -> str:
    if not cmp:
        return ""
    status = cmp.get("status")
    if status == "ok":
        body = "The live server process carries every env key and flag the arm claims."
    elif status == "unknown":
        body = f"Not captured ({cmp.get('reason', '')}): whether the arm ran as claimed is unknown."
    else:
        body = ("The live server process LACKS part of what the arm claims, so the number "
                "may not measure the change:\n" + "\n".join(f"- {i}" for i in items_of(cmp)[:12]))
        if mode != "enforce":
            body += ("\n\nThe gate does not enforce this (`arm_check: record`). Judge whether "
                     "the diff's claim survives it.")
    return f"### Arm check (`effective_config.json`)\n{body}\n\n"
