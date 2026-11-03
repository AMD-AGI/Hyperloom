"""Component mode: Arbor run by a host that owns the node, the GPUs and the clock.

Standalone Arbor owns a node, its container and its clock. In component mode a
host owns them: Hyperloom's multibranch (MB) today, which runs Arbor as an
"autonomous branch" on a lease of GPUs and under its own deadline, gates and
merges what Arbor keeps, and polls session.json's KEEP lessons as the keep
signal (docs/component-mode.md, the design and the operator's decisions).

MB built a sandbox around what standalone Arbor did wrong inside it (its board
topics #100, #126, #148): Arbor stopped MB's servers with a pattern `pkill`,
ran on cards outside its lease, iterated an hour after the session died, hung
for 511 minutes, and loaded kernel tables from paths outside the workspace.
This module is the part of the fix that is not the serving path:

- **the switch**: ``"mode": "standalone" | "component"`` and the ``"component"``
  block, validated here; a standalone config is not touched by any of it;
- **the refusals** (decision 3): no deadline, a ``docker_container`` other than
  ``"@host"``, ``ARBOR_ROLL=1``, a config that does not load or a campaign that does
  not validate exit 2 before anything else touches the session. Exit 2 means refused
  and nothing else does, so a host never restarts it; the reasons go to stderr and to
  ``<session>/component_refused.json``;
- **the deadline**: absolute, or ``seconds`` from the FIRST launch, kept in
  ``<session>/component.json`` so a relaunch never restarts the clock;
- **the iteration gate**: an iteration starts only if the rolling handoff's
  estimate of its length (arbor/cluster/handoff.py) fits before the deadline;
- **the exit record** (decision 1): every component-mode exit writes
  ``<session>/component_exit.json`` with its reason, so a host reads why Arbor
  ended instead of guessing from an exit code;
- **phase timeouts**: no agent turn outlives its cap or the deadline;
- **SIGTERM**: the host's stop (MB: SIGTERM, 30 s grace, then SIGKILL) stops
  Arbor's own server, records the iteration as not measured and exits 0;
- **the heartbeat**: ``<session>/heartbeat.json`` every 5 minutes and at every
  phase boundary, so a host's stall check sees a live run;
- **the KEEP snapshot** (design §4, as agreed with MB on its board, #154):
  ``<session>/keeps/NNNN/`` per KEEP, written before session.json's KEEP count
  moves, with ``keeps/LATEST`` naming the newest;
- **the host's notes and bench command**: what the host has merged, in the
  agent's prompt at every iteration, and the host's own benchmark client for
  every measurement.

The serving path on "@host" is in arbor/frameworks/base.py (the launch, in its own
process group, without GPU masks, its PID and log in ``<session>/server/``),
arbor/execute/preflight.py (stopping that group by its PID file, and nothing
else) and arbor/policy/claude_loop.py (no docker, no VRAM kills).
"""

from __future__ import annotations

import codecs
import hashlib
import itertools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

STANDALONE = "standalone"
COMPONENT = "component"
MODES = (STANDALONE, COMPONENT)

# The only container component mode runs on (arbor/execute/workspace.py's HOST):
# the host runs Arbor where the workspace is, and Arbor never calls docker.
HOST = "@host"

EXIT_REFUSED = 2
# The not-measured streak stopped the run while the model could not be reached (a gateway
# outage): non-zero, so the host resumes it, as it does a crash, instead of retiring it.
EXIT_UNREACHABLE = 3
# ABORTED.md's first line when the model could not be reached; any other abort is a broken
# harness (`aborted`, exit 0), which a restart would not mend.
UNREACHABLE_HEAD = "# Model unreachable"

STATE_FILE = "component.json"        # {mode, host, first_start, deadline_epoch, ...}
EXIT_FILE = "component_exit.json"    # why the run ended
REFUSED_FILE = "component_refused.json"   # why it never started (exit 2): {reasons, ts}
HEARTBEAT_FILE = "heartbeat.json"
SERVER_DIR = "server"              # <session>/server/server.pid and server.log
PID_NAME = "server.pid"
LOG_NAME = "server.log"

OUTBOX = "keeps"                    # <session>/<component.outbox>/NNNN/, one per KEEP
LATEST = "LATEST"                   # the newest snapshot's number, "0007"
KEEP_FILE = "keep.json"
# A table bigger than this is listed in keep.json, not copied (a host uploads tables/).
TABLE_MAX_BYTES = 512 * 1024 * 1024
# The host's notes put into the agent's prompt at most this much; the rest stays in the file.
NOTES_MAX_BYTES = 20 * 1024
NOTES_RULE = ("The host has already measured and merged the following; never redo a change it "
              "lists, and do not copy it into your workspace.")

REASONS = ("deadline", "map_exhausted", "done", "sigterm", "mapped", "aborted", "unreachable",
           "review_rejected")

# The host sets the GPU masks; Arbor never sets or passes them (design §2).
MASK_KEYS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")

BLOCK_KEYS = ("host", "deadline", "seconds", "notes", "bench_command", "outbox", "phase_minutes",
              "root", "plan", "stop_at", "profile", "review")

# component.review: who reviews the campaign before it runs. "host": the host does, before it
# starts this run (MB pauses after its dossier pass, holding no lease), so the review gate
# (arbor/review.py) passes at once and records that the host reviewed (ArborDevelopment #30).
REVIEW_BY = ("host",)

# component.stop_at: where a run ends of itself. "map" is the host's dossier pass (Hyperloom
# #157 c4000): the dossier and its map, then exit 0 with "mapped", before any iteration works
# the map; the host splits it among several Arbors (python3 -m arbor.directions split).
STOP_AT = ("map",)

# component.root: the commit the host gave the worktree (a sha, 7 to 40 hex digits). MB's
# worktree carries its experiment repository's history, so the first commit is not it
# (board #154, c3600); without the key, the first commit is the base (workspace.base_commit).
_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")

# Each phase's cap, in minutes (``component.phase_minutes`` overrides). The
# 511-minute hang MB saw was one agent turn with no bound at all.
PHASE_MINUTES = {"agent": 90, "server": 45, "bench": 60, "critic": 15, "correctness": 30,
                 "short_prompts": 10}

# Every phase ends this long before the deadline, and an iteration starts only
# if it ends by then: time to stop our own server (up to two minutes, never
# SIGKILL) and write the exit record before the host's own deadline.
DEADLINE_MARGIN_S = 180

HEARTBEAT_S = 300

# SIGTERM: MB grants 30 s before SIGKILL. The driver is gone by then whatever
# the code it interrupted does, and waits this long for our server's group.
SIGTERM_EXIT_S = 25
SIGTERM_SERVER_WAIT_S = 15


def log(msg: str) -> None:
    print(f"[arbor] component: {msg}", flush=True)


def _raw_log(msg: str) -> None:
    """``log`` for a signal handler: one os.write, never print. The handler runs on the
    main thread between two bytecodes, possibly inside a print, and a buffered stream
    entered twice raises "reentrant call" instead of writing."""
    try:
        os.write(sys.stdout.fileno(), f"[arbor] component: {msg}\n".encode())
    except Exception:  # noqa: BLE001
        pass


def iso(epoch: float | None) -> str | None:
    """An epoch as ISO 8601 UTC with a Z, to the second."""
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_tmp_ids = itertools.count()


def _write_json(path: Path, doc: dict) -> None:
    """Atomically: a temp file beside it, then a rename. A host polls these files.

    Each write has its own temp name: the SIGTERM handler may write a file while the
    code it interrupted, on the same thread, is half-way through writing it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{next(_tmp_ids)}.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


# ── the switch ─────────────────────────────────────────────────────────────────

def mode_of(config) -> str:
    """The mode a config (or a Campaign) names, lower-case; standalone when it names none."""
    config = getattr(config, "config", config) or {}
    return str(config.get("mode") or STANDALONE).strip().lower()


def is_component(config) -> bool:
    """True for a config, or a Campaign, in component mode. An unknown mode is not."""
    return mode_of(config) == COMPONENT


def block(config) -> dict:
    config = getattr(config, "config", config) or {}
    b = config.get("component")
    return b if isinstance(b, dict) else {}


def session_root(config) -> Path:
    """The session directory a config names, as load_campaign resolves it."""
    config = getattr(config, "config", config) or {}
    return Path(os.path.expandvars(os.path.expanduser(config.get("session_root", ""))))


def parse_deadline(value) -> float:
    """An absolute deadline: epoch seconds (a number or a numeric string), or ISO 8601
    with a timezone ("2026-10-29T03:20:00Z"). ValueError otherwise: a deadline with no
    timezone is read differently on every host."""
    if isinstance(value, bool):
        raise ValueError(f"deadline {value!r} is not a time")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    if s[-1:] in ("Z", "z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(f"deadline {value!r} is neither epoch seconds nor ISO 8601") from None
    if dt.tzinfo is None:
        raise ValueError(f"deadline {value!r} has no timezone (end it with Z or +hh:mm)")
    return dt.timestamp()


def _positive(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _seconds(b: dict):
    """``component.seconds`` as a bound, or None. MB writes ``"seconds": 0`` for "no end"
    (its agent_onboarding.py), so 0 is no bound, and the run still needs a deadline."""
    s = b.get("seconds")
    return None if s is None or (s == 0 and not isinstance(s, bool)) else s


def config_errors(config) -> list[str]:
    """What is wrong with a config's mode and its component block; [] when nothing is.

    A standalone config is checked for its mode alone: a "component" block beside it
    is ignored, as any key standalone does not read is.
    """
    config = getattr(config, "config", config) or {}
    mode = mode_of(config)
    if mode not in MODES:
        return [f"unknown mode {config.get('mode')!r} (one of {', '.join(MODES)})"]
    if mode != COMPONENT:
        return []
    raw = config.get("component", {})
    if not isinstance(raw, dict):
        return [f"\"component\" must be an object, not {type(raw).__name__}"]
    errors = []
    for k in sorted(set(raw) - set(BLOCK_KEYS)):
        errors.append(f"component.{k} is not a component-mode key ({', '.join(BLOCK_KEYS)})")
    for k in ("host", "notes", "bench_command", "outbox", "profile"):
        if k in raw and not isinstance(raw[k], str):
            errors.append(f"component.{k} must be a string")
    if raw.get("stop_at") is not None:
        if raw["stop_at"] not in STOP_AT:
            errors.append(f"component.stop_at is {raw['stop_at']!r}, not one of {', '.join(STOP_AT)}")
        elif raw.get("plan") is not None:
            errors.append("component.stop_at \"map\" writes the map a component.plan would give: "
                          "a dossier pass takes no plan")
    if raw.get("review") is not None and raw["review"] not in REVIEW_BY:
        errors.append(f"component.review is {raw['review']!r}, not one of {', '.join(REVIEW_BY)}")
    if "root" in raw and not (isinstance(raw["root"], str) and _SHA.match(raw["root"])):
        errors.append(f"component.root must be a commit sha (7 to 40 hex digits), not {raw['root']!r}")
    if raw.get("plan") is not None:
        errors += plan_entries(config)[1]
    if raw.get("deadline") is not None:
        try:
            parse_deadline(raw["deadline"])
        except ValueError as exc:
            errors.append(f"component.deadline: {exc}")
    if _seconds(raw) is not None and not _positive(raw["seconds"]):
        errors.append(f"component.seconds must be a positive number (0 for none), "
                      f"not {raw['seconds']!r}")
    pm = raw.get("phase_minutes")
    if pm is not None:
        if not isinstance(pm, dict):
            errors.append("component.phase_minutes must be an object")
        else:
            for k, v in pm.items():
                if k not in PHASE_MINUTES:
                    errors.append(f"component.phase_minutes.{k} is not a phase ({', '.join(PHASE_MINUTES)})")
                elif not _positive(v):
                    errors.append(f"component.phase_minutes.{k} must be a positive number of minutes")
    return errors


# Run by the interpreter the server runs under: exit 3 when the framework's package is not
# found at all (the launch says so itself), 1 with the error when its extension does not import.
_NATIVE_CHECK = r"""
import importlib, importlib.util, sys
top, mod = sys.argv[1], sys.argv[2]
if importlib.util.find_spec(top) is None:
    sys.exit(3)
try:
    importlib.import_module(mod)
except Exception as e:
    print(f"{type(e).__name__}: {e}"[:400])
    sys.exit(1)
"""


def _native_import_error(config, env) -> str | None:
    """A refusal when the framework's package is there but its compiled extension does not
    import, as a source-only worktree's vLLM lacks ``vllm._C`` (Lux, #11 c3939; #11 c3928
    item 2): every server start would fail. Checked with the driver's environment, the
    server_env's PYTHONPATH over it, by the ``python3`` on its PATH. A package that is not
    found at all, or a check that cannot run, refuses nothing."""
    from arbor import frameworks
    module = frameworks.get(config).native_module
    python = shutil.which("python3", path=(env or {}).get("PATH"))
    if not module or not python:
        return None
    run_env = dict(env or {})
    server_env = config.get("server_env") if isinstance(config.get("server_env"), dict) else {}
    if server_env.get("PYTHONPATH"):
        run_env["PYTHONPATH"] = str(server_env["PYTHONPATH"])
    try:
        r = subprocess.run([python, "-c", _NATIVE_CHECK, module.split(".")[0], module],
                           env=run_env, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 1:
        return None
    return (f"the framework's compiled extension {module} does not import ({r.stdout.strip()}): a "
            "workspace without it cannot serve; link or install the image's extension modules "
            "into it, or point workspace_root at an installed tree")


def refusals(config, env=None, session_dir=None) -> list[str]:
    """Why the driver must not start (decision 3), checked before it touches the session.

    Any mode: an unknown mode, and a config whose mode is not the one the session
    recorded (a session keeps its mode). Component mode: a bad component block, no
    ``deadline`` or ``seconds``, a ``docker_container`` other than "@host", and
    ``ARBOR_ROLL=1``. Reads session.json; writes nothing.
    """
    config = getattr(config, "config", config) or {}
    env = os.environ if env is None else env
    out = config_errors(config)
    mode = mode_of(config)
    if mode == COMPONENT and isinstance(config.get("component", {}), dict):
        b = block(config)
        if not config.get("session_root"):
            out.append("component mode needs a session_root: the host reads the KEEP snapshots, "
                       "the heartbeat and the exit record there")
        if b.get("deadline") is None and _seconds(b) is None:
            out.append("component mode needs a deadline: component.deadline (absolute) or "
                       "component.seconds (from the first launch; 0 is none)")
        if config.get("docker_container") != HOST:
            out.append(f"component mode runs on \"{HOST}\" only (docker_container is "
                       f"{config.get('docker_container')!r}): the host owns the node, and Arbor "
                       "never calls docker")
        if env.get("ARBOR_ROLL") == "1":
            out.append("ARBOR_ROLL=1 is refused in component mode: the host owns the clock, and "
                       "a rolling chunk would hand the session to a job the host never started")
        ws = config.get("workspace_root")
        root = b.get("root")
        if (isinstance(root, str) and _SHA.match(root) and ws and config.get("docker_container") == HOST
                and Path(ws).is_dir()):
            r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{root}^{{commit}}"],
                               cwd=ws, capture_output=True, text=True)
            if r.returncode != 0 or not r.stdout.strip():
                out.append(f"component.root {root} is not a commit of the workspace {ws}: every "
                           "patch and snapshot is measured against it")
        # A server with no flags is not the host's launch: one MB onboarding wrote an empty
        # server_flags, and the base server then failed its CUDA-graph capture (#11 c3928).
        if not str(config.get("server_flags") or "").strip():
            out.append("component mode needs server_flags: the launch the host measures (its "
                       "start commit's server args); an empty one serves defaults that are not it")
        if config.get("docker_container") == HOST:
            broken = _native_import_error(config, env)
            if broken:
                out.append(broken)
    if session_dir and mode in MODES:
        raw = _read_json(Path(session_dir) / "session.json") or {}
        recorded = raw.get("mode")
        if isinstance(recorded, str) and recorded and recorded != mode:
            out.append(f"{Path(session_dir) / 'session.json'} was started in {recorded} mode and "
                       f"this config is {mode}: a session keeps its mode (start a new session_root, "
                       f"or set \"mode\": \"{recorded}\")")
    return out


def refuse(reasons: list[str], config=None, session_dir=None) -> int:
    """Exit 2: each reason on stderr, and for a component config in its session's
    ``component_refused.json`` too, the one file a refusal writes. The host shows it and
    never restarts a refused run; the next start that is not refused removes it.
    """
    for r in reasons:
        print(f"[arbor] refusing to start: {r}", file=sys.stderr, flush=True)
    config = getattr(config, "config", config)
    # Only into a session the config names: an empty session_root loads as ".".
    if (session_dir and isinstance(config, dict) and config.get("session_root")
            and mode_of(config) == COMPONENT):
        try:
            _write_json(Path(session_dir) / REFUSED_FILE,
                        {"reasons": list(reasons), "ts": iso(time.time())})
        except OSError as exc:
            print(f"[arbor] could not write {REFUSED_FILE}: {exc}", file=sys.stderr, flush=True)
    return EXIT_REFUSED


# A config that does not load is refused, not crashed: starting it again cannot help. A
# read error that might pass (an NFS hiccup) is not in this list, and stays a crash.
UNLOADABLE = (FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError,
              ValueError, KeyError, TypeError)


def refuse_unloadable(config_path, exc: BaseException) -> int:
    """Exit 2 for a config ``load_campaign`` could not load: JSON that does not parse, no
    ``pack``, a pack whose pack.yaml is missing or broken. The record goes to the session
    the JSON names, when it parses that far.
    """
    raw = _read_json(Path(config_path))
    session = raw.get("session_root") if raw else None
    if isinstance(session, str) and session:
        session = os.path.expandvars(os.path.expanduser(session))
    else:
        session = None
    why = " ".join(str(exc).split())     # one line: a YAML error spans several
    return refuse([f"the config {config_path} does not load: {type(exc).__name__}: {why}"],
                  raw, session)


# ── the host's plan: this campaign's share of the directions ──────────────────

def plan_entries(config) -> tuple[list[dict], list[str]]:
    """``component.plan``: the directions this campaign works, and what is wrong with them.

    A host running several Arbors from one set of directions gives each its share (the
    2026-11-04 release, boards/ArborDevelopment #22). The plan is a list of directions in the
    form a DIRECTIONS line takes (``{"id": "m-<component>", "kind": "increment"|"rewrite",
    "title": ..., "why"?, "rank"?, "ceiling"?}``), or ``{"directions": [...]}``, or the path
    of a JSON file holding either. Every entry is open; ids are unique.
    """
    from arbor import directions
    raw = block(config).get("plan")
    if raw is None:
        return [], []
    if isinstance(raw, str):
        try:
            raw = json.loads(Path(raw).read_text())
        except (OSError, ValueError) as exc:
            return [], [f"component.plan: {raw} cannot be read: {exc}"]
    items = raw.get("directions") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        return [], ['component.plan must be a non-empty list of directions, {"directions": [...]}, '
                    "or the path of a JSON file holding either"]
    out, errors, seen = [], [], set()
    for i, item in enumerate(items):
        e, err = directions._entry(item, listed=True)
        if err:
            errors.append(f"component.plan[{i}]: {err}")
        elif e["status"] != directions.OPEN:
            errors.append(f"component.plan[{i}]: {e['id']} is {e['status']}: a plan gives open directions")
        elif e["id"] in seen:
            errors.append(f"component.plan[{i}]: {e['id']} appears twice")
        else:
            seen.add(e["id"])
            out.append(e)
    return (out, errors) if not errors else ([], errors)


def seed_plan(config, session_dir) -> str | None:
    """The session's map from ``component.plan``, once: a lane of the host's directions.

    A lane's map (arbor/directions.py) is the shape: on whatever the pack's skills, no
    bootstrap or update round, no DIRECTIONS restatement, a direction the agent adds goes to
    ``proposals`` (for the host), each closes on checked evidence (a done increment needs a
    KEEP that cites it), and when all are closed the campaign stops (``done``). A session that
    already has a map keeps it: a resume works the plan it started with. Returns what was done,
    for the log; None without a plan.
    """
    from arbor import directions
    if not is_component(config):
        return None
    entries, errors = plan_entries(config)
    if errors or not entries:
        return None
    session = Path(session_dir)
    session.mkdir(parents=True, exist_ok=True)
    doc, _ = directions.load(session)
    if doc is not None:
        return f"the host's plan: kept the map this session already has ({len(doc.get('directions') or [])} directions)"
    now = time.time()
    host = block(config).get("host") or "host"
    doc = directions.new_doc()
    doc.update(scope="lane", phase="active", map_round=1, round_started=1, round_started_ts=now,
               phase_started_ts=now,
               lane={"run": host, "round": 1, "lane": host, "direction": f"{len(entries)} directions",
                     "kind": "host-plan", "plan": None, "plan_sha256": None, "source": "component.plan"},
               directions=[{**e, "status": directions.OPEN, "why": e.get("why", ""), "added_at": 1,
                            "closed_at": None, "worked": [], "keeps": [], "flagged": False}
                           for e in entries])
    directions.save(session, doc)
    return (f"the host's plan: this campaign works {len(entries)} directions "
            f"({', '.join(e['id'] for e in entries)}) and nothing else")


# ── the deadline ───────────────────────────────────────────────────────────────

def resolve_deadline(config, session_dir, now: float | None = None) -> dict:
    """The run's absolute deadline, from the first launch's clock; written to component.json.

    ``deadline`` is absolute. ``seconds`` runs from the FIRST launch, kept in
    component.json, so a resumed run (MB resumes a crash) does not restart the clock;
    with both, the earlier wins. A session whose component.json is gone falls back to
    session.json's start, never to now.
    """
    session = Path(session_dir)
    now = time.time() if now is None else now
    b = block(config)
    rec = _read_json(session / STATE_FILE) or {}
    first = rec.get("first_start")
    if not isinstance(first, (int, float)):
        state = _read_json(session / "session.json") or {}
        first = state.get("start_epoch") if isinstance(state.get("start_epoch"), (int, float)) else now
    first = float(first)
    candidates = []
    if b.get("deadline") is not None:
        candidates.append((parse_deadline(b["deadline"]), "deadline"))
    if _seconds(b) is not None:
        candidates.append((first + float(b["seconds"]), f"first start + {float(b['seconds']):.0f} s"))
    if not candidates:
        raise ValueError("component mode needs component.deadline or component.seconds")
    deadline, source = min(candidates)
    was = rec.get("deadline_epoch")
    out = {"mode": COMPONENT, "host": b.get("host", ""), "first_start": first,
           "first_start_iso": iso(first), "deadline_epoch": deadline, "deadline": iso(deadline),
           "source": source, "updated": iso(now)}
    _write_json(session / STATE_FILE, out)
    if isinstance(was, (int, float)) and abs(float(was) - deadline) > 1:
        log(f"deadline moved {iso(was)} -> {iso(deadline)} (the config changed; the first start "
            f"{iso(first)} stands)")
    return out


def deadline_of(session_dir) -> float | None:
    rec = _read_json(Path(session_dir) / STATE_FILE) or {}
    v = rec.get("deadline_epoch")
    return float(v) if isinstance(v, (int, float)) else None


def unreachable(session_dir) -> bool:
    """True when ABORTED.md says the run stopped because the model could not be reached."""
    try:
        text = (Path(session_dir) / "ABORTED.md").read_text(errors="replace")
    except OSError:
        return False
    return text.startswith(UNREACHABLE_HEAD)


def clear_unreachable(session_dir, now: float | None = None) -> str | None:
    """A resumed run after an `unreachable` exit: that ABORTED.md moved aside, so the run
    starts instead of exiting at once; the new name, or None when there was none to move.
    Any other ABORTED.md stays, and still ends the run (`aborted`)."""
    if not unreachable(session_dir):
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if now is None else now))
    moved = Path(session_dir) / f"ABORTED-unreachable-{stamp}.md"
    (Path(session_dir) / "ABORTED.md").rename(moved)
    return moved.name


# ── the iteration gate ─────────────────────────────────────────────────────────

def _estimator() -> dict:
    """The rolling handoff's estimator settings (arbor/cluster/handoff.py), Slurm aside."""
    from arbor.cluster.handoff import DEFAULTS
    return {k: DEFAULTS[k] for k in ("est_default_min", "est_floor_min", "est_window",
                                     "est_quantile", "est_factor")}


def iteration_walls(lessons) -> list[float]:
    """Iteration wall times in seconds, oldest first: each gate row's ``wall_s`` (arbor/state.py)."""
    rows = [l for l in lessons or [] if isinstance(l, dict)
            and isinstance(l.get("wall_s"), (int, float)) and not isinstance(l.get("wall_s"), bool)
            and l["wall_s"] > 0]
    rows.sort(key=lambda l: l.get("ts") or 0)
    return [float(l["wall_s"]) for l in rows]


def estimate(lessons) -> tuple[float, str]:
    """The next iteration's length: the 75th percentile of the last 8 iterations x 1.15,
    at least 15 minutes; 60 minutes with none measured (the rolling handoff's rule)."""
    from arbor.cluster.handoff import nearest_rank
    cfg = _estimator()
    xs = iteration_walls(lessons)[-cfg["est_window"]:]
    if not xs:
        return cfg["est_default_min"] * 60.0, "default"
    q = nearest_rank(xs, cfg["est_quantile"])
    est = max(cfg["est_floor_min"] * 60.0, cfg["est_factor"] * q)
    return est, f"q{int(cfg['est_quantile'] * 100)} of {len(xs)} x{cfg['est_factor']}"


def map_counts(session_dir) -> dict | None:
    """The session's map (directions.json) as counts, or None when it has none."""
    try:
        from arbor import directions
        doc, err = directions.load(session_dir)
    except Exception:  # noqa: BLE001 - an exit record must not fail on the map
        return None
    if not doc or err:
        return None
    dirs = doc.get("directions") or []
    return {"open": sum(1 for d in dirs if d.get("status") == directions.OPEN),
            "closed": sum(1 for d in dirs if d.get("status") in directions.CLOSED),
            "exhausted": directions.exhausted(doc),
            "round": doc.get("map_round"), "phase": doc.get("phase")}


def gate(session_dir, lessons, deadline: float | None,
         now: float | None = None) -> tuple[bool, str | None, str]:
    """Whether the next iteration may start: (ok, exit reason or None, the log line).

    It may when its estimate ends by the deadline less the margin. When it may not,
    the reason is ``map_exhausted`` if the map has closed every direction (its update
    round does not fit: decision 1), else ``deadline``.
    """
    now = time.time() if now is None else now
    if deadline is None:
        return True, None, "no deadline"
    est, basis = estimate(lessons)
    end_by = deadline - DEADLINE_MARGIN_S
    line = (f"next iteration est {est / 60:.0f} min ({basis}) would end {iso(now + est)}; "
            f"iterations end by {iso(end_by)} (deadline {iso(deadline)} less {DEADLINE_MARGIN_S} s)")
    if now + est <= end_by:
        return True, None, line
    m = map_counts(session_dir)
    return False, ("map_exhausted" if m and m.get("exhausted") else "deadline"), line


# ── the exit record and the heartbeat ──────────────────────────────────────────

def exit_record(session_dir, reason: str, iteration: int | None = None,
                deadline: float | None = None, detail: str = "", now: float | None = None,
                say=log) -> dict:
    """``component_exit.json`` (decision 1): written, logged as one line, and on the heartbeat."""
    if reason not in REASONS:
        raise ValueError(f"exit reason {reason!r} is not one of {REASONS}")
    session = Path(session_dir)
    raw = _read_json(session / "session.json") or {}
    lessons = raw.get("lessons") if isinstance(raw.get("lessons"), list) else []
    rec = {"reason": reason,
           "iteration": iteration if iteration is not None else raw.get("iteration", 0),
           "keeps": keep_count(lessons),
           "map": map_counts(session),
           "deadline": iso(deadline if deadline is not None else deadline_of(session)),
           "ts": iso(time.time() if now is None else now)}
    if detail:
        rec["detail"] = detail
    try:
        _write_json(session / EXIT_FILE, rec)
    except OSError as exc:
        say(f"could not write {EXIT_FILE}: {exc}")
    say("exit " + json.dumps(rec))
    if _heartbeat is not None:
        _heartbeat.set(phase="exit", reason=reason)
    return rec


class Heartbeat:
    """``<session>/heartbeat.json`` {ts, iteration, phase}: every ``interval`` seconds from a
    daemon thread, and at once whenever the phase changes. MB declares a stall when
    nothing it watches changes for 30 minutes; an agent turn can be quiet longer."""

    def __init__(self, session_dir, interval: float = HEARTBEAT_S):
        self.path = Path(session_dir) / HEARTBEAT_FILE
        self.interval = interval
        self.doc: dict = {"iteration": None, "phase": "start"}
        # Re-entrant: the SIGTERM handler sets the phase on the thread that may hold it.
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def write(self) -> None:
        with self._lock:
            now = time.time()
            doc = {"ts": iso(now), "epoch": round(now, 1), "pid": os.getpid(), **self.doc}
            try:
                _write_json(self.path, doc)
            except OSError:
                pass   # a heartbeat is a signal, never a failure

    def set(self, **fields) -> None:
        with self._lock:
            self.doc.update(fields)
            self.write()

    def start(self) -> "Heartbeat":
        self.write()

        def beat():
            while not self._stop.wait(self.interval):
                self.write()
        self._thread = threading.Thread(target=beat, name="arbor-heartbeat", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


# The running heartbeat, which claude_loop's phases update; None outside component mode.
_heartbeat: Heartbeat | None = None


def phase(name: str, **fields) -> None:
    """Mark a phase boundary on the heartbeat. A no-op when none runs (standalone)."""
    if _heartbeat is not None:
        _heartbeat.set(phase=name, **fields)


# ── phase timeouts ─────────────────────────────────────────────────────────────

class PhaseTimeout(Exception):
    """A phase ran past its cap or the deadline: infrastructure, so NOT MEASURED."""

    def __init__(self, phase_name: str, seconds: float, why: str = ""):
        self.phase = phase_name
        self.seconds = seconds
        super().__init__(why or f"{phase_name} phase timed out after {seconds:.0f} s "
                                f"(component mode: its cap, or the deadline)")


def phase_seconds(config, phase_name: str, session_dir=None, now: float | None = None) -> float | None:
    """The phase's timeout: min(its cap, seconds left before the deadline less the margin).

    None in standalone, where phases have no timeout. Zero or less: no time is left.
    """
    if not is_component(config):
        return None
    minutes = (block(config).get("phase_minutes") or {}).get(phase_name, PHASE_MINUTES[phase_name])
    cap = float(minutes) * 60
    dl = deadline_of(session_dir if session_dir is not None else session_root(config))
    if dl is None:
        return cap
    now = time.time() if now is None else now
    return min(cap, dl - DEADLINE_MARGIN_S - now)


# ── the serving path's files ───────────────────────────────────────────────────

def server_paths(session_dir) -> tuple[Path, Path]:
    """(PID file, log) of the server Arbor launches: ``<session>/server/``, not /tmp."""
    d = Path(session_dir) / SERVER_DIR
    return d / PID_NAME, d / LOG_NAME


def strip_masks(env: dict | None) -> dict:
    """``env`` without ROCR_/HIP_/CUDA_VISIBLE_DEVICES: the host sets them, never Arbor
    (MB found Arbor's servers on cards outside its lease)."""
    return {k: v for k, v in (env or {}).items() if k not in MASK_KEYS}


def seed_env(config) -> dict:
    """The server_env seeded into arbor/config.json: in component mode without the GPU
    masks, which the host owns and the agent's rules forbid it to touch."""
    env = dict((getattr(config, "config", config) or {}).get("server_env") or {})
    return strip_masks(env) if is_component(config) else env


def gpu_list(value) -> list[int] | None:
    """``gpu_ids`` as a list of ints: "4,5,6,7" or [4, 5, 6, 7], since a host's manifest
    may write either (a list crashed every iteration before); None when there are none."""
    items = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
    ids = [int(g) for g in items if str(g).strip()]
    return ids or None


# ── the agent's time ───────────────────────────────────────────────────────────

def turn_block(config, session_dir=None, now: float | None = None) -> str:
    """The main agent's time this turn: its phase cap and what is left before the deadline.

    On Lux (2026-10-29) an agent started an AITER fused-MoE tuning sweep inside a 30-minute
    turn, hit the cap before measuring, and lost half of Arbor's 67 minutes: nothing had
    told it how long its turn was. Empty in standalone, where turns have no cap.
    """
    if not is_component(config):
        return ""
    now = time.time() if now is None else now
    session = session_dir if session_dir is not None else session_root(config)
    cap = phase_seconds(config, "agent", session, now)
    if cap is None:
        return ""
    dl = deadline_of(session)
    left = f"; the run's deadline is in {max(0.0, (dl - now) / 60):.0f} min" if dl is not None else ""
    return (
        f"## THIS TURN'S TIME\n"
        f"You have at most {max(0.0, cap) / 60:.0f} min for this turn{left}. A turn cut off by "
        f"its cap is booked not measured, and its uncommitted edits are discarded. Do not start "
        f"work that cannot finish and be left in the workspace inside it (a kernel-tuning sweep, "
        f"a long build or profile): run a bounded part, keep its results in the workspace, and "
        f"continue in the next iteration.\n"
    )


# ── the host's notes and its bench command ─────────────────────────────────────

def notes_block(config) -> str:
    """The host's notes (``component.notes``) for the agent's prompt; "" when it names none.

    Read at every iteration: the host adds to them as it merges. The first
    NOTES_MAX_BYTES go in, with a line saying the rest is in the file; a file that
    cannot be read is said to be so in the prompt and the log, never fatal.
    """
    raw = block(config).get("notes")
    if not is_component(config) or not isinstance(raw, str) or not raw.strip():
        return ""
    path = Path(os.path.expandvars(os.path.expanduser(raw.strip())))
    head = f"## What the host has already merged\n{NOTES_RULE}\n"
    text, size, cut, exc = _read_head(path)
    if exc is not None:
        log(f"the host's notes {path} could not be read: {exc}")
        return head + f"(Its notes, `{path}`, could not be read this iteration: {exc.strerror or exc}.)\n"
    if not text:
        return head + f"(Its notes, `{path}`, are empty: it lists nothing yet.)\n"
    out = head + f"Its notes, from `{path}`:\n\n{text}\n"
    if cut:
        out += (f"\n[The notes are {size} bytes and only the first {NOTES_MAX_BYTES} are "
                f"above: read the rest in `{path}` before you choose a change.]\n")
    return out


def _read_head(path: Path) -> tuple[str, int, bool, OSError | None]:
    """A host's file for the prompt: (its first NOTES_MAX_BYTES as text, its size, cut, error)."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return "", 0, False, exc
    cut = len(data) > NOTES_MAX_BYTES
    # Incremental, so a multi-byte character the cap splits is left out, not mangled.
    text = codecs.getincrementaldecoder("utf-8")(errors="replace").decode(
        data[:NOTES_MAX_BYTES], final=not cut).rstrip()
    return text, len(data), cut, None


# ── the host's dossier pass: component.stop_at "map" and component.profile ────

PROFILE_RULE = ("The host has profiled this tree already, and these are its numbers. Cite them in "
                "the dossier's profile part, and profile only what they do not cover.")


def host_reviews(config) -> bool:
    """The host reviewed the campaign before it started this run (``component.review``)."""
    return is_component(config) and block(config).get("review") == "host"


def stops_at_map(config) -> bool:
    """The host's dossier pass: the run ends once its map is set (``component.stop_at``)."""
    return is_component(config) and block(config).get("stop_at") == "map"


def profile_block(config) -> str:
    """The host's profile (``component.profile``, a file the host wrote: PRELUDE's op shares for
    MB) for the agent's prompt; "" when it names none. Read at every iteration, like the notes."""
    raw = block(config).get("profile")
    if not is_component(config) or not isinstance(raw, str) or not raw.strip():
        return ""
    path = Path(os.path.expandvars(os.path.expanduser(raw.strip())))
    head = f"## The host's profile\n{PROFILE_RULE}\n"
    text, size, cut, exc = _read_head(path)
    if exc is not None:
        log(f"the host's profile {path} could not be read: {exc}")
        return head + (f"(Its profile, `{path}`, could not be read this iteration: "
                       f"{exc.strerror or exc}. Profile on your own.)\n")
    if not text:
        return head + f"(Its profile, `{path}`, is empty. Profile on your own.)\n"
    out = head + f"From `{path}`:\n\n{text}\n"
    if cut:
        out += (f"\n[The profile is {size} bytes and only the first {NOTES_MAX_BYTES} are above: "
                f"read the rest in `{path}`.]\n")
    return out


def dossier_pass_block(config) -> str:
    """What a dossier pass's agent is told the run is for; "" for any other run."""
    if not stops_at_map(config):
        return ""
    host = block(config).get("host") or "the host"
    return (f"## The host's dossier pass\n"
            f"This run is {host}'s dossier pass. It writes the dossier and its map, and ends "
            "when the map is set: no iteration here works the map. The host splits the map's "
            "open directions among several Arbors, and each works its share alone. So:\n"
            "- LEARN only: read, measure and profile, and change nothing in the served tree.\n"
            "- Make each open direction a component one Arbor can work alone. Join directions "
            "with depends_on only when one cannot start before another is done: joined "
            "directions go to the same Arbor.\n"
            "- Rank every direction: the host deals them out in rank order.\n")


def map_ready(campaign, state, session_dir) -> str | None:
    """A dossier pass's end, checked before each iteration: once the map is set, it is
    promoted as the next iteration would promote it (``directions.advance``), and its summary
    returned; None until then."""
    from arbor import directions
    doc, err = directions.load(session_dir)
    if not doc or err or directions.is_lane(doc):
        return None
    if doc.get("pending"):
        directions.advance(campaign, state, session_dir)
        state.save(Path(session_dir) / "session.json")
        doc, err = directions.load(session_dir)
        if not doc or err:
            return None
    live = [d["id"] for d in doc.get("directions") or [] if d.get("status") == directions.OPEN]
    if doc.get("phase") != "active" or not live:
        return None
    return (f"a map of {len(live)} open directions ({', '.join(live)}) in "
            f"{directions.map_path(session_dir)}")


def bench_command(config) -> str | None:
    """The host's bench command (``component.bench_command``); None: the pack's bench."""
    cmd = block(config).get("bench_command")
    if not is_component(config) or not isinstance(cmd, str) or not cmd.strip():
        return None
    return cmd.strip()


def fill_bench(command: str, port, out) -> str:
    """The host's bench command with ``{port}`` and ``{out}`` filled, nothing else changed:
    str.replace, so any other braces in it (a JSON argument) stay as they are."""
    return command.replace("{port}", str(port)).replace("{out}", shlex.quote(str(out)))


# ── the KEEP snapshot ──────────────────────────────────────────────────────────

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def keep_count(lessons) -> int:
    """The keep signal a host polls: session.json's lessons whose decision is KEEP."""
    return sum(1 for l in lessons or [] if isinstance(l, dict) and l.get("decision") == "KEEP")


def outbox(config, session_dir) -> Path:
    """Where each KEEP's snapshot goes: ``component.outbox`` (default "keeps"), in the session."""
    return Path(session_dir) / (block(config).get("outbox") or OUTBOX)


def keep_name(number: int) -> str:
    """A snapshot's directory, and what LATEST holds: its number in four digits ("0007")."""
    return f"{int(number):04d}"


def _snapshots(out: Path) -> list[tuple[int, Path]]:
    """The outbox's finished snapshots, oldest first: the directories named by digits only."""
    try:
        found = [(int(p.name), p) for p in Path(out).iterdir() if p.name.isdigit() and p.is_dir()]
    except OSError:
        return []
    return sorted(found)


def _write_latest(out: Path, name: str | None) -> None:
    """LATEST: the four digits and no newline, replaced atomically. None removes it."""
    path = Path(out) / LATEST
    if name is None:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_name(f".{LATEST}.{os.getpid()}.{next(_tmp_ids)}.tmp")
    tmp.write_text(name)
    os.replace(tmp, path)


def _copy_tables(server_env: dict, workspace: str, dest: Path, say=log,
                 label: str = "") -> tuple[dict, dict]:
    """Every file an env var points at, copied to ``dest/tables/<VAR>/<basename>``.

    A file is an absolute path to an existing regular file; anything else an env var
    holds (a number, a directory, a relative path) is not a table. Returns
    (``tables``: {VAR: {path, file, sha256, bytes}}, ``tables_outside_workspace``:
    {VAR: path}). A host uploads tables/ as kernel artifacts, so a table need not sit
    in the worktree; one outside it (symlinks resolved) is still flagged, since a host
    that loads only the worktree does not find it (MB's sandbox finding). A file over
    TABLE_MAX_BYTES is listed with ``skipped``, not copied.

    A value that is a ``:``-separated list (aiter's tables: MB gives the start's as its
    private default copies, then the start's rows; the agent appends its own, Hyperloom
    #155 c4101/c4102) is {path: the whole value, files: [...]}: each part that is a file
    copied to ``tables/<VAR>/<k>-<basename>`` (k its place in the list) and listed in
    order. Parts that are not files stay in ``path`` only; a list with no file is not a
    table.
    """
    tables: dict = {}
    outside: dict = {}
    ws = os.path.realpath(workspace) if workspace else ""
    (dest / "tables").mkdir(parents=True, exist_ok=True)
    for var in sorted(server_env):
        value = server_env[var]
        if not isinstance(value, str) or not _ENV_NAME.match(str(var)):
            continue
        if os.pathsep in value.strip():
            files = []
            for k, part in enumerate(value.strip().split(os.pathsep), 1):
                entry = _copy_table(var, Path(part.strip()), f"tables/{var}/{k}-", dest, say, label)
                if entry is not None:
                    files.append(entry)
                    if not _within(entry["path"], ws):
                        outside[var] = value.strip()
            if files:
                tables[var] = {"path": value.strip(), "files": files}
            continue
        entry = _copy_table(var, Path(value.strip()), f"tables/{var}/", dest, say, label)
        if entry is None:
            continue
        if not _within(entry["path"], ws):
            outside[var] = entry["path"]
        tables[var] = entry
    return tables, outside


def _within(path: str, ws: str) -> bool:
    """``path`` (symlinks resolved) is inside the workspace ``ws``, itself resolved."""
    return bool(ws) and os.path.commonpath([os.path.realpath(path), ws]) == ws


def _copy_table(var: str, path: Path, prefix: str, dest: Path, say, label: str) -> dict | None:
    """One table file copied to ``dest/<prefix><basename>``: {path, file, sha256, bytes}, or
    None when ``path`` is not an absolute path to a regular file."""
    if not path.is_absolute():
        return None
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
    except OSError:
        return None
    entry = {"path": str(path), "file": None, "sha256": None, "bytes": size}
    if size > TABLE_MAX_BYTES:
        entry["skipped"] = f"over {TABLE_MAX_BYTES // 2**20} MiB: not copied"
        say(f"{label}{var}={path} is {size} bytes, over {TABLE_MAX_BYTES // 2**20} MiB: "
            "listed in keep.json, not copied")
        return entry
    rel = f"{prefix}{path.name}"
    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
    digest, n = hashlib.sha256(), 0
    with open(path, "rb") as src, open(dest / rel, "wb") as dst:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            digest.update(chunk)
            dst.write(chunk)
            n += len(chunk)
    entry.update(file=rel, sha256=digest.hexdigest(), bytes=n)
    return entry


_ITER_SUBJECT = re.compile(r"iter-(\d+): ")


def carried_changes(container: str, workspace: str, parent: str, commit: str, iteration: int,
                    lessons, spec) -> list[dict]:
    """The earlier iterations' changes a KEEP's tree carries since its parent, oldest first.

    Every ``iter-N:`` commit in ``parent..commit`` other than this KEEP's own, that no
    ``Revert "iter-N: ..."`` undoes, and that changes what is served: the served pathspec,
    or arbor/config.json (flags and env, which the patches leave out). A NOT_MEASURED
    iteration keeps its tree, so its change rides into the next KEEP, whose gate judges
    only the two together (Lux, 2026-10-30: iteration 5's INT4 QuickReduce env in KEEP 2).
    Each is named with its own verdict from session.json, for the host to see.
    """
    from arbor.execute.workspace import git
    subjects = [line.split("\t", 1) for line in git(
        container, workspace, "log", "--format=%H%x09%s", f"{parent}..{commit}").splitlines()
        if "\t" in line]
    reverted = {s[len('Revert "'):-1] for _, s in subjects
                if s.startswith('Revert "') and s.endswith('"')}
    rows = {}
    for row in lessons or []:
        if isinstance(row, dict) and isinstance(row.get("iteration"), int):
            rows[row["iteration"]] = row          # the last row of each iteration
    out = []
    for sha, subject in reversed(subjects):
        m = _ITER_SUBJECT.match(subject)
        if not m or int(m.group(1)) == iteration or subject in reverted:
            continue
        # Two calls: the served pathspec excludes arbor/config.json, and an exclusion wins
        # over any path named beside it.
        served = "".join(git(container, workspace, "diff-tree", "--no-commit-id", "--name-only",
                             "-r", sha, "--", *paths).strip()
                         for paths in (spec, ["arbor/config.json"]))
        if not served:
            continue                              # a probe's eval/ or ledger files only
        n = int(m.group(1))
        row = rows.get(n, {})
        out.append({"iteration": n, "change": subject[m.end():], "commit": sha,
                    "verdict": row.get("verdict") or row.get("decision") or "unknown",
                    "reason": str(row.get("reason") or "")[:300]})
    return out


def snapshot_keep(config, session_dir, *, container: str, workspace: str, number: int,
                  iteration: int, metric, change: str, intent: str,
                  now: float | None = None) -> dict:
    """The KEEP's snapshot (design §4): ``<outbox>/NNNN/``, then LATEST. Returns keep.json.

    The driver calls it before session.json records the KEEP, so the snapshot is
    whole before the count a host polls moves; ``number`` is that count with this
    KEEP. Built in ``.NNNN.tmp`` and renamed, so a host never reads half of one.
    Raises on any failure: the caller logs it and keeps its verdict (bookkeeping, as
    landed.patch is), and the next snapshot's ``previous`` shows the gap.

    - ``previous`` and ``parent``: the newest snapshot numbered below this one and its
      commit; ``null`` and ``root`` for the first. One numbered at or above this one
      was written for a KEEP session.json never recorded (the run stopped between the
      two): it is moved aside and never chained to, so a change it left in the tree
      reaches the host in the next change.patch.
    - ``carries``: the earlier iterations' changes the tree holds since ``parent`` that
      no gate judged on their own, such as a NOT_MEASURED iteration's (carried_changes).
    - Both patches use the served pathspec (no probes, logs, bytecode or arbor/config.json),
      byte for byte as git wrote them (``workspace._write_diff``; CORE-01).
    - ``left_out``: a file over ``workspace.OVERSIZE_BYTES`` stays out of both patches and
      is named here, with its size (a GPU core dump rode in one, Hyperloom #155 c4786).
    - config.json is the live arbor/config.json, what was measured (the server and
      bench agents may edit it after the commit), without the GPU masks the launch drops.
    """
    from arbor.execute.workspace import (
        _write_diff, base_commit, git, landed_pathspec, oversize, read_workspace_config, without,
    )
    now = time.time() if now is None else now
    out = outbox(config, session_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = keep_name(number)
    label = f"KEEP {name}: "

    unrecorded = [d for n, d in _snapshots(out) if n >= number]
    for d in unrecorded:
        aside = out / f".{d.name}.unrecorded.{os.getpid()}.{next(_tmp_ids)}"
        os.rename(d, aside)
        log(f"{label}{d.name} was never recorded in session.json (the run stopped between the "
            f"snapshot and the record): moved aside to {aside.name}")
    previous, prev = None, None
    for n, d in reversed(_snapshots(out)):
        doc = _read_json(d / KEEP_FILE)
        if doc and isinstance(doc.get("commit"), str) and doc["commit"]:
            previous, prev = n, doc
            break
    if unrecorded:   # LATEST named one of them: name the newest recorded one until ours is done
        _write_latest(out, keep_name(previous) if previous is not None else None)

    root = base_commit(container, workspace, getattr(config, "config", config))   # component.root, else the first commit
    commit = git(container, workspace, "rev-parse", "HEAD")
    parent, notes = root, []
    if prev is not None:
        parent = prev["commit"]
        try:
            git(container, workspace, "cat-file", "-e", f"{parent}^{{commit}}")
        except RuntimeError:
            notes.append(f"KEEP {keep_name(previous)}'s commit {parent} is not in the workspace's "
                         f"repository: change.patch is against root")
            log(label + notes[-1])
            parent = root

    tmp = out / f".{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        spec = landed_pathspec(getattr(config, "config", config))
        left_out = oversize(container, workspace, root, commit, spec)
        spec = without(spec, left_out)
        _write_diff(container, workspace, tmp / "change.patch", "--binary", parent, commit, "--", *spec)
        _write_diff(container, workspace, tmp / "served.patch", "--binary", root, commit, "--", *spec)
        live = read_workspace_config(container, workspace)
        env = strip_masks(live.get("server_env") if isinstance(live.get("server_env"), dict) else {})
        (tmp / "config.json").write_text(json.dumps(
            {"server_flags": live.get("server_flags", ""), "server_env": env}, indent=2) + "\n")
        tables, outside = _copy_tables(env, workspace, tmp, label=label)
        lessons = (_read_json(Path(session_dir) / "session.json") or {}).get("lessons")
        carries = carried_changes(container, workspace, parent, commit, iteration, lessons, spec)
        doc = {"number": number, "previous": previous, "iteration": iteration, "metric": metric,
               "change": change, "intent": intent, "root": root, "parent": parent,
               "commit": commit, "carries": carries, "ts": iso(now), "tables": tables,
               "tables_outside_workspace": outside}
        # The links the served tree makes outside itself, and how they are rebuilt
        # (arbor/execute/outside.py), when the driver recorded some for this commit.
        links = _read_json(Path(session_dir) / "outside.json") or {}
        if links.get("head") == commit and (links.get("counts") or {}).get("links"):
            shutil.copyfile(Path(session_dir) / "outside.json", tmp / "outside.json")
            doc["outside"] = links.get("counts")
        # The short-prompt check at this KEEP (arbor/measure/short_prompts.py): its brief
        # always, why it did not run included; its record when it ran.
        sp = Path(session_dir) / f"iter_{iteration}" / "short_prompts_check.json"
        if sp.is_file():
            from arbor.measure.short_prompts import brief
            rec = json.loads(sp.read_text())
            doc["short_prompts"] = brief(rec)
            if rec.get("verdict") in ("floor", "ok"):
                shutil.copyfile(sp, tmp / "short_prompts.json")
        # The full GSM8K this KEEP was held to (arbor/measure/accuracy.py), and the
        # switches it lands, served on (arbor/measure/switches.py): what a host
        # that turns one on later needs to know.
        acc = _read_json(Path(session_dir) / f"iter_{iteration}" / "accuracy" / "accuracy.json")
        if acc:
            from arbor.measure.accuracy import brief as accuracy_brief
            doc["accuracy"] = accuracy_brief(acc)
        sw = _read_json(Path(session_dir) / f"iter_{iteration}" / "switches" / "switches.json")
        if sw:
            from arbor.measure.switches import brief as switches_brief
            doc["switches"] = switches_brief(sw)
        if left_out:
            doc["left_out"] = left_out
        if notes:
            doc["notes"] = notes
        (tmp / KEEP_FILE).write_text(json.dumps(doc, indent=1) + "\n")
        os.rename(tmp, out / name)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _write_latest(out, name)
    if carries:
        log(f"{label}carries {len(carries)} earlier change(s) no gate judged on its own: "
            + "; ".join(f"iteration {c['iteration']} ({c['verdict']}): {c['change']}"[:200]
                        for c in carries))
    for var, path in outside.items():
        log(f"{label}{var}={path} is outside the workspace ({workspace}): copied into tables/, "
            "but a host that loads only the worktree does not find it")
    for f in left_out:
        log(f"{label}{f['path']} ({f['bytes']} bytes) is too large for a patch: left out of both, "
            "named in keep.json's left_out")
    log(f"{label}{out / name} (previous {keep_name(previous) if previous is not None else 'none'}, "
        f"parent {parent[:12]}, commit {commit[:12]}, {len(tables)} table(s))")
    return doc


# ── the driver's side ──────────────────────────────────────────────────────────

class Terminated(BaseException):
    """The host sent SIGTERM. A BaseException, so no `except Exception` on the way out
    (an agent call, the gate's bookkeeping) turns the stop into a verdict."""


class Run:
    """One component-mode launch: the deadline, the gate, the heartbeat, SIGTERM, the exit."""

    def __init__(self, config, session_dir, clock=time.time, heartbeat_s: float = HEARTBEAT_S):
        self.config = getattr(config, "config", config)
        self.session = Path(session_dir)
        self.clock = clock
        self.heartbeat_s = heartbeat_s
        self.deadline: float | None = None
        self.heartbeat: Heartbeat | None = None
        self.iteration: int | None = None       # the iteration in flight
        self.iteration_started: float | None = None
        self.reason: str | None = None          # the gate's exit reason
        # After SIGTERM the process ends this soon whatever happens (None: never; tests).
        self.exit_after: float | None = SIGTERM_EXIT_S
        self._terminating = False
        self._signalled: int | None = None      # our server's group, once SIGTERM reached it
        self._prev_handler = None
        self._watchdog: threading.Timer | None = None

    def start(self, time_budget=None) -> None:
        global _heartbeat
        self.session.mkdir(parents=True, exist_ok=True)
        for name in (EXIT_FILE, REFUSED_FILE):
            stale = self.session / name
            if stale.exists():
                stale.unlink()   # a previous run's: this run writes its own exit record
        rec = resolve_deadline(self.config, self.session, self.clock())
        self.deadline = rec["deadline_epoch"]
        seeded = seed_plan(self.config, self.session)
        if seeded:
            log(seeded)
        left = self.deadline - self.clock()
        log(f"run by host {rec['host'] or '(unnamed)'}; deadline {rec['deadline']} ({rec['source']}; "
            f"{left / 60:.0f} min left; first start {rec['first_start_iso']})")
        if time_budget is not None:
            log(f"time_budget_minutes={time_budget} is ignored: the deadline is the clock")
        log("the target, round ends and the round-end A/B are off: the host gates and merges")
        self.heartbeat = Heartbeat(self.session, self.heartbeat_s).start()
        _heartbeat = self.heartbeat
        if threading.current_thread() is threading.main_thread():
            self._prev_handler = signal.signal(signal.SIGTERM, self._on_sigterm)

    def may_start(self, state) -> bool:
        ok, reason, line = gate(self.session, getattr(state, "lessons", []), self.deadline, self.clock())
        log(line + (" -- starting it" if ok else f" -- exiting ({reason})"))
        self.reason = reason
        return ok

    def begin(self, iteration: int, started: float) -> None:
        self.iteration, self.iteration_started = iteration, started
        phase("iteration", iteration=iteration)

    def end(self) -> None:
        self.iteration = self.iteration_started = None
        phase("between")

    def _stop_server(self, timeout: float) -> dict:
        from arbor.execute import preflight
        try:
            return preflight.stop_own_server(server_paths(self.session)[0], timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - the exit record still has to be written
            log(f"could not stop our server: {exc}")
            return {"status": "error", "error": str(exc)}

    def finish(self, reason: str, detail: str = "") -> int:
        """A normal exit: our own server stopped (the last iteration's stays up until the
        next one starts), the exit record written; 0, or EXIT_UNREACHABLE for `unreachable`,
        so the host resumes the run once its model answers again."""
        self._stop_server(timeout=120)
        exit_record(self.session, reason, iteration=None, deadline=self.deadline, detail=detail)
        self.close()
        return EXIT_UNREACHABLE if reason == "unreachable" else 0

    def abort(self) -> None:
        """A crash (any other exception): our server gets SIGTERM, and no exit record,
        since a crash is the host's to resume (MB resumes non-zero exits). The handler
        goes first, so a SIGTERM during the wait ends the process as it would have."""
        self.close()
        self._stop_server(timeout=30)

    def close(self) -> None:
        """The heartbeat stopped and the SIGTERM handler put back. After a SIGTERM the
        watchdog stays armed (a daemon): if the exit itself hangs, it still ends us."""
        global _heartbeat
        if self.heartbeat:
            self.heartbeat.stop()
        if _heartbeat is self.heartbeat:
            _heartbeat = None
        if self._prev_handler is not None and threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._prev_handler)
            self._prev_handler = None

    def _on_sigterm(self, signum, frame) -> None:
        """Record the exit, signal our server's group, and unwind the driver.

        The record comes first and the server gets SIGTERM before anything waits, so
        both have happened even if the unwinding hangs; a watchdog ends the process
        before the host's SIGKILL whatever the interrupted code does.
        """
        if self._terminating:
            return
        self._terminating = True
        _raw_log("SIGTERM from the host: stopping our own server, recording the iteration as "
                 "not measured, exiting 0")
        try:
            exit_record(self.session, "sigterm", iteration=self.iteration, deadline=self.deadline,
                        say=_raw_log)
        except Exception as exc:  # noqa: BLE001
            _raw_log(f"could not write the exit record: {exc}")
        try:
            from arbor.execute import preflight
            self._signalled = preflight.signal_own_server(server_paths(self.session)[0])
        except Exception as exc:  # noqa: BLE001
            _raw_log(f"could not signal our server: {exc}")
        if self.exit_after is not None:
            self._watchdog = threading.Timer(self.exit_after, os._exit, args=(0,))
            self._watchdog.daemon = True
            self._watchdog.start()
        raise Terminated()

    def terminated(self) -> int:
        """After SIGTERM has unwound the driver: book the iteration in flight as NOT
        MEASURED (once), wait briefly for our server's group, and exit 0."""
        state_path = self.session / "session.json"
        if self.iteration is not None:
            try:
                from arbor.state import CampaignState
                state = CampaignState.load(state_path)
                if state.iteration < self.iteration:
                    state.iteration_started = self.iteration_started or 0.0
                    state.record_invalid(
                        iteration=self.iteration, change=f"iteration {self.iteration} (stopped by the host)",
                        tier=state.tier, reason="SIGTERM from the host: the iteration did not finish",
                        claim_class="PROBE", findings=[])
                    state.save(state_path)
            except Exception as exc:  # noqa: BLE001
                log(f"could not record iteration {self.iteration} as not measured: {exc}")
        if self._signalled is not None:
            # The handler sent our server's group SIGTERM already: wait for it, briefly.
            from arbor.execute import preflight
            gone = preflight.wait_group_gone(self._signalled, SIGTERM_SERVER_WAIT_S)
            if gone:
                server_paths(self.session)[0].unlink(missing_ok=True)
            status = "stopped" if gone else "running"
        else:
            status = self._stop_server(timeout=SIGTERM_SERVER_WAIT_S).get("status")
        exit_record(self.session, "sigterm", iteration=self.iteration, deadline=self.deadline,
                    detail=f"our server: {status}")
        self.close()
        return 0
