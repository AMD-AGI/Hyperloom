"""Checkpoints: a session's state at every pass boundary, immutable and verified; ``restore`` lays one into a workspace
on any node; ``pause`` stops a campaign at the next boundary; ``fork`` starts an isolated session from one.

- **When:** the driver writes one after every pass and after a round end (``write_safely``), unless the config says
  ``"checkpoints": false``; ``checkpoint create`` writes one on demand, and a pause writes ``pause-NNNN``. Writing never
  stops a run: a failure is printed, as for ``landed.patch``.
- **Where:** ``<session>/checkpoints/``:
  - ``objects/<sha256>``: each file's bytes, zlib-compressed, named by the sha256 of its bytes. An object cannot change
    unseen: ``verify`` recomputes every digest. A file that did not change costs nothing.
  - ``NNNN.json`` (``arbor.checkpoint/2``): the checkpoint after iteration NNNN, written atomically, with ``id``, the
    digest of its own content. A later write at the same iteration (a round end, a pause) replaces it.
  - ``names/<name>.json``: a named checkpoint, a frozen copy of its manifest, never replaced (``create --name``,
    ``pause-NNNN``, a fork's ``fork``).
  - ``LATEST``; ``restores.jsonl``, each restore.
- **What a checkpoint holds:**
  - ``state``: the session's files rewritten in place (``STATE_FILES``); ``logs``: each append-only log's byte length;
    ``packages``: the landed packages as the last KEEP captured them;
  - ``tree``: the served tree against the root (``tree.patch``, with the change a NOT MEASURED iteration carries;
    ``scratch.patch``; the live ``arbor/config.json``; ``outside.json``) and ``root_tree``, the root commit's tree hash;
  - ``history``: the root commit's raw object and a bundle of ``root..HEAD``, so ``restore`` rebuilds the exact
    ``HEAD``, commit for commit, in another container made from the same image;
  - ``prompts``: the campaign config (secrets redacted), ``pack.yaml``, the agent's and critic's templates in use
    (``arbor/prompts.py``), and their digests;
  - ``dossier``: the dossier's files, which live in the Arbor checkout and are campaign state all the same;
  - ``fingerprint`` (model, framework, image, point, pack, objective, root tree), ``bars``, and ``parent`` (a fork's
    lineage, from ``fork.json``).
- **Never a secret:** the captured config is redacted by key and by value shape (``redact``); a tree whose patches or
  live config hold a credential-shaped value is not captured (``tree_error`` says where), so it cannot be restored.
- **Compatibility:** a session's format is its latest checkpoint's ``session_format``, or 0 without one.
  ``version.CONTINUES["session"]`` is the range this Arbor continues; ``refusals`` refuses a session outside it at
  start, and ``restore`` and ``fork`` a checkpoint outside it.
- **The session lock** (``session.lock``): a driver holds it shared for its life; ``create``, ``restore`` and ``fork``
  take it exclusive, so they refuse while a driver runs on that session.

    python3 -m arbor restore --config C [--checkpoint REF] [--force]
    python3 -m arbor pause (--config C | --session D) [--reason TEXT]
    python3 -m arbor checkpoint list|show|verify|create|fork ...     (REF: an iteration, a name, or latest)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import zlib
from pathlib import Path

SCHEMA = "arbor.checkpoint/2"
FORK_SCHEMA = "arbor.fork/1"
DIR, OBJECTS, LATEST, RESTORES, NAMES = "checkpoints", "objects", "LATEST", "restores.jsonl", "names"
#: <session>/: the session lock; a pause request; the record of a pause; a fork's lineage.
LOCK, PAUSE, PAUSED, FORK = "session.lock", "PAUSE", "paused.json", "fork.json"
#: The exit code of a run that paused (``arbor pause``): not 0, so a launcher can tell a pause from an end.
EXIT_PAUSED = 76
#: The session's files that are rewritten in place: a checkpoint keeps each one's content.
STATE_FILES = ("session.json", "directions.json", "stages.json", "round_end.json", "switches.json",
               "work_order.json", "worker_report.json", "landed.patch", "scratch.patch",
               "workspace_config.json", "outside.json", "landed_packages.json", "review.json", "objective.json",
               "kept.json", "invalids.json", "trees.json")
#: The session's append-only logs: a checkpoint keeps each one's length, so a fork can cut it back.
LOGS = ("ledger.jsonl", "lifecycle.jsonl")
#: A fork never copies these: markers that end or hold a run, a run's own process records, its clock, its lineage,
#: and its checkpoints (a fork starts its own store).
TRANSIENT = {"STOP", PAUSE, PAUSED, "ABORTED.md", "ROUND_STOP.md", "MAP_STOP.md", "heartbeat.json",
             "component_exit.json", "component_refused.json", "role_exit.json", "invocation.json", "clock.json",
             LOCK, FORK, DIR, "chain", "server", "retraction_conflict.json"}
#: The scheduling counters an exploratory fork may reset (``--reset``); each reset is recorded.
RESETTABLE = ("null_streak", "probe_streak", "tier_iters", "barren_streak", "secondary_only_streak", "invalid_streak")
POLICIES = ("exact", "exploratory")
#: What the bars of a checkpoint are: the session's fields that judge the next change.
BAR_FIELDS = ("metric_name", "metric_direction", "baseline", "best_metric", "baseline_secondary", "best_secondary",
              "guard_ref", "tier", "iteration")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# ── secrets ───────────────────────────────────────────────────────────────────────────────────────────────────────────

#: A key whose value is a credential: redacted from the captured config.
SECRET_KEY = re.compile(r"(?i)(^|[_\-.])(api[_-]?key|access[_-]?key|secret|token|password|passwd|credentials?|"
                        r"auth[_-]?token|subscription[_-]?key)($|[_\-.])|^anthropic_|^ocp[_-]apim")
#: A value shaped like a credential, wherever it appears: what kind it is, and its pattern.
SECRET_VALUE = (
    ("an Anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    ("a Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("a GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("a Slack token", re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}")),
    ("a key assigned to an API-key name", re.compile(
        r"(?i)(ocp-apim-subscription-key|x-api-key|anthropic_api_key|anthropic_auth_token|api[_-]?key)"
        r"[\"']?\s*[:=]\s*[\"']?(bearer\s+)?(?!<redacted>)[A-Za-z0-9_\-.]{16,}")),
    ("a 32-hex key beside a key name", re.compile(r"(?i)(key|token|secret)[^\n]{0,24}\b[0-9a-f]{32}\b")),
)
REDACTED = "<redacted>"


def redact(value, key: str = ""):
    """``value`` with every credential taken out: a secret key's value, and any credential-shaped string."""
    if key and SECRET_KEY.search(str(key)) and isinstance(value, (str, int, float)) and str(value):
        return REDACTED
    if isinstance(value, dict):
        return {k: redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for _, pat in SECRET_VALUE:
            value = pat.sub(REDACTED, value)
    return value


def secrets_in(text: str) -> list[str]:
    """What kinds of credential-shaped values ``text`` holds, with the line of each: never any of the value."""
    found = []
    for kind, pat in SECRET_VALUE:
        for m in pat.finditer(text):
            if REDACTED not in m.group(0):
                found.append(f"{kind} on line {text.count(chr(10), 0, m.start()) + 1}")
    return found


# ── the store ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _put(objects: Path, data: bytes) -> str:
    """Store ``data`` once, zlib-compressed, named by the sha256 of its bytes; the digest."""
    sha = hashlib.sha256(data).hexdigest()
    path = objects / sha
    if not path.exists():
        objects.mkdir(parents=True, exist_ok=True)
        tmp = objects / f".{sha}.{os.getpid()}.tmp"
        tmp.write_bytes(zlib.compress(data, 6))
        os.replace(tmp, path)
    return sha


def _get(objects: Path, sha: str | None, check: bool = False) -> bytes | None:
    """An object's bytes, or None when it is not stored; with ``check``, None too when its bytes are not its name."""
    if not sha:
        return None
    path = objects / sha
    if not path.is_file():
        return None
    try:
        data = zlib.decompress(path.read_bytes())
    except zlib.error:
        return None
    if check and hashlib.sha256(data).hexdigest() != sha:
        return None
    return data


def _write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest_of(doc: dict) -> str:
    """A manifest's id: the sha256 of its canonical JSON without the ``id`` itself."""
    body = {k: v for k, v in doc.items() if k != "id"}
    return "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _objects_of(doc: dict) -> dict[str, str]:
    """Every object a manifest names: ``{where: sha}``."""
    out = {}
    for part in ("state", "packages", "dossier_files"):
        for name, sha in (doc.get(part) or {}).items():
            out[f"{part}/{name}"] = sha
    for name, sha in ((doc.get("tree") or {}).get("files") or {}).items():
        out[f"tree/{name}"] = sha
    for name, sha in ((doc.get("history") or {}).get("files") or {}).items():
        out[f"history/{name}"] = sha
    for name, sha in ((doc.get("prompts") or {}).get("files") or {}).items():
        out[f"prompts/{name}"] = sha
    return out


# ── reading ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def numbers(session_dir) -> list[int]:
    """The session's checkpoints, oldest first."""
    return sorted(int(p.stem) for p in (Path(session_dir) / DIR).glob("[0-9]*.json") if p.stem.isdigit())


def names(session_dir) -> dict[str, dict]:
    """The session's named checkpoints: ``{name: record}``."""
    out = {}
    for p in sorted((Path(session_dir) / DIR / NAMES).glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return out


def load(session_dir, ref=None) -> tuple[int, dict] | None:
    """A checkpoint, ``(iteration, manifest)``: ``ref`` an iteration, a name, or None/``latest`` for the newest; None
    when there is none."""
    base = Path(session_dir) / DIR
    if isinstance(ref, str) and not ref.isdigit() and ref != "latest":
        try:
            rec = json.loads((base / NAMES / f"{ref}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return int(rec.get("checkpoint", 0)), rec.get("manifest") or {}
    number = int(ref) if ref not in (None, "latest") else None
    if number is None:
        try:
            number = int((base / LATEST).read_text().strip())
        except (OSError, ValueError):
            found = numbers(session_dir)
            if not found:
                return None
            number = found[-1]
    try:
        return number, json.loads((base / f"{number:04d}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def label(ref) -> str:
    return "the latest" if ref in (None, "latest") else (f"{int(ref):04d}" if str(ref).isdigit() else str(ref))


# ── compatibility ─────────────────────────────────────────────────────────────────────────────────────────────────────

def session_format(session_dir) -> int:
    """The session's format: its latest checkpoint's ``session_format``, or 0 when it has none."""
    found = load(session_dir) if session_dir else None
    return int(found[1].get("session_format", 1)) if found else 0


def refusals(session_dir) -> list[str]:
    """Why this Arbor must not continue the session: [] when its format is in ``version.CONTINUES``."""
    from arbor import version

    low, high = version.CONTINUES["session"]
    fmt = session_format(session_dir)
    if low <= fmt <= high:
        return []
    return [f"the session {session_dir} is in session format {fmt}; this Arbor ({version.VERSION}) continues formats "
            f"{low} to {high}. Continue it with an Arbor whose `python3 -m arbor version --json` lists {fmt} in "
            f"continues.session"]


def verify(session_dir, ref=None) -> tuple[int | None, dict, list[str]]:
    """Check a checkpoint: ``(iteration, manifest, problems)``; no problems means it is whole, unaltered, in this
    Arbor's range and holds no credential."""
    from arbor import version

    found = load(session_dir, ref)
    if not found:
        return None, {}, [f"no checkpoint {label(ref)} in {Path(session_dir) / DIR}"]
    number, doc = found
    problems = []
    fmt = int(doc.get("session_format", 1))
    low, high = version.CONTINUES["session"]
    if not (max(low, 1) <= fmt <= high):
        problems.append(f"checkpoint {number:04d} is in session format {fmt}; this Arbor ({version.VERSION}) reads "
                        f"formats {max(low, 1)} to {high}")
    if "id" in doc and doc["id"] != digest_of(doc):
        problems.append(f"checkpoint {number:04d}'s manifest does not match its id: altered or partial")
    objects = Path(session_dir) / DIR / OBJECTS
    for where, sha in _objects_of(doc).items():
        data = _get(objects, sha, check=True)
        if data is None:
            problems.append(f"{where}: object {str(sha)[:12]} is missing or does not match its digest")
            continue
        if where.startswith(("prompts/", "state/", "tree/config.json", "tree/tree.patch", "tree/scratch.patch")):
            hits = secrets_in(data.decode("utf-8", errors="replace"))
            if hits:
                problems.append(f"{where}: holds credential-shaped values ({', '.join(sorted(set(hits)))})")
    return number, doc, problems


# ── the session lock ──────────────────────────────────────────────────────────────────────────────────────────────────

class Refused(Exception):
    """The command will not run: exit 2, saying why."""


def _flock(fd: int, exclusive: bool, wait: bool) -> bool:
    try:
        import fcntl
    except ImportError:          # not a POSIX host: no lock, as before locks existed
        return True
    flags = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | (0 if wait else fcntl.LOCK_NB)
    try:
        fcntl.flock(fd, flags)
        return True
    except BlockingIOError:
        return False
    except OSError as exc:       # a filesystem without locks: say so, and go on unlocked
        print(f"[arbor] session lock not taken ({exc}); going on without it", file=sys.stderr, flush=True)
        return True


def hold_session(session_dir, held: dict | None = None) -> None:
    """A driver's share of the session lock, held until the process ends (kept open in ``held``)."""
    if not session_dir:
        return
    path = Path(session_dir) / LOCK
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
    except OSError as exc:
        print(f"[arbor] session lock not taken ({exc})", file=sys.stderr, flush=True)
        return
    if _flock(fd, exclusive=False, wait=True):
        if held is not None:
            held.setdefault("session_lock", []).append(fd)
        with contextlib.suppress(OSError):
            os.write(fd, f"{socket.gethostname()} {os.getpid()} driver\n".encode())


@contextlib.contextmanager
def exclusive(session_dir, what: str):
    """The session lock, exclusive, for ``what``; Refused while a driver holds it."""
    path = Path(session_dir) / LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        if not _flock(fd, exclusive=True, wait=False):
            try:
                who = path.read_text(errors="replace").strip().splitlines()[-1]
            except (OSError, IndexError):
                who = "?"
            raise Refused(f"{what}: a driver is running on {session_dir} ({who}); pause or stop it first")
        yield
    finally:
        os.close(fd)


# ── writing ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def _git_dir(container: str, workspace: str) -> str:
    from arbor.execute import workspace as ws
    return ws.git(container, workspace, "rev-parse", "--absolute-git-dir")


def _history(config: dict, objects: Path, root: str, head: str) -> dict:
    """The root commit's raw object and a bundle of ``root..HEAD``: enough to rebuild ``HEAD`` exactly."""
    from arbor.execute import workspace as ws

    container, workspace = config.get("docker_container") or "", config.get("workspace_root") or ""
    raw = ws.run_in(container, ["git", "cat-file", "commit", root], workdir=workspace, capture_output=True)
    if raw.returncode != 0:
        raise RuntimeError(f"git cat-file commit {root[:12]}: {raw.stderr.decode(errors='replace')[-200:]}")
    files = {"root.commit": _put(objects, raw.stdout)}
    out = {"root": root, "head": head, "root_has_parents": b"\nparent " in raw.stdout.split(b"\n\n", 1)[0]}
    if head != root:
        path = f"{_git_dir(container, workspace)}/arbor-checkpoint.bundle"
        try:
            ws.git(container, workspace, "bundle", "create", path, "HEAD", f"^{root}")
            data = ws.run_in(container, ["cat", path], workdir=workspace, capture_output=True)
            if data.returncode != 0:
                raise RuntimeError(f"reading the bundle: {data.stderr.decode(errors='replace')[-200:]}")
            files["history.bundle"] = _put(objects, data.stdout)
        finally:
            ws.run_in(container, ["rm", "-f", path], workdir=workspace, capture_output=True)
    out["files"] = files
    return out


def _tree(config: dict, objects: Path, previous: dict | None) -> tuple[dict | None, dict | None]:
    """The served tree's part and the history's part of a checkpoint; the previous ones when ``head`` and the live
    config are unchanged. ``(None, None)`` without a workspace."""
    container, workspace = config.get("docker_container") or "", config.get("workspace_root") or ""
    if not (container and workspace):
        return None, None
    from arbor.execute import outside
    from arbor.execute import workspace as ws

    head = ws.git(container, workspace, "rev-parse", "HEAD")
    live = json.dumps(ws.read_workspace_config(container, workspace), indent=2, sort_keys=True).encode()
    live_sha = hashlib.sha256(live).hexdigest()
    last = (previous or {}).get("tree") or {}
    files = last.get("files") or {}
    last_history = (previous or {}).get("history")
    if (last.get("head") == head and files.get("config.json") == live_sha and last_history
            and all((objects / sha).is_file() for sha in [*files.values(), *(last_history.get("files") or {}).values()])):
        return dict(last), dict(last_history)
    root = ws.base_commit(container, workspace, config)
    out = {"head": head, "root": root, "root_tree": ws.git(container, workspace, "rev-parse", f"{root}^{{tree}}"),
           "files": {}}
    texts = {"config.json": live}
    with tempfile.TemporaryDirectory() as tmp:
        for name, save in (("tree.patch", ws.save_landed_patch), ("scratch.patch", ws.save_scratch_patch)):
            path = Path(tmp) / name
            save(container, workspace, path, config)
            texts[name] = path.read_bytes()
        hits = {name: secrets_in(data.decode("utf-8", errors="replace")) for name, data in texts.items()}
        hits = {name: h for name, h in hits.items() if h}
        if hits:
            raise RuntimeError("the served tree holds credential-shaped values ("
                               + "; ".join(f"{n}: {', '.join(sorted(set(h)))}" for n, h in hits.items())
                               + "): not captured, so it cannot be restored. Take the credential out of the tree "
                                 "(pass it in the environment)")
        for name, data in texts.items():
            out["files"][name] = _put(objects, data)
        try:
            path = Path(tmp) / outside.RECORD
            outside.record(container, workspace, config, path)
            out["files"][outside.RECORD] = _put(objects, path.read_bytes())
        except Exception as exc:  # noqa: BLE001 - the links are a note on the tree, not the tree
            out["outside_error"] = str(exc)[:300]
    try:
        history = _history(config, objects, root, head)
    except Exception as exc:  # noqa: BLE001 - the patches still rebuild the tree; only the exact HEAD is lost
        history = {"error": str(exc)[:300]}
    return out, history


def _prompts(campaign, session_dir: Path, arbor_root, objects: Path) -> dict:
    """The instructions the campaign runs under: the config (redacted), the pack's manifest, both templates."""
    from arbor import prompts

    config = getattr(campaign, "config", None) or {}
    files, digests = {}, {}
    doc = redact(config)
    files["config.json"] = _put(objects, json.dumps(doc, indent=2, sort_keys=True).encode())
    pack = getattr(getattr(campaign, "pack", None), "pack_dir", None)
    if pack and (Path(pack) / "pack.yaml").is_file():
        files["pack.yaml"] = _put(objects, (Path(pack) / "pack.yaml").read_bytes())
    for role in prompts.ROLES:
        path = prompts.template(config, role, session_dir, prompts.base_dir(campaign))
        files[f"{role}.md"] = _put(objects, path.read_bytes() if path.is_file() else b"")
        digests[role] = {"path": str(path), "builtin": path == prompts.BUILTIN[role]}
    out = {"files": files, "templates": digests}
    out.update(prompts.provenance(campaign, session_dir, arbor_root))
    return out


def _dossier(campaign, session_dir: Path, arbor_root, objects: Path) -> tuple[dict, dict]:
    """The dossier's place and its files by digest."""
    if campaign is None or not arbor_root:
        return {}, {}
    from arbor import chapters

    place = Path(chapters.places(campaign, session_dir, arbor_root)["dossier"])
    info = {"name": place.name, "path": str(place)}
    if not place.is_dir():
        return info, {}
    files = {str(p.relative_to(place)).replace(os.sep, "/"): _put(objects, p.read_bytes())
             for p in sorted(place.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    return info, files


def _fingerprint(campaign, config: dict, tree: dict | None) -> dict:
    env = config.get("server_env") or {}
    pack = getattr(campaign, "pack", None)
    objective = getattr(pack, "objective", None) or {}
    return {"model_path": config.get("model_path"), "model_name": config.get("model_name"),
            "framework": config.get("framework"), "tp": config.get("tp"), "gpu_type": config.get("gpu_type"),
            "image": config.get("docker_image") or config.get("image"), "mode": config.get("mode", "standalone"),
            "pack": getattr(pack, "name", None) or config.get("pack"),
            "objective": objective.get("metric") if isinstance(objective, dict) else None,
            "workload": {"concurrency": config.get("concurrency"), "num_prompts": config.get("num_prompts"),
                         "isl": env.get("ISL"), "osl": env.get("OSL")},
            "root_tree": (tree or {}).get("root_tree")}


def _bars(session_dir: Path) -> dict:
    try:
        state = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: state.get(k) for k in BAR_FIELDS if k in state}


def write(config: dict, session_dir, iteration: int, campaign=None, arbor_root=None, name: str | None = None,
          reason: str = "", tree_from: dict | None = None) -> Path:
    """Write the checkpoint after ``iteration``; its path. ``name`` also files it under ``names/``; ``tree_from``
    (a manifest) gives its tree and history instead of reading the workspace. Raises on a failure (``write_safely``
    does not)."""
    from arbor import version
    from arbor.execute import packages

    session_dir = Path(session_dir)
    base = session_dir / DIR
    objects = base / OBJECTS
    objects.mkdir(parents=True, exist_ok=True)
    if name is not None and (not NAME.match(name) or name == "latest" or name.isdigit()):
        raise Refused(f"a checkpoint's name is letters, digits, '.', '_' and '-', not all digits, not 'latest': "
                      f"{name!r}")
    previous = load(session_dir)
    doc = {"schema": SCHEMA, "session_format": version.INTERFACES["session"], "iteration": int(iteration),
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "arbor": {"version": version.VERSION, "git": version.git_describe()},
           "state": {}, "logs": {}}
    if reason:
        doc["reason"] = reason
    for fname in STATE_FILES:
        path = session_dir / fname
        if path.is_file():
            doc["state"][fname] = _put(objects, path.read_bytes())
    for fname in LOGS:
        path = session_dir / fname
        if path.is_file():
            doc["logs"][fname] = {"bytes": path.stat().st_size}
    landed = session_dir / packages.FILES
    if landed.is_dir():
        doc["packages"] = {str(p.relative_to(landed)).replace(os.sep, "/"): _put(objects, p.read_bytes())
                           for p in sorted(landed.rglob("*")) if p.is_file()}
    iters = [int(p.name[5:]) for p in session_dir.glob("iter_*") if p.name[5:].isdigit()]
    doc["iter_max"] = max(iters) if iters else 0
    if tree_from is not None:
        doc["tree"], doc["history"] = tree_from.get("tree"), tree_from.get("history")
        if tree_from.get("tree_error"):
            doc["tree_error"] = tree_from["tree_error"]
    else:
        try:
            doc["tree"], doc["history"] = _tree(config, objects, previous[1] if previous else None)
        except Exception as exc:  # noqa: BLE001 - the state is still worth keeping; a restore refuses a treeless one
            doc["tree"], doc["history"], doc["tree_error"] = None, None, str(exc)[:500]
            print(f"[arbor] checkpoint after iteration {iteration}: the tree is not captured ({doc['tree_error']})",
                  file=sys.stderr, flush=True)
    if campaign is not None:
        for part, fn in (("prompts", lambda: _prompts(campaign, session_dir, arbor_root, objects)),):
            try:
                doc[part] = fn()
            except Exception as exc:  # noqa: BLE001 - provenance, not the state
                doc[f"{part}_error"] = str(exc)[:300]
        try:
            doc["dossier"], doc["dossier_files"] = _dossier(campaign, session_dir, arbor_root, objects)
        except Exception as exc:  # noqa: BLE001
            doc["dossier_error"] = str(exc)[:300]
    doc["fingerprint"] = _fingerprint(campaign, config or {}, doc.get("tree"))
    doc["bars"] = _bars(session_dir)
    try:
        doc["parent"] = json.loads((session_dir / FORK).read_text(encoding="utf-8")).get("parent")
    except (OSError, ValueError):
        pass
    doc["id"] = digest_of(doc)
    path = base / f"{int(iteration):04d}.json"
    _write_json(path, doc)
    tmp = base / f".{LATEST}.{os.getpid()}.tmp"
    tmp.write_text(f"{int(iteration):04d}\n")
    os.replace(tmp, base / LATEST)
    if name:
        named = base / NAMES / f"{name}.json"
        if named.exists():
            old = json.loads(named.read_text(encoding="utf-8"))
            if old.get("id") != doc["id"]:
                raise Refused(f"the name {name!r} is taken by checkpoint {int(old.get('checkpoint', 0)):04d} "
                              f"({old.get('id', '')[:19]}); names are never moved")
        else:
            _write_json(named, {"name": name, "checkpoint": int(iteration), "id": doc["id"], "at": doc["at"],
                                "reason": reason, "manifest": doc})
    return path


def write_safely(config: dict, session_dir, iteration: int, campaign=None, arbor_root=None) -> Path | None:
    """``write``, unless the config turns checkpoints off; a failure is printed, never raised."""
    if not session_dir or (config or {}).get("checkpoints") is False:
        return None
    try:
        return write(config, session_dir, iteration, campaign, arbor_root)
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not stop a run
        print(f"[arbor] checkpoint after iteration {iteration} not written: {exc}", file=sys.stderr, flush=True)
        return None


# ── restoring ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _restore_history(config: dict, objects: Path, history: dict) -> str | None:
    """Rebuild the exact ``HEAD`` from the history part; the head, or None when it cannot (the caller falls back to
    the patches)."""
    from arbor.execute import workspace as ws

    container, workspace = config.get("docker_container") or "", config.get("workspace_root") or ""
    files = history.get("files") or {}
    raw, root, head = _get(objects, files.get("root.commit"), check=True), history.get("root"), history.get("head")
    if not raw or not root or not head:
        return None
    have = ws.run_in(container, ["git", "cat-file", "-e", f"{root}^{{commit}}"], workdir=workspace,
                     capture_output=True).returncode == 0
    if not have:
        if history.get("root_has_parents"):
            return None          # a host's root with history behind it: the patches, not a dangling commit
        made = ws.run_in(container, ["git", "hash-object", "-t", "commit", "-w", "--stdin"], workdir=workspace,
                         input=raw, capture_output=True)
        if made.returncode != 0 or made.stdout.decode().strip() != root:
            return None
    if head != root:
        bundle = _get(objects, files.get("history.bundle"), check=True)
        if not bundle:
            return None
        path = f"{_git_dir(container, workspace)}/arbor-restore.bundle"
        try:
            put = ws.run_in(container, ["sh", "-c", f"cat > '{path}'"], workdir=workspace, input=bundle,
                            capture_output=True)
            if put.returncode != 0:
                return None
            got = ws.run_in(container, ["git", "fetch", "-q", path, "HEAD"], workdir=workspace, capture_output=True)
            if got.returncode != 0:
                return None
        finally:
            ws.run_in(container, ["rm", "-f", path], workdir=workspace, capture_output=True)
    ws.git(container, workspace, "reset", "-q", "--hard", head)
    return ws.git(container, workspace, "rev-parse", "HEAD")


def _restore_dossier(objects: Path, doc: dict, campaign, session_dir, arbor_root, force: bool) -> str | None:
    """Put the checkpoint's dossier into this checkout when it has none; what was done."""
    files = doc.get("dossier_files") or {}
    if not files or campaign is None or not arbor_root:
        return None
    from arbor import chapters

    place = Path(chapters.places(campaign, session_dir, arbor_root)["dossier"])
    if place.is_dir() and any(place.iterdir()):
        same = all((place / rel).is_file() and hashlib.sha256((place / rel).read_bytes()).hexdigest() == sha
                   for rel, sha in files.items())
        if same:
            return "the checkout's dossier is the checkpoint's"
        if not force:
            return (f"the checkout's dossier {place} differs from the checkpoint's: kept as it is "
                    "(--force puts the checkpoint's in its place, the old one moved aside)")
        place.rename(place.with_name(f"{place.name}.before-restore-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))
    for rel, sha in files.items():
        dest = place / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_get(objects, sha) or b"")
    return f"the dossier written to {place} ({len(files)} files)"


def restore_into(config: dict, session_dir, number=None, force: bool = False, campaign=None,
                 arbor_root=None) -> dict:
    """Lay checkpoint ``number`` (an iteration or a name; the latest by default) of ``session_dir`` into ``config``'s
    workspace. Returns what it did; raises Refused (exit 2) for a checkpoint or a workspace it must not touch,
    RuntimeError on a failure."""
    from arbor.execute import outside, packages
    from arbor.execute import workspace as ws

    session_dir = Path(session_dir)
    number, doc, problems = verify(session_dir, number)
    if number is None or problems:
        raise Refused("; ".join(problems) or f"no checkpoint in {session_dir / DIR}")
    tree = doc.get("tree")
    if not tree:
        raise Refused(f"checkpoint {number:04d} holds no tree"
                      + (f": {doc['tree_error']}" if doc.get("tree_error") else
                         ": its campaign had no workspace when it was written"))
    container, workspace = config.get("docker_container") or "", config.get("workspace_root") or ""
    if not (container and workspace):
        raise Refused("the config names no docker_container and workspace_root to restore into")
    objects = session_dir / DIR / OBJECTS
    root = ws.base_commit(container, workspace, config)
    root_tree = ws.git(container, workspace, "rev-parse", f"{root}^{{tree}}")
    if root_tree != tree.get("root_tree") and not force:
        raise Refused(f"the workspace's root tree is {root_tree[:12]}, the checkpoint's {str(tree.get('root_tree'))[:12]}: "
                      "another image or base. --force restores onto it anyway")
    head = ws.git(container, workspace, "rev-parse", "HEAD")
    done = {"checkpoint": number, "id": doc.get("id"), "iteration": doc.get("iteration"), "container": container,
            "workspace": workspace, "head_before": head, "forced": force}
    if head == tree.get("head"):
        done.update(head_after=head, action="already at the checkpoint", exact=True, verified=True)
        _record(session_dir, done)
        return done
    dirty = ws.git(container, workspace, "status", "--porcelain").strip()
    if (head != root or dirty) and not force:
        raise Refused(f"the workspace is not at its root (HEAD {head[:12]}, root {root[:12]}"
                      f"{', with uncommitted changes' if dirty else ''}): restore lays a checkpoint on a fresh "
                      "workspace. --force resets it to the root first")
    if head != root or dirty:
        ws.git(container, workspace, "reset", "--hard", root)
        ws.git(container, workspace, "clean", "-fd")
    exact = None
    if doc.get("history") and root_tree == tree.get("root_tree"):
        try:
            exact = _restore_history(config, objects, doc["history"])
        except RuntimeError:
            exact = None
    live = json.loads(_get(objects, (tree.get("files") or {}).get("config.json")) or b"{}")
    if exact:
        done["action"], done["exact"] = "restored (exact history)", exact == tree.get("head")
        if ws.read_workspace_config(container, workspace) != live:
            ws.write_workspace_config(container, workspace, live.get("server_env") or {}, live.get("server_flags") or "")
            ws.commit(container, workspace, f"arbor restore: the live config of checkpoint {number:04d}")
            done["exact"] = False
    else:
        if (ws.git(container, workspace, "rev-parse", "HEAD") != root):
            ws.git(container, workspace, "reset", "--hard", root)
        for name in ("tree.patch", "scratch.patch"):
            data = _get(objects, (tree.get("files") or {}).get(name))
            if data:
                run = ws.run_in(container, ["git", "apply", "--binary", "--index", "-"], workdir=workspace,
                                input=data, capture_output=True)
                if run.returncode != 0:
                    raise RuntimeError(f"{name} did not apply: {(run.stderr or b'').decode(errors='replace')[-400:]}")
        ws.write_workspace_config(container, workspace, live.get("server_env") or {}, live.get("server_flags") or "")
        ws.commit(container, workspace, f"arbor restore: checkpoint {number:04d} of {session_dir.name}")
        done["action"], done["exact"] = "restored (patches)", False
    done["head_after"] = ws.git(container, workspace, "rev-parse", "HEAD")
    # The proof: the restored tree's patch is the checkpoint's, byte for byte.
    with tempfile.TemporaryDirectory() as tmp:
        again = Path(tmp) / "tree.patch"
        ws.save_landed_patch(container, workspace, again, config)
        done["verified"] = hashlib.sha256(again.read_bytes()).hexdigest() == (tree.get("files") or {}).get("tree.patch")
    manifest = _get(objects, (doc.get("state") or {}).get(packages.MANIFEST))
    if doc.get("packages") and manifest:
        with tempfile.TemporaryDirectory() as tmp:
            for rel, sha in doc["packages"].items():
                dest = Path(tmp) / packages.FILES / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(_get(objects, sha) or b"")
            (Path(tmp) / packages.MANIFEST).write_bytes(manifest)
            done["packages"] = packages.restore(container, tmp)
    try:
        done["links_prepared"] = outside.prepare(container, workspace, config)
    except Exception as exc:  # noqa: BLE001 - a link the JIT rebuilds is a note, not the tree
        done["links_error"] = str(exc)[:300]
    dossier = _restore_dossier(objects, doc, campaign, session_dir, arbor_root, force)
    if dossier:
        done["dossier"] = dossier
    _record(session_dir, done)
    return done


def _record(session_dir: Path, done: dict) -> None:
    line = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **done}
    (session_dir / DIR).mkdir(parents=True, exist_ok=True)
    with open(session_dir / DIR / RESTORES, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


# ── pause ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def request_pause(session_dir, reason: str = "") -> Path:
    """Ask the session's driver to pause at its next pass boundary: ``<session>/PAUSE``."""
    path = Path(session_dir) / PAUSE
    _write_json(path, {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reason": reason,
                       "by": f"{os.environ.get('USER') or os.environ.get('USERNAME') or '?'}@{socket.gethostname()}"})
    return path


def pause_requested(session_dir) -> dict | None:
    path = Path(session_dir) / PAUSE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {"reason": path.read_text(errors="replace").strip()[:200]}


def pause_now(config: dict, session_dir, iteration: int, campaign=None, arbor_root=None, stop=None) -> dict:
    """The driver's pause at a pass boundary: its server stopped (``stop``), the tree checked committed, the checkpoint
    ``pause-NNNN`` written, ``paused.json`` recorded and ``PAUSE`` cleared. What was done."""
    from arbor.execute import workspace as ws

    session_dir = Path(session_dir)
    asked = pause_requested(session_dir) or {}
    out = {"iteration": int(iteration), "asked": asked, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if stop is not None:
        try:
            stop(config)
            out["server"] = "stopped"
        except Exception as exc:  # noqa: BLE001 - the checkpoint matters more than the server's stop
            out["server"] = f"not stopped: {exc}"[:300]
    container, workspace = config.get("docker_container") or "", config.get("workspace_root") or ""
    if container and workspace:
        try:
            dirty = ws.git(container, workspace, "status", "--porcelain").strip()
            out["tree_clean"] = not dirty
            if dirty:
                out["dirty"] = dirty.splitlines()[:20]
        except RuntimeError as exc:
            out["tree_clean"] = None
            out["tree_error"] = str(exc)[:300]
    name = f"pause-{int(iteration):04d}"
    if (session_dir / DIR / NAMES / f"{name}.json").exists():
        name = f"{name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    try:
        write(config, session_dir, iteration, campaign, arbor_root, name=name,
              reason=f"pause: {asked.get('reason') or 'asked'}")
        out["checkpoint"] = name
        out["id"] = (load(session_dir, name) or (0, {}))[1].get("id")
    except Exception as exc:  # noqa: BLE001 - the pause still ends the run; the record says the checkpoint failed
        out["checkpoint_error"] = str(exc)[:300]
    _write_json(session_dir / PAUSED, out)
    with contextlib.suppress(OSError):
        (session_dir / PAUSE).unlink()
    return out


# ── fork ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _copy_session(src: Path, dst: Path, doc: dict, number: int) -> dict:
    """The parent's session as of checkpoint ``number``: the state files from the checkpoint, the logs cut back to
    its lengths, the iterations up to it, and the other evidence written before it. Copies, never links."""
    objects = src / DIR / OBJECTS
    dst.mkdir(parents=True, exist_ok=True)
    skipped = []
    for name, sha in (doc.get("state") or {}).items():
        data = _get(objects, sha, check=True)
        if data is None:
            raise Refused(f"state/{name}: object {sha[:12]} is missing or altered")
        (dst / name).write_bytes(data)
    for name, info in (doc.get("logs") or {}).items():
        want = int((info or {}).get("bytes", 0))
        path = src / name
        have = path.stat().st_size if path.is_file() else 0
        if have < want:
            raise Refused(f"{name} is {have} bytes, shorter than the checkpoint's {want}: the parent's log was cut")
        with open(path, "rb") as f:
            (dst / name).write_bytes(f.read(want))
    import calendar
    cutoff = calendar.timegm(time.strptime(doc.get("at", "1970-01-01T00:00:00Z"), "%Y-%m-%dT%H:%M:%SZ"))
    keep = set(doc.get("state") or {}) | set(doc.get("logs") or {})
    for entry in sorted(src.iterdir()):
        name = entry.name
        if name in TRANSIENT or name in keep or name.startswith("."):
            continue
        if name.startswith("iter_") and name[5:].isdigit():
            if int(name[5:]) > number:
                skipped.append(name)
                continue
            shutil.copytree(entry, dst / name, symlinks=True)
            continue
        if entry.is_dir():
            def _ignore(directory, items, _cut=cutoff):
                return [i for i in items if os.path.isfile(os.path.join(directory, i))
                        and os.path.getmtime(os.path.join(directory, i)) > _cut + 60]
            shutil.copytree(entry, dst / name, symlinks=True, ignore=_ignore)
        elif entry.is_file():
            if entry.stat().st_mtime > cutoff + 60:
                skipped.append(name)
                continue
            shutil.copy2(entry, dst / name)
    return {"skipped": skipped}


def _apply_policy(dst: Path, policy: str, directions_file, resets: list[str]) -> dict:
    """An exploratory fork's changes to the copied state: a new map, counters reset. Each recorded."""
    done: dict = {"policy": policy, "resets": {}, "directions": None}
    if policy == "exact":
        if directions_file or resets:
            raise Refused("an exact fork keeps the map and the counters: --directions and --reset need "
                          "--fork-policy exploratory")
        return done
    state_path = dst / "session.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    names_ = list(RESETTABLE) if "all" in resets else resets
    for field_ in names_:
        if field_ not in RESETTABLE:
            raise Refused(f"--reset {field_}: one of {', '.join(RESETTABLE)}, or all")
        done["resets"][field_] = {"was": state.get(field_, 0), "now": 0}
        state[field_] = 0
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    if directions_file:
        new = Path(directions_file).read_bytes()
        try:
            parsed = json.loads(new)
        except ValueError as exc:
            raise Refused(f"--directions {directions_file} is not JSON: {exc}") from None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("directions"), list):
            raise Refused(f"--directions {directions_file}: a map is an object with a `directions` list")
        old = dst / "directions.json"
        done["directions"] = {"file": str(Path(directions_file).resolve()),
                              "was": "sha256:" + hashlib.sha256(old.read_bytes()).hexdigest() if old.exists() else None,
                              "now": "sha256:" + hashlib.sha256(new).hexdigest()}
        old.write_bytes(new)
    return done


def _note(dst: Path, iteration: int, text: str, extra: dict) -> None:
    """A NOTE row on the fork's history, as the dealer and the integration write theirs."""
    path = dst / "session.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state.setdefault("lessons", []).append({"iteration": int(iteration), "decision": "NOTE", "ts": time.time(),
                                            "change": text, "fork": extra})
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def fork(source_session, ref, config_path, branch: str | None = None, policy: str = "exact",
         directions_file=None, resets: list[str] | None = None, prompt_profile=None, restore: bool = True,
         force: bool = False) -> dict:
    """Start the session ``config_path`` names from checkpoint ``ref`` of ``source_session``: see the module's
    docstring and docs/operations.md. Returns what was done; raises Refused (exit 2)."""
    from arbor import master, prompts
    from arbor.pack import load_campaign

    resets = list(resets or [])
    if policy not in POLICIES:
        raise Refused(f"--fork-policy {policy!r}: one of {', '.join(POLICIES)}")
    src = Path(source_session).resolve()
    number, doc, problems = verify(src, ref)
    if number is None or problems:
        raise Refused("; ".join(problems) or f"no checkpoint {label(ref)} in {src / DIR}")
    config_path = str(Path(config_path).resolve())
    arbor_root = str(master._find_arbor_root(Path(config_path)))
    campaign = load_campaign(config_path, arbor_root)
    config = campaign.config
    dst = Path(campaign.session_root).resolve()
    if not str(campaign.session_root):
        raise Refused("the fork's config names no session_root")
    if dst == src:
        raise Refused("the fork's session_root is the parent's: a fork needs a session of its own")
    if dst.exists() and any(p.name != LOCK for p in dst.iterdir()):
        raise Refused(f"the fork's session {dst} is not empty: a fork starts a new session")
    parent_cfg = json.loads(_get(src / DIR / OBJECTS, ((doc.get("prompts") or {}).get("files") or {})
                                 .get("config.json")) or b"{}")
    same_ws = (config.get("docker_container"), config.get("workspace_root")) == (
        parent_cfg.get("docker_container"), parent_cfg.get("workspace_root"))
    if same_ws and config.get("docker_container") and not force:
        raise Refused("the fork's container and workspace are the parent's: each branch needs its own, so neither "
                      "edits the other's tree. --force if the parent will not run again")
    from arbor import chapters
    place = Path(chapters.places(campaign, dst, arbor_root)["dossier"])
    parent_place = (doc.get("dossier") or {}).get("path")
    if parent_place and place.resolve() == Path(parent_place).resolve() and not force:
        raise Refused(f"the fork's dossier is the parent's ({place}): give the fork its own `dossier` name, so the "
                      "two never write the same files. --force shares it")
    branch = branch or dst.name
    with exclusive(dst, "fork"):
        copied = _copy_session(src, dst, doc, number)
        changes = _apply_policy(dst, policy, directions_file, resets)
        if prompt_profile:
            prof = Path(prompt_profile)
            if not prof.is_dir():
                raise Refused(f"--prompt-profile {prompt_profile} is not a directory")
            shutil.copytree(prof, dst / prompts.SESSION_PROFILE)
        if (doc.get("dossier_files") or {}) and not place.exists():
            objects = src / DIR / OBJECTS
            for rel, sha in doc["dossier_files"].items():
                target = place / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_get(objects, sha) or b"")
            changes["dossier"] = str(place)
        # The objects the parent's checkpoint names, so this session's own store can restore its tree.
        for sha in _objects_of(doc).values():
            data = _get(src / DIR / OBJECTS, sha, check=True)
            if data is not None:
                _put(dst / DIR / OBJECTS, data)
        lineage = {"schema": FORK_SCHEMA, "branch_id": branch,
                   "parent": {"session": str(src), "checkpoint": number, "checkpoint_id": doc.get("id"),
                              "identity": parent_cfg.get("identity"), "iteration": doc.get("iteration")},
                   "policy": policy, "resets": changes["resets"], "directions": changes["directions"],
                   "prompt": {"profile": str(Path(prompt_profile).resolve()) if prompt_profile else None,
                              "digest": prompts.prompt_digest(config, dst, prompts.base_dir(campaign))},
                   "skipped": copied["skipped"],
                   "forked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        from arbor import version
        lineage["arbor"] = {"version": version.VERSION, "git": version.git_describe()}
        _write_json(dst / FORK, lineage)
        text = (f"fork {branch}: from {src.name} checkpoint {number:04d} ({policy}"
                + (f"; reset {', '.join(changes['resets'])}" if changes["resets"] else "")
                + ("; a new map" if changes["directions"] else "")
                + ("; its own prompt profile" if prompt_profile else "") + ")")
        _note(dst, int(doc.get("iteration") or number), text, {"branch_id": branch, "parent": lineage["parent"],
                                                               "policy": policy})
        write(config, dst, int(doc.get("iteration") or number), campaign, arbor_root, name="fork",
              reason=text, tree_from=doc)
        done = {"branch_id": branch, "session": str(dst), "parent": lineage["parent"], "policy": policy,
                "resets": changes["resets"], "directions": changes["directions"], "skipped": copied["skipped"],
                "next_iteration": int(doc.get("iteration") or number) + 1}
        if restore and config.get("docker_container") and config.get("workspace_root"):
            try:
                done["restore"] = restore_into(config, dst, "fork", force=force, campaign=campaign,
                                               arbor_root=arbor_root)
            except Refused as exc:
                done["restore_refused"] = str(exc)
    return done


# ── commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def _session_arg(args) -> Path:
    if getattr(args, "session", None):
        return Path(args.session)
    from arbor import master
    return Path(master.session_of(args.config))


def _ref(value):
    return None if value in (None, "latest") else (int(value) if str(value).isdigit() else value)


def restore_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arbor restore",
                                description="Lay a checkpoint of the config's session into its workspace, on any node.")
    p.add_argument("--config", required=True, help="the campaign's config: its session, container and workspace")
    p.add_argument("--checkpoint", default=None,
                   help="the checkpoint: an iteration, a name, or latest (the default)")
    p.add_argument("--force", action="store_true",
                   help="restore onto another image's tree, reset a workspace that is not at its root first, and "
                        "put the checkpoint's dossier in place of a different one")
    p.add_argument("--json", action="store_true", help="what was done, as JSON")
    return p


def main_restore(argv: list[str] | None = None) -> int:
    """``python3 -m arbor restore``."""
    args = restore_parser().parse_args(argv)
    from arbor import master
    from arbor.pack import load_campaign

    try:
        root = str(master._find_arbor_root(Path(args.config).resolve()))
        campaign = load_campaign(str(Path(args.config).resolve()), root)
        session = campaign.session_root
        with exclusive(session, "restore"):
            done = restore_into(campaign.config, session, _ref(args.checkpoint), args.force, campaign, root)
    except Refused as exc:
        print(f"[arbor] restore refused: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"[arbor] restore failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(done, indent=1, sort_keys=True))
    else:
        print(f"[arbor] checkpoint {done['checkpoint']:04d} (iteration {done['iteration']}): {done['action']}; "
              f"HEAD {str(done['head_before'])[:12]} -> {str(done['head_after'])[:12]}"
              + ("; exact" if done.get("exact") else "")
              + ("" if done.get("verified", True) else "; WARNING: the restored tree differs from the checkpoint's")
              + (f"; {done['dossier']}" if done.get("dossier") else ""))
    return 0 if done.get("verified", True) else 1


def pause_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arbor pause", description="Pause a campaign at its next pass boundary: its "
                                "server stopped, a checkpoint named pause-NNNN, exit 76. `arbor run` continues it.")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--config", help="the campaign's config")
    where.add_argument("--session", help="the campaign's session directory")
    p.add_argument("--reason", default="", help="why, for the record")
    p.add_argument("--cancel", action="store_true", help="take back a pause not yet acted on")
    return p


def main_pause(argv: list[str] | None = None) -> int:
    """``python3 -m arbor pause``."""
    args = pause_parser().parse_args(argv)
    try:
        session = _session_arg(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[arbor] pause: {exc}", file=sys.stderr)
        return 2
    if not session.is_dir():
        print(f"[arbor] pause: no session at {session}", file=sys.stderr)
        return 2
    if args.cancel:
        existed = (session / PAUSE).exists()
        with contextlib.suppress(OSError):
            (session / PAUSE).unlink()
        print(f"[arbor] pause {'taken back' if existed else 'was not asked'}: {session}")
        return 0
    request_pause(session, args.reason)
    print(f"[arbor] pause asked: {session / PAUSE}. The driver pauses after the iteration in flight.")
    return 0


ACTIONS = {"list": "every checkpoint, its names and the session's lineage",
           "show": "one checkpoint's manifest",
           "verify": "check a checkpoint's objects, digests and format, and that it holds no credential",
           "create": "write a checkpoint now, optionally named (no driver may be running on the session)",
           "fork": "a new session from a checkpoint, in its own workspace"}


def _action_args(s: argparse.ArgumentParser, name: str) -> None:
    """``name``'s arguments on ``s``: the subcommand's, and the payload's parser's (arbor/cli/payload.py)."""
    if name in ("list", "show", "verify"):
        where = s.add_mutually_exclusive_group(required=True)
        where.add_argument("--config", help="the campaign's config")
        where.add_argument("--session", help="the session directory")
        if name != "list":
            s.add_argument("checkpoint", nargs="?", default=None, help="an iteration, a name, or latest")
        if name == "verify":
            s.add_argument("--all", action="store_true", help="every checkpoint and name")
        return
    if name == "create":
        s.add_argument("--config", required=True, help="the campaign's config")
        s.add_argument("--name", default=None, help="a name that pins it (letters, digits, . _ -)")
        s.add_argument("--reason", default="", help="why, for the record")
        return
    s.add_argument("--from", dest="source", required=True,
                   help="the parent: its session directory, or its config")
    s.add_argument("--checkpoint", default=None, help="the parent's checkpoint: an iteration, a name, or latest")
    s.add_argument("--config", required=True,
                   help="the fork's config: its own session_root, container, workspace and dossier")
    s.add_argument("--branch", default=None, help="the branch's id (default: the session directory's name)")
    s.add_argument("--fork-policy", choices=POLICIES, default="exact",
                   help="exact (the default): bars, counters and map kept; exploratory: also --directions, --reset")
    s.add_argument("--directions", default=None, help="exploratory: a map (directions.json) in place of the parent's")
    s.add_argument("--reset", action="append", default=[], metavar="COUNTER",
                   help=f"exploratory: set a scheduling counter to 0 ({', '.join(RESETTABLE)}, or all); repeatable")
    s.add_argument("--prompt-profile", default=None,
                   help="a directory with claude_loop_prompt.md and/or critic_prompt.md, copied into the fork")
    s.add_argument("--no-restore", action="store_true",
                   help="do not restore the tree into the fork's workspace now (`arbor restore` later)")
    s.add_argument("--force", action="store_true",
                   help="share the parent's workspace or dossier, or restore onto another image's tree")


def parser() -> argparse.ArgumentParser:
    """``arbor checkpoint``: list, show, verify, create, fork."""
    p = argparse.ArgumentParser(prog="arbor checkpoint", description="A session's checkpoints: list, show, verify, "
                                "create one now, or fork a new session from one.")
    sub = p.add_subparsers(dest="action", required=True)
    for name, text in ACTIONS.items():
        _action_args(sub.add_parser(name, help=text, description=text), name)
    return p


def action_parser(action: str) -> argparse.ArgumentParser:
    """One action's parser on its own, for a payload (arbor/cli/payload.py)."""
    if action not in ACTIONS:
        raise ValueError(f"checkpoint: `action` must be one of {', '.join(ACTIONS)}, not {action!r}")
    p = argparse.ArgumentParser(prog=f"arbor checkpoint {action}", description=ACTIONS[action])
    _action_args(p, action)
    return p


def _list(session: Path) -> list[str]:
    lines = []
    lineage = None
    with contextlib.suppress(OSError, ValueError):
        lineage = json.loads((session / FORK).read_text(encoding="utf-8"))
    if lineage:
        par = lineage.get("parent") or {}
        lines.append(f"fork {lineage.get('branch_id')} of {par.get('session')} at checkpoint "
                     f"{int(par.get('checkpoint') or 0):04d} ({lineage.get('policy')})")
    by_number = {}
    for nm, rec in names(session).items():
        by_number.setdefault(int(rec.get("checkpoint", 0)), []).append(nm)
    for n in numbers(session):
        _, doc = load(session, n) or (n, {})
        tree = doc.get("tree") or {}
        lines.append(f"{n:04d}  format {doc.get('session_format')}  {doc.get('at', '')}  "
                     f"head {str(tree.get('head', ''))[:12] or '-'}  {str(doc.get('id', ''))[:19]}"
                     + (f"  names: {', '.join(sorted(by_number[n]))}" if n in by_number else ""))
    for nm, rec in sorted(names(session).items()):
        if int(rec.get("checkpoint", 0)) not in numbers(session) or (load(session, rec.get("checkpoint")) or (0, {}))[1] \
                .get("id") != rec.get("id"):
            lines.append(f"  {nm}: iteration {int(rec.get('checkpoint', 0)):04d}, {str(rec.get('id', ''))[:19]} (frozen)")
    return lines


def main(argv: list[str] | None = None) -> int:
    """``python3 -m arbor checkpoint ...`` (and ``python3 -m arbor.checkpoints``)."""
    args = parser().parse_args(argv)
    try:
        if args.action == "fork":
            src = Path(args.source)
            if src.is_file():
                from arbor import master
                src = Path(master.session_of(str(src)))
            done = fork(src, _ref(args.checkpoint), args.config, args.branch, args.fork_policy, args.directions,
                        args.reset, args.prompt_profile, restore=not args.no_restore, force=args.force)
            print(json.dumps(done, indent=1, sort_keys=True))
            print(f"[arbor] fork {done['branch_id']}: session {done['session']}, next iteration "
                  f"{done['next_iteration']}" + (f"; restore refused: {done['restore_refused']}"
                                                 if done.get("restore_refused") else ""), file=sys.stderr)
            return 0
        if args.action == "create":
            from arbor import master
            from arbor.pack import load_campaign
            from arbor.state import CampaignState
            root = str(master._find_arbor_root(Path(args.config).resolve()))
            campaign = load_campaign(str(Path(args.config).resolve()), root)
            session = Path(campaign.session_root)
            with exclusive(session, "checkpoint create"):
                iteration = CampaignState.load(session / "session.json").iteration
                path = write(campaign.config, session, iteration, campaign, root, name=args.name,
                             reason=args.reason or "create")
            doc = json.loads(path.read_text(encoding="utf-8"))
            print(f"[arbor] checkpoint {iteration:04d} {doc['id'][:19]}" + (f" named {args.name}" if args.name else "")
                  + (f"; the tree is not captured: {doc['tree_error']}" if doc.get("tree_error") else ""))
            return 0
        session = _session_arg(args)
        if args.action == "list":
            print("\n".join(_list(session)) or "no checkpoints")
            return 0
        if args.action == "show":
            found = load(session, _ref(args.checkpoint))
            if not found:
                print(f"no checkpoint {label(args.checkpoint)}", file=sys.stderr)
                return 1
            print(json.dumps(found[1], indent=1, sort_keys=True))
            return 0
        if args.action == "verify":
            refs = ([*numbers(session), *names(session)] if args.all else [_ref(args.checkpoint)])
            bad = 0
            for ref in refs:
                n, doc, problems = verify(session, ref)
                print(f"{label(ref)}: " + ("ok " + str(doc.get("id", ""))[:19] if not problems else
                                           "FAILED: " + "; ".join(problems)))
                bad += bool(problems)
            return 1 if bad else 0
    except Refused as exc:
        print(f"[arbor] checkpoint {args.action} refused: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"[arbor] checkpoint {args.action} failed: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
