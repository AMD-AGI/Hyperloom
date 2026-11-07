"""The worker slots: where a standalone master's workers run, the session
and config each dealt order gets, and the trees that cross between the master's workspace and a slot's.

**A slot** is a share of the node: its own container (the workspace is the container's installed tree, so two workers
never share one), its cards (``tp`` each) and its port. The launcher starts the containers from the campaign's image,
each with its workspace git-inited, as it starts the master's. The config's ``workers`` block says where they are:

- ``count``: how many (default: as many as the cards allow, ``len(gpu_ids) // tp``);
- ``gpu_ids``: the cards the slots share out, ``tp`` each (default 0-7);
- ``container``: each slot's container, with ``{container}`` and ``{k}`` filled (default ``{container}-w{k}``);
- ``port_step``: slot k serves on the config's port plus k times this (default 10);
- ``slots``: each slot in full instead (``docker_container``, ``gpu_ids``, ``port``, and ``workspace_root`` when it
  differs from the campaign's);
- ``size``: the directions dealt in one order (default 2);
- ``reserve_minutes``: the run's time kept for the integration: no order is dealt with less left, and a dealt order's
  budget ends that much before the run's (default 45);
- ``ab_rounds``: the measured pairs of each integration A/B (default ``arbor.ab``'s, 3).

**An order's worker** runs in a session of its own, ``<master session>/workers/<order id>/session``, from a config
written beside it: the campaign's, with the slot's container, cards and port, that session, a dossier of its own there,
and the base's server flags and env. Its workspace starts at the round's base: reset to its first commit (the image),
then the base applied as one commit (``arbor base: ...``), before the worker's own init and baseline.

**The base** leaves the master's workspace as ``landed.patch`` does (the served files, from the first commit to HEAD),
with its ``arbor/config.json``. **An addition** comes back from a slot's workspace as the diff from that base commit to
its HEAD, with its config at both. Editable packages other than the workspace (``arbor/execute/packages.py``) do not
cross yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from arbor import component
from arbor.execute import workspace as ws

KEY = "workers"
BASE_PREFIX = "arbor base:"
DEFAULTS = {"size": 2, "reserve_minutes": 45, "port_step": 10, "container": "{container}-w{k}"}


@dataclass(frozen=True)
class Slot:
    index: int
    container: str
    workspace: str
    gpu_ids: tuple[int, ...]
    port: int

    @property
    def name(self) -> str:
        return f"w{self.index}"


def block(config: dict) -> dict:
    raw = (config or {}).get(KEY)
    return raw if isinstance(raw, dict) else {}


def enabled(config: dict) -> bool:
    return bool(block(config))


def size(config: dict) -> int:
    return int(block(config).get("size") or DEFAULTS["size"])


def reserve_s(config: dict) -> float:
    value = block(config).get("reserve_minutes")
    return 60.0 * float(DEFAULTS["reserve_minutes"] if value is None else value)


def plan(config: dict) -> list[Slot]:
    """The config's slots; ValueError naming everything wrong with them."""

    b, tp = block(config), int(config.get("tp") or 1)
    container, workspace = config.get("docker_container", ""), config.get("workspace_root", "")
    if b.get("slots"):
        out = [Slot(k, str(s.get("docker_container") or ""), str(s.get("workspace_root") or workspace),
                    tuple(component.gpu_list(s.get("gpu_ids")) or ()), int(s.get("port") or 0))
               for k, s in enumerate(b["slots"], 1)]
    else:
        pool = component.gpu_list(b.get("gpu_ids")) or list(range(8))
        count = len(pool) // tp if b.get("count") is None else int(b["count"])
        step, pattern = int(b.get("port_step") or DEFAULTS["port_step"]), str(b.get("container")
                                                                              or DEFAULTS["container"])
        base = int(config.get("port", 8888))
        out = [Slot(k, pattern.format(container=container, k=k), workspace,
                    tuple(pool[(k - 1) * tp:k * tp]), base + step * k) for k in range(1, count + 1)]
    why = []
    if not out:
        why.append("workers: no slot (count, or gpu_ids for tp cards each)")
    for s in out:
        if not s.container or not s.workspace:
            why.append(f"workers: slot {s.name} has no container or workspace")
        if len(s.gpu_ids) != tp:
            why.append(f"workers: slot {s.name} has cards {list(s.gpu_ids)}, not tp {tp}")
        if not s.port:
            why.append(f"workers: slot {s.name} has no port")
    for what, values in (("container", [(s.container, s.workspace) for s in out]), ("port", [s.port for s in out]),
                         ("card", [g for s in out for g in s.gpu_ids])):
        if len(set(values)) != len(values):
            why.append(f"workers: two slots share a {what}")
    if container and workspace and any((s.container, s.workspace) == (container, workspace) for s in out):
        why.append("workers: a slot is the master's own workspace")
    if why:
        raise ValueError("; ".join(why))
    return out


# ── an order's session and config ─────────────────────────────────────────────

def order_dir(session_dir, order_id: str) -> Path:
    return Path(session_dir) / KEY / order_id


def worker_config(config: dict, slot: Slot, where: Path, base_config: dict) -> dict:
    """The config a dealt order's worker runs from: the campaign's, on ``slot``, in its own session under ``where``,
    serving the base's flags and env (without card masks: the slot's ``gpu_ids`` set them)."""

    out = {k: v for k, v in config.items() if k != KEY}
    out.update(docker_container=slot.container, workspace_root=slot.workspace, port=slot.port,
               gpu_ids=",".join(map(str, slot.gpu_ids)), session_root=str(where / "session"),
               dossier=str(where / "dossier"), review_gate=False)
    if base_config:
        out["server_env"] = component.strip_masks(base_config.get("server_env") or {})
        out["server_flags"] = base_config.get("server_flags") or ""
    return out


def write_config(where: Path, doc: dict) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    path = where / "config.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


# ── trees across workspaces ───────────────────────────────────────────────────

def export_base(config: dict, out: Path) -> dict:
    """The master's tree, for a round's workers: ``out/base.patch`` (the served files, its first commit to HEAD), its
    ``arbor/config.json``, and its HEAD."""

    container, root = config.get("docker_container", ""), config.get("workspace_root", "")
    out.mkdir(parents=True, exist_ok=True)
    patch = out / "base.patch"
    ws.save_landed_patch(container, root, patch, config)
    served = ws.read_workspace_config(container, root) or {      # unseeded: what init would seed
        "server_env": component.seed_env(config), "server_flags": config.get("server_flags", "")}
    doc = {"commit": ws.git(container, root, "rev-parse", "HEAD"), "patch": str(patch), "config": served}
    (out / "base.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def _apply(container: str, root: str, patch: bytes, *extra: str) -> tuple[bool, str]:
    r = ws.run_in(container, ["git", "apply", "--binary", *extra, "-"], workdir=root, input=patch, capture_output=True)
    return r.returncode == 0, (r.stderr or r.stdout or b"").decode(errors="replace").strip()


def prepare(slot: Slot, base: dict, label: str) -> str:
    """The slot's workspace at the round's base: reset to its first commit, the base's patch and config committed as
    one commit; that commit. RuntimeError when the base does not apply."""

    c, root = slot.container, slot.workspace
    ws.git(c, root, "reset", "-q", "--hard", ws._root_commit(c, root))
    ws.git(c, root, "clean", "-fdq")
    patch = Path(base["patch"]).read_bytes() if base.get("patch") else b""
    if patch.strip():
        ok, err = _apply(c, root, patch)
        if not ok:
            raise RuntimeError(f"the base does not apply in slot {slot.name}: {err}")
    cfg = base.get("config") or {}
    env = component.strip_masks(cfg.get("server_env") or {})       # the slot's gpu_ids set its cards
    ws.write_workspace_config(c, root, env, cfg.get("server_flags") or "")
    return ws.commit(c, root, f"{BASE_PREFIX} {label}")


def addition(slot: Slot, base_commit: str, config: dict, out: Path) -> dict | None:
    """What a worker kept in ``slot`` on top of ``base_commit``: ``out`` (the served files' diff), the config at both
    ends, its HEAD; None when it kept nothing."""

    c, root = slot.container, slot.workspace
    head = ws.git(c, root, "rev-parse", "HEAD")
    out.parent.mkdir(parents=True, exist_ok=True)
    ws._write_diff(c, root, out, "--binary", base_commit, head, "--", *ws.landed_pathspec(config))
    before = ws.config_at(c, root, base_commit) or {}
    after = ws.config_at(c, root, head) or {}
    from arbor.parallel.trees import flags_from_workspace
    if not out.read_bytes().strip() and flags_from_workspace(before) == flags_from_workspace(after):
        out.unlink()                # no file, no addition (a run's records once read an empty one as one)
        return None
    return {"patch": str(out), "base_config": before, "config": after, "head": head, "base": base_commit}


def stop(slot: Slot) -> None:
    """Stop whatever server the slot's container still runs, so its cards are free for the master's A/Bs."""

    stop_server(slot.container, list(slot.gpu_ids))


def stop_master(config: dict) -> None:
    """Stop whatever server the master's container still runs, before the slots take the cards."""

    stop_server(config.get("docker_container", ""), component.gpu_list(config.get("gpu_ids")))


def stop_server(container: str, gpu_ids: list[int] | None) -> None:
    if not container or container == ws.HOST:
        return
    from arbor.execute.preflight import stop_server as stop_it
    stop_it(container, gpu_ids=gpu_ids)
