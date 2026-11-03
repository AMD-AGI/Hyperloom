"""The review gate: a human approves a campaign before Arbor executes anything.

A fresh session stops once its config and pack are loaded, before ``cmd_init`` pins the
validators, seeds ``arbor/config.json`` or starts an agent, a server or a bench. It writes
``<session>/REVIEW.md`` (what the campaign will do and which files say so) and
``<session>/review.json`` (``pending``), and waits. A human reviews and edits the pack, its
knowledge, its roadmap and the config, then approves:

    python3 -m arbor review --session <dir> show
    python3 -m arbor review --session <dir> approve [--note TEXT] [--by NAME]
    python3 -m arbor review --session <dir> reject  [--note TEXT] [--by NAME]

The driver then reloads the campaign from disk, so the reviewed files are the ones pinned and
measured. A rejection ends the run (``review_rejected``). A session that already has
``session.json`` is never gated again: a resume, a host's relaunch and a crash recovery go on.

On by default, everywhere (ArborDevelopment #30, a human's decision, 2026-10-31). Off with
``run --no-review-gate``, ``ARBOR_REVIEW_GATE=0`` or the config's ``"review_gate": false``;
the flag wins over the environment, the environment over the config. A host that reviews the
campaign itself before it starts the run says so with ``component.review: "host"`` (MB pauses
after its dossier pass, holding no lease); the gate then passes at once. Every start that does
not wait records why, in review.json and session.json's ``review``.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REVIEW_FILE = "review.json"
REVIEW_DOC = "REVIEW.md"
ENV = "ARBOR_REVIEW_GATE"
PENDING, APPROVED, REJECTED = "pending", "approved", "rejected"
POLL_S = 15.0
REMIND_S = 600.0
# What a pack holds that a reviewer may change (docs/packs.md): its results/ and caches are not.
PACK_PARTS = ("pack.yaml", "roadmap.yaml", "knowledge", "validator", "eval")

_FALSE = {"0", "false", "no", "off"}
_TRUE = {"1", "true", "yes", "on"}


def log(msg: str) -> None:
    print(f"[arbor] review: {msg}", flush=True)


# ── the switch ─────────────────────────────────────────────────────────────────

def decision(config: dict, cli: bool | None = None, environ=None) -> tuple[bool, str]:
    """Whether a fresh session waits for a human, and what said so: the host
    (``component.review: "host"``), the run's flag, ``ARBOR_REVIEW_GATE``, the config's
    ``review_gate``, in that order; on (``default``) when none says."""
    from arbor import component
    if component.host_reviews(config):
        return False, "host"
    if cli is not None:
        return cli, "--review-gate" if cli else "--no-review-gate"
    raw = (os.environ if environ is None else environ).get(ENV, "").strip().lower()
    if raw in _FALSE:
        return False, f"{ENV}={raw}"
    if raw in _TRUE:
        return True, f"{ENV}={raw}"
    if "review_gate" not in (config or {}):
        return True, "default"
    value = config["review_gate"]
    on = value.strip().lower() not in _FALSE if isinstance(value, str) else bool(value)
    return on, f"config review_gate={value}"


def enabled(config: dict, cli: bool | None = None, environ=None) -> bool:
    return decision(config, cli, environ)[0]


HOST, SKIPPED = "host", "skipped"


def record_bypass(session_dir, how: str) -> None:
    """A new session that starts without waiting says why in review.json (and so in
    session.json): ``host`` when the host reviewed it, else ``skipped`` and the switch."""
    _save(session_dir, {
        "status": HOST if how == "host" else SKIPPED,
        "decided_at": time.time(),
        "decided_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "by": how,
    })


# ── review.json ────────────────────────────────────────────────────────────────

def load(session_dir) -> dict | None:
    """``review.json``; None when there is none or it cannot be read."""
    p = Path(session_dir) / REVIEW_FILE
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def status(session_dir) -> str | None:
    return (load(session_dir) or {}).get("status")


def _save(session_dir, doc: dict) -> None:
    p = Path(session_dir) / REVIEW_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, p)


# ── what is reviewed ───────────────────────────────────────────────────────────

def reviewed_files(campaign, config_path) -> list[Path]:
    """The config and every file of the pack a reviewer may change, sorted."""
    pack_dir = Path(campaign.pack.pack_dir)
    files = [Path(config_path)]
    for part in PACK_PARTS:
        p = pack_dir / part
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
    return sorted(set(files))


def digests(files) -> dict[str, str]:
    out = {}
    for f in files:
        try:
            out[str(f)] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        except OSError:
            out[str(f)] = "unreadable"
    return out


def _script(arbor_root, name: str):
    """A module of scripts/pack/ (the roadmap's renderer and lint), or None without it."""
    path = Path(arbor_root) / "scripts" / "pack" / f"{name}.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"arbor_review_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def roadmap(campaign, arbor_root) -> tuple[str, list[str]]:
    """The pack's roadmap.yaml rendered as the agent would read it, and the lint's findings;
    ("", []) without one."""
    import yaml

    path = Path(campaign.pack.pack_dir) / "roadmap.yaml"
    if not path.is_file():
        return "", []
    try:
        doc = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        return "", [f"{path} does not load: {exc}"]
    lint = _script(arbor_root, "lint_roadmap")
    found = lint.problems(doc) if lint else []
    render = _script(arbor_root, "render_roadmap")
    text = ""
    if render and not found:
        try:
            text = render.render(doc)
        except Exception as exc:   # a renderer bug must not block the review
            found = [f"render_roadmap.py could not render it: {exc}"]
    return text, found


def _program(campaign, arbor_root) -> str:
    """The program the agent reads at iteration 1, or why it cannot be shown."""
    from arbor.pack import render_program
    from arbor.state import CampaignState
    try:
        return render_program(campaign, CampaignState.from_campaign(campaign), arbor_root)
    except Exception as exc:       # a preview that fails must not block the review
        return f"(The program could not be rendered here: {type(exc).__name__}: {exc}.)"


def problems(campaign) -> list[str]:
    """What keeps an approval: in component mode, what the driver itself refuses
    (``validate_campaign``); in any mode, a roadmap the lint rejects. A standalone campaign's
    scripts live in its container, so its paths are not checked here (driver.py does not either)."""
    from arbor import component
    from arbor.pack import validate_campaign
    found = list(validate_campaign(campaign)) if component.is_component(campaign) else []
    pack_dir = Path(campaign.pack.pack_dir)
    if (pack_dir / "roadmap.yaml").is_file():
        _, lint = roadmap(campaign, _arbor_root(campaign))
        found += [f"roadmap.yaml: {p}" for p in lint]
    return found


def _arbor_root(campaign) -> Path:
    """The checkout the pack lives in (<root>/packs/<name>)."""
    return Path(campaign.pack.pack_dir).resolve().parent.parent


# ── the request ────────────────────────────────────────────────────────────────

def _commands(session_dir) -> str:
    s = str(session_dir)
    return (f"    python3 -m arbor review --session {s} show\n"
            f"    python3 -m arbor review --session {s} approve --note \"...\"\n"
            f"    python3 -m arbor review --session {s} reject --note \"...\"\n")


def _document(campaign, config_path, arbor_root, session_dir, files) -> str:
    pack, cfg = campaign.pack, campaign.config
    obj, meas = pack.objective, pack.objective.get("measurement", {}) or {}
    from arbor.pack.loader import agent_skills
    from arbor import component
    lines = [
        f"# Review before Arbor runs: {campaign.identity}",
        "",
        "Nothing has run yet: no validator is pinned, no workspace commit made, no agent, server "
        "or bench started. Review the campaign below, edit any file listed at the end, then "
        "approve or reject:",
        "",
        _commands(session_dir),
        "## The campaign",
        f"- **Goal:** {campaign.goal}",
        f"- **Mode:** {campaign.mode}"
        + (f" (host `{component.block(cfg).get('host', '')}`)" if component.is_component(campaign) else ""),
        f"- **Pack:** `{pack.name}` in `{pack.pack_dir}`",
        f"- **Config:** `{config_path}`",
        f"- **Session:** `{session_dir}`",
        "",
        "## What it optimizes and the bar",
        f"- **Metric:** `{obj.get('metric')}` ({obj.get('direction', 'max')}, {obj.get('unit', '')})",
        f"- **Bar:** `threshold_rel` {meas.get('threshold_rel', 1.01)}, noise band "
        f"{meas.get('noise_band_pct', 1.0)} %",
    ]
    if meas.get("secondary_metric"):
        lines.append(f"- **Second metric:** `{meas['secondary_metric']}` "
                     f"({meas.get('secondary_direction', 'max')})")
    if cfg.get("target") is not None:
        lines.append(f"- **Target:** {cfg.get('target')}")
    lines += [
        "",
        "## What it runs",
        "- **Workload:** " + "; ".join(
            part for part in (
                pack.workload.get("kind", ""),
                f"model `{cfg['model_path']}`" if cfg.get("model_path") else "",
                f"framework `{cfg['framework']}`" if cfg.get("framework") else "",
                f"TP {cfg['tp']}" if cfg.get("tp") is not None else "",
                f"concurrency {cfg['concurrency']}" if cfg.get("concurrency") is not None else "",
            ) if part),
        f"- **Server flags:** `{cfg.get('server_flags', '')}`",
        f"- **Server env:** `{json.dumps(cfg.get('server_env', {}))}`",
    ]
    bench = component.bench_command(cfg) if component.is_component(campaign) else None
    lines.append(f"- **Bench:** `{bench}`" if bench else
                 f"- **Bench:** `{cfg.get('bench_script', '')}`; correctness "
                 f"`{cfg.get('correctness_script', '')}`")
    lines += ["", "## The rules"]
    lines += [f"- {c}" for c in (pack.constraints or [])] or ["- (no constraints)"]
    if cfg.get("directives"):
        lines += ["", "**Directives (config):**"] + [f"- {d}" for d in cfg["directives"]]
    skills = agent_skills(pack)
    lines += ["", f"**Skills:** {', '.join(skills) if skills else '(none)'}"]
    policy = {**(pack.policy or {}), **(cfg.get("policy") or {})}
    lines.append(f"**Policy overrides:** `{json.dumps(policy)}`" if policy else
                 "**Policy overrides:** none (the defaults in arbor/state.py)")
    text, lint = roadmap(campaign, arbor_root)
    if text or lint:
        lines += ["", "## The roadmap (`roadmap.yaml`)"]
        lines += [f"- **Lint:** {p}" for p in lint] or ["- **Lint:** clean"]
        if text:
            lines += ["", text.strip()]
    found = problems(campaign)
    if found:
        lines += ["", "## Problems an approval will refuse until fixed"] + [f"- {p}" for p in found]
    lines += ["", "## The program the agent reads at iteration 1", "", "```markdown",
              _program(campaign, arbor_root).strip(), "```", "",
              "## Files you may edit", ""]
    lines += [f"- `{f}`" for f in files]
    lines += ["", "The driver reloads all of them from disk after the approval.", ""]
    return "\n".join(lines)


def request(campaign, config_path, arbor_root, session_dir) -> Path:
    """Write REVIEW.md and a pending review.json; the path of REVIEW.md."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    files = reviewed_files(campaign, config_path)
    doc_path = session_dir / REVIEW_DOC
    doc_path.write_text(_document(campaign, config_path, arbor_root, session_dir, files))
    _save(session_dir, {
        "status": PENDING,
        "requested_at": time.time(),
        "requested_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": str(config_path),
        "arbor_root": str(arbor_root),
        "identity": campaign.identity,
        "requested_digests": digests(files),
    })
    return doc_path


# ── the wait ───────────────────────────────────────────────────────────────────

def wait(session_dir, deadline: float | None = None, margin: float = 0.0, clock=None,
         sleep=None, poll_s: float = POLL_S, remind_s: float = REMIND_S) -> str:
    """Poll review.json until a human decides: ``approved`` or ``rejected``; ``stop`` when
    ``<session>/STOP`` appears; ``deadline`` when the host's deadline less ``margin`` passes.
    SIGTERM interrupts the sleep through the caller's handler (component.Terminated)."""
    from arbor import component
    clock = clock or time.time       # looked up at the call, so a test can replace them
    sleep = sleep or time.sleep
    session_dir = Path(session_dir)
    started = last = clock()
    component.phase("review")
    log(f"waiting for a human: read {session_dir / REVIEW_DOC}, then\n{_commands(session_dir)}")
    while True:
        st = status(session_dir)
        if st in (APPROVED, REJECTED):
            return st
        if (session_dir / "STOP").exists():
            return "stop"
        now = clock()
        if deadline is not None and now >= deadline - margin:
            return "deadline"
        if now - last >= remind_s:
            log(f"still waiting for a review ({(now - started) / 60:.0f} min): {session_dir / REVIEW_DOC}")
            last = now
        sleep(poll_s)


def outcome_detail(session_dir, outcome: str) -> str:
    """The exit record's detail for a run that ended at the gate."""
    doc = load(session_dir) or {}
    if outcome == REJECTED:
        who, note = doc.get("by", ""), doc.get("note", "")
        return f"rejected by {who or 'a reviewer'}" + (f": {note}" if note else "")
    if outcome == "stop":
        return "STOP while awaiting review"
    return "the deadline passed while awaiting review"


# ── approve, reject, show ──────────────────────────────────────────────────────

def _who(by: str | None) -> str:
    if by:
        return by
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "")


def approve(session_dir, config_path=None, note: str = "", by: str | None = None) -> list[str]:
    """Approve a pending review after re-loading and checking the edited campaign; [] when
    approved, else what keeps it (the status stays pending)."""
    from arbor.pack import load_campaign
    doc = load(session_dir)
    if not doc:
        return [f"no review is pending in {session_dir}"]
    if doc.get("status") != PENDING:
        return [f"the review is already {doc.get('status')}"]
    config_path = config_path or doc.get("config")
    arbor_root = doc.get("arbor_root") or str(Path(config_path).resolve().parent.parent)
    try:
        campaign = load_campaign(config_path, arbor_root)
    except Exception as exc:
        return [f"the edited campaign does not load: {type(exc).__name__}: {exc}"]
    found = problems(campaign)
    if found:
        return found
    files = reviewed_files(campaign, config_path)
    now = digests(files)
    before = doc.get("requested_digests", {})
    doc.update({
        "status": APPROVED,
        "decided_at": time.time(),
        "decided_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "by": _who(by),
        "note": note,
        "approved_digests": now,
        "changed": sorted(f for f in set(now) | set(before) if now.get(f) != before.get(f)),
    })
    _save(session_dir, doc)
    return []


def reject(session_dir, note: str = "", by: str | None = None) -> list[str]:
    doc = load(session_dir)
    if not doc:
        return [f"no review is pending in {session_dir}"]
    if doc.get("status") != PENDING:
        return [f"the review is already {doc.get('status')}"]
    doc.update({"status": REJECTED, "decided_at": time.time(),
                "decided_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "by": _who(by), "note": note})
    _save(session_dir, doc)
    return []


def summary(session_dir) -> dict:
    """The review as session.json keeps it: approved (who, when, the note, which files
    changed), skipped (by which switch) or reviewed by the host; {} when there was none."""
    doc = load(session_dir) or {}
    if doc.get("status") not in (APPROVED, SKIPPED, HOST):
        return {}
    keys = ("status", "requested_at_iso", "decided_at_iso", "by", "note", "changed")
    return {k: doc[k] for k in keys if k in doc}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arbor review",
                                     description="Review a campaign before Arbor runs it.")
    parser.add_argument("action", choices=("show", "approve", "reject"))
    parser.add_argument("--session", required=True, help="the campaign's session directory")
    parser.add_argument("--config", help="the campaign config (default: the one review.json names)")
    parser.add_argument("--note", default="", help="why, for the record")
    parser.add_argument("--by", default=None, help="who decides (default: the login name)")
    args = parser.parse_args(argv)
    session_dir = Path(args.session)
    if args.action == "show":
        doc = load(session_dir)
        if not doc:
            print(f"No review in {session_dir}.")
            return 1
        path = session_dir / REVIEW_DOC
        if path.is_file():
            print(path.read_text())
        print(f"Status: {doc.get('status')}"
              + (f" by {doc.get('by')} at {doc.get('decided_at_iso')}" if doc.get("by") else ""))
        return 0
    if args.action == "approve":
        found = approve(session_dir, args.config, args.note, args.by)
    else:
        found = reject(session_dir, args.note, args.by)
    if found:
        print(f"Not {args.action}d:", file=sys.stderr)
        for p in found:
            print(f"  - {p}", file=sys.stderr)
        return 2
    doc = load(session_dir) or {}
    changed = doc.get("changed") or []
    print(f"{args.action.capitalize()}d by {doc.get('by')}."
          + (f" Files changed since the request: {len(changed)}." if args.action == "approve" else ""))
    return 0
