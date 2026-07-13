from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("optimize-submit")

from . import config as _config
from . import records as _records

globals().update({k: v for k, v in vars(_config).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_records).items() if not k.startswith("__")})

from . import delivery as _delivery

globals().update({k: v for k, v in vars(_delivery).items() if not k.startswith("__")})

def _sandbox_duration_seconds(last_task: dict) -> float | None:
    """SaFE-side sandbox wallclock = finishedAt - startedAt (from the task API).
    None when either field is missing/unparseable so we don't fabricate a duration.

    Args:
        last_task (dict): SaFE task record carrying ``startedAt``/``finishedAt``.

    Returns:
        float | None: Rounded duration in seconds, or ``None`` when either
        timestamp is missing/unparseable or the delta is negative.
    """
    from datetime import datetime

    start = (last_task or {}).get("startedAt") or ""
    end = (last_task or {}).get("finishedAt") or ""
    if not start or not end:
        return None
    try:
        # SaFE serializes UTC with trailing 'Z'; fromisoformat needs '+00:00'.
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except Exception:
        return None
    delta = (e - s).total_seconds()
    return round(delta, 1) if delta >= 0 else None


def _find_hyperloom_commit_sha(start: Path) -> str:
    """Resolve the Hyperloom git SHA the sandbox cloned (for audit fields).

    First hit wins: (1) hyperloom_source_commit.txt written by the agent (depth
    varies by which fallback collected it), then (2) the CI runner env
    (HYPERLOOM_SOURCE_REF, else GITHUB_SHA).

    Args:
        start (Path): A path near the artifact, used to derive sibling
            ``hyperloom_source_commit.txt`` candidates.

    Returns:
        str: The resolved SHA-shaped commit string, or ``""`` when none is
        found.
    """
    candidates = [
        start.parent / "hyperloom_source_commit.txt",
        start.parent / "session" / "hyperloom_source_commit.txt",
        start.parent.parent / "hyperloom_source_commit.txt",
    ]
    for sha_path in candidates:
        if not sha_path.exists():
            continue
        try:
            sha = sha_path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        # Accept only SHA-shaped strings; never trust a corrupted file.
        if 7 <= len(sha) <= 80 and all(c in "0123456789abcdef" for c in sha.lower()):
            return sha

    # Fallback: CI runner env. HYPERLOOM_SOURCE_REF is the pinned commit
    # (preferred); GITHUB_SHA is the unconditional fallback.
    for env_var in ("HYPERLOOM_SOURCE_REF", "GITHUB_SHA"):
        env_sha = (os.environ.get(env_var) or "").strip()
        if 7 <= len(env_sha) <= 80 and all(c in "0123456789abcdef" for c in env_sha.lower()):
            return env_sha
    return ""


def _backfill_ci_metrics_file(path: Path, rec: SubmissionRecord) -> None:
    """Backfill task metadata (model, image, hyperloom_commit, category,
    sandbox_duration_seconds) into ci_metrics.json / session_breakdown.json /
    manifest.json so each artifact is self-describing.

    Writes the right shape per filename: ci_metrics.json → flat top-level;
    session_breakdown.json → under session_meta; manifest.json → flat top-level
    (V2 cli schema; category/duration are extra keys V2 ignores on re-read).

    Args:
        path (Path): Target JSON file (ci_metrics / session_breakdown /
            manifest); other names and missing files are no-ops.
        rec (SubmissionRecord): Record supplying the metadata to backfill.
    """
    if path.name not in ("ci_metrics.json", "session_breakdown.json", "manifest.json"):
        return
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return

    changed = False
    detected = rec.detected or {}
    image = detected.get("image") or ""
    hyperloom_sha = _find_hyperloom_commit_sha(path)
    image_tag = image.split("/")[-1] if image else ""

    if path.name == "ci_metrics.json":
        for key, value in {
            "model": rec.model,
            "task_id": rec.task_id,
            "claw_session_id": rec.claw_session_id,
        }.items():
            if value and not data.get(key):
                data[key] = value
                changed = True
        for key in ("framework", "tp"):
            if detected.get(key) is not None and data.get(key) is None:
                data[key] = detected.get(key)
                changed = True
        if image and not data.get("image"):
            data["image"] = image
            changed = True
        if hyperloom_sha and not data.get("hyperloom_commit"):
            data["hyperloom_commit"] = hyperloom_sha
            changed = True
        if rec.category and not data.get("category"):
            data["category"] = rec.category
            changed = True
        if rec.sandbox_duration_seconds is not None and not data.get("sandbox_duration_seconds"):
            data["sandbox_duration_seconds"] = rec.sandbox_duration_seconds
            changed = True

    elif path.name == "session_breakdown.json":
        # Keyed by a `session_meta` sub-dict; only write empty fields so we don't
        # overwrite what the V2 collectors filled in.
        meta = data.get("session_meta")
        if not isinstance(meta, dict):
            meta = {}
            data["session_meta"] = meta
        if image and not meta.get("image"):
            meta["image"] = image
            changed = True
        if image_tag and not meta.get("image_id"):
            meta["image_id"] = image_tag
            changed = True
        if hyperloom_sha and not meta.get("code_revision"):
            meta["code_revision"] = hyperloom_sha
            changed = True
        if rec.sandbox_duration_seconds is not None and not meta.get("session_duration_seconds"):
            meta["session_duration_seconds"] = rec.sandbox_duration_seconds
            changed = True
        # `category` isn't in the schema but unknown fields are tolerated.
        if rec.category and not meta.get("category"):
            meta["category"] = rec.category
            changed = True

    elif path.name == "manifest.json":
        # V2 cli schema: flat top-level keys, often null when the sandbox didn't
        # set HYPERLOOM_IMAGE or git rev-parse failed; backfill from the CI side.
        for key, value in {
            "model_name": rec.model,
            "claw_session_id": rec.claw_session_id,
        }.items():
            if value and not data.get(key):
                data[key] = value
                changed = True
        for key in ("framework", "tp"):
            if detected.get(key) is not None and not data.get(key):
                data[key] = detected.get(key)
                changed = True
        if image and not data.get("image"):
            data["image"] = image
            changed = True
        if hyperloom_sha and not data.get("code_revision"):
            data["code_revision"] = hyperloom_sha
            changed = True
        if rec.category and not data.get("category"):
            data["category"] = rec.category
            changed = True
        if rec.sandbox_duration_seconds is not None and not data.get("sandbox_duration_seconds"):
            data["sandbox_duration_seconds"] = rec.sandbox_duration_seconds
            changed = True

    if changed:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _backfill_wekafs_in_place(rec: SubmissionRecord) -> int:
    """Reverse-write audit fields back into the wekafs SOURCE files so operators
    see them under /wekafs/users/<uid>/<sess>/ without GHA artifact zips
    (image/category/duration are SaFE-side facts the agent/V2 cli never had).

    Match (mirrors Stage B): exact `model` field, else conservative session-dir
    match; only sessions modified in the last 24h. Updates ci_metrics.json,
    manifest.json, session_breakdown[_v2].json across subdirs via
    _backfill_ci_metrics_file. No-op when wekafs isn't mounted.

    Args:
        rec (SubmissionRecord): Record used to match sessions and supply the
            metadata to backfill.

    Returns:
        int: Number of files updated; ``0`` when wekafs is unmounted or nothing
        matched.
    """
    nfs_root = os.environ.get("NFS_ROOT", "/wekafs")
    users_root = os.path.join(nfs_root, "users")
    if not os.path.isdir(users_root):
        return 0
    target = _norm_token((rec.model or "").split("/")[-1])
    if not target:
        return 0

    fresh_cutoff = time.time() - 24 * 3600
    targets = ("ci_metrics.json", "manifest.json", "session_breakdown.json", "session_breakdown_v2.json")
    subdirs = ("", "phase10_report", "results", "v2_session")

    n = 0

    def _session_has_matching_json(sess_path: str) -> bool:
        """Return whether a session dir holds JSON matching the target model.

        Args:
            sess_path: Session directory to scan.

        Returns:
            ``True`` if any target JSON file under the known subdirs has a
            model field matching ``rec``; otherwise ``False``.
        """
        for sub in subdirs:
            base = os.path.join(sess_path, sub) if sub else sess_path
            if not os.path.isdir(base):
                continue
            for fn in targets:
                p = Path(base) / fn
                if not p.is_file():
                    continue
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                mf = _json_model_field(d) if isinstance(d, dict) else ""
                if mf and _record_model_field_matches(rec, mf):
                    return True
        return False

    def _backfill_files(sess_path: str) -> int:
        """Backfill the model field into target JSON files in a session dir.

        Args:
            sess_path: Session directory whose target JSON files should be
                updated.

        Returns:
            The number of files updated.
        """
        updated = 0
        for sub in subdirs:
            base = os.path.join(sess_path, sub) if sub else sess_path
            if not os.path.isdir(base):
                continue
            for fn in targets:
                p = Path(base) / fn
                if not p.is_file():
                    continue
                try:
                    before = p.read_bytes()
                    _backfill_ci_metrics_file(p, rec)
                    if p.read_bytes() != before:
                        updated += 1
                        log.info("[task %s] wekafs backfill: %s", rec.task_id, p)
                except Exception as e:
                    log.warning("[task %s] wekafs backfill failed for %s: %s", rec.task_id, p, e)
        return updated

    for uid_dir in os.listdir(users_root):
        uid_path = os.path.join(users_root, uid_dir)
        if not os.path.isdir(uid_path):
            continue
        for sess in os.listdir(uid_path):
            sess_path = os.path.join(uid_path, sess)
            if not os.path.isdir(sess_path):
                continue
            try:
                if os.path.getmtime(sess_path) < fresh_cutoff:
                    continue
            except OSError:
                continue
            # Confirm session ownership: `model` field match (strongest), else
            # the conservative session-dir-name heuristic.
            matched = False
            for sub in subdirs:
                ci = (
                    os.path.join(sess_path, sub, "ci_metrics.json")
                    if sub
                    else os.path.join(sess_path, "ci_metrics.json")
                )
                if not os.path.isfile(ci):
                    continue
                try:
                    d = json.loads(Path(ci).read_text(encoding="utf-8"))
                except Exception:
                    continue
                mf = str(d.get("model") or d.get("model_name") or "")
                if mf and _record_model_field_matches(rec, mf):
                    matched = True
                    break
            if not matched and _session_has_matching_json(sess_path):
                matched = True
            if not matched and _record_matches_session_dir(rec, sess):
                matched = True
            if not matched:
                continue
            n += _backfill_files(sess_path)

        # Current layout: /wekafs/users/<uid>/<model-basename>/<YYYYmmddTHHMMSSZ>/.
        # A deleted ci_metrics.json must not block manifest.json backfill.
        if rec.safe_user_id and uid_dir != rec.safe_user_id:
            continue
        for model_dir_name in _candidate_model_dir_names(rec):
            model_dir = os.path.join(uid_path, model_dir_name)
            if not os.path.isdir(model_dir):
                continue
            try:
                ts_entries = sorted(os.listdir(model_dir), reverse=True)
            except Exception:
                continue
            for ts_entry in ts_entries:
                sess_path = os.path.join(model_dir, ts_entry)
                if not os.path.isdir(sess_path):
                    continue
                try:
                    if os.path.getmtime(sess_path) < fresh_cutoff:
                        continue
                except OSError:
                    continue
                ts = _session_timestamp_from_path(ts_entry)
                matched = _record_has_task_window(rec) and _timestamp_in_task_window(ts, rec)
                if not matched and _session_has_matching_json(sess_path):
                    matched = True
                if not matched:
                    continue
                n += _backfill_files(sess_path)
    return n


def _record_matches_session_dir(rec: SubmissionRecord, sess_name: str) -> bool:
    """Conservative directory-name match for the /wekafs/users fallback. Prefer
    the displayName slug; fall back to basename with strict-term guards to avoid
    cross-wiring adjacent repos (Qwen2.5 vs -AWQ, Nano vs Super, bnb, etc.).

    Args:
        rec (SubmissionRecord): Record providing the model id / display name.
        sess_name (str): Candidate session directory name.

    Returns:
        bool: True when ``sess_name`` conservatively matches the record.
    """
    sess_norm = _norm_token(sess_name)
    display = rec.display_name or ""
    if display and _norm_token(display) in sess_norm:
        return True

    model = rec.model or ""
    base = model.split("/")[-1]
    base_norm = _norm_token(base)
    if not base_norm or base_norm not in sess_norm:
        return False

    identity = f"{model} {display}".lower()
    lower_dir = sess_name.lower()
    strict_terms = ("awq", "gptq", "bnb", "4bit", "abliterated", "geneticlemonade", "nano", "super")
    for term in strict_terms:
        in_identity = term in identity
        in_dir = term in lower_dir
        if in_identity != in_dir:
            return False
    return True


def _record_model_field_matches(rec: SubmissionRecord, model_field: str) -> bool:
    """Compare a JSON model field with a SubmissionRecord conservatively.

    Args:
        rec (SubmissionRecord): Record supplying the allowed model names.
        model_field (str): The model id read from a result JSON.

    Returns:
        bool: True when ``model_field`` normalizes to one of the record's
        allowed names.
    """
    observed = _norm_token(str(model_field or ""))
    if not observed:
        return False
    allowed = {
        _norm_token((rec.model or "").split("/")[-1]),
        _norm_token((rec.model or "").replace("/", "-")),
        _norm_token((rec.model_path or "").rstrip("/\\").split("/")[-1]),
        _norm_token(rec.display_name or ""),
    }
    allowed.discard("")
    return observed in allowed or _norm_token(str(model_field).split("/")[-1]) in allowed


def _candidate_model_dir_names(rec: SubmissionRecord) -> list[str]:
    """Derive plausible per-model directory basenames for a record.

    Args:
        rec (SubmissionRecord): The record supplying model path / id / display
            name candidates.

    Returns:
        list[str]: De-duplicated basename candidates, in priority order.
    """
    names: list[str] = []
    for value in (
        rec.model_path or "",
        (rec.model or "").replace("/", "-"),
        (rec.model or "").split("/")[-1],
        rec.display_name or "",
    ):
        name = str(value or "").strip().rstrip("/\\").split("/")[-1]
        if name and name not in names:
            names.append(name)
    return names


def _json_positive_perf(data: dict) -> bool:
    """Report whether a breakdown JSON has positive baseline AND optimized perf.

    Args:
        data (dict): A parsed session-breakdown/metrics JSON object.

    Returns:
        bool: True only if at least one positive baseline throughput and one
            positive optimized throughput are present.
    """

    def positive(value: object) -> bool:
        """Return True for a positive, non-bool numeric value.

        Args:
            value (object): Candidate value to test.

        Returns:
            bool: True if ``value`` is an ``int``/``float`` (not ``bool``)
                greater than zero.
        """
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0

    baseline = data.get("baseline") if isinstance(data.get("baseline"), dict) else {}
    final = data.get("final") if isinstance(data.get("final"), dict) else {}
    best = data.get("best") if isinstance(data.get("best"), dict) else {}
    base_values = (
        data.get("baseline_tput"),
        data.get("baseline_throughput"),
        data.get("tok_per_gpu_baseline"),
        baseline.get("throughput_tok_s_per_gpu"),
        baseline.get("output_throughput"),
    )
    opt_values = (
        data.get("best_tput"),
        data.get("optimized_throughput"),
        data.get("tok_per_gpu_optimized"),
        final.get("throughput_tok_s_per_gpu"),
        final.get("output_throughput"),
        best.get("throughput_tok_s_per_gpu"),
        best.get("output_throughput"),
    )
    return any(positive(v) for v in base_values) and any(positive(v) for v in opt_values)


def _json_model_field(data: dict) -> str:
    """Extract the model id from a breakdown/metrics JSON, trying known keys.

    Args:
        data (dict): A parsed session-breakdown/metrics JSON object.

    Returns:
        str: The first non-empty model field found, or ``""``.
    """
    workload = data.get("workload") if isinstance(data.get("workload"), dict) else {}
    session = data.get("session") if isinstance(data.get("session"), dict) else {}
    meta = data.get("session_meta") if isinstance(data.get("session_meta"), dict) else {}
    for value in (
        data.get("model"),
        data.get("model_name"),
        workload.get("model_name"),
        workload.get("model"),
        session.get("model"),
        meta.get("model"),
    ):
        if value not in (None, ""):
            return str(value)
    return ""


def _json_claw_session_id(data: dict) -> str:
    """Extract the Claw session id from a breakdown/metrics JSON.

    Args:
        data (dict): A parsed session-breakdown/metrics JSON object.

    Returns:
        str: The first non-empty ``claw_session_id`` found, or ``""``.
    """
    session = data.get("session") if isinstance(data.get("session"), dict) else {}
    meta = data.get("session_meta") if isinstance(data.get("session_meta"), dict) else {}
    for value in (
        data.get("claw_session_id"),
        session.get("claw_session_id"),
        meta.get("claw_session_id"),
    ):
        if value not in (None, ""):
            return str(value)
    return ""


def _resolve_record_claw_session_id(
    safe: SafeOptimizeClient,
    rec: SubmissionRecord,
    last_task: dict | None = None,
) -> str | None:
    """Resolve a Claw session id for a submitted record.

    SaFE's terminal task payload can occasionally omit ``clawSessionId`` even
    though a subsequent task GET has it. Prefer the freshest terminal payload,
    but fall back to the cached resolver before leaving the manifest without a
    Claw id; otherwise Pulse classifies the dispatch as ``no_claw_session``.
    """
    for value in (
        (last_task or {}).get("clawSessionId"),
        rec.claw_session_id,
    ):
        sid = str(value or "").strip()
        if sid:
            return sid
    if rec.task_id:
        try:
            sid = safe._claw_session_id_for(rec.task_id)
            if sid:
                return sid
        except Exception as e:
            log.debug("[task %s] clawSessionId fallback lookup failed: %s", rec.task_id, e)
    return None


def _env_truthy(name: str) -> bool:
    """Report whether an environment variable is set to a truthy value.

    Args:
        name (str): The environment variable name.

    Returns:
        bool: True if the value is one of 1/true/yes/y/on (case-insensitive).
    """
    return (os.environ.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _copy_session_tree(src_dir: str, dst_dir: Path) -> int:
    """Copy an entire persisted session directory into ``dst_dir`` (existing files
    untouched). Returns the number of files copied.

    Args:
        src_dir (str): Source session directory to copy from.
        dst_dir (Path): Destination directory (created if absent).

    Returns:
        int: Number of files copied.
    """
    copied = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for root, dirnames, filenames in os.walk(src_dir):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".pytest_cache"}]
        rel_root = os.path.relpath(root, src_dir)
        out_root = dst_dir if rel_root == "." else dst_dir / rel_root
        out_root.mkdir(parents=True, exist_ok=True)
        for fname in filenames:
            src = os.path.join(root, fname)
            dst = out_root / fname
            if dst.exists():
                continue
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                log.warning("full-session copy failed %s -> %s: %s", src, dst, e)
                continue
            copied += 1
    return copied
