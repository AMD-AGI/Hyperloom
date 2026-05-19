"""Cortex KB client — KnowledgePlane facade write surface (v0.8 M1).

This module is the **single point of contact** between
``inference_optimizer`` and the Cortex KB service. The design intent
(KB_design §3.6 + §3.13 M1) makes three things explicit:

1. **All writes are channeled through the Coordinator**, never the
   reactor LLMs. PolicyGate enforces this; the facade itself doesn't
   check ACLs but every entrypoint takes a logical operation name so
   audit logs ascribe writes to ``inference_optimizer.coordinator``.
2. **Failure modes are well-defined** — synchronous CLI failures fall
   through to a per-session NDJSON queue (see ``runtime/cortex/``);
   the queue is later drained by ``cortex_kb_flusher`` or by the
   ``drain_pending()`` helper at T4.
3. **The facade is stdlib-only** so it works in stripped sandboxes
   that don't yet have a vendored ``primus-kb-client``.

This first cut implements the writers needed by M1 (T0 begin /
propose_point / hypothesize / ingest_attempt / verify / commit /
abort). The reader surface (``traverse`` / ``find_recipe`` actually
returning parsed data into prompts) is deferred to M4 / M5.

Cortex CLI contract reference: ``/wekafs/haiskong/cortex-for-hyperloom-2026-05-18.md``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .session_paths import (
    cortex_audit_jsonl,
    cortex_dir,
    cortex_pending_ndjson,
    cortex_pitfalls_json,
    cortex_sid_file,
    cortex_warm_json,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CortexKBError(RuntimeError):
    """Raised for unrecoverable interactions with the Cortex KB.

    The synchronous T0/T4 hooks treat this as fail-fast (PRELUDE
    rejection / ``stop_reason=cortex_drain_failed``). Async T2/T3 hooks
    catch it and downgrade to an NDJSON enqueue.
    """


class CortexBinaryNotFound(CortexKBError):
    """``cortex-kb`` is not on PATH (or override path missing)."""


# ---------------------------------------------------------------------------
# Canonical id derivation
# ---------------------------------------------------------------------------
# Centralized so every producer (Coordinator T2/T3, breakdown collector,
# resume probe) agrees on the canonical-id shape. Changing one of these
# shapes is a KB-wide migration concern.
def workload_canonical_id(model_name: str, gpu_type: str) -> str:
    """``workload.<model_slug>.<gpu_type>`` — single workload node per
    (model, hw) pair across all sessions."""
    slug = (model_name or "unknown_model").strip().replace("/", "_").replace(" ", "_")
    gpu = (gpu_type or "unknown_gpu").strip().lower() or "unknown_gpu"
    return f"workload.{slug}.{gpu}"


def optimization_canonical_id(cortex_session_id: str, proposal_msg_id: str) -> str:
    """``opt.session-{sid}.proposal-{msg_id}`` — per-proposal optimization
    node. msg_id is unique per session so canonical_id is naturally
    idempotent against retries."""
    sid = (cortex_session_id or "unknown").strip() or "unknown"
    mid = (proposal_msg_id or "unknown").strip() or "unknown"
    return f"opt.session-{sid}.proposal-{mid}"


def attempt_canonical_id(cortex_session_id: str, task_id: str) -> str:
    """``attempt.session-{sid}.task-{task_id}`` — per-task attempt node."""
    sid = (cortex_session_id or "unknown").strip() or "unknown"
    tid = (task_id or "unknown").strip() or "unknown"
    return f"attempt.session-{sid}.task-{tid}"


# ---------------------------------------------------------------------------
# NDJSON envelope
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _ndjson_envelope(
    *,
    op: str,
    payload: Mapping[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    """Build one NDJSON row.

    Shape (KB_design §3.6.7): ``{"op", "payload", "created_at",
    "idempotency_key", "attempts"}``. ``attempts`` counts NDJSON
    flusher retries so robustness can alert on stuck rows.
    """
    return {
        "op":              op,
        "payload":         dict(payload),
        "created_at":      _now_iso(),
        "idempotency_key": idempotency_key,
        "attempts":        0,
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
@dataclass
class CortexKBClient:
    """Single-process Cortex KB client used by the Coordinator + cli T0 hook.

    Construction is cheap — no network I/O happens until the first call.
    All entrypoints are synchronous; T2/T3 callers should be tolerant of
    the synchronous CLI taking ~200ms per invocation. NDJSON enqueue is
    O(append) regardless.

    Args:
        session_dir: hyperloom session root (writable). Used for the
            audit log + NDJSON queue.
        kb_url: ``CORTEX_KB_URL`` override; ``None`` keeps the env value.
        binary: ``cortex-kb`` executable name; ``None`` resolves via
            PATH. Override for offline testing (e.g. a stub shell).
        timeout_sec: per-CLI-call timeout. Defaults match the cortex
            doc's recommendation (10s for write ops).
        enabled: when ``False``, every entrypoint becomes a no-op
            (used by ``--no-cortex``). Audit log still records the
            skip so breakdown collection can flag the bypass.
        env_extra: extra env vars to splice into every CLI invocation
            (e.g. ``CORTEX_KB_INITIATOR``).
    """

    session_dir: Path
    kb_url: str | None = None
    binary: str | None = None
    timeout_sec: float = 10.0
    enabled: bool = True
    env_extra: dict[str, str] = field(default_factory=dict)

    # Internal: resolved binary path (populated lazily by _resolve_binary).
    _resolved_binary: str | None = field(default=None, init=False)

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.session_dir = Path(self.session_dir)
        cortex_dir(self.session_dir).mkdir(parents=True, exist_ok=True)

    @property
    def pending_path(self) -> Path:
        return cortex_pending_ndjson(self.session_dir)

    @property
    def audit_path(self) -> Path:
        return cortex_audit_jsonl(self.session_dir)

    @property
    def sid_path(self) -> Path:
        return cortex_sid_file(self.session_dir)

    # ------------------------------------------------------------------
    # CLI invocation primitives
    # ------------------------------------------------------------------
    def _resolve_binary(self) -> str:
        if self._resolved_binary:
            return self._resolved_binary
        candidate = self.binary or os.environ.get("CORTEX_KB_BIN") or "cortex-kb"
        resolved = shutil.which(candidate)
        if not resolved:
            raise CortexBinaryNotFound(
                f"cortex-kb binary {candidate!r} not found on PATH; "
                f"set CORTEX_KB_BIN or pass --cortex-kb-bin"
            )
        self._resolved_binary = resolved
        return resolved

    def _cli_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.kb_url:
            env["CORTEX_KB_URL"] = self.kb_url
        env.update(self.env_extra)
        return env

    def _run_cli(self, *args: str) -> str:
        """Invoke ``cortex-kb`` synchronously, returning stdout text.

        Raises :class:`CortexKBError` on non-zero exit / timeout /
        binary-not-found. Always records an audit entry — both success
        and failure — so breakdown collection has a complete trace.
        """
        bin_path = self._resolve_binary()
        cmd = [bin_path, *args, "--format", "text"]
        started = time.monotonic()
        proc: subprocess.CompletedProcess | None = None
        err: Exception | None = None
        try:
            proc = subprocess.run(
                cmd,
                env=self._cli_env(),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            err = exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if err is not None:
            self._audit_record(
                op="cli", status="error",
                args=list(args), elapsed_ms=elapsed_ms,
                error=type(err).__name__,
                error_message=str(err)[:512],
            )
            raise CortexKBError(
                f"cortex-kb {' '.join(args)} failed: "
                f"{type(err).__name__}: {err}"
            ) from err
        assert proc is not None
        if proc.returncode != 0:
            self._audit_record(
                op="cli", status="non_zero_exit",
                args=list(args), elapsed_ms=elapsed_ms,
                returncode=proc.returncode,
                stderr_tail=(proc.stderr or "")[-512:],
            )
            raise CortexKBError(
                f"cortex-kb {' '.join(args)} exit={proc.returncode}: "
                f"{(proc.stderr or '').strip()[:512]}"
            )
        self._audit_record(
            op="cli", status="ok",
            args=list(args), elapsed_ms=elapsed_ms,
        )
        return proc.stdout

    def _audit_record(self, **fields: Any) -> None:
        """Append one structured line to ``.kb_audit.jsonl``.

        Best-effort: a write failure is logged but never raised — the
        audit log is a forensic aid, not a correctness invariant.
        """
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"ts": _now_iso(), **fields}
            with self.audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError as exc:
            log.warning("cortex audit append failed (%s): %s", self.audit_path, exc)

    # ------------------------------------------------------------------
    # Stdout text-format parsers (cortex-kb --format text uses
    # "key: value" lines; the doc's awk recipes work because of this.)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_kv(stdout: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in (stdout or "").splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
        return out

    # ------------------------------------------------------------------
    # NDJSON enqueue (fallback path for async ops)
    # ------------------------------------------------------------------
    def _enqueue(
        self, *, op: str, payload: Mapping[str, Any], idempotency_key: str,
    ) -> None:
        envelope = _ndjson_envelope(
            op=op, payload=payload, idempotency_key=idempotency_key,
        )
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(envelope, sort_keys=True) + "\n"
        # Open + append + flush + fsync to survive a coordinator crash
        # between the .write and a graceful close.
        with self.pending_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync is best-effort: tmpfs / certain wekafs mounts
                # reject it but the write is still visible.
                pass
        self._audit_record(
            op="enqueue", status="ok",
            envelope_op=op, idempotency_key=idempotency_key,
        )

    # ==================================================================
    # Public API — read side (T0)
    # ==================================================================
    def session_begin(
        self,
        *,
        task: str,
        workload: str,
        hw: str,
        image_digest: str = "",
        stack_fingerprint: Mapping[str, str] | None = None,
        extra_attrs: Mapping[str, Any] | None = None,
        goal: str = "find_recommendation",
        thinking_style: str = "recommendation",
    ) -> str:
        """T0 — ``cortex-kb session begin``.

        Synchronous; failures bubble as :class:`CortexKBError` so the cli
        layer can fail-fast unless ``--no-cortex`` was passed. Caller is
        responsible for writing the returned sid into SharedState +
        ``.kb_sid``.

        Args mirror the CLI's required attrs (see KB_design §3.6.5.1).
        """
        if not self.enabled:
            self._audit_record(op="session_begin", status="skip_disabled")
            return ""
        attrs: dict[str, Any] = {
            "workload":     workload,
            "hw":           hw,
            "image_digest": image_digest or "unknown",
            "stack_fingerprint": dict(stack_fingerprint or {}),
        }
        if extra_attrs:
            attrs.update(extra_attrs)
        stdout = self._run_cli(
            "session", "begin",
            "--goal", goal,
            "--task", task,
            "--thinking-style", thinking_style,
            "--attrs", json.dumps(attrs, sort_keys=True),
        )
        kv = self._parse_kv(stdout)
        sid = kv.get("session_id", "").strip()
        if not sid:
            raise CortexKBError(
                f"cortex-kb session begin returned no session_id; stdout={stdout!r}"
            )
        try:
            self.sid_path.write_text(sid + "\n", encoding="utf-8")
        except OSError as exc:
            log.warning("failed to persist .kb_sid: %s", exc)
        self._audit_record(
            op="session_begin", status="ok",
            session_id=sid, workload=workload, hw=hw,
        )
        return sid

    def find_recipe(self, *, workload: str, hw: str) -> str:
        """T0 — ``cortex-kb find-recipe``. Returns raw stdout (text)
        snapshotted to ``.kb_warm.json`` by the caller.

        Failures are non-fatal in M1 (warm_start is consumed by M5);
        callers should swallow :class:`CortexKBError`.
        """
        if not self.enabled:
            return ""
        return self._run_cli(
            "find-recipe", "--workload", workload, "--chip", hw,
        )

    def traps(self, *, symptom: str) -> str:
        """T0 — ``cortex-kb traps``. Returns raw stdout (text)
        snapshotted to ``.kb_pitfalls.json`` by the caller."""
        if not self.enabled:
            return ""
        return self._run_cli("traps", "--symptom", symptom)

    # ==================================================================
    # Public API — write side (T2 / T3 / T4)
    # ==================================================================
    def propose_point(
        self,
        *,
        canonical_id: str,
        kind: str,
        attrs: Mapping[str, Any] | None = None,
        authority: str = "HYPOTHESIZED",
        evidence: list[str] | None = None,
        source: str = "agent_observation",
        idempotency_key: str | None = None,
        prefer_sync: bool = True,
    ) -> dict[str, Any]:
        """T0 mint / T2 mint — create or upsert a Cortex point.

        Sync first (so the caller gets a point_id back); on CLI failure,
        falls back to NDJSON enqueue (returns ``{"status": "queued"}``).

        ``canonical_id`` is the natural idempotency key on the Cortex
        side; ``idempotency_key`` here only feeds the NDJSON queue's
        per-row de-dup.
        """
        if not self.enabled:
            return {"status": "skip_disabled"}
        idem = idempotency_key or f"propose_point:{canonical_id}"
        payload = {
            "canonical_id": canonical_id,
            "kind":         kind,
            "authority":    authority,
            "attrs":        dict(attrs or {}),
            "evidence":     list(evidence or []),
            "source":       source,
        }
        if prefer_sync:
            try:
                args = [
                    "propose-point",
                    "--canonical-id", canonical_id,
                    "--kind", kind,
                    "--authority", authority,
                    "--source", source,
                ]
                if attrs:
                    args.extend(["--attrs", json.dumps(dict(attrs), sort_keys=True)])
                for ev in (evidence or []):
                    args.extend(["--evidence", ev])
                stdout = self._run_cli(*args)
                kv = self._parse_kv(stdout)
                # Structured audit entry (in addition to the generic
                # ``op="cli"`` row that ``_run_cli`` already wrote) so the
                # breakdown collector can surface ``points_created[]``
                # without re-parsing argv. KB_design §3.13 M4 §4 +
                # §3.12 §4.4 (kb_provenance.points_created).
                self._audit_record(
                    op="propose_point", status=kv.get("status", "ok"),
                    canonical_id=canonical_id, kind=kind,
                    authority=authority, source=source,
                    point_id=kv.get("point_id", ""),
                )
                return {
                    "status":      kv.get("status", "ok"),
                    "point_id":    kv.get("point_id", ""),
                    "proposal_id": kv.get("proposal_id", ""),
                }
            except CortexKBError as exc:
                log.info(
                    "propose_point sync failed (%s); enqueueing NDJSON", exc,
                )
        self._audit_record(
            op="propose_point", status="queued",
            canonical_id=canonical_id, kind=kind,
            authority=authority, source=source,
        )
        self._enqueue(op="propose_point", payload=payload, idempotency_key=idem)
        return {"status": "queued"}

    def hypothesize(
        self,
        *,
        sid: str,
        from_canonical: str,
        to_canonical: str,
        edge_type: str = "hypothetical",
        reason: str = "",
        attrs: Mapping[str, Any] | None = None,
        evidence: list[str] | None = None,
        idempotency_key: str | None = None,
        prefer_sync: bool = True,
    ) -> dict[str, Any]:
        """T2 — ``cortex-kb session hypothesize``.

        Returns ``{"tentative_edge_id": "..."}`` on sync success; on
        failure enqueues NDJSON and returns ``{"status": "queued",
        "tentative_edge_id": ""}``. The empty edge id signals to T3 it
        should fall back to ``propose-edge + late_verified``.
        """
        if not self.enabled or not sid:
            return {"status": "skip_disabled", "tentative_edge_id": ""}
        idem = idempotency_key or f"hypothesize:{sid}:{from_canonical}->{to_canonical}"
        payload = {
            "sid":            sid,
            "from":           from_canonical,
            "to":             to_canonical,
            "type":           edge_type,
            "reason":         reason,
            "attrs":          dict(attrs or {}),
            "evidence":       list(evidence or []),
        }
        if prefer_sync:
            try:
                args = [
                    "session", "hypothesize",
                    "--sid", sid,
                    "--from", from_canonical,
                    "--to", to_canonical,
                    "--type", edge_type,
                ]
                if reason:
                    args.extend(["--reason", reason])
                if attrs:
                    args.extend(["--attrs", json.dumps(dict(attrs), sort_keys=True)])
                for ev in (evidence or []):
                    args.extend(["--evidence", ev])
                stdout = self._run_cli(*args)
                kv = self._parse_kv(stdout)
                edge_id = kv.get("tentative_edge_id", "")
                return {
                    "status":            kv.get("status", "ok"),
                    "tentative_edge_id": edge_id,
                }
            except CortexKBError as exc:
                log.info(
                    "hypothesize sync failed (%s); enqueueing NDJSON", exc,
                )
        self._enqueue(op="hypothesize", payload=payload, idempotency_key=idem)
        return {"status": "queued", "tentative_edge_id": ""}

    def ingest_attempt(
        self,
        *,
        sid: str,
        iter_id: int,
        outcome: str,
        metrics: Mapping[str, Any],
        plan_edge: str = "",
        evidence: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """T3 — ``cortex-kb ingest-attempt``.

        Async-only per KB_design §3.6: always enqueues NDJSON. Returns
        immediately with ``{"status": "queued"}``. The flusher (or T4
        drain) picks it up.

        ``outcome ∈ {"PASS", "FAIL", "PARTIAL"}``.
        """
        if not self.enabled or not sid:
            return {"status": "skip_disabled"}
        idem = idempotency_key or f"ingest_attempt:{sid}:{iter_id}"
        payload = {
            "sid":       sid,
            "iter":      int(iter_id),
            "outcome":   outcome,
            "metrics":   dict(metrics or {}),
            "plan_edge": plan_edge,
            "evidence":  list(evidence or []),
        }
        self._enqueue(op="ingest_attempt", payload=payload, idempotency_key=idem)
        return {"status": "queued"}

    def verify(
        self,
        *,
        sid: str,
        edge_id: str,
        outcome: str,
        evidence: list[str] | None = None,
        promote_authority: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """T3 — ``cortex-kb session verify``.

        Async per KB_design §3.6; always enqueues. ``outcome ∈
        {"confirmed", "refuted"}``. ``promote_authority="EXPERIENTIAL"``
        is the standard promotion for KEEP outcomes.
        """
        if not self.enabled or not sid or not edge_id:
            return {"status": "skip_disabled"}
        idem = idempotency_key or f"verify:{sid}:{edge_id}"
        payload = {
            "sid":               sid,
            "edge":              edge_id,
            "outcome":           outcome,
            "evidence":          list(evidence or []),
            "promoted_authority": promote_authority or "",
        }
        self._enqueue(op="verify", payload=payload, idempotency_key=idem)
        return {"status": "queued"}

    def session_commit(self, sid: str) -> dict[str, Any]:
        """T4 — ``cortex-kb session commit``.

        Synchronous. Caller must :meth:`drain_pending` first so all
        queued T2/T3 rows land before commit closes the session.
        Returns parsed commit summary (status / promoted_edges /
        derived_summary_id) on success.
        """
        if not self.enabled or not sid:
            return {"status": "skip_disabled"}
        stdout = self._run_cli("session", "commit", "--sid", sid)
        kv = self._parse_kv(stdout)
        promoted: list[str] = []
        raw_promoted = kv.get("promoted_edges", "")
        if raw_promoted and raw_promoted not in ("None", "[]"):
            stripped = raw_promoted.strip().strip("[]")
            for piece in stripped.split(","):
                v = piece.strip()
                if v:
                    promoted.append(v)
        summary = {
            "status":             kv.get("status", "committed"),
            "promoted_edges":     promoted,
            "derived_summary_id": kv.get("derived_summary_id", "") or "",
            "raw":                stdout,
        }
        self._audit_record(
            op="session_commit", status=summary["status"],
            session_id=sid, promoted_count=len(promoted),
        )
        return summary

    def session_abort(self, sid: str, *, reason: str = "") -> dict[str, Any]:
        """``cortex-kb session abort`` — fail-fast escape hatch.

        Called from cli failure paths (T0 succeeded but PRELUDE crashed)
        so the KB-side session doesn't linger in HYPOTHESIZED limbo.
        """
        if not self.enabled or not sid:
            return {"status": "skip_disabled"}
        args = ["session", "abort", "--sid", sid]
        if reason:
            args.extend(["--reason", reason])
        try:
            self._run_cli(*args)
            self._audit_record(op="session_abort", status="ok", session_id=sid)
            return {"status": "aborted"}
        except CortexKBError as exc:
            self._audit_record(
                op="session_abort", status="error",
                session_id=sid, error=str(exc)[:512],
            )
            return {"status": "abort_failed", "error": str(exc)}

    # ==================================================================
    # NDJSON drain (T4 + flusher)
    # ==================================================================
    def drain_pending(self, *, timeout_sec: float = 60.0) -> dict[str, Any]:
        """Process every row in ``.kb_pending.ndjson`` synchronously.

        Used by T4 before ``session commit`` so the commit closes a
        complete view of the session. The flusher daemon may run in
        parallel; we rely on append-only semantics + per-row exclusive
        consumption (rename → ``.kb_pending.processing.<pid>``) to
        avoid double-flush.

        Returns ``{"drained": N, "remaining": M, "dead_letter": K,
        "elapsed_ms": ...}``.
        """
        if not self.enabled:
            return {"drained": 0, "remaining": 0, "dead_letter": 0, "elapsed_ms": 0}
        started = time.monotonic()
        pending = self.pending_path
        if not pending.exists() or pending.stat().st_size == 0:
            return {"drained": 0, "remaining": 0, "dead_letter": 0, "elapsed_ms": 0}
        # Snapshot the file to avoid races with the flusher daemon.
        snapshot = pending.with_suffix(
            f".processing.{os.getpid()}.{uuid.uuid4().hex[:6]}",
        )
        try:
            os.rename(pending, snapshot)
        except FileNotFoundError:
            return {"drained": 0, "remaining": 0, "dead_letter": 0, "elapsed_ms": 0}
        drained = 0
        dead_letter = 0
        leftover_lines: list[str] = []
        deadline = started + max(0.0, timeout_sec)
        with snapshot.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if time.monotonic() > deadline:
                    leftover_lines.append(stripped)
                    continue
                try:
                    envelope = json.loads(stripped)
                except json.JSONDecodeError:
                    dead_letter += 1
                    self._audit_record(
                        op="drain", status="malformed_row",
                        line=stripped[:512],
                    )
                    continue
                outcome = self._flush_one(envelope)
                if outcome == "ok":
                    drained += 1
                elif outcome == "permanent":
                    dead_letter += 1
                else:
                    leftover_lines.append(stripped)
        # Put unprocessed rows back at the head of the queue so the
        # flusher can retry them. (Append semantics: write a fresh
        # pending file containing the leftover lines + whatever the
        # daemon may have appended in the meantime.)
        if leftover_lines:
            new_pending = pending.with_suffix(f".restore.{os.getpid()}")
            with new_pending.open("w", encoding="utf-8") as f:
                for ln in leftover_lines:
                    f.write(ln + "\n")
            # If the daemon appended new rows to ``pending`` while we
            # processed, concatenate them after the leftovers.
            try:
                if pending.exists():
                    with pending.open("r", encoding="utf-8") as src, new_pending.open("a", encoding="utf-8") as dst:
                        for line in src:
                            dst.write(line)
                    pending.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("drain leftover concat failed: %s", exc)
            os.replace(new_pending, pending)
        try:
            snapshot.unlink(missing_ok=True)
        except OSError:
            pass
        elapsed_ms = int((time.monotonic() - started) * 1000)
        self._audit_record(
            op="drain", status="done",
            drained=drained, dead_letter=dead_letter,
            remaining=len(leftover_lines), elapsed_ms=elapsed_ms,
        )
        return {
            "drained":     drained,
            "remaining":   len(leftover_lines),
            "dead_letter": dead_letter,
            "elapsed_ms":  elapsed_ms,
        }

    def _flush_one(self, envelope: Mapping[str, Any]) -> str:
        """Convert one NDJSON envelope back into a synchronous CLI call.

        Returns ``"ok"`` / ``"transient"`` / ``"permanent"``.
        """
        op = str(envelope.get("op", ""))
        payload = envelope.get("payload", {}) or {}
        try:
            if op == "propose_point":
                self.propose_point(
                    canonical_id=str(payload.get("canonical_id", "")),
                    kind=str(payload.get("kind", "")),
                    attrs=payload.get("attrs") or {},
                    authority=str(payload.get("authority") or "HYPOTHESIZED"),
                    evidence=list(payload.get("evidence") or []),
                    source=str(payload.get("source") or "agent_observation"),
                    prefer_sync=True,
                )
            elif op == "hypothesize":
                self.hypothesize(
                    sid=str(payload.get("sid", "")),
                    from_canonical=str(payload.get("from", "")),
                    to_canonical=str(payload.get("to", "")),
                    edge_type=str(payload.get("type") or "hypothetical"),
                    reason=str(payload.get("reason", "")),
                    attrs=payload.get("attrs") or {},
                    evidence=list(payload.get("evidence") or []),
                    prefer_sync=True,
                )
            elif op == "ingest_attempt":
                # Bypass enqueue path on flush — go straight to CLI.
                self._ingest_attempt_sync(
                    sid=str(payload.get("sid", "")),
                    iter_id=int(payload.get("iter") or 0),
                    outcome=str(payload.get("outcome") or "PARTIAL"),
                    metrics=payload.get("metrics") or {},
                    plan_edge=str(payload.get("plan_edge") or ""),
                    evidence=list(payload.get("evidence") or []),
                )
            elif op == "verify":
                self._verify_sync(
                    sid=str(payload.get("sid", "")),
                    edge_id=str(payload.get("edge", "")),
                    outcome=str(payload.get("outcome") or "confirmed"),
                    evidence=list(payload.get("evidence") or []),
                    promote_authority=str(payload.get("promoted_authority") or "") or None,
                )
            else:
                # Unknown op — bury it in dead letter rather than burn cycles.
                return "permanent"
        except CortexKBError as exc:
            # Distinguish business-rejection (4xx-style: bad payload) from
            # transient (5xx-style: connection / timeout). The CLI doesn't
            # expose status codes directly so we string-match: timeouts +
            # connection issues + binary-not-found are transient, everything
            # else (including non_zero_exit with a clear error message) is
            # treated as transient too on first pass — the dead-letter
            # promotion happens once ``attempts`` crosses a threshold which
            # the flusher daemon owns.
            log.info("flush_one %s deferred: %s", op, exc)
            return "transient"
        return "ok"

    def _ingest_attempt_sync(
        self, *, sid: str, iter_id: int, outcome: str,
        metrics: Mapping[str, Any], plan_edge: str,
        evidence: list[str],
    ) -> None:
        args = [
            "ingest-attempt",
            "--sid", sid,
            "--iter", str(iter_id),
            "--outcome", outcome,
            "--metrics", json.dumps(dict(metrics or {}), sort_keys=True),
        ]
        if plan_edge:
            args.extend(["--plan-edge", plan_edge])
        for ev in evidence or []:
            args.extend(["--evidence", ev])
        self._run_cli(*args)

    def _verify_sync(
        self, *, sid: str, edge_id: str, outcome: str,
        evidence: list[str], promote_authority: str | None,
    ) -> None:
        args = [
            "session", "verify",
            "--sid", sid,
            "--edge", edge_id,
            "--outcome", outcome,
        ]
        if promote_authority:
            args.extend(["--promoted-authority", promote_authority])
        for ev in evidence or []:
            args.extend(["--evidence", ev])
        self._run_cli(*args)


__all__ = [
    "CortexBinaryNotFound",
    "CortexKBClient",
    "CortexKBError",
    "attempt_canonical_id",
    "optimization_canonical_id",
    "workload_canonical_id",
]
