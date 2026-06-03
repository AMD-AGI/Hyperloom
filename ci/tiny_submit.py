#!/usr/bin/env python3
"""Tiny CI — long-running queue controller for the full 3B-12B pool.

A single process keeps up to ``--sandbox-cap`` + ``--hyperloom-cap`` SaFE
optimization tasks in flight at once across two *independent* submit
workspaces (default 150 + 150 = 300 concurrent), pulls the next model from a
shared queue the instant a slot frees, and emits an incremental CI summary +
Teams webhook every ``--report-every`` completed models (default every 10).

Why a controller instead of a GitHub Actions matrix
----------------------------------------------------
GitHub Actions caps a single workflow's matrix at ~256 concurrent jobs, and a
5341-model matrix would be unmanageable. Instead this runs as ONE job that
manages its own 300-wide thread pool, so concurrency, replenishment and
progressive reporting are all controlled in-process.

Everything heavy is reused verbatim from ``optimize_submit.py``:
  * ``process_model``        — auto-detect + register + download + submit.
  * ``wait_and_collect_one`` — wait for the task (4h cap) + SaFE-artifact and
                               NFS-fallback collection + wekafs in-place backfill.
  * ``write_manifest``       — the same ``submission_manifest.{json,md}`` the
                               existing summarize stage feeds to build_summary.

The only new logic is the scheduler: two worker pools (one per submit
workspace), a shared work queue, a registration-concurrency gate so the
download burst stays bounded, crash-safe progress checkpointing, and the
"every K completions -> build_summary.py + send_webhook.py" reporting hook.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

# tiny_submit.py lives next to optimize_submit.py in ci/. Make the import
# work regardless of the caller's CWD (workflow uses working-directory: ci,
# but local debugging may run from the repo root).
_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

from optimize_submit import (  # noqa: E402
    DEFAULT_API_URL,
    DEFAULT_GPU_TYPE,
    DEFAULT_INFERENCEX_PATH,
    DEFAULT_OOB_PATH,
    DEFAULT_REGISTER_WORKSPACE,
    DEFAULT_RESULTS_PATH,
    DEFAULT_TARGET_GAIN,
    DEFAULT_TRACELENS_ROOT,
    DEFAULT_VOLUME,
    HuggingFaceClient,
    SafeOptimizeClient,
    SubmissionRecord,
    _load_default_prompt_prefix,
    parse_kernel_backends,
    process_model,
    wait_and_collect_one,
    write_manifest,
)

log = logging.getLogger("tiny")

# Default workspaces for the two independent pools. Register happens in the
# RW workspace (where /wekafs writes + the canonical Model CR live), submit
# spreads across sandbox + hyperloom so each can run its own GPU pool.
DEFAULT_SANDBOX_WORKSPACE = "core42-sandbox"
DEFAULT_HYPERLOOM_WORKSPACE = "core42-hyperloom"
DEFAULT_CANDIDATES_FILE = "candidates/hf_3b_12b_base_inference_2026-06-01.json"

# Tiny runs each model for 4h (vs optimize-submit's 12h) per the user's
# spec, so a 5341-model pool at 300-wide finishes in ~3 days instead of ~9.
DEFAULT_MAX_HOURS = 4.0
DEFAULT_TASK_TIMEOUT_MIN = 240  # == 4h; the per-task wait deadline.


# ── Candidate loading ─────────────────────────────────────────────────────────

def load_candidates(path: Path) -> list[dict]:
    """Read a candidates JSON (schema: {pool_id, candidates:[{repo_id,...}]}).

    Returns the candidate dicts in pool order, dropping any without a repo_id.
    Each dict keeps pool_id / pool_index so the manifest can carry provenance.

    Args:
        path (Path): Path to the candidates JSON file.

    Returns:
        list[dict]: De-duplicated candidate dicts in pool order.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("candidates") or data.get("models") or []
    out: list[dict] = []
    seen: set[str] = set()
    for c in raw:
        repo = (c.get("repo_id") or c.get("model") or "").strip()
        if not repo or repo in seen:
            continue
        seen.add(repo)
        out.append(c)
    return out


# ── Crash-safe progress checkpoint ──────────────────────────────────────────────

def load_done_records(progress_path: Path) -> dict[str, SubmissionRecord]:
    """Reconstruct already-finished records from a previous run's progress log.

    The progress file is append-only NDJSON, one ``asdict(record)`` per line.
    On resume we skip any repo that already reached a terminal state so the
    controller can pick up a multi-day run after an interruption.

    Args:
        progress_path (Path): Path to the append-only NDJSON progress log.

    Returns:
        dict[str, SubmissionRecord]: Map of model -> reconstructed record for
            every valid line; empty if the file is absent.
    """
    done: dict[str, SubmissionRecord] = {}
    if not progress_path.is_file():
        return done
    valid_fields = set(SubmissionRecord.__dataclass_fields__.keys())
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        model = d.get("model")
        if not model:
            continue
        clean = {k: v for k, v in d.items() if k in valid_fields}
        try:
            done[model] = SubmissionRecord(**clean)
        except TypeError:
            continue
    return done


# ── The controller ──────────────────────────────────────────────────────────────

class TinyController:
    """Long-running scheduler that keeps many SaFE optimization tasks in flight.

    Drives two independent submit-workspace worker pools off a shared work
    queue, throttles the register/download burst with a semaphore, checkpoints
    progress for crash-safe resume, and emits incremental per-batch CI
    summaries plus webhooks while building a full ranked summary at the end.

    Attributes:
        args (argparse.Namespace): Parsed CLI arguments controlling the run.
        artifacts_dir (Path): Where downloaded task artifacts are stored.
        manifests_dir (Path): Where manifests + progress NDJSON are written.
        summary_out (Path): Where the full ranked summary is written.
        records (list[SubmissionRecord]): All completed submission records.
        register_sem (threading.Semaphore): Bounds the register/download burst.
        stop (threading.Event): Set to ask workers to stop after the current
            model.
        completed (int): Count of completed models.
        total_planned (int): Total models planned (done + pending).
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize controller state, output dirs, and SaFE clients.

        Args:
            args (argparse.Namespace): Parsed CLI arguments; also consulted
                with env-var fallbacks for SaFE connection and cluster fields.
        """
        self.args = args
        self.artifacts_dir = Path(args.artifacts_dir)
        self.manifests_dir = Path(args.output_dir)
        self.summary_out = Path(args.summary_out_dir)
        # Per-batch webhook manifests live OUTSIDE manifests_dir: build_summary
        # scans for submission_manifest.json *recursively*, so keeping them under
        # manifests_dir would make the full-summary build double-count records.
        self.batch_root = self.manifests_dir.parent / "tiny-batch-reports"
        self.progress_path = self.manifests_dir / "progress.ndjson"
        for d in (self.artifacts_dir, self.manifests_dir, self.summary_out,
                  self.batch_root):
            d.mkdir(parents=True, exist_ok=True)

        self.records: list[SubmissionRecord] = []
        self.records_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.emit_lock = threading.Lock()
        self.manifest_lock = threading.Lock()
        # Bound the register+download+submit burst. The 4h optimization wait
        # runs OUTSIDE this gate, so we still reach full 300-wide concurrency;
        # only the storage-heavy download fan-out is throttled.
        self.register_sem = threading.Semaphore(max(1, args.register_concurrency))
        self.stop = threading.Event()

        self.completed = 0
        self.last_report = 0      # count of records already webhook-reported
        self._batch_seq = 0
        self.submitted_ok = 0
        self.succeeded = 0
        self.start_ts = time.time()
        self.total_planned = 0

        self.prompt_prefix = self._resolve_prompt_prefix()
        self.kernel_backends = parse_kernel_backends(args.kernel_opt_backends)

        # Resolve SaFE connection + cluster prompt fields with env fallbacks,
        # mirroring optimize_submit.main() so a dispatch can configure either.
        self.base_url = (args.api_url or os.environ.get("SAFE_BASE_URL")
                         or os.environ.get("SAFE_API_URL") or DEFAULT_API_URL)
        self.api_key = (args.api_key or os.environ.get("CLAW_API_KEY")
                        or os.environ.get("SAFE_API_KEY") or "")
        self.register_workspace = (args.register_workspace
                                   or os.environ.get("SAFE_OPTIMIZE_REGISTER_WORKSPACE")
                                   or DEFAULT_REGISTER_WORKSPACE)
        self.sandbox_workspace = (args.sandbox_workspace
                                  or os.environ.get("SAFE_OPTIMIZE_SUBMIT_WORKSPACE")
                                  or DEFAULT_SANDBOX_WORKSPACE)
        self.hyperloom_workspace = (args.hyperloom_workspace
                                    or DEFAULT_HYPERLOOM_WORKSPACE)
        self.volume = (args.volume or os.environ.get("SAFE_OPTIMIZE_VOLUME")
                       or DEFAULT_VOLUME)
        self.gpu_type = (args.gpu_type or os.environ.get("SAFE_OPTIMIZE_GPU_TYPE")
                         or DEFAULT_GPU_TYPE)
        self.inferencex_path = (args.inferencex_path
                                or os.environ.get("SAFE_OPTIMIZE_INFERENCEX_PATH")
                                or DEFAULT_INFERENCEX_PATH)
        self.oob_path = (args.oob_path or os.environ.get("SAFE_OPTIMIZE_OOB_PATH")
                         or DEFAULT_OOB_PATH)
        self.tracelens_root = (args.tracelens_root
                               or os.environ.get("SAFE_OPTIMIZE_TRACELENS_ROOT")
                               or DEFAULT_TRACELENS_ROOT)
        self.hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
        self.webhook_url = args.webhook_url or os.environ.get("WEBHOOK_URL", "")
        self.dashboard_url = args.dashboard_url or os.environ.get("DASHBOARD_URL", "")

        self.hf = HuggingFaceClient(self.hf_token)
        self.safe_sandbox = self._make_client(self.sandbox_workspace, args.sandbox_cap)
        self.safe_hyperloom = self._make_client(self.hyperloom_workspace, args.hyperloom_cap)

    def _resolve_prompt_prefix(self) -> str:
        """Resolve the prompt prefix from file, inline arg, or the default.

        Returns:
            str: The prompt prefix text (the packaged default if neither a
                file nor an inline value is provided).
        """
        a = self.args
        if a.prompt_prefix_file:
            p = Path(a.prompt_prefix_file)
            if p.is_file():
                return p.read_text(encoding="utf-8")
            log.warning("prompt-prefix-file %s missing; falling back", p)
        if a.prompt_prefix:
            return a.prompt_prefix
        return _load_default_prompt_prefix()

    def _make_client(self, submit_workspace: str, cap: int) -> SafeOptimizeClient:
        """Create a SaFE client sized for one submit workspace's worker pool.

        Args:
            submit_workspace (str): Workspace to submit tasks into.
            cap (int): Worker count for this pool; used to size the HTTP
                connection pool.

        Returns:
            SafeOptimizeClient: A configured client with a per-workspace cap.
        """
        client = SafeOptimizeClient(
            self.base_url, self.api_key or "dry-run",
            register_workspace=self.register_workspace,
            submit_workspace=submit_workspace,
            volume=self.volume,
            submit_workspaces_pool=None,  # hard per-workspace cap, no round-robin
        )
        # Size the urllib3 pool to the worker count so 150 concurrent requests
        # don't serialize behind a default 10-connection pool.
        try:
            from requests.adapters import HTTPAdapter
            adapter = HTTPAdapter(pool_connections=2, pool_maxsize=cap + 16,
                                  max_retries=0)
            client._sess.mount("https://", adapter)
            client._sess.mount("http://", adapter)
        except Exception as e:  # pragma: no cover - best effort tuning
            log.debug("pool sizing skipped: %s", e)
        return client

    # ── progress / reporting ────────────────────────────────────────────────

    def _append_progress(self, rec: SubmissionRecord) -> None:
        """Append one record to the crash-safe NDJSON progress log.

        Args:
            rec (SubmissionRecord): The completed record to checkpoint.

        Returns:
            None
        """
        line = json.dumps(asdict(rec), separators=(",", ":"))
        with self.progress_lock:
            with self.progress_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _write_manifest_snapshot(self) -> None:
        """Write a submission manifest from a snapshot of all current records.

        Returns:
            None
        """
        with self.records_lock:
            snapshot = list(self.records)
        with self.manifest_lock:
            write_manifest(
                self.manifests_dir, snapshot,
                self.base_url, self.register_workspace,
                f"{self.sandbox_workspace}+{self.hyperloom_workspace}",
                self.volume,
            )

    def _emit_batch_report(self, seq: int, batch: list[SubmissionRecord]) -> None:
        """Webhook ONLY this freshly-completed batch (~report_every models),
        never the cumulative table.

        build_summary is manifest-driven (it reads submission_manifest.json and
        looks up artifacts by task_id), so a manifest holding just this batch
        yields just these rows -> a single 10-row Teams card. The full ranked
        table is produced separately by _write_full_summary() as the run
        artifact; it is never pushed to the webhook.

        Args:
            seq (int): Monotonic batch sequence number (used for the dir name).
            batch (list[SubmissionRecord]): Records completed since the last
                report.

        Returns:
            None
        """
        if not batch:
            return
        if not self.webhook_url:
            log.info("[batch %d] WEBHOOK_URL unset; %d new rows not sent",
                     seq, len(batch))
            return
        with self.emit_lock:
            bdir = self.batch_root / f"batch_{seq:05d}"
            bout = bdir / "out"
            bout.mkdir(parents=True, exist_ok=True)
            write_manifest(
                bdir, batch, self.base_url, self.register_workspace,
                f"{self.sandbox_workspace}+{self.hyperloom_workspace}",
                self.volume,
            )
            try:
                subprocess.run(
                    [sys.executable, str(_CI_DIR / "build_summary.py"),
                     "--artifacts-dir", str(self.artifacts_dir),
                     "--manifests-dir", str(bdir),
                     "--target-gpu", self.args.target_gpu,
                     "--isl", str(self.args.isl),
                     "--osl", str(self.args.osl),
                     "--out-dir", str(bout)],
                    check=False, timeout=300,
                )
            except Exception as e:
                log.warning("[batch %d] build_summary failed: %s", seq, e)
                return
            summary_json = bout / "ci_summary.json"
            if not summary_json.is_file():
                log.warning("[batch %d] ci_summary.json missing; webhook skipped", seq)
                return
            env = os.environ.copy()
            env["WEBHOOK_URL"] = self.webhook_url
            if self.dashboard_url:
                env["DASHBOARD_URL"] = self.dashboard_url
            try:
                subprocess.run(
                    [sys.executable, str(_CI_DIR / "send_webhook.py"),
                     "--summary", str(summary_json),
                     "--url", self.webhook_url,
                     "--rows-per-card", "10"],
                    check=False, env=env, timeout=180,
                )
                log.info("[batch %d] webhook sent: %d new rows", seq, len(batch))
            except Exception as e:
                log.warning("[batch %d] send_webhook failed: %s", seq, e)

    def _write_full_summary(self) -> None:
        """Build the full ranked ci_summary.{md,json} over ALL records for the
        run artifact + GHA job summary. Never webhooked — the webhook only ever
        receives the per-batch increments via _emit_batch_report()."""
        self._write_manifest_snapshot()
        try:
            subprocess.run(
                [sys.executable, str(_CI_DIR / "build_summary.py"),
                 "--artifacts-dir", str(self.artifacts_dir),
                 "--manifests-dir", str(self.manifests_dir),
                 "--target-gpu", self.args.target_gpu,
                 "--isl", str(self.args.isl),
                 "--osl", str(self.args.osl),
                 "--out-dir", str(self.summary_out)],
                check=False, timeout=1800,
            )
            log.info("[summary] full ranked table (%d records) written to %s",
                     len(self.records), self.summary_out)
        except Exception as e:
            log.warning("[summary] full build_summary failed: %s", e)

    def _on_complete(self, rec: SubmissionRecord, pool_label: str) -> None:
        """Record a finished model, update counters, and maybe emit a batch.

        Thread-safe: appends to the progress log and records list, bumps the
        submitted/succeeded tallies, logs progress, and triggers a webhook
        batch report once ``report_every`` new completions accumulate.

        Args:
            rec (SubmissionRecord): The completed record.
            pool_label (str): Label of the pool that handled it (for logging).

        Returns:
            None
        """
        self._append_progress(rec)
        batch: list[SubmissionRecord] | None = None
        seq = 0
        with self.records_lock:
            self.records.append(rec)
            self.completed += 1
            n = self.completed
            if rec.status == "submitted" and rec.task_id:
                self.submitted_ok += 1
            if rec.final_status == "Succeeded":
                self.succeeded += 1
            if (self.args.report_every > 0
                    and n - self.last_report >= self.args.report_every):
                # Only the models completed since the previous report — never
                # re-send the cumulative table.
                batch = self.records[self.last_report:n]
                self.last_report = n
                seq = self._batch_seq
                self._batch_seq += 1
        elapsed_h = (time.time() - self.start_ts) / 3600.0
        log.info("[progress] %d/%d done (%s) pool=%s model=%s status=%s/%s elapsed=%.1fh",
                 n, self.total_planned,
                 f"submitted_ok={self.submitted_ok} succeeded={self.succeeded}",
                 pool_label, rec.model, rec.status, rec.final_status or "-", elapsed_h)
        if batch:
            self._emit_batch_report(seq, batch)

    # ── per-model work ────────────────────────────────────────────────────────

    def _try_prewarm(self, repo: str) -> None:
        """Best-effort pre-pull of a model into ``/wekafs`` before register.

        No-op when prewarm is disabled or the NFS models root is not writable;
        timeouts/errors are logged and SaFE downloads the model instead.

        Args:
            repo (str): HuggingFace repo id to prewarm.

        Returns:
            None
        """
        if not self.args.prewarm:
            return
        nfs_root = os.environ.get("NFS_ROOT", "/wekafs")
        root = f"{nfs_root}/models"
        if not os.path.isdir(root) or not os.access(root, os.W_OK):
            return  # runner can't see/write /wekafs — SaFE will download instead
        try:
            subprocess.run(
                [sys.executable, str(_CI_DIR / "prewarm_models.py"),
                 "--repos", repo,
                 "--target-root", root,
                 "--concurrency", "1",
                 "--inner-workers", str(self.args.prewarm_inner_workers)],
                check=False, timeout=self.args.prewarm_timeout_s,
            )
        except subprocess.TimeoutExpired:
            log.warning("[%s] prewarm timed out after %ds — SaFE will download",
                        repo, self.args.prewarm_timeout_s)
        except Exception as e:
            log.warning("[%s] prewarm error %s — SaFE will download", repo, e)

    def _handle_one(self, safe: SafeOptimizeClient, cand: dict) -> SubmissionRecord:
        """Process one candidate: prewarm, register/submit, then wait+collect.

        Register/download/submit run under the concurrency gate; the long
        per-task wait runs outside it so full concurrency is reached.

        Args:
            safe (SafeOptimizeClient): Client for this candidate's pool.
            cand (dict): Candidate dict (requires ``repo_id``).

        Returns:
            SubmissionRecord: The resulting record (possibly ``skipped`` when
                the controller is stopping, or unwaited in dry-run mode).
        """
        repo = cand["repo_id"]
        pool_metadata = {
            "pool_id": cand.get("pool_id") or self.args.pool_id,
            "pool_index": cand.get("pool_index"),
            "batch_index": "",
            "batch_size": "",
            "source_task_id": "",
        }
        # Register + download + submit happen under the concurrency gate so the
        # storage fan-out stays bounded. The long 4h wait is OUTSIDE the gate.
        with self.register_sem:
            if self.stop.is_set():
                return SubmissionRecord(model=repo, status="skipped",
                                        error="controller stopping")
            self._try_prewarm(repo)
            rec = process_model(
                repo, self.hf, safe, overrides={},
                isl=self.args.isl, osl=self.args.osl,
                dry_run=self.args.dry_run, hf_token=self.hf_token,
                manual_mode=False, mode=self.args.mode,
                gpu_type=self.gpu_type, inferencex_path=self.inferencex_path,
                oob_path=self.oob_path, tracelens_root=self.tracelens_root,
                prompt_prefix=self.prompt_prefix or None,
                prompt_suffix=None,
                kernel_backends=self.kernel_backends,
                max_hours=self.args.max_hours,
                target_gain=self.args.target_gain,
                results_path=self.args.results_path,
                pool_metadata=pool_metadata,
            )
        if self.args.dry_run:
            return rec
        if rec.status == "submitted" and rec.task_id:
            wait_and_collect_one(
                safe, rec, self.artifacts_dir,
                task_timeout_min=self.args.task_timeout_min,
                poll_s=self.args.poll_interval_s,
                collect=True, all_artifacts=self.args.all_artifacts,
            )
        return rec

    def _worker(self, safe: SafeOptimizeClient, work_q: "queue.Queue[dict]",
                pool_label: str) -> None:
        """Worker loop: pull candidates off the queue and process each one.

        Exceptions from a single model are caught so one failure never kills
        the worker; each result is forwarded to ``_on_complete``.

        Args:
            safe (SafeOptimizeClient): Client for this worker's pool.
            work_q (queue.Queue[dict]): Shared queue of candidate dicts.
            pool_label (str): Label of this worker's pool (for logging).

        Returns:
            None
        """
        while not self.stop.is_set():
            try:
                cand = work_q.get_nowait()
            except queue.Empty:
                return
            repo = cand.get("repo_id", "?")
            try:
                rec = self._handle_one(safe, cand)
            except Exception as e:  # never let one model kill its worker
                log.exception("[%s] worker error on %s", pool_label, repo)
                rec = SubmissionRecord(model=repo, status="failed",
                                       error=f"worker: {type(e).__name__}: {e}")
            finally:
                work_q.task_done()
            try:
                self._on_complete(rec, pool_label)
            except Exception:
                log.exception("[%s] on_complete error for %s", pool_label, repo)

    def _heartbeat(self) -> None:
        """Periodic liveness log so a multi-day run shows progress in GHA."""
        interval = max(30, self.args.heartbeat_s)
        while not self.stop.wait(interval):
            with self.records_lock:
                n = self.completed
            elapsed_h = (time.time() - self.start_ts) / 3600.0
            rate = (n / elapsed_h) if elapsed_h > 0 else 0.0
            remaining = self.total_planned - n
            eta_h = (remaining / rate) if rate > 0 else float("nan")
            log.info("[heartbeat] %d/%d done, %.1f models/h, elapsed=%.1fh, ETA~%.1fh",
                     n, self.total_planned, rate, elapsed_h, eta_h)

    # ── run ─────────────────────────────────────────────────────────────────

    def run(self, candidates: list[dict]) -> int:
        """Run the full controller over a candidate list until drained.

        Optionally resumes from the progress log, starts the heartbeat and
        both worker pools, joins them, flushes the final partial batch to the
        webhook, and writes the full ranked summary.

        Args:
            candidates (list[dict]): Candidate dicts to process (pool order).

        Returns:
            int: Exit code — ``0`` if at least one task submitted (or there is
                nothing to do), ``2`` on missing API key / empty prompt
                prefix, else ``1``.
        """
        if not self.api_key and not self.args.dry_run:
            log.error("no API key (CLAW_API_KEY / SAFE_API_KEY / --api-key)")
            return 2
        if not self.prompt_prefix and not self.args.dry_run:
            log.error("prompt prefix empty — refusing to submit with empty promptPrefix")
            return 2

        # Resume: drop already-finished repos, pre-load their records so the
        # manifest + summary still include the full picture.
        done = load_done_records(self.progress_path) if self.args.resume else {}
        if done:
            log.info("resume: %d models already complete; skipping them", len(done))
            with self.records_lock:
                self.records.extend(done.values())
                self.completed = len(done)
                self.last_report = len(done)
                self.submitted_ok = sum(1 for r in done.values()
                                        if r.status == "submitted" and r.task_id)
                self.succeeded = sum(1 for r in done.values()
                                     if r.final_status == "Succeeded")
        pending = [c for c in candidates if c["repo_id"] not in done]
        self.total_planned = len(done) + len(pending)

        log.info("=" * 70)
        log.info("Tiny CI start: %d pending (%d already done), total=%d",
                 len(pending), len(done), self.total_planned)
        log.info("SaFE=%s register_ws=%s sandbox_ws=%s(cap=%d) hyperloom_ws=%s(cap=%d)",
                 self.base_url, self.register_workspace,
                 self.sandbox_workspace, self.args.sandbox_cap,
                 self.hyperloom_workspace, self.args.hyperloom_cap)
        log.info("max_hours=%.1f task_timeout=%dm register_gate=%d report_every=%d "
                 "kernel_backends=%s prewarm=%s all_artifacts=%s",
                 self.args.max_hours, self.args.task_timeout_min,
                 self.args.register_concurrency, self.args.report_every,
                 ",".join(self.kernel_backends), self.args.prewarm,
                 self.args.all_artifacts)

        if not pending:
            log.info("nothing to do")
            if self.completed:
                self._write_full_summary()
            return 0

        if self.args.dry_run:
            for c in pending[:20]:
                log.info("[dry-run] would submit %s (pool_index=%s)",
                         c["repo_id"], c.get("pool_index"))
            log.info("[dry-run] %d models total (showing first 20). No submit.",
                     len(pending))
            return 0

        work_q: "queue.Queue[dict]" = queue.Queue()
        for c in pending:
            work_q.put(c)

        threads: list[threading.Thread] = []
        hb = threading.Thread(target=self._heartbeat, name="heartbeat", daemon=True)
        hb.start()
        for i in range(self.args.sandbox_cap):
            t = threading.Thread(target=self._worker,
                                 args=(self.safe_sandbox, work_q, "sandbox"),
                                 name=f"sandbox-{i}", daemon=True)
            t.start()
            threads.append(t)
        for i in range(self.args.hyperloom_cap):
            t = threading.Thread(target=self._worker,
                                 args=(self.safe_hyperloom, work_q, "hyperloom"),
                                 name=f"hyperloom-{i}", daemon=True)
            t.start()
            threads.append(t)

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            log.warning("interrupted — signalling workers to stop after current model")
            self.stop.set()
            for t in threads:
                t.join(timeout=30)

        self.stop.set()
        # Flush the last partial batch to the webhook (the < report_every models
        # finished since the previous batch report), then build the full ranked
        # table as the run artifact. The full table is NOT webhooked.
        with self.records_lock:
            n = self.completed
            if self.last_report < n:
                tail = self.records[self.last_report:n]
                self.last_report = n
                seq = self._batch_seq
                self._batch_seq += 1
            else:
                tail, seq = None, -1
        if tail:
            self._emit_batch_report(seq, tail)
        self._write_full_summary()

        with self.records_lock:
            total = len(self.records)
            submitted_ok = self.submitted_ok
            succeeded = self.succeeded
            non_success = [r for r in self.records
                           if r.task_id and r.final_status != "Succeeded"]
        log.info("=" * 70)
        log.info("Tiny CI done: total=%d submitted_ok=%d succeeded=%d non_success=%d",
                 total, submitted_ok, succeeded, len(non_success))
        return 0 if submitted_ok > 0 else 1


# ── CLI ──────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build the Tiny CI command-line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with all controller, pool, SaFE,
            prewarm, and reporting options.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--candidates-file", default=DEFAULT_CANDIDATES_FILE,
                   help="Candidates JSON (default: the 3B-12B base pool).")
    p.add_argument("--start-index", type=int, default=0,
                   help="Skip the first N candidates (pool order). For splitting runs.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N candidates after --start-index. 0 = all.")

    p.add_argument("--sandbox-cap", type=int, default=150,
                   help="Concurrent in-flight tasks on the sandbox workspace.")
    p.add_argument("--hyperloom-cap", type=int, default=150,
                   help="Concurrent in-flight tasks on the hyperloom workspace.")
    p.add_argument("--register-concurrency", type=int, default=32,
                   help="Max concurrent register+download+submit operations "
                        "(bounds the storage fan-out; the 4h wait is unbounded).")
    p.add_argument("--report-every", type=int, default=10,
                   help="Emit a summary + webhook every K completed models. 0 = final only.")

    p.add_argument("--sandbox-workspace", default="")
    p.add_argument("--hyperloom-workspace", default="")
    p.add_argument("--register-workspace", default="")
    p.add_argument("--volume", default="")
    p.add_argument("--gpu-type", default="")
    p.add_argument("--inferencex-path", default="")
    p.add_argument("--oob-path", default="")
    p.add_argument("--tracelens-root", default="")

    p.add_argument("--api-url", default="")
    p.add_argument("--api-key", default="")

    p.add_argument("--isl", type=int, default=1024)
    p.add_argument("--osl", type=int, default=1024)
    p.add_argument("--mode", choices=["local", "claw"], default="local")
    p.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS,
                   help="Per-model optimizer budget (default 4h for Tiny).")
    p.add_argument("--target-gain", type=float, default=DEFAULT_TARGET_GAIN)
    p.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    p.add_argument("--kernel-opt-backends", default="geak,claude,codex")

    p.add_argument("--prompt-prefix", default="",
                   help="Override prompt prefix. Empty -> ci/prompt_prefix.txt.")
    p.add_argument("--prompt-prefix-file", default="",
                   help="Read the prompt prefix from this file (takes precedence; "
                        "the workflow uses it to inject the CI source pin without "
                        "leaking the token onto argv).")

    p.add_argument("--task-timeout-min", type=int, default=DEFAULT_TASK_TIMEOUT_MIN,
                   help="Per-task wait timeout in minutes (default 240 = 4h).")
    p.add_argument("--poll-interval-s", type=int, default=60)

    prewarm = p.add_mutually_exclusive_group()
    prewarm.add_argument("--prewarm", dest="prewarm", action="store_true", default=True,
                         help="Pre-pull each repo into /wekafs/models before register (default).")
    prewarm.add_argument("--no-prewarm", dest="prewarm", action="store_false",
                         help="Skip prewarm; let SaFE download each model itself.")
    p.add_argument("--prewarm-inner-workers", type=int, default=8)
    p.add_argument("--prewarm-timeout-s", type=int, default=3600)

    aa = p.add_mutually_exclusive_group()
    aa.add_argument("--all-artifacts", dest="all_artifacts", action="store_true", default=True,
                    help="Download the full session (default for Tiny — no re-fetch later).")
    aa.add_argument("--no-all-artifacts", dest="all_artifacts", action="store_false",
                    help="Only optimization_report + ci_metrics.")

    p.add_argument("--target-gpu", default="mi300x",
                   help="Reference GPU for the InferenceX comparison column.")
    p.add_argument("--webhook-url", default="")
    p.add_argument("--dashboard-url", default="")

    p.add_argument("--pool-id", default="",
                   help="Fallback pool id if the candidate row lacks one.")
    p.add_argument("--output-dir", default="tiny-output",
                   help="Where submission_manifest.{json,md} + progress.ndjson land.")
    p.add_argument("--artifacts-dir", default="tiny-artifacts")
    p.add_argument("--summary-out-dir", default="tiny-summary-out")

    p.add_argument("--resume", action="store_true",
                   help="Skip models already terminal in output-dir/progress.ndjson.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan; never register or submit.")
    p.add_argument("--heartbeat-s", type=int, default=300)
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> int:
    """CLI entrypoint: load candidates, build the controller, and run it.

    Returns:
        int: Process exit code — ``2`` if the candidates file is missing or no
            candidates are selected, otherwise the controller's exit code.
    """
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cand_path = Path(args.candidates_file)
    if not cand_path.is_absolute() and not cand_path.is_file():
        # Allow passing a path relative to ci/ (the default).
        alt = _CI_DIR / args.candidates_file
        if alt.is_file():
            cand_path = alt
    if not cand_path.is_file():
        log.error("candidates file not found: %s", args.candidates_file)
        return 2

    candidates = load_candidates(cand_path)
    if args.start_index:
        candidates = candidates[args.start_index:]
    if args.limit and args.limit > 0:
        candidates = candidates[:args.limit]
    if not candidates:
        log.error("no candidates selected (file=%s start=%d limit=%d)",
                  cand_path, args.start_index, args.limit)
        return 2
    log.info("loaded %d candidates from %s", len(candidates), cand_path)

    controller = TinyController(args)
    return controller.run(candidates)


if __name__ == "__main__":
    sys.exit(main())
