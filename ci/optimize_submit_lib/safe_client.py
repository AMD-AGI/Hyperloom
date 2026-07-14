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

globals().update({k: v for k, v in vars(_config).items() if not k.startswith("__")})

# ── SaFE client ─────────────────────────────────────────────────────────────────


class SafeOptimizeClient:
    """Thin wrapper for SaFE playground/optimization endpoints.

    Reuses the same bearer token as the rest of Hyperloom CI. The API contract
    here mirrors SaFE/scripts/optimize_submit.py (2026-05-06), in particular
    the ``target.volume`` field added to /api/v1/playground/models.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        register_workspace: str,
        submit_workspace: str,
        volume: str,
        timeout: int = 30,
        submit_workspaces_pool: list[str] | None = None,
    ):
        """Initialise the SaFE client and its authenticated HTTP session.

        Args:
            base_url (str): SaFE base URL (trailing slash trimmed).
            token (str): Bearer token for the Authorization header.
            register_workspace (str): Workspace where models are registered and
                downloaded (must allow RW writes to ``volume``).
            submit_workspace (str): Workspace where optimization tasks run (must
                allow the Sandbox scope).
            volume (str): Wekafs volume mounted RW in ``register_workspace``.
            timeout (int): Per-request timeout in seconds.
            submit_workspaces_pool (list[str] | None): Optional round-robin pool
                of submit workspaces; when set, each submit cycles through it.
        """
        self.base_url = base_url.rstrip("/")
        # Where the model registers + downloads (needs RW to the volume).
        self.register_workspace = register_workspace
        # Where the task is created (needs Sandbox scope); may equal register.
        self.submit_workspace = submit_workspace
        # Optional round-robin pool: each submit_task picks the next workspace,
        # letting a batch span multiple workspaces without manual splitting.
        self.submit_workspaces_pool = [w.strip() for w in (submit_workspaces_pool or []) if w and w.strip()] or None
        self._submit_ws_counter = 0
        self.volume = volume
        self.timeout = timeout
        self._sess = requests.Session()
        self._sess.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )
        # Honor CA bundle env so corp proxies don't break HTTPS.
        self._sess.verify = os.environ.get("SSL_CERT_FILE", os.environ.get("REQUESTS_CA_BUNDLE", True))

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        """Send an HTTP request to the SaFE API and return the parsed JSON.

        Args:
            method: HTTP method (e.g. ``GET``, ``POST``).
            path: API path appended to ``base_url``.
            body: Optional JSON request body.
            timeout: Optional per-request timeout; falls back to the client
                default.

        Returns:
            The decoded JSON response, or an empty dict when there is no body.

        Raises:
            RuntimeError: If the response status code is ``>= 400``.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._sess.request(method, url, json=body, timeout=timeout or self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else {}

    def find_model(self, repo_id: str) -> dict | None:
        """Look up an existing SaFE Model by HF source URL, scoped to
        register_workspace (where the canonical Model CR + LocalPaths live).

        Args:
            repo_id (str): HuggingFace repo id to match by source URL.

        Returns:
            dict | None: The matching SaFE model record, or ``None`` when none
            matches or the listing fails.
        """
        hf_url = f"https://huggingface.co/{repo_id}".rstrip("/")
        from urllib.parse import quote

        try:
            data = self._request(
                "GET",
                f"api/v1/playground/models?limit=200&workspace={quote(self.register_workspace)}",
            )
        except Exception as e:
            log.warning("list models failed: %s", e)
            return None
        for m in data.get("items", []):
            if (m.get("sourceURL") or "").rstrip("/") == hf_url:
                return m
        return None

    def register_model(
        self,
        repo_id: str,
        hf_token: str = "",
        local_path: str = "",
    ) -> str:
        """Register a model record with SaFE so submit_task has a model_id.

        local_path set → accessMode=local_path: SaFE skips its Download Job
        (phase=Ready immediately) since files are already on disk (prewarm path).
        local_path empty → accessMode=local: SaFE downloads from HF (slow
        fallback when prewarm can't run).

        Args:
            repo_id (str): HuggingFace repo id to register.
            hf_token (str): Optional HF token for gated downloads (local mode).
            local_path (str): On-disk model path; when set, registers in
                ``local_path`` mode and skips SaFE's Download Job.

        Returns:
            str: The registered SaFE model id (empty string when absent).
        """
        if local_path:
            # local_path mode bypasses SaFE's HF metadata fetch, so we MUST
            # provide displayName. SaFE feeds it into GenerateName → K8s
            # metadata.name, which must satisfy RFC 1123 (lowercase [a-z0-9-.],
            # 1-63 chars); sanitize here since the backend doesn't.

            raw = repo_id.split("/")[-1] or repo_id
            cleaned = re.sub(r"[^a-z0-9.-]+", "-", raw.lower()).strip(".-") or "model"
            # Trim to 50 to leave headroom for GenerateName's -xxxxx suffix (max 63).
            display_name = cleaned[:50].rstrip(".-") or "model"
            body = {
                "displayName": display_name,
                "source": {
                    "url": repo_id,
                    "accessMode": "local_path",
                    "localPath": local_path,
                },
                "workspace": self.register_workspace,
            }
            log.info(
                "[%s] register (local_path mode): workspace=%s displayName=%s localPath=%s",
                repo_id,
                self.register_workspace,
                display_name,
                local_path,
            )
        else:
            body = {
                "source": {
                    "url": repo_id,
                    "accessMode": "local",
                    **({"token": hf_token} if hf_token else {}),
                },
                "workspace": self.register_workspace,
                "target": {"volume": self.volume},
            }
            log.info(
                "[%s] register (local mode — SaFE will download): workspace=%s volume=%s",
                repo_id,
                self.register_workspace,
                self.volume,
            )
        result = self._request("POST", "api/v1/playground/models", body)
        return result.get("id", "")

    def wait_ready(
        self,
        model_id: str,
        timeout_min: int = 480,
        poll_s: int = 30,
    ) -> bool:
        """Poll a SaFE model until it reaches the Ready phase.

        Args:
            model_id (str): SaFE model id to poll.
            timeout_min (int): Maximum minutes to wait before giving up.
            poll_s (int): Seconds between polls.

        Returns:
            bool: True once the model is Ready; False if it reaches Failed or
                the timeout elapses.
        """
        log.info("waiting for model %s to be Ready (timeout=%dm)", model_id, timeout_min)
        deadline = time.time() + timeout_min * 60
        last_phase = ""
        while time.time() < deadline:
            try:
                m = self._request("GET", f"api/v1/playground/models/{model_id}")
                phase = m.get("phase", "")
                if phase != last_phase:
                    log.info("model %s phase: %s", model_id, phase or "(empty)")
                    last_phase = phase
                if phase == "Ready":
                    return True
                if phase == "Failed":
                    log.error("model %s Failed: %s", model_id, m.get("message", ""))
                    return False
            except Exception as e:
                log.debug("phase poll error (will retry): %s", e)
            time.sleep(poll_s)
        log.error("model %s wait timed out after %dm", model_id, timeout_min)
        return False

    def submit_task(
        self,
        model_id: str,
        display_name: str,
        framework: str,
        precision: str,
        tp: int,
        concurrency: int,
        isl: int,
        osl: int,
        image: str | None,
        mode: str = "local",
        gpu_type: str | None = None,
        inferencex_path: str | None = None,
        oob_path: str | None = None,
        tracelens_root: str | None = None,
        prompt_prefix: str | None = None,
        prompt_suffix: str | None = None,
        kernel_backends: list[str] | None = None,
        max_hours: float | None = None,
        target_gain: float | None = None,
        results_path: str | None = None,
        env: dict | None = None,
    ) -> dict:
        """Submit an optimization task to SaFE and return the API response.

        Builds the task request body from the model and benchmark parameters,
        choosing a target workspace (single or round-robin across the pool).

        Args:
            model_id: Registered SaFE model id.
            display_name: Human-readable task name.
            framework: Serving framework (``vllm`` / ``sglang``).
            precision: Model precision (e.g. ``BF16``, ``FP8``).
            tp: Tensor-parallel size.
            concurrency: Benchmark concurrency level.
            isl: Input sequence length.
            osl: Output sequence length.
            image: Server image to run, or ``None`` for the framework default.
            mode: Submission mode (e.g. ``local``).
            gpu_type: Target GPU type.
            inferencex_path: Optional InferenceX checkout path.
            oob_path: Optional out-of-box baseline path.
            tracelens_root: Optional TraceLens root path.
            prompt_prefix: Optional prompt prefix override.
            prompt_suffix: Optional prompt suffix override.
            kernel_backends: Optional list of kernel backends to enable.
            max_hours: Optional wall-clock budget for the task.
            target_gain: Optional target performance gain.
            results_path: Optional path where results should be written.

        Returns:
            The decoded API response for the submitted task.
        """
        # Pick the workspace: single submit_workspace, or round-robin across the
        # pool. Counter is per-instance, not thread-safe — fine since submit_task
        # runs serially (only wait_and_collect is parallel, after submit returns).
        if self.submit_workspaces_pool:
            chosen_ws = self.submit_workspaces_pool[self._submit_ws_counter % len(self.submit_workspaces_pool)]
            self._submit_ws_counter += 1
            log.info(
                "[submit] round-robin chose workspace=%s (pool=%s, idx=%d)",
                chosen_ws,
                ",".join(self.submit_workspaces_pool),
                self._submit_ws_counter - 1,
            )
        else:
            chosen_ws = self.submit_workspace
        body = {
            "displayName": display_name,
            "modelId": model_id,
            "workspace": chosen_ws,
            "mode": mode,
            "framework": framework,
            "precision": precision,
            "tp": tp,
            "ep": 1,
            "isl": isl,
            "osl": osl,
            "concurrency": concurrency,
            "kernelBackends": list(kernel_backends or DEFAULT_KERNEL_BACKENDS),
        }
        if max_hours and max_hours > 0:
            body["maxHours"] = max_hours
        if target_gain and target_gain > 0:
            body["targetGain"] = target_gain
        if results_path:
            body["resultsPath"] = results_path
        if image:
            body["image"] = image
        # Override SaFE's wrong-for-core42 MI355X default (see DEFAULT_GPU_TYPE).
        if gpu_type:
            body["gpuType"] = gpu_type
        # Always send inferencexPath (even empty) to suppress SaFE's Zod default
        # "/hyperloom/InferenceX"; empty lets install.sh clone a writable copy.
        body["inferencexPath"] = inferencex_path or ""
        if oob_path:
            body["oobPath"] = oob_path
        if tracelens_root:
            body["tracelensRoot"] = tracelens_root
        # Optional prefix/suffix forwarded to BuildHyperloomPrompt on the SaFE side.
        if prompt_prefix:
            body["promptPrefix"] = prompt_prefix
        if prompt_suffix:
            body["promptSuffix"] = prompt_suffix
        # Optional session-scoped env forwarded to SaFE (body.env). SaFE relays
        # it to Claw as session_env, injected into the sandbox so the
        # inference_optimizer process sees it (e.g. CLAUDE_MODEL override).
        # Logged so each CI job records exactly which session env it submitted,
        # making "did the env reach Claw" verifiable from the job log alone.
        if env:
            body["env"] = env
            log.info("[%s] session env forwarded to SaFE: %s", display_name, env)
        attempts = 8
        # Captured before the first POST so the dedup lookup only matches a task
        # this call created (not an unrelated older one for the same model).
        submit_started_at = time.time()
        for attempt in range(1, attempts + 1):
            try:
                # The submit POST can be slow when the core42 apiserver is
                # under load from many parallel daily jobs. Give it a generous
                # read timeout so a busy-but-alive backend doesn't trip the
                # default 30s and get misreported as a hard submit failure.
                return self._request("POST", "api/v1/optimization/tasks", body, timeout=120)
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                transient = (
                    "HTTP 500" in msg
                    or "HTTP 502" in msg
                    or "HTTP 503" in msg
                    or "HTTP 504" in msg
                    or "timed out" in low  # requests ReadTimeout/ConnectTimeout
                    or "timeout" in low
                    or "connection" in low  # ConnectionError / HTTPSConnectionPool
                )
                if not transient:
                    raise
                # The POST is NOT idempotent: a slow/dropped response can hide a
                # task the backend actually created. Blindly retrying then spawns
                # a SECOND Claw session for the same model (the duplicate /
                # abandoned-PRELUDE dirs). Before retrying, look the task up by
                # modelId+workspace; if it already exists, reuse it.
                existing = self._find_recent_submitted_task(model_id, chosen_ws, submit_started_at)
                if existing and existing.get("id"):
                    log.warning(
                        "[submit] transient submit failure but a task for modelId=%s "
                        "already exists server-side (id=%s) — reusing it instead of "
                        "re-POSTing to avoid a duplicate session: %s",
                        model_id,
                        existing.get("id"),
                        msg,
                    )
                    return existing
                if attempt >= attempts:
                    raise
                delay = random.uniform(10, 60)
                log.warning(
                    "[submit] transient SaFE/Claw submit failure (attempt %d/%d, workspace=%s); retrying in %.1fs: %s",
                    attempt,
                    attempts,
                    chosen_ws,
                    delay,
                    msg,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable submit retry loop exit")

    def _find_recent_submitted_task(self, model_id: str, workspace: str, since_ts: float) -> dict | None:
        """Find a task this submit call may have created before its POST failed.

        A transient submit failure (slow/timeout/dropped response) can hide a
        task the backend actually created. Look it up by ``modelId`` +
        ``workspace`` so the retry path can reuse it instead of creating a
        duplicate Claw session.

        Args:
            model_id (str): Registered SaFE model id used in the submit body.
            workspace (str): Submit workspace used in the submit body.
            since_ts (float): Unix time captured just before the first POST;
                only tasks created at/after this (minus clock-skew slack) match.

        Returns:
            dict | None: The most recently-created matching task, or ``None``.
        """
        from urllib.parse import quote

        try:
            data = self._request(
                "GET",
                f"api/v1/optimization/tasks?modelId={quote(model_id)}"
                f"&workspace={quote(workspace)}&limit=20",
            )
        except Exception as e:  # noqa: BLE001
            log.debug("[submit] dedup lookup failed (will fall back to retry): %s", e)
            return None
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return None
        floor = since_ts - 120.0  # 2-min slack for client/server clock skew
        best = None
        best_ct = floor
        for it in items:
            if not isinstance(it, dict) or (it.get("modelId") or "") != model_id:
                continue
            created = str(it.get("createdAt") or "")
            try:
                ct = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
            except Exception:  # noqa: BLE001
                continue
            if ct >= best_ct:
                best_ct = ct
                best = it
        return best

    # ── Task lifecycle ──

    # Lifecycle states from SaFE types.go OptimizationTaskStatus.
    TERMINAL_TASK_STATUSES = {"Succeeded", "Failed", "Interrupted"}

    def get_task(self, task_id: str) -> dict:
        """Fetch the current state of an optimization task.

        Args:
            task_id (str): SaFE optimization task id.

        Returns:
            dict: The task record JSON.
        """
        return self._request("GET", f"api/v1/optimization/tasks/{task_id}")

    def wait_task_done(
        self,
        task_id: str,
        timeout_min: int = 480,
        poll_s: int = 60,
    ) -> tuple[str, dict]:
        """Wait until the task reaches a terminal status. Returns (status, last_task).

        Prefer the Claw SSE stream (SaFE's task status lags Claw by minutes), and
        fall back to SaFE polling when no clawSessionId exists or SSE fails.
        Returns ('Timeout', {}) if neither sees a terminal status by the deadline.

        Args:
            task_id (str): SaFE optimization task id to wait on.
            timeout_min (int): Maximum minutes to wait for a terminal status.
            poll_s (int): Seconds between SaFE status polls.

        Returns:
            tuple[str, dict]: ``(status, last_task)`` — the terminal status (or
            ``"Timeout"``) and the last task record observed.
        """
        log.info("[task %s] waiting for completion (timeout=%dm, poll=%ds)", task_id, timeout_min, poll_s)
        deadline = time.time() + timeout_min * 60

        # Wait briefly (cap 60s) for clawSessionId to materialize, else fall
        # through to polling.
        sid = None
        for _ in range(12):
            sid = self._claw_session_id_for(task_id)
            if sid:
                break
            time.sleep(5)

        sse_used = False
        sf_status = ""
        # On idle_timeout/stream_error the Claw SSE merely went quiet (a long tool
        # call, or the agent paused between phases). SaFE often *prematurely* marks
        # such a task Failed/Interrupted ("optimization report not found") even
        # though it is still well within budget and may resume to finalize (and
        # write session_breakdown.json). Do NOT accept an idle-induced terminal as
        # final: resubscribe and keep waiting, up to a bounded number of
        # consecutive idle re-entries, before concluding. Only a real Stopped
        # (sandbox exit), a Succeeded, the deadline, or exhausted retries end it.
        idle_retries = 0
        # Default 0: a single 1h idle window (idle_grace_s) is the whole budget;
        # no resubscribe. Raise SAFE_OPTIMIZE_SSE_IDLE_RETRIES to re-enable retries.
        max_idle_retries = int(_env_float("SAFE_OPTIMIZE_SSE_IDLE_RETRIES", 0.0))
        while sid and time.time() < deadline:
            sse_used = True
            log.info("[task %s] using SSE on clawSessionId=%s", task_id, sid)
            sse_reason = self._sse_wait_until_done(sid, deadline)
            log.info("[task %s] SSE finished: reason=%s", task_id, sse_reason)
            try:
                last_task = self.get_task(task_id)
            except Exception:
                last_task = {}
            sf_status = last_task.get("status", "") if last_task else ""
            # A real success is always final.
            if sf_status == "Succeeded":
                return sf_status, last_task
            # Stopped = sandbox pod exited (real end-of-task). SaFE's controller
            # lags shutdown by 10-180s, so short-poll up to 5min for its verdict.
            if sse_reason == "Stopped":
                log.info("[task %s] sandbox stopped — short-polling SaFE for terminal status (up to 5min)", task_id)
                for _ in range(30):
                    time.sleep(10)
                    if time.time() > deadline:
                        break
                    try:
                        last_task = self.get_task(task_id)
                    except Exception:
                        continue
                    sf_status = last_task.get("status", "") if last_task else ""
                    if sf_status in self.TERMINAL_TASK_STATUSES:
                        log.info("[task %s] SaFE settled on %s after sandbox stop", task_id, sf_status)
                        return sf_status, last_task
                # Sandbox gone but SaFE hasn't settled — treat as Succeeded so
                # collect_artifacts can still read whatever the agent wrote.
                log.info(
                    "[task %s] SaFE never settled within 5min after "
                    "sandbox stop — returning Succeeded (collect "
                    "step will read ci_metrics.json directly)",
                    task_id,
                )
                return "Succeeded", last_task
            if sse_reason == "deadline":
                return "Timeout", last_task
            # idle_timeout / stream_error: inconclusive. Resubscribe and keep
            # waiting (bounded) instead of accepting an idle-induced terminal.
            if sse_reason in ("idle_timeout", "stream_error") and idle_retries < max_idle_retries and time.time() < deadline:
                idle_retries += 1
                log.info(
                    "[task %s] SSE %s within budget (sf_status=%s) — resubscribing "
                    "(idle retry %d/%d) instead of concluding on a non-Stopped status",
                    task_id,
                    sse_reason,
                    sf_status or "?",
                    idle_retries,
                    max_idle_retries,
                )
                time.sleep(min(poll_s, 60))
                continue
            # Retries exhausted (or another reason): accept a terminal SaFE
            # verdict if one exists, else fall through to SaFE polling.
            if sf_status in self.TERMINAL_TASK_STATUSES:
                return sf_status, last_task
            log.info(
                "[task %s] SSE inconclusive (reason=%s, sf_status=%s) — "
                "falling back to SaFE polling for terminal status",
                task_id,
                sse_reason,
                sf_status or "?",
            )
            break

        # Fallback / continuation: SaFE optimization-API polling.
        if not sse_used:
            log.info("[task %s] no clawSessionId yet — using SaFE polling", task_id)
        last_status = ""
        last_phase = -1
        last_task: dict = {}
        while time.time() < deadline:
            try:
                t = self.get_task(task_id)
                last_task = t
                status = t.get("status", "")
                phase = t.get("currentPhase", -1)
                if status != last_status or phase != last_phase:
                    log.info(
                        "[task %s] status=%s phase=%s message=%s",
                        task_id,
                        status or "?",
                        phase,
                        (t.get("message") or "")[:120],
                    )
                    last_status, last_phase = status, phase
                if status in self.TERMINAL_TASK_STATUSES:
                    return status, t
            except Exception as e:
                log.debug("[task %s] poll error (will retry): %s", task_id, e)
            time.sleep(poll_s)
        log.warning("[task %s] wait timed out after %dm", task_id, timeout_min)
        return "Timeout", last_task

    def _sse_wait_until_done(self, session_id: str, deadline: float) -> str:
        """Subscribe to the Claw session SSE stream, return when the agent ends.

        IMPORTANT: do NOT return on ResultMessage — it fires at the end of EVERY
        agent turn (dozens over 1-3h), so the first turn was being mistaken for
        completion. The only reliable signal is sandboxStatus
        phase=Stopped/Terminated/Failed (sandbox pod actually exits).

        Returns: "Stopped" | "idle_timeout" (no events > idle_grace_s, default
        1h) | "deadline" (per-task wall clock) | "stream_error" (caller falls
        back).
        """
        url = f"{self.base_url}/claw-api/v1/chat/sessions/{session_id}/messages"
        last_evt = time.time()
        # Idle grace: how long the SSE may be silent (keepalive-only) before we
        # treat it as idle. Default 1h to cover slow model download + inference
        # server startup, which emit no agent events. Configurable via env.
        idle_grace_s = int(_env_float("SAFE_OPTIMIZE_SSE_IDLE_GRACE_S", 3600.0))
        try:
            with self._sess.get(url, stream=True, timeout=(10, 60)) as r:
                if not r.ok:
                    log.warning("SSE stream HTTP %d for session %s", r.status_code, session_id[:8])
                    return "stream_error"
                current_event = None
                for raw in r.iter_lines(decode_unicode=True):
                    now = time.time()
                    if now > deadline:
                        return "deadline"
                    if raw is None:
                        continue
                    if raw == "" or raw.startswith(":"):
                        # Blank-line separator or `: keepalive` heartbeat.
                        if now - last_evt > idle_grace_s:
                            return "idle_timeout"
                        continue
                    if raw.startswith("id:"):
                        continue
                    if raw.startswith("event:"):
                        current_event = raw[6:].strip()
                        continue
                    if not raw.startswith("data:"):
                        continue
                    payload = raw[5:].strip()
                    try:
                        d = json.loads(payload)
                    except Exception:
                        continue
                    last_evt = now
                    et = d.get("type") or current_event
                    # ResultMessage is intentionally not a return signal (see
                    # docstring); it still refreshes last_evt above.
                    if et == "sandboxStatus":
                        ph = (d.get("phase") or "").lower()
                        if ph in ("stopped", "terminated", "failed"):
                            return "Stopped"
        except requests.exceptions.RequestException as e:
            log.warning("SSE stream error for session %s: %s", session_id[:8], type(e).__name__)
            return "stream_error"
        except Exception as e:
            log.warning("SSE stream unexpected error for session %s: %s", session_id[:8], e)
            return "stream_error"
        # Stream closed cleanly without a Stopped signal — treat as idle.
        return "idle_timeout"

    def _claw_session_id_for(self, task_id: str) -> str | None:
        """Resolve clawSessionId for a task (per-instance cached). None when
        SaFE has no session attached (e.g. task failed before session creation).

        Args:
            task_id (str): SaFE optimization task id.

        Returns:
            str | None: The Claw session id, or ``None`` when none is attached
            or the lookup fails.
        """
        if not hasattr(self, "_claw_session_cache"):
            self._claw_session_cache = {}
        if task_id in self._claw_session_cache:
            return self._claw_session_cache[task_id]
        try:
            t = self.get_task(task_id)
        except Exception as e:
            log.warning("[task %s] get_task failed while resolving clawSessionId: %s", task_id, e)
            return None
        sid = (t.get("clawSessionId") or "").strip()
        self._claw_session_cache[task_id] = sid or None
        return sid or None

    def list_artifacts(self, task_id: str) -> list[dict]:
        """List task artifacts via the SaFE standard endpoint.

        Returns items shaped {path, size, lastModified, downloadPath}.
        downloadPath is server-relative; download_artifact() uses it when present,
        else builds the /artifacts/download?path= URL from path.

        Args:
            task_id (str): SaFE optimization task id.

        Returns:
            list[dict]: Artifact item dicts, or ``[]`` on error / unexpected
            shape.
        """
        try:
            data = self._request("GET", f"api/v1/optimization/tasks/{task_id}/artifacts")
        except Exception as e:
            log.warning("[task %s] list_artifacts failed: %s", task_id, e)
            return []
        items = (data.get("items") or data.get("data")) if isinstance(data, dict) else data
        if isinstance(items, dict) and isinstance(items.get("items"), list):
            items = items["items"]
        if not isinstance(items, list):
            return []
        return items

    def download_artifact(self, task_id: str, path_or_item: "str | dict") -> bytes:
        """Download a single task artifact. Accepts a string path or a
        list_artifacts item dict; prefers the item's downloadPath when present.

        Args:
            task_id (str): SaFE optimization task id.
            path_or_item (str | dict): Artifact path string or a
                :meth:`list_artifacts` item dict.

        Returns:
            bytes: The downloaded artifact content.

        Raises:
            RuntimeError: If the download response status is ``>= 400``.
        """
        if isinstance(path_or_item, dict):
            download_path = (path_or_item.get("downloadPath") or "").strip()
            if download_path:
                url = f"{self.base_url}/{download_path.lstrip('/')}"
            else:
                path = path_or_item.get("path", "")
                encoded = requests.utils.quote(path, safe="")
                url = f"{self.base_url}/api/v1/optimization/tasks/{task_id}/artifacts/download?path={encoded}"
        else:
            encoded = requests.utils.quote(path_or_item, safe="")
            url = f"{self.base_url}/api/v1/optimization/tasks/{task_id}/artifacts/download?path={encoded}"
        resp = self._sess.get(url, timeout=120)
        if resp.status_code >= 400:
            raise RuntimeError(f"SaFE artifact download -> HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.content

    def download_artifact_to(
        self,
        task_id: str,
        path_or_item: "str | dict",
        local_path: str,
    ) -> int:
        """Download an artifact and write it to a local file.

        Creates parent directories as needed.

        Args:
            task_id (str): SaFE optimization task id.
            path_or_item (str | dict): Artifact path string or item dict.
            local_path (str): Destination file path.

        Returns:
            int: Number of bytes written.
        """
        data = self.download_artifact(task_id, path_or_item)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        return len(data)
