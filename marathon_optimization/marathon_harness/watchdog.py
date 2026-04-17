"""Watchdog — triage (Level 2A-2P + merge/accuracy), 17 pattern signatures, RCA dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

from . import ipc, rca_bridge

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLL_INTERVAL_S = 30
PATTERN_THRESHOLD = 3
MAX_CONCURRENT_RCA = 2
RCA_TIMEOUT_MIN = 15
EVIDENCE_SNIPPET_CHARS = 5000

# ---------------------------------------------------------------------------
# 17 pattern signatures (from watchdog/SKILL.md)
# ---------------------------------------------------------------------------

SIGNATURES: dict[str, re.Pattern[str]] = {
    "triton_register_alloc": re.compile(r"register allocation|VGPR", re.I),
    "segfault":              re.compile(r"segfault|signal 11|exit.*139", re.I),
    "hipcc_compilation":     re.compile(r"hipcc|hip compilation", re.I),
    "import_error":          re.compile(r"ImportError|ModuleNotFoundError", re.I),
    "sgl_kernel_build":      re.compile(r"sgl.kernel|setup_rocm", re.I),
    "rebuild":               re.compile(r"rebuild|pip install|cmake", re.I),
    "rccl_timeout":          re.compile(r"RCCL|NCCL.*timeout", re.I),
    "rdma_error":            re.compile(r"RDMA|InfiniBand|ibverbs", re.I),
    "comm_failure":          re.compile(r"all.reduce|broadcast|scatter", re.I),
    "server_oom":            re.compile(r"out of memory|OOM|CUDA.*alloc", re.I),
    "server_hang":           re.compile(r"hang|deadlock|unresponsive", re.I),
    "server_crash":          re.compile(r"server.*crash|sglang.*exit", re.I),
    "triton_codegen":        re.compile(r"triton.*codegen|triton.*compile", re.I),
    "inductor_codegen":      re.compile(r"inductor|torchinductor", re.I),
    "codegen_fail":          re.compile(r"codegen.*fail", re.I),
    "tuning_fail":           re.compile(r"tuning.*crash|autotune.*fail", re.I),
    "other":                 re.compile(r".*"),
}


def _error_signature(event: dict[str, Any]) -> str:
    """Match event details against known signatures."""
    msg = (event.get("details") or {}).get("error_message", "")
    crash = (event.get("details") or {}).get("crash_log_snippet", "")
    text = f"{msg} {crash}"
    for name, pattern in SIGNATURES.items():
        if name == "other":
            continue
        if pattern.search(text):
            return name
    return f"other_{hashlib.md5(text.encode()).hexdigest()[:8]}"


# ---------------------------------------------------------------------------
# Triage verdicts
# ---------------------------------------------------------------------------

def _inv(priority: str, reason: str, systemic: bool = False) -> dict[str, Any]:
    return {"action": "investigate", "priority": priority, "systemic": systemic, "reason": reason}

def _skip(reason: str) -> dict[str, Any]:
    return {"action": "skip", "priority": "low", "systemic": False, "reason": reason}

def _watch(reason: str) -> dict[str, Any]:
    return {"action": "pattern-watch", "priority": "low", "systemic": False, "reason": reason}


def triage(event: dict[str, Any], pattern_tracker: dict[str, int]) -> dict[str, Any]:
    """Hardcoded triage (Level 2A-2L, anomaly 2M-2P, merge/accuracy)."""
    t = event.get("type", "")
    d = event.get("details") or {}
    exit_code = d.get("exit_code")
    micro = d.get("micro_speedup_before_crash", 0) or 0
    gpu_pct = d.get("gpu_pct", 0) or 0
    msg = (d.get("error_message") or "").lower()
    round_num = d.get("round_number", 0) or 0

    # 2A: segfault
    if t == "segfault" or exit_code == 139:
        if micro > 1:
            return _inv("high", "segfault with prior improvement")
        sh = d.get("session_history") or []
        if any(r.get("outcome") == "PASS" for r in sh):
            return _inv("high", "segfault after prior success in session")
        return _inv("medium", "segfault")

    # 2B: crash
    if t == "crash":
        if micro > 1:
            return _inv("high", "crash with prior improvement")
        if exit_code == 134:
            return _inv("medium", "abort signal")
        if micro <= 1 and round_num <= 1:
            return _skip("crash R1 no improvement")
        if round_num >= 3:
            return _watch("crash late round")
        return _watch("crash")

    # 2C: regression
    if t == "regression":
        if micro > 1.5:
            return _inv("high", "regression after 1.5x micro gain")
        if micro > 1.05:
            return _inv("medium", "regression after modest gain")
        return _skip("regression without improvement")

    # 2D: compilation-fail
    if t == "compilation-fail":
        sig = _error_signature(event)
        if pattern_tracker.get(sig, 0) >= PATTERN_THRESHOLD - 1:
            return _inv("high", f"compilation systemic: {sig}", systemic=True)
        if "register allocation" in msg:
            return _inv("medium", "register allocation failure")
        if "hipcc" in msg:
            return _watch("hipcc compilation")
        return _watch("compilation failure")

    # 2E: merge-revert / merge-fail
    if t in ("merge-revert", "merge-fail"):
        if d.get("rebuild_required"):
            return _inv("high", "merge-revert after rebuild")
        if exit_code in (139, 134):
            return _inv("high", "server crashed after patch")
        return _watch("E2E regression no crash")

    # 2F: exhausted
    if t == "exhausted":
        if gpu_pct > 5:
            return _inv("medium", "exhausted high-value kernel")
        return _skip("exhausted low-value kernel")

    # 2G: rebuild-fail / rebuild-crash
    if t in ("rebuild-fail", "rebuild-crash"):
        if t == "rebuild-crash":
            return _inv("high", "rebuild crash")
        sig = _error_signature(event)
        if pattern_tracker.get(sig, 0) >= 1:
            return _inv("high", f"rebuild systemic: {sig}", systemic=True)
        if "hipcc" in msg:
            return _inv("medium", "hipcc rebuild failure")
        if "setup_rocm" in msg:
            return _inv("medium", "setup_rocm rebuild failure")
        return _watch("rebuild failure")

    # 2H: tuning-crash / tuning-fail
    if t == "tuning-crash":
        if exit_code == 139:
            return _inv("high", "tuning segfault")
        if "oom" in msg or "out of memory" in msg:
            return _inv("medium", "tuning OOM")
        return _watch("tuning crash")
    if t == "tuning-fail":
        if "server" in msg and "crash" in msg:
            return _inv("high", "server crash on tuned config load")
        return _skip("tuning no improvement")

    # 2I: comm-hang / comm-fail
    if t == "comm-hang":
        return _inv("high", "communication hang")
    if t == "comm-fail":
        if "timeout" in msg:
            return _inv("high", "comm timeout")
        sig = _error_signature(event)
        if pattern_tracker.get(sig, 0) >= 1:
            return _inv("high", f"comm systemic: {sig}", systemic=True)
        return _watch("comm failure")

    # 2J: codegen-fail / cache-corrupt
    if t == "cache-corrupt":
        return _inv("medium", "cache corruption")
    if t == "codegen-fail":
        sig = _error_signature(event)
        if pattern_tracker.get(sig, 0) >= PATTERN_THRESHOLD - 1:
            return _inv("high", f"codegen systemic: {sig}", systemic=True)
        return _watch("codegen failure")

    # 2K: server-crash / server-hang
    if t == "server-crash":
        if exit_code == 139:
            return _inv("high", "server segfault")
        if "oom" in msg:
            return _inv("medium", "server OOM")
        if d.get("last_config_change"):
            return _inv("high", "server crash after config change")
        return _watch("server crash")
    if t == "server-hang":
        return _inv("medium", "server hang")

    # 2L: dispatch-fix-fail
    if t == "dispatch-fix-fail":
        if d.get("fix_type") == "git-revert":
            return _inv("medium", "git-revert fix failed")
        return _watch("dispatch fix failure")

    # 2M: micro-e2e-gap — micro speedup didn't translate to E2E
    if t == "micro-e2e-gap":
        micro = d.get("micro_speedup", 0) or 0
        e2e_gain = d.get("e2e_gain_pct", 0) or 0
        if micro > 1.2 and e2e_gain < 1.0:
            return _inv("medium", f"micro {micro:.1f}x but E2E only +{e2e_gain:.1f}%")
        return _skip("micro-e2e gap within tolerance")

    # 2N: regime-divergence — different behavior at different concurrency levels
    if t == "regime-divergence":
        return _inv("medium", "regime-dependent performance detected")

    # 2O: stale-change — change made but 0% E2E delta suggests JIT/cache issue
    if t == "stale-change":
        return _inv("high", "code changed but no E2E impact — likely stale JIT/cache")

    # 2P: interaction-regression — combined changes regress despite individual gains
    if t == "interaction-regression":
        return _inv("high", "interaction regression between stacked changes")

    # merge-keep → SKIP
    if t == "merge-keep":
        return _skip("successful merge")

    # accuracy-fail → SKIP
    if t == "accuracy-fail":
        return _skip("accuracy handled by orchestrator")

    return _watch(f"unknown: {t}")


# ---------------------------------------------------------------------------
# Watchdog class — main async loop
# ---------------------------------------------------------------------------

class Watchdog:
    """Event monitor + triage + RCA dispatch."""

    def __init__(
        self,
        state: Any,
        llm: Any,
        session_dir: str,
        env: dict[str, str],
        dashboard: Any,
        shutdown: asyncio.Event,
    ):
        self.state = state
        self.llm = llm
        self.session_dir = session_dir
        self.env = env
        self.dashboard = dashboard
        self.shutdown = shutdown
        self.pattern_tracker: dict[str, int] = {}
        self.active_investigations = 0
        self.last_seen_event_id = getattr(state, "watchdog_last_seen_event_id", "") or ""
        self.investigation_queue: list[dict[str, Any]] = []
        self._rca_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RCA)

    async def run(self) -> None:
        log.info("Watchdog started — polling every %ds", POLL_INTERVAL_S)
        while not self.shutdown.is_set():
            try:
                await self._poll_cycle()
            except Exception as exc:
                log.exception("Watchdog poll error: %s", exc)
            try:
                await asyncio.wait_for(self.shutdown.wait(), timeout=POLL_INTERVAL_S)
                break
            except asyncio.TimeoutError:
                pass

    async def _poll_cycle(self) -> None:
        events = ipc.read_new_events(self.session_dir, after_id=self.last_seen_event_id)
        if not events:
            return

        for event in events:
            self.last_seen_event_id = event.get("id", self.last_seen_event_id)

            sig = _error_signature(event)
            self.pattern_tracker[sig] = self.pattern_tracker.get(sig, 0) + 1
            if len(self.pattern_tracker) > 500:
                min_count = sorted(self.pattern_tracker.values())[len(self.pattern_tracker) // 2]
                self.pattern_tracker = {
                    k: v for k, v in self.pattern_tracker.items() if v >= min_count
                }

            # Triage
            verdict = triage(event, self.pattern_tracker)
            log.info("Triage %s [%s]: %s (%s)",
                     event.get("id"), event.get("type"),
                     verdict["action"], verdict["reason"])
            self.dashboard.update_wd(
                f"triage: {event.get('type')} → {verdict['action']}",
                self.active_investigations,
                dict(self.pattern_tracker),
            )

            if verdict["action"] == "investigate":
                self.investigation_queue.append({"event": event, "verdict": verdict})
                if event.get("type") == "regression":
                    ipc.write_insight(self.session_dir, {
                        "type": "anomaly-detected",
                        "source": "watchdog-triage-2C",
                        "event_id": event.get("id"),
                        "task_id": event.get("task_id"),
                        "kernel_name": event.get("kernel_name"),
                        "summary": verdict.get("reason", ""),
                    })
            elif verdict["action"] == "pattern-watch":
                # Check promotion threshold
                if self.pattern_tracker.get(sig, 0) >= PATTERN_THRESHOLD:
                    log.info("Pattern %s promoted to investigate (count=%d)", sig, self.pattern_tracker[sig])
                    self.investigation_queue.append({
                        "event": event,
                        "verdict": _inv("high", f"pattern promoted: {sig}", systemic=True),
                    })

        # Persist event cursor to state
        self.state.watchdog_last_seen_event_id = self.last_seen_event_id

        # Process investigation queue (priority ordered, bounded task creation)
        max_tasks = MAX_CONCURRENT_RCA * 3
        self.investigation_queue.sort(key=lambda x: self._priority_key(x["verdict"]))
        if len(self.investigation_queue) > max_tasks:
            dropped = len(self.investigation_queue) - max_tasks
            self.investigation_queue = self.investigation_queue[:max_tasks]
            log.warning("Investigation queue capped: dropped %d lowest-priority items", dropped)

        rca_tasks = []
        while self.investigation_queue:
            item = self.investigation_queue.pop(0)
            rca_tasks.append(asyncio.create_task(self._run_investigation(item["event"])))
        if rca_tasks:
            await asyncio.gather(*rca_tasks, return_exceptions=True)

    async def _run_investigation(self, event: dict[str, Any]) -> None:
        """Semaphore-gated wrapper for concurrent RCA investigations."""
        async with self._rca_semaphore:
            self.active_investigations += 1
            try:
                finding = await asyncio.wait_for(
                    self._investigate(event), timeout=RCA_TIMEOUT_MIN * 60,
                )
                ipc.write_finding(self.session_dir, finding)
                log.info("Finding written for %s: %s — resubmit=%s",
                         finding.get("event_id"), finding.get("classification"),
                         finding.get("resubmit"))
            except asyncio.TimeoutError:
                log.error("RCA investigation timed out for %s after %d min",
                          event.get("id"), RCA_TIMEOUT_MIN)
                ipc.write_finding(self.session_dir, rca_bridge._fallback_finding(event))
            except Exception as exc:
                log.exception("Investigation failed for %s: %s", event.get("id"), exc)
                ipc.write_finding(self.session_dir, {
                    "event_id": event.get("id", ""),
                    "classification": "investigation-error",
                    "kernel_name": event.get("kernel_name", ""),
                    "target_type": "unknown",
                    "resubmit": False,
                    "error": str(exc),
                })
            finally:
                self.active_investigations -= 1

    async def _investigate(self, event: dict[str, Any]) -> dict[str, Any]:
        """Dispatch investigation to RCA agent via rca_bridge."""
        self.dashboard.update_wd(
            f"investigating: {event.get('type')} {event.get('kernel_name', '')}",
            self.active_investigations,
            dict(self.pattern_tracker),
        )

        # Read work_queue for context (read-only, per IR-1)
        wq_context = ipc.read_work_queue_entry(self.session_dir, event.get("task_id", ""))

        # Dispatch to RCA agent
        issue_dir = await rca_bridge.prepare_and_run(event, wq_context, self.state, self.env)

        # Parse output
        finding = rca_bridge.parse_rca_output(issue_dir, event)

        # Track hw-blocked kernels
        if finding.get("classification") == "hardware":
            kernel = finding.get("kernel_name")
            if kernel and kernel not in self.state.watchdog_hw_blocked_kernels:
                self.state.watchdog_hw_blocked_kernels.append(kernel)

        return finding

    @staticmethod
    def _priority_key(verdict: dict[str, Any]) -> tuple[int, int]:
        """Lower = higher priority: (priority_rank, systemic_rank)."""
        prio_map = {"high": 0, "medium": 1, "low": 2}
        prio = prio_map.get(verdict.get("priority", "low"), 2)
        systemic = 0 if verdict.get("systemic") else 1
        return (prio, systemic)
