"""Kernel Manager — 8-row classification, Deep OOB Loop (5 rounds, 4 backends),
local test pipeline (compile→correct→bench→verdict), patch-gen.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator


@asynccontextmanager
async def _noop_ctx() -> AsyncIterator[None]:
    """No-op async context manager when gpu_lock is not available."""
    yield

from . import ipc
from .oob_backends import OOBBackends, OOBResult, CLASSIFICATION

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (from kernel-manager/SKILL.md)
# ---------------------------------------------------------------------------

POLL_INTERVAL_S = 30
MAX_OOB_ROUNDS = 5
FINDINGS_POLL_INTERVAL_S = 30
MICRO_BENCHMARK_THRESHOLD = 1.05
CORRECTNESS_ATOL = 1e-2
CORRECTNESS_RTOL = 1e-2
MAX_CONCURRENT_BACKENDS = 4
EVENT_SNIPPET_CHARS = 2000
OOB_ROUND_TIMEOUT_S = 3600

# Risk defaults by patch_type
RISK_DEFAULTS: dict[str, tuple[float, float]] = {
    "python-dispatch": (0.02, 0.01),
    "triton-source":   (0.05, 0.05),
    "inductor-triton": (0.05, 0.05),
    "cpp-rebuild":     (0.15, 0.10),
    "config-only":     (0.01, 0.01),
    "jit-source":      (0.05, 0.05),
}

MERGE_READY_REQUIRED = {"task_id", "kernel_name", "patch_type", "target_file",
                        "rollback_command", "apply_instructions"}

# If all OOB rounds for a target complete faster than this, the dispatch is broken
ZOMBIE_ROUND_THRESHOLD_S = 5.0
COMPACT_EVERY_N_CYCLES = 20


class KernelManager:
    """Async kernel optimization manager — poll work_queue, run Deep OOB, write results."""

    def __init__(
        self,
        state: Any,
        llm: Any,
        session_dir: str,
        oob: OOBBackends,
        dashboard: Any,
        shutdown: asyncio.Event,
        gpu_lock: Any = None,
    ):
        self.state = state
        self.llm = llm
        self.session_dir = session_dir
        self.oob = oob
        self.dashboard = dashboard
        self.shutdown = shutdown
        self.gpu_lock = gpu_lock
        self.last_seen_id = getattr(state, "kernel_manager_last_seen_id", "") or ""
        self.last_finding_id = getattr(state, "watchdog_last_seen_finding_id", "") or ""
        self.failure_counts: dict[str, int] = {}
        self._cycle_count = 0
        self._consecutive_zombies = 0
        self._processed_set: set[str] = set(getattr(state, "kernel_manager_processed_ids", []) or [])

    async def run(self) -> None:
        log.info("Kernel Manager started — polling every %ds", POLL_INTERVAL_S)
        self._dedup_processed_ids()
        while not self.shutdown.is_set():
            try:
                await self._poll_cycle()
            except Exception as exc:
                log.exception("KM poll error: %s", exc)
            try:
                await asyncio.wait_for(self.shutdown.wait(), timeout=POLL_INTERVAL_S)
                break
            except asyncio.TimeoutError:
                pass

    def _dedup_processed_ids(self) -> None:
        """Deduplicate kernel_manager_processed_ids (can grow unbounded with repeats)."""
        ids = self.state.kernel_manager_processed_ids
        if not ids:
            return
        before = len(ids)
        self.state.kernel_manager_processed_ids = list(dict.fromkeys(ids))
        after = len(self.state.kernel_manager_processed_ids)
        self._processed_set = set(self.state.kernel_manager_processed_ids)
        if before != after:
            log.info("Deduped processed_ids: %d → %d", before, after)

    async def _poll_cycle(self) -> None:
        self._cycle_count += 1

        # Periodic compaction: dedup + prune terminal entries from the WQ file
        if self._cycle_count % COMPACT_EVERY_N_CYCLES == 0:
            processed = set(self.state.kernel_manager_processed_ids)
            removed = ipc.compact_work_queue(self.session_dir, processed)
            if removed:
                self._dedup_processed_ids()

        # Read ALL pending targets (ignore last_seen_id to avoid stale cursor skips)
        all_targets = ipc.read_work_queue(self.session_dir, after_id="")
        already_done = self._processed_set
        targets = [t for t in all_targets
                   if t.get("status") == "pending" and t.get("id") not in already_done]

        # Dedup by ID (keep first occurrence)
        seen_ids: set[str] = set()
        deduped: list[dict] = []
        for t in targets:
            tid = t.get("id", "")
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                deduped.append(t)
            elif not tid:
                deduped.append(t)
        targets = deduped
        if not targets:
            return

        def _pri(t: dict) -> float:
            gpu_pct = float(t.get("gpu_pct", 0) or 0)
            priority = float(t.get("priority", 0) or 0)
            return -(gpu_pct * 10 + priority)
        targets.sort(key=_pri)

        target = targets[0]
        log.info("KM queue: %d pending targets (top: %s, gpu=%.1f%%)",
                 len(targets), target.get("kernel_name", "?")[:50],
                 float(target.get("gpu_pct", 0) or 0))

        self.last_seen_id = target.get("id", self.last_seen_id)
        log.info("Processing target: %s (priority=%s, strategy=%s)",
                 target.get("kernel_name"), target.get("priority"), target.get("strategy"))
        try:
            strategy, backends = self._classify(target)

            if strategy in ("dispatch-fix", "config-only"):
                result = await self._self_fix(target)
                if (result.get("status") in ("no_change", "failed")
                        and (target.get("gpu_pct", 0) or 0) >= 5):
                    log.info("Self-fix found no change for high-value %s (%.1f%% GPU) — escalating to OOB",
                             target.get("kernel_name"), target.get("gpu_pct", 0))
                    findings = ipc.read_new_findings(self.session_dir, after_event_id=self.last_finding_id)
                    if findings:
                        self.last_finding_id = findings[-1].get("event_id", self.last_finding_id)
                    result = await self._deep_oob_loop(target, backends or ["claude"], findings)
            else:
                findings = ipc.read_new_findings(self.session_dir, after_event_id=self.last_finding_id)
                if findings:
                    self.last_finding_id = findings[-1].get("event_id", self.last_finding_id)
                result = await self._deep_oob_loop(target, backends, findings)

            ipc.write_result(self.session_dir, result)
            self._mark_processed(target)
            self.dashboard.update_km(target.get("kernel_name", "?"), 0, "", result.get("status", "?"))
        except Exception as exc:
            log.exception("KM target %s failed: %s", target.get("id"), exc)
            ipc.write_result(self.session_dir, self._build_result(
                "failed", target, error=str(exc),
            ))
            self._mark_processed(target)

    def _mark_processed(self, target: dict[str, Any]) -> None:
        tid = target.get("id", "")
        self.state.kernel_manager_merges_completed += 1
        if tid and tid not in self._processed_set:
            self._processed_set.add(tid)
            self.state.kernel_manager_processed_ids.append(tid)

    # ------------------------------------------------------------------
    # Classification (8-row table)
    # ------------------------------------------------------------------

    def _classify(self, target: dict[str, Any]) -> tuple[str, list[str]]:
        strategy = target.get("strategy", "oob-rewrite")
        dispatch = target.get("dispatch_analysis", {})

        if dispatch.get("dispatch_bug"):
            return "dispatch-fix", ["local"]
        if strategy in CLASSIFICATION:
            return strategy, CLASSIFICATION[strategy]

        # Fallback classification from source_type
        source_type = target.get("source_type", "")
        if source_type in ("cpp_cuda", "cpp_hip"):
            return "hip-kernel", ["geak", "claude"]
        if "triton" in source_type:
            return "triton-rewrite", ["geak", "codex", "claude"]

        return strategy, CLASSIFICATION.get(strategy, ["claude"])

    # ------------------------------------------------------------------
    # Self-fix (7-step protocol)
    # ------------------------------------------------------------------

    async def _self_fix(self, target: dict[str, Any]) -> dict[str, Any]:
        from . import prompts
        log.info("Self-fix for %s", target.get("kernel_name"))
        self.dashboard.update_km(target.get("kernel_name", "?"), 0, "local", "self-fix")

        tid = target.get("id") or target.get("task_id") or "unknown"
        output_file = Path(self.session_dir) / "kernel_manager" / f"selffix_{tid}.json"
        prompt = prompts.prompt_execute_dispatch_fix(self.state.state_summary(), target)
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=15)

        if result.is_error:
            return self._build_result("failed", target, error=result.error_message)

        output = result.output
        if output.get("status") == "success" or output.get("patch_applied"):
            patch_dir, meta = await self._generate_patch(target, "", output)
            return self._build_result("merge-ready", target, patch_dir=str(patch_dir), metadata=meta)
        if output.get("status") in ("no_change_needed", "already_applied", "not_applicable"):
            log.info("Self-fix for %s: %s — %s",
                     target.get("kernel_name"), output.get("status"), output.get("analysis", {}).get("dispatch_logic", ""))
            return self._build_result("no_change", target, error=None)
        return self._build_result("failed", target, error=output.get("error", "self-fix failed"))

    # ------------------------------------------------------------------
    # Deep OOB Loop (5 rounds)
    # ------------------------------------------------------------------

    async def _deep_oob_loop(
        self,
        target: dict[str, Any],
        backends: list[str],
        findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        session_history: list[dict[str, Any]] = []
        kernel_findings = [f for f in findings if f.get("kernel_name") == target.get("kernel_name")]
        loop_t0 = time.monotonic()

        for round_num in range(1, MAX_OOB_ROUNDS + 1):
            log.info("OOB round %d/%d for %s", round_num, MAX_OOB_ROUNDS, target.get("kernel_name"))

            # Snapshot source file before this round so we can detect changes
            # even when the LLM edits files via tool calls instead of returning code
            pre_round_snapshot = self._capture_source_snapshot(target.get("source_file", ""))

            # Check findings before each round (IR-10)
            new_findings = ipc.get_findings_for_kernel(self.session_dir, target.get("kernel_name", ""))
            kernel_findings = new_findings if new_findings else kernel_findings

            # Select backends
            round_backends = self.oob.select_backends(
                target.get("strategy", "oob-rewrite"), round_num, session_history,
            )
            round_backends = [b for b in round_backends if b in backends]
            if not round_backends:
                round_backends = ["claude"]

            self.dashboard.update_km(
                target.get("kernel_name", "?"), round_num,
                ",".join(round_backends), "dispatching",
            )

            # Build prompt with accumulated context (IR-11)
            from . import prompts
            prompt_texts: list[tuple[str, str]] = []
            for backend in round_backends:
                p = prompts.prompt_oob_round(target, session_history, kernel_findings, round_num, backend)
                prompt_texts.append((backend, p))

            # Dispatch (parallel, with timeout)
            tasks = [asyncio.ensure_future(self.oob.dispatch(be, p, target))
                     for be, p in prompt_texts]
            try:
                raw_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=OOB_ROUND_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                log.warning("OOB round %d timed out after %ds", round_num, OOB_ROUND_TIMEOUT_S)
                raw_results = []
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            results: list[OOBResult] = [r for r in raw_results if isinstance(r, OOBResult)]

            if not results:
                session_history.append({
                    "round": round_num, "backend": "none",
                    "outcome": "COMPILE_FAIL", "error_analysis": "All backends failed to respond",
                })
                continue

            best = self._pick_best(results)
            if not best.code:
                # Even without explicit code block, Claude may have edited files
                # via tool calls. Check for changes on disk.
                source_file = target.get("source_file", "")
                after_content = self._capture_source_snapshot(source_file)
                disk_diff = self._generate_diff(pre_round_snapshot, after_content, source_file)
                if disk_diff:
                    log.info("No code block returned but detected file changes on disk for %s",
                             target.get("kernel_name"))
                    best = OOBResult(
                        backend=best.backend, status="success",
                        code=after_content or "", duration_s=best.duration_s,
                    )
                else:
                    session_history.append({
                        "round": round_num, "backend": best.backend,
                        "outcome": "COMPILE_FAIL",
                        "error_analysis": best.error or "No code returned and no file changes detected",
                    })
                    continue

            # Local test (4-step pipeline)
            test_result = await self._local_test(target, best.code)
            outcome = test_result.get("verdict", "FAIL")

            self.dashboard.update_km(
                target.get("kernel_name", "?"), round_num, best.backend, outcome,
            )

            if outcome == "PASS-DEFERRED":
                log.info("Local test deferred (GPU busy) for %s round %d — will retry next poll",
                         target.get("kernel_name", "?"), round_num)
                session_history.append({
                    "round": round_num, "backend": best.backend,
                    "outcome": "DEFERRED",
                    "reason": test_result.get("reason", "GPU_BUSY"),
                })
                continue

            if outcome == "PASS":
                patch_dir, meta = await self._generate_patch(target, best.code, test_result)
                self.oob.record_result(best.backend, True)
                ipc.write_insight(self.session_dir, {
                    "source": "kernel-manager",
                    "type": "pattern-discovery",
                    "pattern": target.get("strategy", "oob-rewrite"),
                    "confidence": "high" if test_result.get("speedup", 0) > 1.3 else "medium",
                    "details": {
                        "kernel_name": target.get("kernel_name"),
                        "speedup": test_result.get("speedup"),
                        "backend_used": best.backend,
                        "strategy": target.get("strategy"),
                        "source_file": target.get("source_file"),
                        "applicable_to": [],
                    },
                })
                self._snapshot_repo_diffs(target, test_result)
                return self._build_result(
                    "merge-ready", target,
                    micro_speedup=test_result.get("speedup"),
                    patch_dir=str(patch_dir),
                    strategy_used=target.get("strategy"),
                    backend_used=best.backend,
                    metadata=meta,
                )

            elif outcome.startswith("FAIL-COMPILE"):
                constraints = self._extract_compile_constraints(test_result)
                session_history.append({
                    "round": round_num, "backend": best.backend,
                    "outcome": "COMPILE_FAIL",
                    "error_analysis": test_result.get("error", ""),
                    "constraints_used": constraints,
                    "attempt_summary": (best.code or "")[:200],
                })

            elif outcome.startswith("FAIL-CORRECT"):
                session_history.append({
                    "round": round_num, "backend": best.backend,
                    "outcome": "CORRECTNESS_FAIL",
                    "error_analysis": test_result.get("error", ""),
                    "attempt_summary": (best.code or "")[:200],
                })

            elif outcome.startswith("FAIL-REGRESS"):
                session_history.append({
                    "round": round_num, "backend": best.backend,
                    "outcome": "REGRESSION",
                    "micro_speedup": test_result.get("speedup"),
                    "error_analysis": str(test_result.get("per_shape", [])),
                    "attempt_summary": (best.code or "")[:200],
                })

            elif outcome == "FAIL-ADVERSARIAL":
                session_history.append({
                    "round": round_num, "backend": best.backend,
                    "outcome": "ADVERSARIAL_FAIL",
                    "error_analysis": str(test_result.get("adversarial_failures", [])),
                    "attempt_summary": (best.code or "")[:200],
                })

            elif outcome == "SEGFAULT":
                ipc.write_event(self.session_dir, {
                    "source": "kernel-manager",
                    "type": "segfault",
                    "kernel_name": target.get("kernel_name"),
                    "task_id": target.get("id"),
                    "severity": "error",
                    "promising": (test_result.get("micro_speedup_before_crash", 0) or 0) > 1,
                    "details": {
                        "crash_log_snippet": (test_result.get("crash_log", "") or "")[:EVENT_SNIPPET_CHARS],
                        "session_history": session_history[-3:],
                        "round_number": round_num,
                        "backend_used": best.backend,
                        "exit_code": test_result.get("exit_code", 139),
                        "source_file": target.get("source_file"),
                        "gpu_pct": target.get("gpu_pct"),
                        "strategy_used": target.get("strategy"),
                    },
                })
                self.state.events_written += 1
                await asyncio.sleep(FINDINGS_POLL_INTERVAL_S)
                session_history.append({
                    "round": round_num, "backend": best.backend,
                    "outcome": "SEGFAULT",
                    "crash_log_snippet": (test_result.get("crash_log", "") or "")[:200],
                })
            elif outcome.startswith("FAIL-BENCH"):
                session_history.append({
                    "round": round_num, "backend": best.backend,
                    "outcome": "BENCH_FAIL",
                    "error_analysis": test_result.get("error", outcome),
                    "attempt_summary": (best.code or "")[:200],
                })

            elif outcome.startswith("FAIL-NOIMPROV") or outcome == "FAIL-NOIMPROV":
                session_history.append({
                    "round": round_num, "backend": best.backend,
                    "outcome": "NO_IMPROVEMENT",
                    "micro_speedup": test_result.get("speedup"),
                    "error_analysis": test_result.get("error", outcome),
                })

            else:
                session_history.append({
                    "round": round_num, "backend": best.backend,
                    "outcome": outcome,
                    "error_analysis": test_result.get("error", outcome),
                    "attempt_summary": (best.code or "")[:200],
                })

            self.oob.record_result(best.backend, False)

        # All rounds exhausted — check if any round was close
        best_outcome = "COMPILE_FAIL"
        best_speedup = 0.0
        for h in session_history:
            if h.get("outcome") in ("REGRESSION", "ADVERSARIAL_FAIL"):
                best_outcome = h["outcome"]
                best_speedup = max(best_speedup, h.get("micro_speedup", 0) or 0)
            elif h.get("outcome") == "CORRECTNESS_FAIL" and best_outcome == "COMPILE_FAIL":
                best_outcome = "CORRECTNESS_FAIL"

        is_promising = best_outcome in ("REGRESSION", "ADVERSARIAL_FAIL") or best_speedup > 0.9
        ipc.write_event(self.session_dir, {
            "source": "kernel-manager",
            "type": "exhausted",
            "kernel_name": target.get("kernel_name"),
            "task_id": target.get("id"),
            "severity": "warning",
            "promising": is_promising,
            "details": {
                "session_history": session_history,
                "gpu_pct": target.get("gpu_pct"),
                "source_file": target.get("source_file"),
                "best_outcome": best_outcome,
                "best_speedup": best_speedup,
            },
        })
        self.state.events_written += 1
        loop_elapsed = time.monotonic() - loop_t0
        if loop_elapsed < ZOMBIE_ROUND_THRESHOLD_S:
            self._consecutive_zombies += 1
            log.error(
                "ZOMBIE DISPATCH: all %d rounds for %s completed in %.1fs "
                "(threshold %.1fs). Dispatch is broken! (consecutive: %d)",
                MAX_OOB_ROUNDS, target.get("kernel_name", "?")[:50],
                loop_elapsed, ZOMBIE_ROUND_THRESHOLD_S,
                self._consecutive_zombies,
            )
            if self._consecutive_zombies >= 3:
                log.error(
                    "ZOMBIE HALT: %d consecutive zombie targets — halting KM "
                    "for 5 minutes to avoid burning through the entire queue",
                    self._consecutive_zombies,
                )
                await asyncio.sleep(300)
        else:
            self._consecutive_zombies = 0

        return self._build_result("failed", target, error=f"exhausted after 5 rounds (best: {best_outcome}, speedup: {best_speedup:.2f})")

    # ------------------------------------------------------------------
    # Local test pipeline (4 steps)
    # ------------------------------------------------------------------

    async def _local_test(self, target: dict[str, Any], optimized_code: str) -> dict[str, Any]:
        from . import prompts, server

        # Step 1: Compile check
        tid = target.get("id") or target.get("task_id") or "unknown"
        output_file = Path(self.session_dir) / "kernel_manager" / f"compile_{tid}.json"
        compile_prompt = prompts.prompt_local_test_compile(target, optimized_code)
        compile_result = await self.llm.call(compile_prompt, output_file=str(output_file), max_turns=10)

        if compile_result.is_error:
            return {"verdict": "FAIL-COMPILE-LLM", "error": compile_result.error_message}
        if not compile_result.output.get("compiled", False):
            return {
                "verdict": f"FAIL-COMPILE-{compile_result.output.get('error_type', 'UNKNOWN')}",
                "error": compile_result.output.get("error_message", ""),
            }

        # Step 2: Correctness + Micro-benchmark (GPU required)
        # With GPU lock: wait briefly, then defer. After repeated deferrals,
        # switch to a real blocking wait so KM doesn't starve.
        if self.gpu_lock:
            can_proceed = await self.gpu_lock.wait_or_defer(
                "local-test", "kernel-manager",
                quick_timeout_s=30, full_timeout_s=300,
            )
            if not can_proceed:
                log.info("GPU busy (%s/%s held %.0fs), deferring %s (defer #%d)",
                         self.gpu_lock.state.holder, self.gpu_lock.state.phase,
                         self.gpu_lock.state.held_seconds,
                         target.get("kernel_name"),
                         self.gpu_lock._defer_counts.get("kernel-manager", 0))
                return {"verdict": "PASS-DEFERRED", "reason": "GPU_BUSY",
                        "micro_benchmark": "deferred"}
            return await self._local_test_gpu_phase(target, optimized_code, prompts)

        if not await server.gpu_available():
            return {"verdict": "PASS-DEFERRED", "reason": "GPU_BUSY", "micro_benchmark": "deferred"}
        return await self._local_test_gpu_phase(target, optimized_code, prompts)

    async def _local_test_gpu_phase(self, target: dict[str, Any],
                                     optimized_code: str, prompts: Any) -> dict[str, Any]:
        """GPU-dependent phase of local test: correctness + bench + adversarial.

        Runs under GPU lock when available to prevent contention with
        orchestrator merge-ops (which kill the server to rebuild).
        """
        tid = target.get("id") or target.get("task_id") or "unknown"
        lock_ctx = (self.gpu_lock.acquire("local-test", "kernel-manager", timeout_s=1800)
                     if self.gpu_lock else _noop_ctx())

        async with lock_ctx:
            output_file = Path(self.session_dir) / "kernel_manager" / f"correct_{tid}.json"
            correct_prompt = prompts.prompt_local_test_correctness(target, optimized_code)
            correct_result = await self.llm.call(correct_prompt, output_file=str(output_file), max_turns=40)

            if correct_result.is_error:
                return {"verdict": "FAIL-CORRECT-LLM", "error": correct_result.error_message}
            if not correct_result.output.get("correct", False):
                return {
                    "verdict": "FAIL-CORRECT",
                    "error": correct_result.output.get("error_message", ""),
                }

            # Step 3: Micro-benchmark
            output_file = Path(self.session_dir) / "kernel_manager" / f"bench_{tid}.json"
            bench_prompt = prompts.prompt_local_test_benchmark(target, optimized_code)
            bench_result = await self.llm.call(bench_prompt, output_file=str(output_file), max_turns=30)

            if bench_result.is_error:
                return {"verdict": "FAIL-BENCH-LLM", "error": bench_result.error_message}

            speedup = bench_result.output.get("avg_speedup", 0)
            per_shape = bench_result.output.get("per_shape", [])
            any_regression = any(s.get("speedup", 1) < 0.95 for s in per_shape)

            # Step 4: Verdict
            if speedup > MICRO_BENCHMARK_THRESHOLD and not any_regression:
                # Step 5: Adversarial stress test (edge cases)
                adversarial_result = await self._adversarial_test(target, optimized_code)
                if adversarial_result.get("failed"):
                    failures = adversarial_result.get("failures", [])
                    reachable = [f for f in failures
                                 if not self._is_unreachable_edge_case(f, target)]
                    if reachable:
                        return {
                            "verdict": "FAIL-ADVERSARIAL",
                            "speedup": speedup,
                            "per_shape": per_shape,
                            "adversarial_failures": failures,
                        }
                    log.info(
                        "Adversarial failures for %s are all on unreachable edge cases — accepting as PASS-DEFERRED",
                        target.get("kernel_name"),
                    )
                    return {
                        "verdict": "PASS-DEFERRED",
                        "speedup": speedup,
                        "per_shape": per_shape,
                        "deferred_adversarial": failures,
                        "note": "Adversarial failures on unreachable shapes accepted",
                    }
                return {"verdict": "PASS", "speedup": speedup, "per_shape": per_shape}
            elif any_regression:
                return {"verdict": "FAIL-REGRESS", "speedup": speedup, "per_shape": per_shape}
            else:
                return {"verdict": "FAIL-NOIMPROV", "speedup": speedup}

    @staticmethod
    def _is_unreachable_edge_case(failure: dict[str, Any], target: dict[str, Any]) -> bool:
        """Determine if an adversarial failure is on a shape unreachable in production.

        Returns True for cases that should NOT block acceptance:
        - Speculative decode kernels (qseqlen>1) with ctx < qseqlen
          (impossible: KV cache always has prompt tokens)
        - Batch size 0 or context length 0 (no real workload)
        - Shapes explicitly smaller than the kernel's minimum tile size
        """
        case_name = str(failure.get("case", "")).lower()
        error_msg = str(failure.get("error", "")).lower()

        # Speculative decode with degenerate context
        qseqlen = 4
        constraints = target.get("constraints", {})
        spec_tokens = constraints.get("speculative_tokens", 0)
        if spec_tokens > 1 or "qseqlen" in target.get("kernel_name", ""):
            qseqlen = spec_tokens or 4

        if qseqlen > 1 and ("ctx1" in case_name or "ctx_1" in case_name
                            or "bs1_ctx1" in case_name or "small_ctx" in case_name):
            return True

        # Zero-dimension inputs
        if "bs0" in case_name or "ctx0" in case_name or "empty" in case_name:
            return True

        # NaN/Inf injected into inputs (testing robustness, not correctness)
        if "nan_input" in case_name or "inf_input" in case_name:
            return True

        return False

    async def _adversarial_test(
        self, target: dict[str, Any], optimized_code: str
    ) -> dict[str, Any]:
        """Quick edge-case stress test: zero-dim, max-shape, NaN inputs."""
        from . import prompts
        tid = target.get("id") or target.get("task_id") or "unknown"
        output_file = (
            Path(self.session_dir) / "kernel_manager" / f"adversarial_{tid}.json"
        )
        prompt = prompts.prompt_adversarial_test(target, optimized_code)
        result = await self.llm.call(prompt, output_file=str(output_file), max_turns=10)
        if result.is_error:
            log.warning("Adversarial test LLM call failed: %s", result.error_message)
            return {"failed": True, "failures": [{"reason": "LLM call failed — treating as unsafe", "error": result.error_message}]}
        output = result.output
        failures = output.get("failures", [])
        if failures:
            log.warning("Adversarial test found %d failures for %s",
                        len(failures), target.get("kernel_name"))
            return {"failed": True, "failures": failures}
        return {"failed": False}

    # ------------------------------------------------------------------
    # Patch generation
    # ------------------------------------------------------------------

    async def _generate_patch(
        self,
        target: dict[str, Any],
        optimized_code: str,
        test_result: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        task_id = target.get("id", "unknown")
        kernel_name = target.get("kernel_name", "unknown")
        source_file = target.get("source_file", "")
        source_type = target.get("source_type", "python")
        patch_type = self._infer_patch_type(source_file, source_type)

        a_risk, c_risk = RISK_DEFAULTS.get(patch_type, (0.05, 0.05))
        if kernel_name and ("reduction" in kernel_name or "norm" in kernel_name):
            a_risk = 0.15

        filename = Path(source_file).name if source_file else "kernel.py"
        rebuild_required = patch_type in ("cpp-rebuild",)
        rebuild_cmd = None
        if rebuild_required:
            if "sgl-kernel" in source_file or "sgl_kernel" in source_file:
                rebuild_cmd = "cd /sgl-workspace/sglang/sgl-kernel && python setup_rocm.py install"
            elif "aiter" in source_file:
                rebuild_cmd = "cd /sgl-workspace/aiter && pip install -e . --no-deps"

        metadata = {
            "task_id": task_id,
            "kernel_name": kernel_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy_used": target.get("strategy", "oob-rewrite"),
            "backend_used": test_result.get("backend") or target.get("backend_used"),
            "backend_model": test_result.get("model"),
            "patch_type": patch_type,
            "target_file": source_file,
            "backup_file": f"original_{filename}.bak",
            "apply_method": "file-replace",
            "apply_instructions": [
                f"cp {source_file} $PATCH_DIR/original_{filename}.bak",
                f"cp $PATCH_DIR/optimized_{filename} {source_file}",
            ],
            "rebuild_required": rebuild_required,
            "rebuild_command": rebuild_cmd,
            "cache_clear_commands": self._cache_clear_commands(patch_type),
            "rollback_command": f"cp $PATCH_DIR/original_{filename}.bak {source_file}",
            "rollback_rebuild_command": rebuild_cmd,
            "verification_command": f"python -c 'import importlib; exec(open(\"{source_file}\").read())'",
            "micro_speedup": test_result.get("speedup"),
            "micro_benchmark_status": "passed" if test_result.get("verdict") == "PASS" else "deferred",
            "correctness_status": (
                "passed" if test_result.get("verdict") == "PASS"
                else "deferred" if test_result.get("verdict") == "PASS-DEFERRED"
                else "failed"
            ),
            "git_archaeology": {
                "commits_checked": [],
                "finding": "",
                "fix_approach": target.get("strategy", ""),
            },
            "risk_assessment": {
                "accuracy_risk": a_risk,
                "crash_risk": c_risk,
                "notes": f"patch_type={patch_type}",
            },
        }

        artifacts = {}

        # Always capture the optimized file content — either from the code
        # block the LLM returned, or by reading the file from disk (the LLM
        # may have edited it directly via tool calls).
        effective_code = optimized_code
        if not effective_code and source_file:
            try:
                effective_code = Path(source_file).read_text()
                log.info("Code block was empty — captured %d bytes from %s on disk",
                         len(effective_code), source_file)
            except Exception as exc:
                log.warning("Failed to read source file %s for patch backup: %s", source_file, exc)

        if effective_code:
            artifacts[f"optimized_{filename}"] = effective_code

        # Always capture a git diff so the change is recoverable even if
        # the merge_ready directory is cleaned up or lost.
        git_diff = self._capture_git_diff(source_file) if source_file else None
        if git_diff:
            artifacts["change.diff"] = git_diff
            metadata["git_diff"] = git_diff[:5000]

        bench_data = test_result.get("per_shape", [])
        artifacts["micro_benchmark.json"] = json.dumps({
            "avg_speedup": test_result.get("speedup"),
            "per_shape": bench_data,
            "verdict": test_result.get("verdict"),
        }, indent=2)

        patch_dir = ipc.write_merge_ready(self.session_dir, task_id, metadata, artifacts)
        return patch_dir, metadata

    @staticmethod
    def _infer_patch_type(source_file: str, source_type: str) -> str:
        if source_type in ("cpp_cuda", "cpp_hip"):
            return "cpp-rebuild"
        if source_file.startswith("/tmp/torchinductor"):
            return "inductor-triton"
        if "jit" in source_file.lower():
            return "jit-source"
        if "config" in source_file.lower():
            return "config-only"
        return "triton-source"

    @staticmethod
    def _cache_clear_commands(patch_type: str) -> list[str]:
        cmds: list[str] = []
        if patch_type in ("triton-source", "inductor-triton"):
            cmds.append("rm -rf ~/.triton/cache")
        if patch_type == "inductor-triton":
            cmds.append("rm -rf /tmp/torchinductor_root")
        if patch_type == "jit-source":
            cmds.append("find /sgl-workspace/aiter/aiter/jit/build -name '*.so' -delete 2>/dev/null || true")
        cmds.append("find /sgl-workspace -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true")
        return cmds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot_repo_diffs(self, target: dict, test_result: dict) -> None:
        """Save git diffs of affected repos when a merge-ready result is produced."""
        try:
            import time as _time
            ts = _time.strftime("%Y%m%d_%H%M%S")
            task_id = target.get("id", "unknown")
            diff_dir = Path(self.session_dir) / "diffs" / f"km_{ts}_{task_id}"
            diff_dir.mkdir(parents=True, exist_ok=True)

            source_file = target.get("source_file", "")
            repos = [("aiter", "/sgl-workspace/aiter"), ("sglang", "/sgl-workspace/sglang")]
            for name, repo_path in repos:
                if not Path(repo_path).joinpath(".git").exists():
                    continue
                try:
                    diff = subprocess.run(
                        ["git", "diff"], capture_output=True, text=True,
                        cwd=repo_path, timeout=30,
                    )
                    if diff.stdout.strip():
                        (diff_dir / f"{name}.patch").write_text(diff.stdout)
                except Exception:
                    pass

            import json as _json
            manifest = {
                "task_id": task_id,
                "kernel": target.get("kernel_name"),
                "source_file": source_file,
                "speedup": test_result.get("speedup"),
                "timestamp": ts,
            }
            (diff_dir / "manifest.json").write_text(_json.dumps(manifest, indent=2))
            log.info("KM diff snapshot saved to %s", diff_dir)
        except Exception as exc:
            log.warning("Failed to save KM diff snapshot: %s", exc)

    @staticmethod
    def _capture_source_snapshot(source_file: str) -> str | None:
        """Read the current content of a source file for before/after diffing."""
        if not source_file:
            return None
        try:
            return Path(source_file).read_text()
        except Exception:
            return None

    @staticmethod
    def _generate_diff(before: str | None, after: str | None, filepath: str) -> str | None:
        """Generate a unified diff between before and after content."""
        if before is None or after is None or before == after:
            return None
        import difflib
        diff_lines = difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{Path(filepath).name}",
            tofile=f"b/{Path(filepath).name}",
        )
        return "".join(diff_lines) or None

    @staticmethod
    def _capture_git_diff(source_file: str) -> str | None:
        """Capture git diff for a file against HEAD."""
        if not source_file:
            return None
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD", "--", source_file],
                capture_output=True, text=True, timeout=10,
                cwd=str(Path(source_file).parent),
            )
            return result.stdout.strip() if result.stdout.strip() else None
        except Exception:
            return None

    @staticmethod
    def _pick_best(results: list[OOBResult]) -> OOBResult:
        successes = [r for r in results if r.status == "success" and r.code]
        if successes:
            return max(successes, key=lambda r: len(r.code or ""))
        return results[0] if results else OOBResult(status="error", error="No results")

    @staticmethod
    def _extract_compile_constraints(test_result: dict[str, Any]) -> dict[str, Any]:
        error = test_result.get("error", "")
        constraints: dict[str, Any] = {}
        if "register" in error.lower() or "vgpr" in error.lower():
            constraints["max_vgprs"] = 64
            constraints["target_waves"] = 4
        if "block" in error.lower():
            constraints["reduce_block_dims"] = True
        return constraints

    @staticmethod
    def _build_result(
        status: str,
        target: dict[str, Any],
        *,
        error: str = "",
        micro_speedup: float | None = None,
        patch_dir: str | None = None,
        strategy_used: str | None = None,
        backend_used: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        meta = metadata or {}
        result = {
            "id": target.get("id", ""),
            "status": status,
            "kernel_name": target.get("kernel_name"),
            "source_file": target.get("source_file"),
            "strategy_used": strategy_used or target.get("strategy"),
            "backend_used": backend_used,
            "micro_speedup": micro_speedup,
            "patch_dir": patch_dir,
            "patch_type": meta.get("patch_type") or target.get("source_type"),
            "rebuild_required": meta.get("rebuild_required", target.get("rebuild_required", False)),
            "rollback_command": meta.get("rollback_command"),
            "verification_command": meta.get("verification_command"),
            "error_message": error or None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Persist the git diff in the result so the change is always
        # recoverable from results.jsonl even if merge_ready/ is lost.
        if meta.get("git_diff"):
            result["git_diff"] = meta["git_diff"]
        return result
