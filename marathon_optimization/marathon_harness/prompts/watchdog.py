"""Watchdog prompt templates — triage (full Level 2A-2L), investigate (12 templates A-L)."""

from __future__ import annotations

import json
from typing import Any


# -----------------------------------------------------------------------
# Triage — used when hardcoded rules need LLM disambiguation
# -----------------------------------------------------------------------

def prompt_triage(event: dict[str, Any], pattern_counts: dict[str, int]) -> str:
    return (
        f"TRIAGE this event from the Marathon event log.\n\n"
        f"Event:\n```json\n{json.dumps(event, indent=2, default=str)[:3000]}\n```\n\n"
        f"Pattern tracker counts: {json.dumps(pattern_counts)}\n\n"
        f"Decision tree (Level 2):\n"
        f"2A SEGFAULT: exit 139 or type=segfault\n"
        f"  - micro>1 → INV HIGH (improvement before crash)\n"
        f"  - session_history has PASS → INV HIGH\n"
        f"  - else → INV MEDIUM\n"
        f"2B CRASH:\n"
        f"  - micro>1 → INV HIGH\n"
        f"  - exit 134 → INV MEDIUM (abort)\n"
        f"  - micro<=1 and round<=1 → SKIP\n"
        f"  - round>=3 → WATCH\n"
        f"2C REGRESSION:\n"
        f"  - micro>1.5 → INV HIGH\n"
        f"  - micro>1.05 → INV MEDIUM\n"
        f"  - else → SKIP\n"
        f"2D COMPILATION-FAIL:\n"
        f"  - pattern>=3 → INV HIGH systemic\n"
        f"  - 'register allocation' → INV MEDIUM\n"
        f"  - 'hipcc' → WATCH\n"
        f"  - else → WATCH\n"
        f"2E MERGE-REVERT/MERGE-FAIL:\n"
        f"  - rebuild_required → INV HIGH\n"
        f"  - exit 139/134 → INV HIGH\n"
        f"  - else → WATCH\n"
        f"2F EXHAUSTED:\n"
        f"  - gpu_pct>5 → INV MEDIUM\n"
        f"  - else → SKIP\n"
        f"2G REBUILD:\n"
        f"  - rebuild-crash → INV HIGH\n"
        f"  - pattern>=2 → INV HIGH systemic\n"
        f"  - hipcc → INV MEDIUM\n"
        f"  - setup_rocm → INV MEDIUM\n"
        f"  - else → WATCH\n"
        f"2H TUNING:\n"
        f"  - crash+exit139 → INV HIGH\n"
        f"  - OOM → INV MEDIUM\n"
        f"  - server crash on load → INV HIGH\n"
        f"  - no improvement → SKIP\n"
        f"2I COMM:\n"
        f"  - hang → INV HIGH\n"
        f"  - timeout → INV HIGH\n"
        f"  - pattern>=2 → INV HIGH systemic\n"
        f"  - else → WATCH\n"
        f"2J CODEGEN:\n"
        f"  - cache-corrupt → INV MEDIUM\n"
        f"  - pattern>=3 → INV HIGH systemic\n"
        f"  - else → WATCH\n"
        f"2K SERVER:\n"
        f"  - exit 139 → INV HIGH\n"
        f"  - OOM → INV MEDIUM\n"
        f"  - last_config_change → INV HIGH\n"
        f"  - hang → INV MEDIUM\n"
        f"  - else → WATCH\n"
        f"2L DISPATCH-FIX-FAIL:\n"
        f"  - fix_type=git-revert → INV MEDIUM\n"
        f"  - else → WATCH\n"
        f"merge-keep → SKIP\n"
        f"accuracy-fail → SKIP\n"
        f"unknown → WATCH\n\n"
        f"Write to $OUTPUT_FILE: action (investigate|skip|pattern-watch), "
        f"priority (high|medium|low), systemic (bool), reason\n"
    )


# -----------------------------------------------------------------------
# Investigate — dispatched to RCA agent, but this prompt provides
# context for cases where LLM-based investigation is used as fallback
# -----------------------------------------------------------------------

INVESTIGATION_TEMPLATES = {
    "segfault": (
        "Template A — Segfault Investigation:\n"
        "1. Read crash_log_snippet, identify faulting address\n"
        "2. Check session_history for pattern (which rounds crash)\n"
        "3. grep work_queue for kernel context\n"
        "4. Check dmesg for amdgpu faults\n"
        "5. Classify: register pressure, memory conflict, or hardware\n"
        "6. Constraint: max_vgprs = floor(256 / target_waves)\n"
    ),
    "compilation-fail": (
        "Template B — Compilation Failure Investigation:\n"
        "1. Parse error message for specific failure type\n"
        "2. Check if register allocation exceeded VGPR budget\n"
        "3. Check Triton/hipcc version compatibility\n"
        "4. Look for systemic pattern across kernels\n"
        "5. Constraint: VGPR <= 64 for >=4 waves occupancy\n"
    ),
    "correctness": (
        "Template C — Correctness Failure Investigation:\n"
        "1. Identify which shapes fail\n"
        "2. Check precision (bf16/fp16/fp8 accumulation)\n"
        "3. Check reduction order sensitivity\n"
        "4. Compare against torch reference\n"
    ),
    "regression": (
        "Template D — E2E Regression Investigation:\n"
        "1. Compare micro-benchmark vs E2E results\n"
        "2. Check if kernel interferes with scheduling\n"
        "3. Profile cache behavior\n"
        "4. Check for increased memory pressure\n"
    ),
    "systemic-build": (
        "Template E — Systemic Build Failure:\n"
        "1. Check hipcc version and ROCm installation\n"
        "2. Verify setup_rocm.py configuration\n"
        "3. Check for corrupted build artifacts\n"
        "4. Test minimal rebuild\n"
    ),
    "exhausted": (
        "Template F — Exhausted Investigation:\n"
        "1. Review all 5 round session_history\n"
        "2. Identify common failure patterns\n"
        "3. Check if kernel is fundamentally hardware-limited\n"
        "4. Consider alternative optimization strategies\n"
    ),
    "rebuild": (
        "Template G — Rebuild Failure:\n"
        "1. Check build logs for specific error\n"
        "2. Verify dependencies (cmake, hipcc, pip)\n"
        "3. Check for file permission issues\n"
        "4. Test incremental vs clean rebuild\n"
    ),
    "server-config": (
        "Template H — Server Crash After Config Change:\n"
        "1. Identify last config change\n"
        "2. Check if change caused OOM or incompatibility\n"
        "3. Revert config and test\n"
        "4. Constraint: revert-last-change\n"
    ),
    "comm-hang": (
        "Template I — Communication Hang:\n"
        "1. Check RCCL/NCCL logs\n"
        "2. Verify topology (rocm-smi --showtopoweight)\n"
        "3. Check for RDMA issues (ibstat)\n"
        "4. Test with different algorithms\n"
    ),
    "codegen": (
        "Template J — Codegen Failure:\n"
        "1. Check Triton/Inductor cache corruption\n"
        "2. Verify codegen flags compatibility\n"
        "3. Clear caches and retry\n"
        "4. Constraint: clear-cache-retry\n"
    ),
    "server-hang": (
        "Template K — Server Hang:\n"
        "1. Check for deadlock in scheduling\n"
        "2. Verify GPU utilization (rocm-smi)\n"
        "3. Check memory pressure\n"
        "4. Constraint: restart-and-retry\n"
    ),
    "dispatch-fix-fail": (
        "Template L — Dispatch Fix Failure:\n"
        "1. Check git history for the revert target\n"
        "2. Verify file hasn't diverged from expected state\n"
        "3. Try alternative fix approach\n"
    ),
}


def prompt_investigate(
    event: dict[str, Any],
    work_queue_context: dict[str, Any] | None = None,
    session_history: list[dict[str, Any]] | None = None,
) -> str:
    event_type = event.get("type", "unknown")
    template_key = _map_event_to_template(event_type)
    template = INVESTIGATION_TEMPLATES.get(template_key, "Generic investigation.")

    parts: list[str] = [
        f"INVESTIGATE event: {event.get('id', '?')}",
        f"Type: {event_type}",
        f"Kernel: {event.get('kernel_name', 'N/A')}",
        "",
        f"Event details:\n```json\n{json.dumps(event, indent=2, default=str)[:3000]}\n```",
    ]

    if work_queue_context:
        parts.append(f"\nWork queue context:\n```json\n{json.dumps(work_queue_context, indent=2, default=str)[:1500]}\n```")

    if session_history:
        parts.append("\nSession history:")
        for entry in session_history[-5:]:
            parts.append(f"  Round {entry.get('round')}: {entry.get('outcome')} via {entry.get('backend')}")

    parts.append(f"\n{template}")

    parts.append(
        "\n6-Phase RCA methodology:\n"
        "1. Discovery: gather logs, context, kernel source\n"
        "2. Log Analysis: parse error patterns\n"
        "3. Metrics: GPU%, shapes, performance data\n"
        "4. HW/SW Classification: ECC→HW, dmesg fault→HW, Python→SW\n"
        "5. Infra Deep Dive: rocm-smi, dmesg, build system\n"
        "6. Root Cause Synthesis: actionable finding\n\n"
        "Write to $OUTPUT_FILE: classification (software|hardware|build-system|toolchain|unknown), "
        "root_cause, constraint, approach, avoid, confidence, resubmit, systemic\n"
    )

    return "\n".join(parts)


def _map_event_to_template(event_type: str) -> str:
    mapping = {
        "segfault": "segfault",
        "crash": "segfault",
        "compilation-fail": "compilation-fail",
        "regression": "regression",
        "exhausted": "exhausted",
        "rebuild-fail": "rebuild",
        "rebuild-crash": "rebuild",
        "tuning-crash": "segfault",
        "tuning-fail": "regression",
        "comm-hang": "comm-hang",
        "comm-fail": "comm-hang",
        "codegen-fail": "codegen",
        "cache-corrupt": "codegen",
        "server-crash": "server-config",
        "server-hang": "server-hang",
        "dispatch-fix-fail": "dispatch-fix-fail",
        "merge-fail": "rebuild",
        "merge-revert": "regression",
    }
    return mapping.get(event_type, "segfault")
