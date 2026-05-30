"""RCA (Root Cause Analysis) — LLM-powered diagnosis triggered by alerts.

Only invoked when the rule engine detects anomalies that need deeper analysis.
The LLM receives structured evidence and returns a diagnosis + action recommendation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from .config import Config
from .models import Alert, RcaFinding, Severity

log = logging.getLogger(__name__)

RCA_SYSTEM_PROMPT = """\
You are the Robustness Agent's diagnostic engine for an inference optimization system.

You receive structured evidence from multiple monitoring sources:
- GPU metrics (VRAM, utilization, temperature, ECC errors)
- Process state (server PIDs, benchmark processes, zombies)
- Application logs (error patterns from server.log, build logs)
- Conductor events (agent activities, task states, lease states)
- Infrastructure faults (from Primus-Robust-Internal, if available)

Your job:
1. Analyze the evidence to identify the root cause
2. Determine the most appropriate corrective action
3. Output a structured JSON response

Valid action_type values:
- "kill_task": terminate a specific stuck/failing task
- "prune_branch": stop all tasks in a failing action family
- "escalate_strategy_change": suggest the orchestration agent change approach
- "recover": trigger recovery from checkpoint
- "no_action": alert only, no intervention needed

Respond with JSON only:
{
    "root_cause": "description of root cause",
    "confidence": 0.0-1.0,
    "action_type": "one of the valid types above",
    "action_payload": {},
    "evidence_summary": "brief summary of key evidence"
}
"""


class RcaEngine:
    """Invoke LLM for root cause analysis when alerts accumulate."""

    def __init__(self, config: Config):
        self._config = config
        self._pending_alerts: list[Alert] = []
        self._last_rca_time: float = 0
        self._min_interval_s: float = 120.0
        self._trigger_threshold: int = 2

    def ingest_alerts(self, alerts: list[Alert]) -> None:
        critical = [a for a in alerts if a.severity in (Severity.CRITICAL, Severity.WARNING)]
        self._pending_alerts.extend(critical)

    def should_trigger(self) -> bool:
        if not self._pending_alerts:
            return False
        critical_count = sum(
            1 for a in self._pending_alerts if a.severity == Severity.CRITICAL
        )
        if critical_count < self._trigger_threshold:
            return False
        if time.time() - self._last_rca_time < self._min_interval_s:
            return False
        return True

    async def run_rca(self, extra_context: dict[str, Any] | None = None) -> Optional[RcaFinding]:
        if not self._pending_alerts:
            return None

        self._last_rca_time = time.time()
        evidence = self._build_evidence(extra_context)

        try:
            result = await self._call_llm(evidence)
            finding = self._parse_result(result)
            self._pending_alerts.clear()
            return finding
        except Exception as exc:
            log.error("RCA failed: %s", exc)
            return None

    def _build_evidence(self, extra: dict[str, Any] | None) -> str:
        sections: list[str] = ["## Alerts\n"]
        for alert in self._pending_alerts[-20:]:
            sections.append(
                f"- [{alert.severity.value}] {alert.check_name}: {alert.summary}\n"
                f"  evidence: {json.dumps(alert.evidence, default=str)}\n"
            )
        if extra:
            sections.append("\n## Additional Context\n")
            sections.append(json.dumps(extra, indent=2, default=str))
        return "\n".join(sections)

    async def _call_llm(self, evidence: str) -> str:
        if not self._config.llm_base_url or not self._config.llm_api_key:
            log.warning("LLM not configured, returning stub RCA")
            return json.dumps({
                "root_cause": "LLM not configured — manual review required",
                "confidence": 0.0,
                "action_type": "no_action",
                "action_payload": {},
                "evidence_summary": evidence[:500],
            })

        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url=self._config.llm_base_url,
            api_key=self._config.llm_api_key,
        )
        response = await client.chat.completions.create(
            model=self._config.llm_model,
            messages=[
                {"role": "system", "content": RCA_SYSTEM_PROMPT},
                {"role": "user", "content": evidence},
            ],
            max_tokens=2000,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _parse_result(self, raw: str) -> RcaFinding:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}

        return RcaFinding(
            trigger_alerts=list(self._pending_alerts),
            root_cause=data.get("root_cause", "unknown"),
            suggested_action=data.get("action_type", "no_action"),
            action_type=data.get("action_type", "no_action"),
            action_payload=data.get("action_payload", {}),
            confidence=float(data.get("confidence", 0)),
            evidence_summary=data.get("evidence_summary", ""),
        )
