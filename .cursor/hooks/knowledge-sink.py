#!/usr/bin/env python3
"""
Cursor stop hook: Detect inference optimization runs and prompt the agent
to contribute new knowledge to the RAG knowledge base.

Fires when the conversation contains evidence of:
  1. A benchmark run (tok/s results)
  2. An optimization action (backend switch, param change, kernel patch)
  3. A measurable outcome (gain/regression percentage)

Returns a followup_message instructing the agent to ingest findings into
the KB via kb_ingest.py. Only fires once per conversation.

Modeled after Primus-Conductor's node-pod-knowledge-sink.py pattern.
"""

import json
import os
import re
import sys

OPTIMIZATION_INDICATORS = [
    r"tok/s",
    r"tput_per_gpu",
    r"output_throughput",
    r"TPOT.*ms",
    r"TTFT.*ms",
    r"gain.*%",
    r"\+\d+(\.\d+)?%",
    r"-\d+(\.\d+)?%",
    r"KEEP|REVERT|DISCARD",
]

ACTION_INDICATORS = [
    "backend",
    "decode-backend",
    "prefill-backend",
    "attention-backend",
    "torch.compile",
    "GEAK",
    "TraceLens",
    "kernel",
    "cuda-graph-max-bs",
    "decode-steps",
    "mem-fraction",
    "mixed-chunk",
    "allreduce",
    "run_baseline.sh",
    "run_profile.sh",
    "run_sweep.sh",
    "benchmark_serving",
    "kb_ingest",
]

BENCHMARK_INDICATORS = [
    r"benchmark.*results?",
    r"output_throughput.*\d+",
    r"mean_tpot_ms",
    r"run_benchmark_serving",
    r"baseline.*tok/s",
    r"optimized.*tok/s",
]

SELF_FOLLOWUP_MARKER = "kb_ingest.py"

SKILL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "skills", "inference-optimization",
)
KB_INGEST_PATH = os.path.join(SKILL_ROOT, "kb", "kb_ingest.py")

STATE_DIR = os.path.expanduser("~/.cursor/hooks-state")
STATE_FILE = os.path.join(STATE_DIR, "inference-opt-kb-sink-fired.json")

FOLLOWUP_MESSAGE = f"""This conversation involved inference optimization work with measurable results.
Please contribute any new findings to the knowledge base before ending.

For each significant finding (backend switch, parameter change, kernel optimization, pitfall discovered), run:

```bash
python3 {KB_INGEST_PATH} \\
    --category <backend_exploration|kernel_optimization|server_params|pitfall|lesson|target_comparison|benchmark_methodology|architecture_constraint|framework_comparison> \\
    --model "$MODEL_NAME" \\
    --framework "$FRAMEWORK" \\
    --action "Brief description of what was done" \\
    --lesson "Key takeaway - what worked, what didn't, why" \\
    --tags relevant,comma,separated,tags \\
    --gain <percentage if applicable> \\
    --status <KEEP|REVERT|DISCARD> \\
    --context "Controlled A/B test, $(date +%Y-%m-%d), additional methodology notes"
```

Guidelines:
- Only ingest findings with clear methodology (controlled A/B comparisons preferred)
- Include throughput numbers (tok/s/GPU before and after) in the lesson
- Tag with model architecture traits (MoE, MLA, SWA, NSA) for future KB queries
- If a finding contradicts existing KB entries, the conflict resolution system will handle it
- Mark pitfalls and failures too — they prevent future agents from repeating mistakes"""


def already_fired(conversation_id: str) -> bool:
    if not conversation_id:
        return False
    try:
        with open(STATE_FILE, "r") as f:
            fired = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        fired = {}
    return conversation_id in fired


def mark_fired(conversation_id: str) -> None:
    if not conversation_id:
        return
    os.makedirs(STATE_DIR, exist_ok=True)
    try:
        with open(STATE_FILE, "r") as f:
            fired = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        fired = {}
    fired[conversation_id] = True
    if len(fired) > 200:
        keys = sorted(fired.keys())
        fired = {k: fired[k] for k in keys[-100:]}
    with open(STATE_FILE, "w") as f:
        json.dump(fired, f)


def read_transcript(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(500_000)
    except Exception:
        return ""


def has_optimization_results(content: str) -> bool:
    """At least 2 optimization indicators present."""
    count = sum(1 for p in OPTIMIZATION_INDICATORS if re.search(p, content))
    return count >= 2


def has_actions(content: str) -> bool:
    """At least 2 action indicators present."""
    content_lower = content.lower()
    count = sum(1 for ind in ACTION_INDICATORS if ind.lower() in content_lower)
    return count >= 2


def has_benchmarks(content: str) -> bool:
    """At least 1 benchmark indicator present."""
    return any(re.search(p, content, re.IGNORECASE) for p in BENCHMARK_INDICATORS)


def main():
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, ValueError):
        payload = {}

    conversation_id = payload.get("conversation_id", "")
    transcript_path = payload.get("transcript_path", "")

    if already_fired(conversation_id):
        sys.stdout.write("{}\n")
        return

    content = read_transcript(transcript_path)

    if SELF_FOLLOWUP_MARKER in content:
        mark_fired(conversation_id)
        sys.stdout.write("{}\n")
        return

    response = {}

    if content and has_optimization_results(content) and has_actions(content) and has_benchmarks(content):
        response["followup_message"] = FOLLOWUP_MESSAGE
        mark_fired(conversation_id)

    sys.stdout.write(json.dumps(response) + "\n")


if __name__ == "__main__":
    main()
