"""Accuracy gating — ensures optimizations don't break model quality.

Accuracy is NEVER optional. Every optimization must be gated on quality.

For LLM serving: uses lm-evaluation-harness with GSM8K (same as InferenceX).
For non-LLM workloads (diffusion, etc.): user provides a custom eval script
via --accuracy-script that outputs JSON with a score and pass/fail.

The interface is the same regardless of domain — the bench/accuracy script
defines what "correct" means for that workload.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import SessionConfig
from .plugins.base import AccuracyPlugin, AccuracyResult
from .plugins.custom import CustomAccuracyPlugin
from .plugins.lm_eval import LMEvalAccuracyPlugin, GSM8K_THRESHOLD

log = logging.getLogger(__name__)


def resolve_accuracy_plugin(config: SessionConfig) -> AccuracyPlugin:
    """Resolve accuracy plugin. Always returns a plugin — accuracy is mandatory.

    Resolution order:
      1. Custom script (--accuracy-script): for non-LLM workloads (diffusion,
         audio, etc.) or custom LLM evals. User provides any executable that
         outputs JSON with {score, passed}.
      2. Built-in lm_eval/GSM8K: default for LLM serving. Runs lm-evaluation-
         harness against the model's OpenAI-compatible endpoint.
    """
    if config.accuracy.script:
        log.info("Accuracy gate: custom script (%s)", config.accuracy.script)
        return CustomAccuracyPlugin(
            script=config.accuracy.script,
            output_format=config.accuracy.output_format,
            score_key=config.accuracy.score_key,
            threshold=config.accuracy.threshold,
            env=config.accuracy.env,
            timeout=config.accuracy.timeout,
        )

    threshold = config.accuracy.threshold if config.accuracy.threshold > 0 else GSM8K_THRESHOLD
    log.info("Accuracy gate: lm_eval/gsm8k (threshold=%.2f)", threshold)
    return LMEvalAccuracyPlugin(
        port=config.port,
        task="gsm8k",
        threshold=threshold,
        max_model_len=4096,
    )


def run_accuracy_gate(
    plugin: AccuracyPlugin,
    config: SessionConfig,
    baseline_score: float | None = None,
) -> AccuracyResult:
    """Run accuracy eval and check if it passes the gate.

    Always returns an AccuracyResult — accuracy is never skipped.
    The optimization loop MUST reject any patch where passed=False.
    """
    config_dict: dict[str, Any] = {
        "model_path": config.model_path,
        "gpu_type": config.gpu_type,
        "port": config.port,
    }
    config_dict.update(config.extra)

    log.info("Running accuracy eval (%s)...", plugin.name)
    result = plugin.run(config_dict)

    if result.passed:
        log.info("Accuracy gate PASSED: %s = %.4f (threshold: %.4f)",
                 result.metric_name, result.score, result.threshold)
    else:
        log.warning("Accuracy gate FAILED: %s = %.4f (threshold: %.4f)",
                    result.metric_name, result.score, result.threshold)

    return result
