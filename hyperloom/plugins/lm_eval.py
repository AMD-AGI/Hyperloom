"""Built-in accuracy plugin using lm-evaluation-harness (GSM8K).

Mirrors InferenceX's eval methodology: runs lm_eval against the serving
endpoint's OpenAI-compatible chat completions API with the same task
configs, thresholds, and filters.

This is the default accuracy gate - accuracy is never optional.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import AccuracyPlugin, AccuracyResult

log = logging.getLogger(__name__)

INFERENCEX_EVALS_DIR = None
GSM8K_THRESHOLD = 0.85
GPQA_THRESHOLD = 0.30

DEFAULT_THRESHOLDS = {
    "gsm8k": GSM8K_THRESHOLD,
    "gpqa_diamond_cot_n_shot": GPQA_THRESHOLD,
}


def _find_eval_tasks_dir() -> str:
    """Find the InferenceX evals directory with task YAML configs."""
    inferencex = os.environ.get("INFERENCEX_PATH", "")
    if inferencex:
        evals_dir = os.path.join(inferencex, "utils", "evals")
        gsm8k_yaml = os.path.join(evals_dir, "gsm8k.yaml")
        if os.path.isfile(gsm8k_yaml):
            return gsm8k_yaml
    return ""


def _install_lm_eval() -> bool:
    """Ensure lm-eval is installed."""
    try:
        import lm_eval  # noqa: F401
        return True
    except ImportError:
        pass

    log.info("Installing lm-eval[api]...")
    result = subprocess.run(
        ["python3", "-m", "pip", "install", "-q", "--no-cache-dir",
         "--break-system-packages", "lm-eval[api]"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        log.warning("Failed to install lm-eval: %s", result.stderr[-500:])
        return False

    lm_eval_ref = "b315ef3b05176acc9732bb7fdec116abe1ecc476"
    subprocess.run(
        ["python3", "-m", "pip", "install", "-q", "--no-cache-dir",
         "--no-deps", "--force-reinstall", "--break-system-packages",
         f"https://github.com/EleutherAI/lm-evaluation-harness/archive/{lm_eval_ref}.tar.gz"],
        capture_output=True, text=True, timeout=300,
    )
    return True


class LMEvalAccuracyPlugin(AccuracyPlugin):
    """Runs lm-evaluation-harness GSM8K against the served model endpoint."""

    def __init__(
        self,
        port: int = 8000,
        task: str = "gsm8k",
        threshold: float = GSM8K_THRESHOLD,
        concurrent_requests: int = 8,
        max_model_len: int = 4096,
    ):
        self._port = port
        self._task = task
        self._threshold = threshold
        self._concurrent = concurrent_requests
        self._max_model_len = max_model_len

    @property
    def name(self) -> str:
        return f"lm_eval/{self._task}"

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors = []
        if not _install_lm_eval():
            errors.append("Cannot install lm-evaluation-harness")
        return errors

    def run(self, config: dict[str, Any]) -> AccuracyResult:
        if not _install_lm_eval():
            return AccuracyResult(
                score=0.0, passed=False, threshold=self._threshold,
                raw_output="Failed to install lm-eval",
            )

        tasks_path = _find_eval_tasks_dir()
        if not tasks_path:
            tasks_path = self._task

        model_name = config.get("model_path", "")
        results_dir = tempfile.mkdtemp(prefix="hyperloom_eval_")
        base_url = f"http://0.0.0.0:{self._port}/v1/chat/completions"

        max_gen_tokens = min(self._max_model_len // 2, 2048)

        cmd = [
            "python3", "-m", "lm_eval",
            "--model", "local-chat-completions",
            "--apply_chat_template",
            "--tasks", tasks_path,
            "--output_path", results_dir,
            "--log_samples",
            "--model_args", (
                f"model={model_name},"
                f"base_url={base_url},"
                f"api_key=EMPTY,"
                f"eos_string=</s>,"
                f"max_retries=5,"
                f"num_concurrent={self._concurrent},"
                f"timeout=1800,"
                f"tokenized_requests=False,"
                f"max_length={self._max_model_len}"
            ),
            "--gen_kwargs", f"max_tokens={max_gen_tokens},temperature=0,top_p=1.0",
        ]

        env = {**os.environ, "OPENAI_API_KEY": "EMPTY"}

        log.info("Running lm_eval: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=7200, env=env,
        )

        output = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            log.error("lm_eval failed (rc=%d): %s", result.returncode, output[-1000:])
            return AccuracyResult(
                score=0.0, passed=False, threshold=self._threshold,
                raw_output=output[-2000:],
                extra={"returncode": result.returncode},
            )

        return self._parse_results(results_dir, output)

    def _parse_results(self, results_dir: str, raw_output: str) -> AccuracyResult:
        """Parse lm-eval JSON results and extract the score."""
        result_files = glob.glob(os.path.join(results_dir, "**", "results*.json"), recursive=True)
        if not result_files:
            result_files = glob.glob(os.path.join(results_dir, "results*.json"))

        best_score = 0.0
        task_found = ""
        all_metrics: dict[str, float] = {}

        for f in result_files:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                for task, metrics in data.get("results", {}).items():
                    for metric_name, val in metrics.items():
                        if not metric_name.startswith("exact_match"):
                            continue
                        if "stderr" in metric_name:
                            continue
                        if not isinstance(val, (int, float)):
                            continue
                        all_metrics[f"{task}/{metric_name}"] = val
                        if val > best_score:
                            best_score = val
                            task_found = task
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Could not parse %s: %s", f, e)

        threshold = DEFAULT_THRESHOLDS.get(self._task, self._threshold)
        passed = best_score >= threshold

        log.info("lm_eval result: %s exact_match=%.4f (threshold=%.4f) → %s",
                 task_found or self._task, best_score, threshold,
                 "PASS" if passed else "FAIL")

        return AccuracyResult(
            score=best_score,
            metric_name=f"{task_found or self._task}/exact_match",
            passed=passed,
            threshold=threshold,
            raw_output=raw_output[-2000:],
            extra={"all_metrics": all_metrics, "results_dir": results_dir},
        )
