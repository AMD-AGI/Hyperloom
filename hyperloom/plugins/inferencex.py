"""InferenceX/Magpie benchmark plugin.

Wraps the Magpie benchmarking tool for supported frameworks
(SGLang, vLLM, Atom via InferenceX). Invocation pattern preserved
from inference_optimizer/orchestrator/action_executors/baseline.py.

Features:
  - Automatic YAML config materialization from session parameters
  - Full environment variable plumbing (INFERENCEX_PATH, RESULT_DIR, etc.)
  - Magpie atomic-write patcher (fixes concurrent-run race condition)
  - Auto-routing: sglang/vllm/atom frameworks all route through Magpie
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .base import BenchmarkPlugin, BenchResult

log = logging.getLogger(__name__)

MAGPIE_DEFAULT_TIMEOUT = 7800

_DEFAULT_YAML_TEMPLATE = """\
benchmark:
  backend: "{backend}"
  model: "{model_path}"
  server:
    host: "{host}"
    port: {port}
  workload:
    dataset: "random"
    num_prompts: {num_prompts}
    input_len: {isl}
    output_len: {osl}
    concurrency: {concurrency}
  output_dir: "{output_dir}"
  run_mode: "local"
"""


class InferenceXPlugin(BenchmarkPlugin):
    """Benchmark via Magpie (InferenceX).

    Handles SGLang, vLLM, and Atom frameworks through Magpie's unified interface.
    """

    def __init__(self, config: Any = None):
        self._config = config
        self._magpie_python = self._resolve_magpie_python()

    @property
    def name(self) -> str:
        return "inferencex"

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        errors = []
        if not self._magpie_python:
            errors.append("Magpie not found (set MAGPIE_PYTHON or ensure python3 can import Magpie)")
        return errors

    def run(self, config: dict[str, Any]) -> BenchResult:
        if self._magpie_python:
            return self._run_magpie(config)
        return self._run_bench_script(config)

    def _run_bench_script(self, config: dict[str, Any]) -> BenchResult:
        """Run benchmark_serving.py directly (InferenceX without Magpie module)."""
        import re

        bench_script = os.environ.get("VLLM_BENCH_SCRIPT", "")
        if not bench_script or not Path(bench_script).exists():
            inferencex_path = os.environ.get("INFERENCEX_PATH", "")
            if inferencex_path:
                candidate = Path(inferencex_path) / "utils" / "bench_serving" / "benchmark_serving.py"
                if candidate.exists():
                    bench_script = str(candidate)

        if not bench_script:
            return BenchResult(raw_output="Neither Magpie module nor benchmark_serving.py found")

        model_path = config.get("model_path", "")
        port = config.get("port", 8000)
        isl = config.get("isl", int(os.environ.get("ISL", "1024")))
        osl = config.get("osl", int(os.environ.get("OSL", "1024")))
        concurrency = config.get("concurrency", int(os.environ.get("CONCURRENCY", "64")))
        num_prompts = config.get("num_prompts", concurrency * 10)

        cmd = [
            "python3", bench_script,
            "--backend", "vllm",
            "--base-url", f"http://localhost:{port}",
            "--model", model_path,
            "--dataset-name", "random",
            "--random-input-len", str(isl),
            "--random-output-len", str(osl),
            "--random-range-ratio", "1.0",
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(concurrency),
            "--request-rate", "inf",
            "--ignore-eos",
        ]

        env = self._build_env(config, tempfile.mkdtemp(prefix="hyperloom_bench_"))
        timeout = config.get("timeout", MAGPIE_DEFAULT_TIMEOUT)

        log.info("Running benchmark_serving.py: %s", " ".join(cmd[:6]))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            return BenchResult(raw_output="benchmark_serving.py timed out")

        output = result.stdout + "\n" + result.stderr
        log.info("benchmark_serving.py exit code: %d, output length: %d", result.returncode, len(output))
        if result.returncode != 0:
            log.warning("benchmark_serving.py stderr (last 500): %s", result.stderr[-500:])

        throughput = 0.0
        m = re.search(r"Output token throughput.*?:\s*([\d.]+)", output)
        if m:
            throughput = float(m.group(1))
        else:
            m = re.search(r"output_throughput.*?:\s*([\d.]+)", output, re.IGNORECASE)
            if m:
                throughput = float(m.group(1))
        latency = 0.0
        m = re.search(r"Mean TPOT.*?:\s*([\d.]+)", output)
        if m:
            latency = float(m.group(1))

        return BenchResult(
            throughput=throughput,
            latency_mean_ms=latency,
            raw_output=output,
        )

    def _run_magpie(self, config: dict[str, Any]) -> BenchResult:
        """Run benchmark via Magpie Python module."""
        self._apply_magpie_patcher()

        output_dir = config.get("output_dir", tempfile.mkdtemp(prefix="hyperloom_bench_"))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        benchmark_config = config.get("benchmark_config", "")
        if not benchmark_config:
            benchmark_config = self._materialize_yaml(config, output_dir)
            if not benchmark_config:
                return BenchResult(raw_output="No benchmark_config and failed to materialize YAML")

        cmd = [
            self._magpie_python, "-m", "Magpie", "-v", "benchmark",
            "--benchmark-config", benchmark_config,
            "--output-dir", output_dir,
            "--run-mode", "local",
        ]

        env = self._build_env(config, output_dir)
        timeout = config.get("timeout", MAGPIE_DEFAULT_TIMEOUT)

        log.info("Running Magpie: %s", " ".join(cmd[:6]))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            return BenchResult(raw_output="Magpie benchmark timed out")

        output = result.stdout + "\n" + result.stderr

        report_path = Path(output_dir) / "benchmark_report.json"
        if report_path.exists():
            return self._parse_report(report_path, output)

        for candidate in Path(output_dir).rglob("benchmark_report.json"):
            return self._parse_report(candidate, output)

        return BenchResult(raw_output=output)

    def _build_env(self, config: dict[str, Any], output_dir: str) -> dict[str, str]:
        """Build full environment for Magpie subprocess."""
        env = os.environ.copy()

        env["RESULT_DIR"] = output_dir

        model_path = config.get("model_path", os.environ.get("MODEL_PATH", ""))
        if model_path:
            env["MODEL_PATH"] = model_path

        inferencex_path = config.get("inferencex_path", os.environ.get("INFERENCEX_PATH", ""))
        if inferencex_path:
            env["INFERENCEX_PATH"] = inferencex_path

        magpie_dir = os.environ.get("MAGPIE_DIR", "")
        if magpie_dir:
            env["MAGPIE_DIR"] = magpie_dir

        gpu_type = config.get("gpu_type", os.environ.get("GPU_TYPE", ""))
        if gpu_type:
            env["GPU_TYPE"] = gpu_type

        serving_gpus = config.get("gpus", os.environ.get("SERVING_GPUS", ""))
        if serving_gpus:
            env["CUDA_VISIBLE_DEVICES"] = serving_gpus
            env.pop("ROCR_VISIBLE_DEVICES", None)
            env.pop("HIP_VISIBLE_DEVICES", None)

        server_log = config.get("server_log", "")
        if server_log:
            env["SERVER_LOG"] = server_log

        return env

    def _materialize_yaml(self, config: dict[str, Any], output_dir: str) -> str:
        """Generate a Magpie benchmark YAML from session parameters."""
        framework = config.get("framework", "").lower()
        backend_map = {
            "sglang": "sglang",
            "vllm": "vllm",
            "atom": "atom",
            "inferencex": "sglang",
        }
        backend = backend_map.get(framework, framework or "sglang")

        model_path = config.get("model_path", os.environ.get("MODEL_PATH", "unknown"))
        host = config.get("host", "localhost")
        port = config.get("port", int(os.environ.get("PORT", "8888")))
        isl = config.get("isl", int(os.environ.get("ISL", "1024")))
        osl = config.get("osl", int(os.environ.get("OSL", "1024")))
        concurrency = config.get("concurrency", int(os.environ.get("CONCURRENCY", "64")))
        num_prompts = config.get("num_prompts", concurrency * 10)

        yaml_content = _DEFAULT_YAML_TEMPLATE.format(
            backend=backend,
            model_path=model_path,
            host=host,
            port=port,
            num_prompts=num_prompts,
            isl=isl,
            osl=osl,
            concurrency=concurrency,
            output_dir=output_dir,
        )

        config_path = Path(output_dir) / "benchmark_config.yaml"
        config_path.write_text(yaml_content)
        log.info("Materialized Magpie YAML: %s", config_path)
        return str(config_path)

    def _parse_report(self, report_path: Path, raw_output: str) -> BenchResult:
        """Parse Magpie's benchmark_report.json."""
        try:
            data = json.loads(report_path.read_text())
            throughput = float(
                data.get("output_token_throughput",
                         data.get("output_throughput",
                                  data.get("throughput", 0)))
            )
            latency = float(
                data.get("mean_e2e_latency_ms",
                         data.get("mean_tpot_ms", 0))
            )
            return BenchResult(
                throughput=throughput,
                latency_mean_ms=latency,
                raw_output=raw_output,
                extra=data,
            )
        except (json.JSONDecodeError, ValueError) as e:
            return BenchResult(raw_output=f"Failed to parse report: {e}\n{raw_output}")

    def _resolve_magpie_python(self) -> str:
        """Resolve Python interpreter that can import Magpie."""
        env_python = os.environ.get("MAGPIE_PYTHON", "")
        if env_python and shutil.which(env_python):
            return env_python

        candidates = [
            shutil.which("python3"),
            shutil.which("python"),
        ]
        venv_python = os.environ.get("MAGPIE_VENV", "")
        if venv_python:
            candidates.append(venv_python)
        for python in candidates:
            if not python or not Path(python).exists():
                continue
            try:
                result = subprocess.run(
                    [python, "-c", "import Magpie"],
                    capture_output=True, timeout=10,
                )
                if result.returncode == 0:
                    return python
            except (subprocess.TimeoutExpired, OSError):
                continue
        return ""

    def _apply_magpie_patcher(self) -> None:
        """Apply atomic-write fix to Magpie benchmarker to prevent race conditions.

        Magpie's benchmarker.py uses non-atomic writes when copying scripts,
        which causes crashes under concurrent runs. This patches the write
        to use tempfile + os.replace for atomicity.
        """
        magpie_dir = os.environ.get("MAGPIE_DIR", "")
        if not magpie_dir:
            if self._magpie_python:
                try:
                    result = subprocess.run(
                        [self._magpie_python, "-c",
                         "import Magpie; import os; print(os.path.dirname(Magpie.__file__))"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if result.returncode == 0:
                        magpie_dir = result.stdout.strip()
                except (subprocess.TimeoutExpired, OSError):
                    pass

        if not magpie_dir:
            return

        benchmarker_path = Path(magpie_dir) / "modes" / "benchmark" / "benchmarker.py"
        if not benchmarker_path.exists():
            return

        try:
            content = benchmarker_path.read_text()
            # Only patch if the non-atomic pattern exists and hasn't been patched
            if "shutil.copy2" in content and "os.replace" not in content:
                patched = content.replace(
                    "shutil.copy2(src, dst)",
                    (
                        "import tempfile as _tf\n"
                        "            _tmp = _tf.NamedTemporaryFile(dir=os.path.dirname(dst), delete=False)\n"
                        "            shutil.copy2(src, _tmp.name)\n"
                        "            os.replace(_tmp.name, dst)"
                    ),
                )
                if patched != content:
                    benchmarker_path.write_text(patched)
                    log.info("Applied Magpie atomic-write patch to %s", benchmarker_path)
        except (OSError, PermissionError):
            log.debug("Could not patch Magpie benchmarker (read-only?)")
