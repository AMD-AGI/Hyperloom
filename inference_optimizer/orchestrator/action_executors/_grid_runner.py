"""Shared helper for backends / params executors.

Each runner's job is essentially: take a base Magpie YAML + a list of
(name, extra_sglang_args, extra_envs) variants, run Magpie once per
variant, parse `benchmark_report.json`, return the winners.

We share the "run one Magpie variant" loop here so backends.py / params.py
stay tiny and only declare the grid (the marathon DFS playbook).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .benchmark_result import extract_benchmark_measurement


log = logging.getLogger(__name__)


def _resolve_magpie_python() -> str:
    """Resolve the Python interpreter for Magpie subprocesses.

    Order: $MAGPIE_PYTHON env > first `python3` on PATH that can
    ``import Magpie`` > /opt/venv/bin/python (if it exists).
    """
    env_val = os.environ.get("MAGPIE_PYTHON", "").strip()
    if env_val:
        return env_val

    def _can_import_magpie(py: str) -> bool:
        try:
            return subprocess.run(
                [py, "-c", "import Magpie"],
                capture_output=True, timeout=10,
            ).returncode == 0
        except Exception:
            return False

    candidate = shutil.which("python3")
    if candidate and _can_import_magpie(candidate):
        return candidate

    fallback = Path("/opt/venv/bin/python")
    if fallback.exists():
        return str(fallback)

    return candidate or "/opt/venv/bin/python"


def _resolve_session_dir() -> Path:
    """Resolve the active session_dir for executors that need an output root.

    Reads :func:`inference_optimizer.paths.session_dir`; this honors
    ``$INFERENCE_OPTIMIZER_SESSION_DIR`` and otherwise returns
    ``/workspace/hyperloom``. Used by executor-class fallback paths when
    ``ctx.extra["workspace"]`` was not pre-mkdir'd by SubAgentRunner.
    """
    from ...paths import session_dir as _sd
    return _sd()


_MAGPIE_CWD_DEFAULT = "/tmp"
_VARIANT_TIMEOUT_SEC_DEFAULT = 900


@dataclass
class GridVariant:
    """One row of the grid we're going to test."""

    name: str                                    # human-readable label
    extra_sglang_args: str = ""                  # appended via EXTRA_SGLANG_ARGS env
    extra_envs: dict[str, str] = field(default_factory=dict)
    note: str = ""                                # optional reason / category


@dataclass
class VariantResult:
    """One bench run's parsed result."""

    name: str
    extra_sglang_args: str
    extra_envs: dict[str, str]
    status: str
    output_throughput: float | None = None
    request_throughput: float | None = None
    total_token_throughput: float | None = None
    completed_requests: int | None = None
    duration_seconds: float | None = None
    ttft_mean_ms: float | None = None
    e2el_mean_ms: float | None = None
    workspace: str | None = None
    report_path: str | None = None
    raw_result_path: str | None = None
    reported_success: bool | None = None
    returncode: int | None = None
    nonfatal_warnings: list[str] = field(default_factory=list)
    error: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":               self.name,
            "extra_sglang_args":  self.extra_sglang_args,
            "extra_envs":         self.extra_envs,
            "status":             self.status,
            "output_throughput":  self.output_throughput,
            "request_throughput": self.request_throughput,
            "total_token_throughput": self.total_token_throughput,
            "completed_requests": self.completed_requests,
            "duration_seconds":   self.duration_seconds,
            "ttft_mean_ms":       self.ttft_mean_ms,
            "e2el_mean_ms":       self.e2el_mean_ms,
            "workspace":          self.workspace,
            "report_path":        self.report_path,
            "raw_result_path":    self.raw_result_path,
            "reported_success":   self.reported_success,
            "returncode":         self.returncode,
            "nonfatal_warnings":  self.nonfatal_warnings,
            "error":              self.error,
            "note":               self.note,
        }


# ---------------------------------------------------------------------------
def server_args_env_name(framework: str | None) -> str:
    """Return the Magpie env var used to append backend server args."""
    name = str(framework or "").strip().lower()
    if "vllm" in name:
        return "EXTRA_VLLM_ARGS"
    return "EXTRA_SGLANG_ARGS"


def merge_server_args(*parts: str | None) -> str:
    """Merge server arg strings preserving left-to-right override semantics.

    vLLM/SGLang command lines are assembled by shell-appending
    ``EXTRA_{VLLM,SGLANG}_ARGS`` after the default server args. Some flags are
    intentionally repeated so later variants can override base args (e.g. a
    model-specific ``--block-size 1`` plus a grid candidate ``--block-size
    256``). Therefore this helper only removes empty chunks; it does not try to
    de-duplicate option names.
    """
    return " ".join(str(p).strip() for p in parts if str(p or "").strip())


def apply_runtime_benchmark_overrides(
    bench: dict[str, Any],
    *,
    model_path: str | None = None,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    """Apply runtime env/CLI overrides to a Magpie benchmark YAML.

    This is the single shared path used by baseline/profile and grid
    executors. Historically only ``baseline.py`` applied these overrides, so
    backends/params/sweep silently fell back to shipped YAML defaults like
    ``TP=1`` and ``ROCR_VISIBLE_DEVICES="1"``. Large models (DeepSeek-R1-0528)
    then OOM-failed even though the launch environment had ``TP=8``.
    """
    if model_path:
        bench["model"] = str(model_path)

    precision = os.environ.get("PRECISION", "").strip()
    if precision:
        bench["precision"] = precision

    if gpu_type:
        bench["runner_type"] = str(gpu_type)
        # Magpie priority: explicit benchmark_script > native script >
        # runner_type-derived generic script. Drop stale explicit scripts so
        # runtime GPU selection actually wins.
        bench.pop("benchmark_script", None)

    envs = bench.setdefault("envs", {})
    for env_key in ("ISL", "OSL", "MAX_MODEL_LEN", "TP", "CONC"):
        val = os.environ.get(env_key, "").strip()
        if val:
            envs[env_key] = int(val)

    explicit_rocr = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if explicit_rocr:
        envs["ROCR_VISIBLE_DEVICES"] = explicit_rocr
    else:
        tp_val = int(envs.get("TP", 1) or 1)
        existing_rocr = str(envs.get("ROCR_VISIBLE_DEVICES", "")).strip()
        existing_count = (
            len([x for x in existing_rocr.split(",") if x.strip()])
            if existing_rocr else 0
        )
        if tp_val > 1 and existing_count < tp_val:
            envs["ROCR_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp_val))

    return envs


# ---------------------------------------------------------------------------
def _build_variant_yaml(
    base_yaml_path: Path,
    base_extra_args: str,
    variant: GridVariant,
    *,
    output_subdir: Path,
    model_path: str | None = None,
    gpu_type: str | None = None,
) -> Path:
    """Materialize a per-variant Magpie YAML on disk.

    Magpie's sglang_mi300x.sh honors ``EXTRA_SGLANG_ARGS`` from envs to
    append flags after the auto-generated server args, so we just inject
    the variant's flags there.

    ``model_path`` (when non-empty) overrides ``benchmark.model``; the
    shipped configs all have a legacy hardcoded Qwen-Qwen3-8B path that
    would otherwise win over the user's runtime selection.

    ``gpu_type`` (when non-empty) injects ``benchmark.runner_type`` and
    pops any explicit ``benchmark.benchmark_script`` so Magpie's
    runner_type -> script logic actually fires for the requested GPU.
    """
    with base_yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bench = cfg.setdefault("benchmark", {})
    envs = apply_runtime_benchmark_overrides(
        bench, model_path=model_path, gpu_type=gpu_type,
    )
    extra_args_env = server_args_env_name(bench.get("framework"))

    combined = merge_server_args(
        str(envs.get(extra_args_env, "")),
        base_extra_args,
        variant.extra_sglang_args,
    )
    if combined:
        envs[extra_args_env] = combined
    for k, v in variant.extra_envs.items():
        envs[str(k)] = str(v)

    output_subdir.mkdir(parents=True, exist_ok=True)
    out_path = output_subdir / "config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def _parse_report(workspace: Path) -> dict[str, Any] | None:
    report = workspace / "benchmark_report.json"
    if not report.exists():
        return None
    try:
        with report.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _kill_stale_servers() -> None:
    """Deep-clean any lingering inference server processes + shared memory.

    Magpie's server_cleanup.sh only kills the process group leader and waits,
    but vLLM::Worker / EngineCore children often escape the pgrp (Ray-spawned
    or multiprocessing.spawn). Without this pre-clean, the next vLLM startup
    hangs for 5 minutes on zmq socket / shared mem conflicts:
      "Did not receive response from front-end process within 5 minutes"

    We call this BEFORE every Magpie invocation so each grid variant starts
    on a pristine server state.

    NOTE: uses /proc scan instead of `subprocess.run(["pgrep",...])` to avoid
    conflicting with test mocks that patch subprocess.run for Magpie calls.
    """
    import signal
    import glob
    import time

    _KILL_PATTERNS = ("VLLM::Worker", "VLLM::EngineCore", "vllm.entrypoints",
                      "vllm serve", "sglang.srt", "sglang.launch_server")

    my_pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == my_pid:
            continue
        try:
            cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
        except (OSError, PermissionError):
            continue
        text = cmdline.replace(b"\0", b" ").decode("utf-8", "replace")
        if any(pat in text for pat in _KILL_PATTERNS):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # Clear /dev/shm vllm/nccl/cuda segments that prevent re-binding.
    for pattern in ("/dev/shm/vllm*", "/dev/shm/nccl*", "/dev/shm/cuda*"):
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass

    # Brief pause for KFD (ROCm kernel driver) async VRAM release.
    time.sleep(2)


def _run_magpie(
    *,
    magpie_python: str,
    config_path: Path,
    output_dir: Path,
    timeout_sec: int,
    cwd: str,
) -> tuple[int, str, str]:
    """Blocking subprocess wrapper. Returns (rc, stdout, stderr)."""
    # Pre-clean: kill lingering server processes + clear shared memory so the
    # next vLLM/SGLang startup doesn't collide with stale resources.
    # Skip in test environments to avoid 2s sleep per variant.
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        _kill_stale_servers()

    env = os.environ.copy()
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
    magpie_dir = os.environ.get("MAGPIE_DIR", "")
    if magpie_dir:
        env["PYTHONPATH"] = f"{magpie_dir}:{env.get('PYTHONPATH', '')}"
    cmd = [
        magpie_python, "-m", "Magpie", "-v", "benchmark",
        "--benchmark-config", str(config_path),
        "--output-dir", str(output_dir),
        "--run-mode", "local",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_sec,
        env=env, cwd=cwd,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


# ---------------------------------------------------------------------------
async def run_grid(
    *,
    base_yaml_path: Path,
    base_extra_args: str,
    grid: list[GridVariant],
    output_root: Path,
    magpie_python: str | None = None,
    cwd: str = _MAGPIE_CWD_DEFAULT,
    variant_timeout_sec: int = _VARIANT_TIMEOUT_SEC_DEFAULT,
    keep_going_on_failure: bool = True,
    model_path: str | None = None,
    gpu_type: str | None = None,
) -> list[VariantResult]:
    """Execute every variant in ``grid`` once, in order.

    Returns the per-variant :class:`VariantResult` list (all attempts,
    including failed ones). Caller decides which variants are "winners".

    Synchronous subprocess call wrapped in ``asyncio.to_thread`` so the
    Coordinator reactor isn't blocked.

    ``model_path`` and ``gpu_type`` are forwarded to every variant's YAML
    render; pass the values resolved by the executor (task.params or
    $MODEL_PATH / $GPU_TYPE) so each Magpie invocation benchmarks the
    user's actual model on the user's actual GPU rather than the YAML's
    legacy default.
    """
    if not magpie_python:
        magpie_python = _resolve_magpie_python()
    results: list[VariantResult] = []
    for i, variant in enumerate(grid):
        slot = output_root / f"variant_{i:02d}_{_safe(variant.name)}"
        try:
            cfg_path = _build_variant_yaml(
                base_yaml_path, base_extra_args, variant, output_subdir=slot,
                model_path=model_path,
                gpu_type=gpu_type,
            )
        except Exception as exc:  # noqa: BLE001
            results.append(VariantResult(
                name=variant.name, extra_sglang_args=variant.extra_sglang_args,
                extra_envs=dict(variant.extra_envs),
                status="failed", error=f"yaml_build_error: {exc!r}",
                note=variant.note,
            ))
            if not keep_going_on_failure:
                break
            continue

        log.info(
            "grid_runner: variant %d/%d name=%s args=%s",
            i + 1, len(grid), variant.name, variant.extra_sglang_args,
        )

        try:
            rc, stdout, stderr = await asyncio.to_thread(
                _run_magpie,
                magpie_python=magpie_python,
                config_path=cfg_path,
                output_dir=slot,
                timeout_sec=variant_timeout_sec,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as exc:
            results.append(VariantResult(
                name=variant.name, extra_sglang_args=variant.extra_sglang_args,
                extra_envs=dict(variant.extra_envs),
                status="failed", error=f"timeout: {exc}", note=variant.note,
            ))
            if not keep_going_on_failure:
                break
            continue

        # Locate workspace inside slot.
        candidates = sorted(slot.glob("benchmark_*"))
        if not candidates:
            results.append(VariantResult(
                name=variant.name, extra_sglang_args=variant.extra_sglang_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                returncode=rc,
                error=(
                    (stderr or stdout)[-2000:]
                    if rc != 0 else "no benchmark_* workspace produced"
                ),
                note=variant.note,
            ))
            if rc != 0 and not keep_going_on_failure:
                break
            continue
        workspace = candidates[-1]
        report = _parse_report(workspace)
        report_path = workspace / "benchmark_report.json"
        measurement = extract_benchmark_measurement(report, workspace=workspace)
        warnings = list(measurement.pop("nonfatal_warnings", []) or [])
        if rc != 0:
            warnings.append("magpie_nonzero_after_valid_measurement")

        if not measurement.get("valid_measurement"):
            if rc != 0:
                error = (stderr or stdout)[-2000:]
            elif not report:
                error = "benchmark_report missing"
            else:
                error = "benchmark_report missing valid throughput/completed requests"
            results.append(VariantResult(
                name=variant.name, extra_sglang_args=variant.extra_sglang_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                workspace=str(workspace),
                report_path=str(report_path) if report_path.exists() else None,
                raw_result_path=measurement.get("raw_result_path"),
                reported_success=measurement.get("reported_success"),
                returncode=rc,
                nonfatal_warnings=warnings,
                error=error,
                note=variant.note,
            ))
            if rc != 0 and not keep_going_on_failure:
                break
            continue

        results.append(VariantResult(
            name=variant.name, extra_sglang_args=variant.extra_sglang_args,
            extra_envs=dict(variant.extra_envs),
            status="succeeded",
            output_throughput=measurement.get("output_throughput"),
            request_throughput=measurement.get("request_throughput"),
            total_token_throughput=measurement.get("total_token_throughput"),
            completed_requests=measurement.get("completed_requests"),
            duration_seconds=measurement.get("duration_seconds"),
            ttft_mean_ms=measurement.get("ttft_mean_ms"),
            e2el_mean_ms=measurement.get("e2el_mean_ms"),
            workspace=str(workspace),
            report_path=str(report_path) if report_path.exists() else None,
            raw_result_path=measurement.get("raw_result_path"),
            reported_success=measurement.get("reported_success"),
            returncode=rc,
            nonfatal_warnings=warnings,
            error=(stderr or stdout)[-2000:] if rc != 0 else None,
            note=variant.note,
        ))
        log.info(
            "grid_runner: variant %s tput=%.1f tok/s",
            variant.name, results[-1].output_throughput or 0.0,
        )
    return results


def pick_winners(
    results: list[VariantResult],
    baseline_tput: float,
    *,
    keep_threshold_pct: float = 1.0,
) -> list[VariantResult]:
    """Filter the variants whose throughput beats ``baseline_tput`` by
    ``keep_threshold_pct`` percent (marathon §params: > 1% = KEEP)."""
    cutoff = baseline_tput * (1.0 + keep_threshold_pct / 100.0)
    return [
        r for r in results
        if r.status == "succeeded"
        and isinstance(r.output_throughput, (int, float))
        and r.output_throughput > cutoff
    ]


def _safe(name: str) -> str:
    """Filesystem-safe slug for variant directory names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]


__all__ = [
    "GridVariant",
    "VariantResult",
    "pick_winners",
    "run_grid",
    "server_args_env_name",
]
