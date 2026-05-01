"""Shared helper for backends / params executors.

Each executor's job is essentially: take a base Magpie YAML + a list of
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
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


log = logging.getLogger(__name__)


_MAGPIE_PYTHON_DEFAULT = "/opt/venv/bin/python"
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
    ttft_mean_ms: float | None = None
    e2el_mean_ms: float | None = None
    workspace: str | None = None
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
            "ttft_mean_ms":       self.ttft_mean_ms,
            "e2el_mean_ms":       self.e2el_mean_ms,
            "workspace":          self.workspace,
            "error":              self.error,
            "note":               self.note,
        }


# ---------------------------------------------------------------------------
def _build_variant_yaml(
    base_yaml_path: Path,
    base_extra_args: str,
    variant: GridVariant,
    *,
    output_subdir: Path,
) -> Path:
    """Materialize a per-variant Magpie YAML on disk.

    Magpie's sglang_mi300x.sh honors ``EXTRA_SGLANG_ARGS`` from envs to
    append flags after the auto-generated server args, so we just inject
    the variant's flags there.
    """
    with base_yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bench = cfg.setdefault("benchmark", {})
    envs = bench.setdefault("envs", {})

    combined = " ".join(part for part in (base_extra_args.strip(),
                                            variant.extra_sglang_args.strip()) if part)
    if combined:
        envs["EXTRA_SGLANG_ARGS"] = combined
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
    with report.open(encoding="utf-8") as f:
        return json.load(f)


def _run_magpie(
    *,
    magpie_python: str,
    config_path: Path,
    output_dir: Path,
    timeout_sec: int,
    cwd: str,
) -> tuple[int, str, str]:
    """Blocking subprocess wrapper. Returns (rc, stdout, stderr)."""
    env = os.environ.copy()
    env["PATH"] = f"/opt/venv/bin:{env.get('PATH', '')}"
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
    magpie_python: str = _MAGPIE_PYTHON_DEFAULT,
    cwd: str = _MAGPIE_CWD_DEFAULT,
    variant_timeout_sec: int = _VARIANT_TIMEOUT_SEC_DEFAULT,
    keep_going_on_failure: bool = True,
) -> list[VariantResult]:
    """Execute every variant in ``grid`` once, in order.

    Returns the per-variant :class:`VariantResult` list (all attempts,
    including failed ones). Caller decides which variants are "winners".

    Synchronous subprocess call wrapped in ``asyncio.to_thread`` so the
    Conductor reactor isn't blocked.
    """
    results: list[VariantResult] = []
    for i, variant in enumerate(grid):
        slot = output_root / f"variant_{i:02d}_{_safe(variant.name)}"
        try:
            cfg_path = _build_variant_yaml(
                base_yaml_path, base_extra_args, variant, output_subdir=slot,
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

        if rc != 0:
            results.append(VariantResult(
                name=variant.name, extra_sglang_args=variant.extra_sglang_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                error=(stderr or stdout)[-2000:], note=variant.note,
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
                error="no benchmark_* workspace produced",
                note=variant.note,
            ))
            continue
        workspace = candidates[-1]
        report = _parse_report(workspace)
        if not report or not report.get("success"):
            results.append(VariantResult(
                name=variant.name, extra_sglang_args=variant.extra_sglang_args,
                extra_envs=dict(variant.extra_envs),
                status="failed",
                workspace=str(workspace),
                error="benchmark_report missing or success=false",
                note=variant.note,
            ))
            continue

        tput = (report.get("throughput") or {})
        latency = (report.get("latency") or {})
        ttft = latency.get("ttft", {}) or {}
        e2el = latency.get("e2el", {}) or {}
        results.append(VariantResult(
            name=variant.name, extra_sglang_args=variant.extra_sglang_args,
            extra_envs=dict(variant.extra_envs),
            status="succeeded",
            output_throughput=tput.get("output_throughput"),
            request_throughput=tput.get("request_throughput"),
            ttft_mean_ms=ttft.get("mean_ms"),
            e2el_mean_ms=e2el.get("mean_ms"),
            workspace=str(workspace),
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
]
