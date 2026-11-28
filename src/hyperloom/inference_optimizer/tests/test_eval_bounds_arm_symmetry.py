# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Both arms must truncate at the same place."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _accuracy_gate
from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    materialized_run_eval_disabled,
)


def _write_config(path: Path, **envs) -> Path:
    """Write a materialized benchmark YAML carrying ``envs`` in benchmark.envs."""
    base_envs = {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256}
    base_envs.update(envs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "/path/models/Qwen-Qwen3-8B",
                    "precision": "bf16",
                    "run_mode": "local",
                    "benchmark_script": "sglang.sh",
                    "envs": base_envs,
                    "timeout_seconds": 600,
                    "profiler": {
                        "torch_profiler": {"enabled": False},
                        "system_profiler": {"enabled": False},
                        "tracelens": {"enabled": False},
                    },
                    "gpu_selection": {"auto": False},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


# Grid arm: assert the bounds landed, on the same terms as the baseline arm


# The shared RUN_EVAL reader both arms now key off


@pytest.mark.parametrize("spelling", ["false", "0", "no", "off", ""])
def test_run_eval_disabled_recognizes_every_falsey_spelling(tmp_path, spelling):
    cfg = _write_config(tmp_path / f"cfg_{spelling or 'empty'}.yaml", RUN_EVAL=spelling)
    assert materialized_run_eval_disabled(cfg) is True


def test_run_eval_absent_reads_as_enabled(tmp_path):
    """Matches the materialize default, which is "true" when the key is absent."""
    assert materialized_run_eval_disabled(_write_config(tmp_path / "cfg.yaml")) is False


def test_an_unreadable_config_reads_as_eval_enabled(tmp_path):
    """Fail closed: an unreadable config must not silently skip the eval guards."""
    assert materialized_run_eval_disabled(tmp_path / "does_not_exist.yaml") is False

    broken = tmp_path / "broken.yaml"
    broken.write_text("benchmark: [unclosed\n", encoding="utf-8")
    assert materialized_run_eval_disabled(broken) is False


def test_the_shared_reader_lives_in_a_module_every_arm_can_import():
    """All three arms reach one reader, none of them through an import cycle."""
    import ast
    from pathlib import Path as _Path

    executors = _Path(_accuracy_gate.__file__).parent

    def siblings(module: str) -> set[str]:
        tree = ast.parse((executors / f"{module.replace('.', '/')}.py").read_text(encoding="utf-8"))
        package = module.split(".")[:-1]
        found = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level and node.module):
                continue
            if node.level - 1 > len(package):
                continue
            target = ".".join([*package[: len(package) - node.level + 1], *node.module.split(".")])
            if (executors / f"{target.replace('.', '/')}.py").is_file():
                found.add(target)
        return found

    reached, frontier = set(), siblings("_accuracy_gate")
    while frontier:
        reached |= frontier
        frontier = {module for name in frontier for module in siblings(name)} - reached
    assert "_accuracy_gate" not in reached, f"_accuracy_gate is imported back through {sorted(reached)}"

    from hyperloom.orchestrator.actions.executors import baseline

    assert baseline.materialized_run_eval_disabled is materialized_run_eval_disabled
