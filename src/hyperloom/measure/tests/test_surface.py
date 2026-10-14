# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from hyperloom.measure.surface import (
    POINT_ENV_KEYS,
    REQUEST_ENV_KEYS,
    SURFACE_ENV_KEYS,
    SurfaceViolation,
    check_launch_envs,
    check_point,
    check_surface,
    extract_surface,
    hash_model_files,
    point_values,
)


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"hidden": 1}')
    (root / "tokenizer.json").write_text("{}")
    (root / "model-00001.safetensors").write_bytes(b"weights")
    return root


@pytest.fixture()
def inferencex(tmp_path: Path) -> Path:
    scripts = tmp_path / "ix" / "benchmarks"
    scripts.mkdir(parents=True)
    (scripts / "sglang_mi355x.sh").write_text("run --isl $ISL\n")
    return tmp_path / "ix"


@pytest.fixture()
def config(model_dir: Path, inferencex: Path) -> dict:
    return {
        "benchmark": {
            "framework": "sglang",
            "model": str(model_dir),
            "benchmark_script": "sglang_mi355x.sh",
            "inferencex_path": str(inferencex),
            "envs": {
                "ISL": 1024,
                "OSL": 1024,
                "RANDOM_RANGE_RATIO": 1.0,
                "MAGPIE_EVAL_LIMIT": 250,
                "CONC": "64",
                "RUN_EVAL": "true",
                "NUM_PROMPTS": 320,
                "NUM_WARMUPS": 8,
            },
        }
    }


def changed(config: dict, **overrides) -> dict:
    other = copy.deepcopy(config)
    for key, value in overrides.items():
        other["benchmark"]["envs"][key] = value
    return other


def test_surface_hash_is_stable_and_ignores_what_the_request_decides(config):
    baseline = extract_surface(config).sha256

    assert extract_surface(copy.deepcopy(config)).sha256 == baseline
    assert extract_surface(changed(config, CONC="128", RUN_EVAL="false", NUM_PROMPTS=640)).sha256 == baseline
    assert extract_surface(changed(config, RANDOM_RANGE_RATIO=1)).sha256 == baseline


@pytest.mark.parametrize("name", [*SURFACE_ENV_KEYS])
def test_every_workload_and_eval_field_changes_the_hash(config, name):
    baseline = extract_surface(config)
    value = "7" if name != "MAGPIE_EVAL_TASKS" else "easy_task"

    assert extract_surface(changed(config, **{name: value})).sha256 != baseline.sha256


def test_check_surface_names_the_field_that_differs(config):
    canonical = extract_surface(config)

    check_surface(changed(config, CONC="128"), canonical)
    with pytest.raises(SurfaceViolation, match=r"envs\.MAGPIE_EVAL_LIMIT: canonical '250', test '5'"):
        check_surface(changed(config, MAGPIE_EVAL_LIMIT=5), canonical)
    with pytest.raises(SurfaceViolation, match=r"envs\.OSL: canonical '1024', test '16'"):
        check_surface(changed(config, OSL=16), canonical)


def test_benchmark_script_content_and_model_files_are_part_of_the_surface(config, inferencex, model_dir):
    canonical = extract_surface(config)

    (inferencex / "benchmarks" / "sglang_mi355x.sh").write_text("run --isl 1\n")
    with pytest.raises(SurfaceViolation, match="benchmark_script_sha256"):
        check_surface(config, canonical)

    (inferencex / "benchmarks" / "sglang_mi355x.sh").write_text("run --isl $ISL\n")
    check_surface(config, canonical)
    (model_dir / "tokenizer.json").write_text('{"chat_template": "x"}')
    with pytest.raises(SurfaceViolation, match="model_files"):
        check_surface(config, canonical)
    (model_dir / "model-00001.safetensors").write_bytes(b"other weights")


def test_other_script_name_is_a_surface_change(config):
    other = copy.deepcopy(config)
    other["benchmark"]["benchmark_script"] = "sglang_easy.sh"

    with pytest.raises(SurfaceViolation, match=r"benchmark\.benchmark_script"):
        check_surface(other, extract_surface(config))


def test_model_hashes_cover_configuration_and_tokenizer_but_not_weights(model_dir):
    assert set(hash_model_files(model_dir)) == {"config.json", "tokenizer.json"}


@pytest.mark.parametrize("name", [*SURFACE_ENV_KEYS, *POINT_ENV_KEYS, *REQUEST_ENV_KEYS])
def test_launch_environment_may_not_set_guarded_variables(name):
    with pytest.raises(SurfaceViolation, match=name.lower()):
        check_launch_envs(["HSA_X", name.lower()])  # the check is case-insensitive and reports the name as given


def test_launch_environment_with_ordinary_variables_passes():
    check_launch_envs(["HSA_FORCE_FINE_GRAIN_PCIE", "AITER_CONFIG_GEMM_BF16"])


def test_point_is_held_to_concurrency_run_eval_and_canonical_counts(config):
    reference = point_values(config)
    check_point(config, concurrency=64, accuracy=True, reference=reference)

    with pytest.raises(SurfaceViolation, match=r"CONC: expected '32', point has '64'"):
        check_point(config, concurrency=32, accuracy=True, reference=reference)
    with pytest.raises(SurfaceViolation, match=r"RUN_EVAL: expected 'false', point has 'true'"):
        check_point(config, concurrency=64, accuracy=False, reference=reference)
    with pytest.raises(SurfaceViolation, match=r"NUM_PROMPTS: expected '320', point has '8'"):
        check_point(changed(config, NUM_PROMPTS=8), concurrency=64, accuracy=True, reference=reference)
