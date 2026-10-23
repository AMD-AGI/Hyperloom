# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measurement surface: the inputs that decide what a test measures (design 4.20).

A specialist runs in the same container as the application and can write to shared storage, so anything that shapes
the measurement must not come from a place it can change unnoticed: the workload shape, the evaluation task and its
limit, the benchmark script, and the model's configuration and tokenizer files. The values for these come from the
canonical measurement configuration, are hashed into ``surface_sha256``, and every test's YAML is held to them:

* a request that sets one of these through its launch environment is refused before anything is generated;
* the YAML generated for a test is compared with the canonical value, and a difference refuses the request;
* a runner takes the same fields from the YAML it actually executes and signs the hash into its result.

The concurrency level and ``RUN_EVAL`` are decided by the test request, so they stay out of the hash. The number of
prompts and warmups follow from the concurrency, so they are compared per point against the canonical rule rather
than hashed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.outside_facts import (
    ENV_CONC,
    ENV_ISL,
    ENV_MAGPIE_EVAL_LIMIT,
    ENV_MAGPIE_EVAL_TASKS,
    ENV_MAGPIE_EVAL_TOKENIZED_REQUESTS,
    ENV_NUM_PROMPTS,
    ENV_NUM_WARMUPS,
    ENV_OSL,
    ENV_RANDOM_RANGE_RATIO,
    ENV_RUN_EVAL,
)
from hyperloom.contracts.recipe import RecipeError

#: ``benchmark`` keys that fix what is measured.
SURFACE_BENCH_KEYS: tuple[str, ...] = ("framework", "model", "benchmark_script")
#: ``benchmark.envs`` keys that fix the workload shape and the evaluation.
SURFACE_ENV_KEYS: tuple[str, ...] = (
    ENV_ISL,
    ENV_OSL,
    ENV_RANDOM_RANGE_RATIO,
    ENV_MAGPIE_EVAL_TASKS,
    ENV_MAGPIE_EVAL_LIMIT,
    ENV_MAGPIE_EVAL_TOKENIZED_REQUESTS,
)
#: Request counts that follow from the concurrency level; compared per point, not hashed.
POINT_ENV_KEYS: tuple[str, ...] = (ENV_NUM_PROMPTS, ENV_NUM_WARMUPS)
#: Decided by the test request; recorded beside the hash, never part of it.
REQUEST_ENV_KEYS: tuple[str, ...] = (ENV_CONC, ENV_RUN_EVAL)
#: Model files whose content shapes the measurement: configuration and tokenizer, never weights.
MODEL_FILE_NAMES: frozenset[str] = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "chat_template.jinja",
    }
)
_SCRIPT_FIELD = "benchmark_script_sha256"
_MODEL_FILES_FIELD = "model_files"


class SurfaceViolation(RecipeError):
    """A test would not measure the canonical workload or evaluation; the request is refused."""


def _text(value: object) -> str | None:
    """One spelling per value, so ``1`` and ``1.0`` and ``"1"`` compare equal."""

    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_files(model_dir: Path) -> dict[str, str]:
    """``{relative path: sha256}`` of the configuration and tokenizer files in a model directory."""

    return {
        path.relative_to(model_dir).as_posix(): _sha256_file(path)
        for path in sorted(model_dir.rglob("*"))
        if path.name in MODEL_FILE_NAMES and path.is_file()
    }


@dataclass(frozen=True)
class MeasurementSurface:
    """The surface fields of one configuration, as text; :attr:`sha256` is ``surface_sha256``."""

    fields: Mapping[str, str | None]

    @property
    def sha256(self) -> str:
        document = json.dumps(dict(sorted(self.fields.items())), separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(document.encode()).hexdigest()

    def differences(self, other: MeasurementSurface) -> dict[str, tuple[str | None, str | None]]:
        names = sorted({*self.fields, *other.fields})
        return {
            n: (self.fields.get(n), other.fields.get(n)) for n in names if self.fields.get(n) != other.fields.get(n)
        }


def _benchmark(config: Mapping[str, Any]) -> Mapping[str, Any]:
    bench = config.get("benchmark")
    return bench if isinstance(bench, Mapping) else {}


def benchmark_envs(config: Mapping[str, Any]) -> Mapping[str, Any]:
    envs = _benchmark(config).get("envs")
    return envs if isinstance(envs, Mapping) else {}


def extract_surface(config: Mapping[str, Any]) -> MeasurementSurface:
    """The surface fields of a materialized Magpie configuration, with the script and model files hashed.

    The model files are those of the directory the server loads: an AgentX configuration names it in ``envs.MODEL_PATH``
    and keeps the model's identity in ``benchmark.model``; any other names the directory in ``benchmark.model``.
    """

    bench, envs = _benchmark(config), benchmark_envs(config)
    fields: dict[str, str | None] = {f"benchmark.{key}": _text(bench.get(key)) for key in SURFACE_BENCH_KEYS}
    fields.update({f"envs.{key}": _text(envs.get(key)) for key in SURFACE_ENV_KEYS})
    script, root = _text(bench.get("benchmark_script")), _text(bench.get("inferencex_path"))
    script_path = Path(root) / "benchmarks" / script if script and root else None
    fields[_SCRIPT_FIELD] = _sha256_file(script_path) if script_path and script_path.is_file() else None
    model = _text(envs.get("MODEL_PATH")) or _text(bench.get("model"))
    model_files = hash_model_files(Path(model)) if model and Path(model).is_dir() else None
    fields[_MODEL_FILES_FIELD] = json.dumps(model_files, sort_keys=True) if model_files is not None else None
    return MeasurementSurface(fields)


def point_values(config: Mapping[str, Any]) -> dict[str, str | None]:
    """The request counts of one point, which follow from its concurrency."""

    envs = benchmark_envs(config)
    return {key: _text(envs.get(key)) for key in POINT_ENV_KEYS}


def check_launch_envs(names: Iterable[str]) -> None:
    """Refuse a launch environment that sets a workload, evaluation or request-decided variable."""

    guarded = {*SURFACE_ENV_KEYS, *POINT_ENV_KEYS, *REQUEST_ENV_KEYS}
    offending = sorted(name for name in names if name.upper() in guarded)
    if offending:
        raise SurfaceViolation(
            f"the launch environment sets {', '.join(offending)}: workload and evaluation settings belong to the "
            "canonical measurement configuration, and concurrency and RUN_EVAL to the test request"
        )


def check_surface(config: Mapping[str, Any], canonical: MeasurementSurface, *, ignore: Collection[str] = ()) -> None:
    """Refuse a configuration whose surface differs from the canonical one, naming each differing field.

    ``ignore`` names fields that may differ (a profile test lowers its output length so the capture fits).
    """

    differences = {
        name: pair for name, pair in canonical.differences(extract_surface(config)).items() if name not in ignore
    }
    if differences:
        detail = "; ".join(f"{name}: canonical {want!r}, test {got!r}" for name, (want, got) in differences.items())
        raise SurfaceViolation(f"the test changes the measurement surface ({detail})")


def check_point(
    config: Mapping[str, Any], *, concurrency: int, accuracy: bool, reference: Mapping[str, str | None]
) -> None:
    """Hold one point to its request: concurrency, ``RUN_EVAL`` and the canonical request counts for it."""

    envs = benchmark_envs(config)
    expected = {ENV_CONC: str(concurrency), ENV_RUN_EVAL: "true" if accuracy else "false", **reference}
    wrong = {key: (want, _text(envs.get(key))) for key, want in expected.items() if _text(envs.get(key)) != want}
    if wrong:
        detail = "; ".join(f"{key}: expected {want!r}, point has {got!r}" for key, (want, got) in wrong.items())
        raise SurfaceViolation(f"the generated point does not match its request ({detail})")


__all__ = [
    "MODEL_FILE_NAMES",
    "POINT_ENV_KEYS",
    "REQUEST_ENV_KEYS",
    "SURFACE_BENCH_KEYS",
    "SURFACE_ENV_KEYS",
    "MeasurementSurface",
    "SurfaceViolation",
    "benchmark_envs",
    "check_launch_envs",
    "check_point",
    "check_surface",
    "extract_surface",
    "hash_model_files",
    "point_values",
]
