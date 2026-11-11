# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Serve a measured matmul_ogs flag table through a startup import hook."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

FLAGS_ENV = "HYPERLOOM_MATMUL_OGS_FLAGS"
_TARGET = "triton_kernels.matmul_ogs_details.opt_flags"
_SHA256_HEX_LENGTH = 64
_TILE_64 = 64
_FOUR = 4
_EIGHT = 8
_ROUTING_ARG_INDEX = _EIGHT
_ROUTING_ARG_NAME = "routing_data"
_RDNA4_VERSION = _FOUR
_FLAG_FIELDS = frozenset({"block_m", "block_n", "block_k", "num_warps", "num_stages", "split_k", "xcd_swizzle"})
_ALLOWED_FLAGS = {
    "block_m": {16, 32, _TILE_64},
    "block_n": {_TILE_64, 128},
    "block_k": {128, 256, 512},
    "num_warps": {2, _FOUR, _EIGHT},
    "num_stages": {1, 2},
    "split_k": {1},
    "xcd_swizzle": {0, _EIGHT},
}


def _validate_row(row: Any) -> None:
    if not isinstance(row, dict) or not isinstance(row.get("flags"), dict):
        raise ValueError("malformed matmul_ogs flag row")
    flags = row["flags"]
    if set(flags) != _FLAG_FIELDS or any(
        type(value) is not int or value not in _ALLOWED_FLAGS[key] for key, value in flags.items()
    ):
        raise ValueError("matmul_ogs flag row has unsupported values")
    if not all(type(row.get(key)) is int and row[key] > 0 for key in ("m", "n", "k")):
        raise ValueError("matmul_ogs flag row has invalid dimensions")


def _validate_unique_rows(rows: list[dict[str, Any]]) -> None:
    keys = {(row["m"], row["n"], row["k"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("matmul_ogs flag table has duplicate routed shapes")


def _read_table(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    table = json.loads(raw)
    if not isinstance(table, dict) or table.get("schema_version") != 1 or table.get("gpu_target") != "gfx1201":
        raise ValueError("unsupported matmul_ogs flag table")
    if not isinstance(table.get("gpu_name"), str) or not table["gpu_name"].strip():
        raise ValueError("matmul_ogs flag table lacks a GPU name")
    digest = table.get("opt_flags_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != _SHA256_HEX_LENGTH
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("matmul_ogs flag table lacks a source digest")
    if not all(type(table.get(field)) is int and table[field] > 0 for field in ("experts", "top_k")):
        raise ValueError("matmul_ogs flag table lacks expert routing")
    if table["top_k"] > table["experts"]:
        raise ValueError("matmul_ogs flag table has invalid expert routing")
    rows = table.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("matmul_ogs flag table has no measured rows")
    for row in rows:
        _validate_row(row)
        if row["m"] % table["top_k"] or row.get("tokens") != row["m"] // table["top_k"]:
            raise ValueError("matmul_ogs flag row disagrees with expert routing")
    _validate_unique_rows(rows)
    return table, hashlib.sha256(raw).hexdigest()


def _vendor_triton_kernels_path() -> None:
    """Make vLLM's bundled triton_kernels visible before its imports begin."""
    if importlib.util.find_spec("triton_kernels") is not None:
        return
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vLLM is unavailable for the matmul_ogs startup hook")
    third_party = Path(next(iter(spec.submodule_search_locations))) / "third_party"
    if not (third_party / "triton_kernels" / "matmul_ogs_details" / "opt_flags.py").is_file():
        raise RuntimeError("vLLM does not bundle the required triton_kernels opt_flags API")
    sys.path.insert(1, str(third_party))


def _arg(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str) -> Any:
    return args[index] if len(args) > index else kwargs.get(name)


def _matching_row(table: dict[str, Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any] | None:
    m = _arg(args, kwargs, 5, "m")
    n = _arg(args, kwargs, 6, "n")
    k = _arg(args, kwargs, 7, "k")
    if not all(isinstance(value, int) and value > 0 for value in (m, n, k)):
        return None
    return next((row for row in table["rows"] if (row["m"], row["n"], row["k"]) == (m, n, k)), None)


def _is_supported_call(module: ModuleType, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    precision = _arg(args, kwargs, 3, "precision_config")
    routing = _arg(args, kwargs, _ROUTING_ARG_INDEX, _ROUTING_ARG_NAME)
    return (
        _arg(args, kwargs, 0, "out_dtype") == module.torch.bfloat16
        and _arg(args, kwargs, 1, "lhs_dtype") == module.torch.bfloat16
        and _arg(args, kwargs, 2, "rhs_dtype") == module.FP4
        and getattr(precision, "weight_scale", None) is not None
        and getattr(precision, "act_scale", None) is None
        and getattr(precision, "out_scale", None) is None
        and not getattr(precision, "enforce_bitwise_invariance", False)
        and _arg(args, kwargs, _FOUR, "batch_size") == 1
        and getattr(routing, "expt_hist", None) is not None
        and not _arg(args, kwargs, 12, "x_transpose")
        and not _arg(args, kwargs, 13, "has_y_acc_in")
        and _arg(args, kwargs, 14, "block_k") is None
        and not getattr(module, "_opt_flags_constraints", {})
        and getattr(module, "_opt_flags", None) is None
        and module.get_rdna_version() == _RDNA4_VERSION
    )


def _install_module(module: ModuleType, table: dict[str, Any], table_digest: str) -> None:
    source_digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if source_digest != table["opt_flags_sha256"]:
        raise RuntimeError("matmul_ogs opt_flags source differs from the tuned source")
    original = module.make_opt_flags
    reported: set[tuple[int, int, int]] = set()

    def tuned_make_opt_flags(*args: Any, **kwargs: Any) -> Any:
        default = original(*args, **kwargs)
        if not _is_supported_call(module, args, kwargs) or default.is_persistent or default.split_k != 1:
            return default
        routing = _arg(args, kwargs, _ROUTING_ARG_INDEX, _ROUTING_ARG_NAME)
        if routing.n_expts_tot != table["experts"] or routing.n_expts_act != table["top_k"]:
            return default
        row = _matching_row(table, args, kwargs)
        if row is None:
            return default
        if module.torch.cuda.get_device_name() != table["gpu_name"]:
            return default
        tuned = replace(default, **row["flags"])
        key = (row["m"], row["n"], row["k"])
        if key not in reported:
            print(
                f"HL_MATMUL_OGS_FLAGS_APPLIED sha256={table_digest} m={key[0]} n={key[1]} k={key[2]}",
                flush=True,
            )
            reported.add(key)
        return tuned

    module.make_opt_flags = tuned_make_opt_flags


class _Loader(importlib.abc.Loader):
    def __init__(self, original: Any, table: dict[str, Any], digest: str) -> None:
        self.original = original
        self.table = table
        self.digest = digest

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self.original, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module: ModuleType) -> None:
        self.original.exec_module(module)
        _install_module(module, self.table, self.digest)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, table: dict[str, Any], digest: str) -> None:
        self.table = table
        self.digest = digest

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname != _TARGET:
            return None
        index = sys.meta_path.index(self)
        sys.meta_path.pop(index)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(index, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, self.table, self.digest)
        sys.meta_path.remove(self)
        return spec


def install() -> None:
    """Install only when a tuning table was explicitly delivered to serving."""
    path = os.environ.get(FLAGS_ENV)
    if not path:
        return
    table, digest = _read_table(Path(path))
    _vendor_triton_kernels_path()
    loaded = sys.modules.get(_TARGET)
    if loaded is not None:
        _install_module(loaded, table, digest)
    else:
        sys.meta_path.insert(0, _Finder(table, digest))


def chain_sitecustomize(current_file: str) -> None:
    """Run the next startup hook when another candidate already supplied one."""
    current = Path(current_file).resolve()
    for root in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not root:
            continue
        candidate = Path(root) / "sitecustomize.py"
        if not candidate.is_file() or candidate.resolve() == current:
            continue
        spec = importlib.util.spec_from_file_location("_hyperloom_prior_sitecustomize", candidate)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return
