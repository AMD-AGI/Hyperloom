# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A two-workload SIKL schema, and a SIKL stand-in with the contract calls and the `validate` report the driver
reads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCHEMA_NAME = "gemm_n32_k64"
CASES = ("m=8", "m=16")

# The contract surface the driver and the api-params probe call, with the schema text as plain JSON.
FAKE_CONTRACT = """import hashlib, json
from pathlib import PurePosixPath


class _Param:
    def __init__(self, name):
        self.name = name


class _Narrowed:
    def __init__(self, raw):
        self.raw = raw

    def schema_text(self):
        return json.dumps(self.raw, indent=2) + "\\n"


class _Contract:
    api_params = [_Param(name) for name in ("a", "b", "trans_b", "bias", "out_dtype", "out")]

    def point_from_json(self, raw, axes):
        return axes

    def narrow(self, point, *, workloads, kernel):
        return _Narrowed(dict(self.raw, workloads=workloads, kernel=kernel))


class ContractRegistry:
    @staticmethod
    def default():
        return ContractRegistry()

    def for_schema_json(self, raw):
        contract = _Contract()
        contract.raw = raw
        return contract


class OpContract:
    @staticmethod
    def authored_parts(schema):
        return {"kernel": schema["kernel"], "workloads": schema["workloads"]}


def schema_relative_path(name, language, uuid):
    return PurePosixPath("impls", language, name, f"{uuid}.json")


def schema_uuid(name, parts):
    return hashlib.sha256(json.dumps([name, parts], sort_keys=True).encode()).hexdigest()[:16]
"""

# `python -m sikl validate`: passes a kernel whose source contains {passing!r} on every workload, finds a numeric
# mismatch on every workload otherwise, and counts its calls.
FAKE_VALIDATE = """import json, pathlib, sys
argv = sys.argv[1:]
out = pathlib.Path(argv[argv.index("--out") + 1])
candidate = json.loads(pathlib.Path(argv[-1]).read_text())
calls = pathlib.Path("validate_calls")
calls.write_text(str(int(calls.read_text()) + 1 if calls.exists() else 1))
passing = {passing!r} in candidate["kernel"]["sources"][0]["content"]
findings, records = [], []
for index, workload in enumerate(candidate["workloads"]):
    if not passing:
        findings.append({{"code": "numeric_mismatch", "axes": workload["axes"], "detail": "off"}})
    timing = {{"mode": "graph", "versus": "baseline", "ratio_lower_bound": 0.9,
               "candidate_milliseconds": [0.01 * (index + 1)] * 3, "versus_milliseconds": [0.009 * (index + 1)] * 3}}
    records.append({{"axes": workload["axes"], "worst_violation_fraction": 0.0, "timings": [timing]}})
out.mkdir(parents=True, exist_ok=True)
report = {{"stages": [{{"stage_name": "audit", "findings": []}},
                      {{"stage_name": "bench", "findings": findings, "workload_records": records}}]}}
(out / "report.json").write_text(json.dumps(report))
"""


def install_sikl(root: Path, *, passing: str = "torch.matmul") -> Path:
    """Plant the SIKL stand-in at ``root``, a checkout `--sikl-root` can name."""
    contract = root / "sikl" / "core" / "contract"
    contract.mkdir(parents=True, exist_ok=True)
    for package in (root / "sikl", root / "sikl" / "core", contract):
        (package / "__init__.py").write_text("", encoding="utf-8")
    for module in ("registry", "op", "locate"):
        (contract / f"{module}.py").write_text(FAKE_CONTRACT, encoding="utf-8")
    (root / "sikl" / "__main__.py").write_text(FAKE_VALIDATE.format(passing=passing), encoding="utf-8")
    return root


def schema_payload(name: str = SCHEMA_NAME, *, op_type: str = "gemm", rank: int = 2, sources: list | None = None):
    batch = ["batch"] if rank == 3 else []
    return {
        "schema_version": 6,
        "name": name,
        "contract_id": f"{op_type}@1.0",
        "description": "16-bit gemm.",
        "op_type": op_type,
        "axes": {
            "m": {"type": "var", "description": "Tokens."},
            "n": {"type": "const", "value": 32},
            "k": {"type": "const", "value": 64},
        },
        "attrs": {"trans_b": True, "has_bias": False},
        "inputs": {
            "a": {"shape": [*batch, "m", 64], "dtype": "bfloat16"},
            "b": {"shape": [*batch, 32, 64], "dtype": "bfloat16"},
        },
        "outputs": {"out": {"shape": [*batch, "m", 32], "dtype": "bfloat16"}},
        "workloads": [{"axes": {"m": int(case.split("=")[1])}} for case in CASES],
        "compare": {"max_violation_fraction": 0.0, "determinism": "bitwise", "element_order": "fixed"},
        "kernel": {
            "language": "flydsl",
            "targets": [{"arch": "gfx950", "hardware": "MI355X"}],
            "entry_point": "kernel.py::run",
            "sources": sources or [],
        },
    }


def write_schema(sikl_root: Path, payload: dict, uuid: str = "0123456789abcdef") -> Path:
    path = sikl_root / "sikl" / "ops" / payload["op_type"] / "impls" / "flydsl" / payload["name"] / f"{uuid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def sikl_root(tmp_path: Path) -> Path:
    root = install_sikl(tmp_path / "sikl_checkout")
    write_schema(root, schema_payload())
    return root
