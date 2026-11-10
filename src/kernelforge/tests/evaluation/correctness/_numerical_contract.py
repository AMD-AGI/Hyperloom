# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Numerical-validation tasks and the evidence a measurement produces, shared by the correctness and loop tests."""

from __future__ import annotations

import copy
import sys

import yaml


def contract():
    return {
        "schema_version": 1,
        "repetitions": 5,
        "cases": {
            "moe/output/graph": {"max_oracle_error": 10 ** (-30 / 20), "max_error_ratio": 1.25, "error_floor": 1e-6}
        },
    }


def evidence(*, candidate_oracle_db=45, candidate_repeat_db=45):
    source = {"finite": True, "oracle_errors": [10 ** (-45 / 20)] * 5, "repeat_errors": [10 ** (-45 / 20)] * 4}
    candidate = {
        "finite": True,
        "oracle_errors": [10 ** (-candidate_oracle_db / 20)] * 5,
        "repeat_errors": [10 ** (-candidate_repeat_db / 20)] * 4,
    }
    return {
        "schema_version": 1,
        "request_id": "fresh",
        "cases": [
            {
                "id": "moe/output/graph",
                "source_before": copy.deepcopy(source),
                "candidate": candidate,
                "source_after": copy.deepcopy(source),
            }
        ],
    }


def _task(tmp_path, record=None, *, declared=True):
    task = {
        "compile_command": [f'"{sys.executable}" compile.py'],
        "correctness_command": [f'"{sys.executable}" driver.py'],
    }
    if declared:
        task["numerical_validation"] = contract()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(task))
    (tmp_path / "compile.py").write_text("pass\n")
    text = "print('SNR: 31 dB')\n"
    if record is not None:
        text += (
            "import json,os\nrecord="
            + repr(record)
            + "\nrecord['request_id']=os.environ['FORGE_NUMERICAL_REQUEST']\nprint('__FORGE_NUMERICAL__'+json.dumps(record))\n"
        )
    (tmp_path / "driver.py").write_text(text)
