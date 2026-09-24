# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A scripted ``kth-qualify`` executable that speaks the KTH wire contract.

It runs as a real subprocess, so the tests exercise the same exit-code, file
and timeout paths a real KTH does. Each scenario starts from a well-formed,
correctly bound attestation and then applies the one defect under test.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any

REVISION = "4" * 40

_SCRIPT = """#!{python}
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, {tests_dir!r})
from kth_fakes import attest

scenario = json.loads(Path({scenario!r}).read_text())
args = sys.argv[1:]
request_path = Path(args[args.index("--request") + 1])
out = Path(args[args.index("--out") + 1])
request = json.loads(request_path.read_text())
with open({calls!r}, "a") as calls:
    calls.write(json.dumps(request) + "\\n")
for name in scenario.get("echo_env", []):
    print(name, os.environ.get(name, "<unset>"))
    print(name, os.environ.get(name, "<unset>"), file=sys.stderr)
sys.stdout.write(scenario.get("stdout", ""))
sys.stderr.write(scenario.get("stderr", ""))
sys.stdout.flush()
time.sleep(scenario.get("sleep", 0))
if "raw" in scenario:
    out.write_text(scenario["raw"])
elif "replay" in scenario:
    stale = json.loads(Path(scenario["replay"]).read_text())
    if scenario.get("rebind_request_id"):
        stale["request_id"] = request["request_id"]
    out.write_text(json.dumps(stale))
elif scenario.get("write", True):
    out.write_text(json.dumps(attest(request, scenario)))
sys.exit(scenario.get("exit", {{"Eligible for performance evaluation": 0, "Blocked": 2, "Inconclusive": 3}}[scenario["verdict"]]))
"""


class FakeKth:
    """One fake executable whose next answer is set with :meth:`answer`."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.scenario_path = root / "scenario.json"
        self.calls_path = root / "calls.jsonl"
        self.executable = root / "kth-qualify"
        self.executable.write_text(
            _SCRIPT.format(
                python=sys.executable,
                tests_dir=str(Path(__file__).parent),
                scenario=str(self.scenario_path),
                calls=str(self.calls_path),
            ),
            encoding="utf-8",
        )
        self.executable.chmod(self.executable.stat().st_mode | stat.S_IXUSR)
        self.answer("Eligible for performance evaluation")

    def answer(self, verdict: str, **scenario: Any) -> None:
        self.scenario_path.write_text(json.dumps({"verdict": verdict, **scenario}), encoding="utf-8")

    @property
    def requests(self) -> list[dict[str, Any]]:
        if not self.calls_path.exists():
            return []
        return [json.loads(line) for line in self.calls_path.read_text(encoding="utf-8").splitlines()]


def attest(request: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """A contract-conformant attestation for ``request``, then the scenario's defect."""
    attestation = _adaptive(request, scenario) if "envelope" in request else _reviewed(request, scenario)
    for dotted, value in scenario.get("set", {}).items():
        target = attestation
        *parents, leaf = dotted.split(".")
        for key in parents:
            target = target[key]
        target[leaf] = value
    for key in scenario.get("drop", []):
        attestation.pop(key, None)
    return attestation


def _reviewed(request: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    import base64
    import hashlib

    from hyperloom.orchestrator.kernel.kth_contract import reviewed_subject_digest

    candidate = request["candidate"]
    patch = base64.b64decode(candidate["patch_base64"])
    patch_sha = hashlib.sha256(patch).hexdigest()
    plan = {
        "plan_id": request["plan_id"],
        "plan_version": "1",
        "kernel_path": candidate["kernel_path"],
        "bind_source": scenario.get("bind_source", False),
        "allowed_paths": scenario.get("allowed_paths", []),
    }
    verdict = scenario["verdict"]
    return {
        "schema_version": request["schema_version"],
        "request_id": request["request_id"],
        "subject_digest": reviewed_subject_digest(
            base_commit=candidate["base_commit"],
            patch=patch,
            kernel_path=candidate["kernel_path"],
            qualification_plan=plan,
        ),
        "kth_sha": REVISION,
        "qualification_plan": plan,
        "candidate_identity": {
            "candidate_id": candidate["candidate_id"],
            "base_commit": candidate["base_commit"],
            "kernel_path": candidate["kernel_path"],
            "patch_sha256": patch_sha,
            "plan_id": request["plan_id"],
        },
        "execution_binding": {
            "base_commit": candidate["base_commit"],
            "patch_sha256": patch_sha,
            "bind_source": False,
            "artifact_sha256": None,
        },
        "execution_mode": "real",
        "hardware_identity": {"device": "cpu"},
        "verdict": verdict,
        "primary_detector": None if verdict.startswith("Eligible") else "OUTPUT_WRITTEN",
        "mandatory_oracle_coverage": {
            "complete": True,
            "missing_oracles": [],
            "executed_cases": ["case.0"],
        },
        "findings": [{"check_id": "REF", "status": "consistent"}],
        "unexplored_regions": [],
        "duration_s": 0.01,
        "replay": {},
        "repair_feedback": {} if verdict.startswith("Eligible") else {"primary_mechanism": {"meaning": "stale rows"}},
    }


def _adaptive(request: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    from hyperloom.orchestrator.kernel.kth_contract import (
        adaptive_subject_digest,
        envelope_digest,
        environment_identity,
    )

    envelope = request["envelope"]
    verdict = scenario["verdict"]
    trust_class = scenario.get(
        "trust_class", "fully_verified" if verdict.startswith("Eligible") else "partial_inferred"
    )
    binding = {
        "envelope_digest": envelope_digest(envelope),
        "patch_digest": envelope["patch_digest"],
        "base_commit": envelope["base_commit"],
        "binary_digest": "",
        "module_digest": "",
        "loaded_identity": "",
        "pre_patch_module_digest": "",
        "build_digest": "",
        "compiler": "",
        "autospec_digest": "sha256:" + "1" * 64,
        "resolved_digest": "sha256:" + "2" * 64,
        "plan_digest": "sha256:" + "3" * 64,
        "kth_revision": REVISION,
        "environment_identity": environment_identity(envelope["environment"]),
        "adapter_key": "",
    }
    binding["subject_digest"] = adaptive_subject_digest(binding)
    return {
        "schema_version": request["schema_version"],
        "request_id": request["request_id"],
        "subject_digest": binding["subject_digest"],
        "kth_revision": REVISION,
        "envelope_digest": binding["envelope_digest"],
        "autospec": {"digest": binding["autospec_digest"], "spec_trust_class": trust_class},
        "resolved_spec": {"digest": binding["resolved_digest"], "trust_class": trust_class},
        "evidence_plan": {
            "digest": binding["plan_digest"],
            "selected": [{"check_id": "REF"}],
            "uncovered": [] if trust_class == "fully_verified" else ["semantics.reduction_axis"],
            "required_unavailable": [],
        },
        "binding": binding,
        "verdict": verdict,
        "reason": "specification or mandatory obligation remains unresolved" if verdict == "Inconclusive" else verdict,
        "findings": [{"check_id": "REF", "status": "consistent"}],
        "autospec_uncertainty": trust_class != "fully_verified",
    }
