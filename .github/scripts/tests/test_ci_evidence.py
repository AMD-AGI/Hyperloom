# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Portable self-tests for the isolated CI evidence tools, not project tests."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import os
import platform
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ci_evidence as evidence
import ci_evidence_gate as gate


class TestMeasurement(unittest.TestCase):
    def test_shell_selection_matches_source_literal_parameter_ids(self):
        source = Path(__file__).resolve().parents[3] / evidence.SHELL_FILE
        functions = {
            node.name: node
            for node in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertEqual(len(evidence.SHELL_NODEIDS), 17)
        names = {node.split("::")[1].split("[")[0] for node in evidence.SHELL_NODEIDS}
        expected = []
        for name in names:
            groups = []
            for decorator in reversed(functions[name].decorator_list):
                self.assertEqual(ast.unparse(decorator.func), "pytest.mark.parametrize")
                self.assertFalse(decorator.keywords)
                values = ast.literal_eval(decorator.args[1])
                groups.append(["-".join(map(str, row)) if isinstance(row, tuple) else str(row) for row in values])
            suffixes = [f"[{'-'.join(parts)}]" for parts in itertools.product(*groups)] if groups else [""]
            expected.extend(f"{evidence.SHELL_FILE}::{name}{suffix}" for suffix in suffixes)
        self.assertCountEqual(evidence.SHELL_NODEIDS, expected)

    def test_diagnostics_ignore_order_and_root_but_preserve_meaning_and_duplicates(self):
        root = Path(__file__).resolve().parents[3]
        other_root = root / "other-worktree"
        diagnostic = dict(path=str(root / "src" / "sample.py"), line=1, column=2, endLine=1, endColumn=3)
        diagnostic.update({"message-id": "E0602", "symbol": "undefined-variable", "message": "Unknown 'name'"})
        normalized = evidence.normalize_diagnostics([diagnostic, diagnostic, dict(diagnostic, line=9)], root)
        relocated = dict(diagnostic, path=str(other_root / "src" / "sample.py"))
        self.assertEqual(
            normalized, evidence.normalize_diagnostics([dict(relocated, line=9), relocated, relocated], other_root)
        )
        self.assertEqual(len(normalized), 3)
        expected = ("src/sample.py", 1, 2, 1, 3, "E0602", "undefined-variable", "Unknown 'name'")
        self.assertEqual(normalized.count(expected), 2)
        changes = {"message": "Changed", "line": 2, "endColumn": 4, "message-id": "E0100", "symbol": "other"}
        for key, value in changes.items():
            with self.subTest(field=key):
                actual = evidence.normalize_diagnostics([dict(diagnostic, **{key: value})], root)
                self.assertNotEqual([expected], actual)
        self.assertNotEqual(normalized, evidence.normalize_diagnostics([diagnostic, dict(diagnostic, line=9)], root))
        for malformed in ({}, [dict(diagnostic, line=None)]):
            with self.assertRaises(ValueError):
                evidence.normalize_diagnostics(malformed, root)

    def test_pytest_requires_every_node_and_successful_phase(self):
        node = "tests/test_sample.py::test_ok"
        data = {
            "schema_version": 1,
            "nodeids": [node],
            "exitstatus": 0,
            "phases": [
                {"nodeid": node, "when": phase, "outcome": "passed", "duration": 0.01}
                for phase in ("setup", "call", "teardown")
            ],
        }
        evidence.validate_pytest_result(data, [node])
        invalid = [{}, dict(data, nodeids=[]), dict(data, exitstatus=1), dict(data, phases=data["phases"][:-1])]
        for field, value in (
            ("outcome", "skipped"),
            ("outcome", "failed"),
            ("duration", -1),
            ("duration", float("nan")),
        ):
            sample = json.loads(json.dumps(data))
            sample["phases"][1][field] = value
            invalid.append(sample)
        for sample in invalid:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                evidence.validate_pytest_result(sample, [node])

    def test_pylint_exit_status_cannot_hide_crashes_or_lost_diagnostics(self):
        diagnostic = ("src/sample.py", 1, 0, 1, 3, "E0602", "undefined-variable", "Undefined variable")
        evidence.validate_pylint_result(0, [])
        evidence.validate_pylint_result(2, [diagnostic])
        for code, messages in ((1, []), (32, []), (2, []), (0, [diagnostic])):
            with self.subTest(code=code, messages=messages), self.assertRaises(ValueError):
                evidence.validate_pylint_result(code, messages)

    def test_constraints_exclude_editable_project_and_detect_version_drift(self):
        project = Mock(metadata={"Name": "Hyperloom-Inference_Optimizer"}, version="1.2.3")
        project.read_text.return_value = '{"url":"file:///local/worktree","dir_info":{"editable":true}}'
        dependency = Mock(metadata={"Name": "Example_Package"}, version="2.0")
        dependency.read_text.return_value = None
        with patch.object(evidence.metadata, "distributions", return_value=[project, dependency]):
            packages = evidence.installed_packages()
            self.assertEqual(packages, {"example-package": "2.0"})
            self.assertEqual(evidence.constraints_text(packages), "example-package==2.0\n")
            dependency.read_text.return_value = project.read_text.return_value
            with self.assertRaises(ValueError):
                evidence.installed_packages()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            data = dict(schema_version=1, baseline=evidence.BASELINE, packages=packages)
            data["python_version"] = platform.python_version()
            manifest.write_text(json.dumps(data), encoding="utf-8")
            manifest.with_name("constraints.txt").write_text(evidence.constraints_text(packages), encoding="utf-8")
            with patch.object(evidence, "installed_packages", return_value=packages) as installed:
                with patch.object(evidence, "check_source_guard"):
                    evidence.verify(manifest)
                    installed.return_value = {"example-package": "2.1"}
                    with self.assertRaises(ValueError):
                        evidence.verify(manifest)

    def test_nonzero_subprocess_is_recorded_with_complete_json_and_logs(self):
        command = [sys.executable, "-c", "print('intentional failure'); raise SystemExit(7)"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "sample"
            result = evidence.measure_command(command, repo=root, output=output, env=dict(os.environ), timeout=10)
            self.assertEqual(json.loads((output / "result.json").read_text(encoding="utf-8")), result)
            self.assertEqual(result["schema_version"], 1)
            self.assertEqual(result["command"], command)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["returncode"], 7)
            self.assertGreaterEqual(result["elapsed_seconds"], 0)
            self.assertIn("peak_tree_rss_bytes", result)
            self.assertIn("intentional failure", (output / "stdout.txt").read_text(encoding="utf-8"))
            self.assertTrue((output / "stderr.txt").is_file())


class TestGate(unittest.TestCase):
    def setUp(self):
        self.expected = dict(run="true", run_scope="full", tasks="", reuse="", base_version="1.2.3", ci_version=None)
        self.start = datetime(2026, 9, 21, 0, 0, 30, tzinfo=timezone.utc)
        self.end = datetime(2026, 9, 21, 0, 1, 5, tzinfo=timezone.utc)
        self.result = dict(self.expected, ci_version="1.2.3.dev202609210001+ci")

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML unavailable locally; bootstrap tools require it")
    def test_extracts_exact_script_and_rejects_nonunique_step(self):
        script = "set -euo pipefail\nprintf 'run=true\\n'\n"
        document = {"jobs": {"resolve": {"steps": [{"id": "decide", "run": script}]}}}
        self.assertEqual(gate.extract_decide_script(json.dumps(document)), script)
        for steps in ([], [{"id": "decide", "run": script}] * 2):
            document["jobs"]["resolve"]["steps"] = steps
            with self.subTest(steps=steps), self.assertRaises(gate.GateValidationError):
                gate.extract_decide_script(json.dumps(document))

    def test_fixture_expectations_pin_negative_controls_and_full_scope(self):
        cases = {case.name: case for case in gate.CASES}
        self.assertEqual(len(cases), len(gate.CASES))
        for case in gate.CASES:
            for path in case.changes:
                self.assertNotIn("\\", path)
        for name in ("pyproject_comment", "changelog_url", "unrelated_file"):
            self.assertEqual(cases[name].expected, dict(self.expected, run="false", run_scope="none"))
        for name in ("version_bump", "version_and_ci", "manual_default"):
            self.assertEqual(cases[name].expected, self.expected)
        tasks = "baremetal-vllm-3h,baremetal-sglang-3h,docker-vllm-3h,docker-sglang-3h"
        for name in ("ci_workflow", "ci_script", "ci_prompt"):
            self.assertEqual(cases[name].expected, dict(self.expected, run_scope="scripts-only", tasks=tasks))

    def test_output_parser_rejects_duplicate_missing_and_extra_fields(self):
        text = "".join(f"{key}={value}\n" for key, value in self.result.items())
        self.assertEqual(gate.parse_outputs(text), self.result)
        for invalid in (text + "run=true\n", text.replace("tasks=\n", ""), text + "unexpected=1\n"):
            with self.subTest(text=invalid), self.assertRaises(gate.GateValidationError):
                gate.parse_outputs(invalid)

    def test_generated_version_requires_calendar_and_observed_minute_window(self):
        gate.validate_outputs(self.result, self.expected, self.start, self.end)
        for timestamp in ("202609210000", "202609210001"):
            gate.validate_outputs(
                dict(self.result, ci_version=f"1.2.3.dev{timestamp}+ci"), self.expected, self.start, self.end
            )
        for version in (
            "1.2.3.dev202609202359+ci",
            "1.2.3.dev202609210002+ci",
            "1.2.3.dev202609310000+ci",
            "1.2.3.devbad+ci",
        ):
            with self.subTest(version=version), self.assertRaises(gate.GateValidationError):
                gate.validate_outputs(dict(self.result, ci_version=version), self.expected, self.start, self.end)
        for field in ("run", "run_scope", "tasks", "reuse", "base_version"):
            with self.subTest(field=field), self.assertRaises(gate.GateValidationError):
                gate.validate_outputs(dict(self.result, **{field: "wrong"}), self.expected, self.start, self.end)

    def test_reuse_and_tasks_are_preserved_exactly(self):
        reuse = "1.2.3.dev202609200000+ci"
        expected = dict(self.expected, reuse=reuse, ci_version=reuse, tasks="docker-vllm-3h,baremetal-vllm-3h")
        gate.validate_outputs(expected.copy(), expected, self.start, self.end)
        for field, value in (
            ("ci_version", self.result["ci_version"]),
            ("reuse", reuse + "x"),
            ("tasks", "baremetal-vllm-3h,docker-vllm-3h"),
        ):
            with self.subTest(field=field), self.assertRaises(gate.GateValidationError):
                gate.validate_outputs(dict(expected, **{field: value}), expected, self.start, self.end)


if __name__ == "__main__":
    unittest.main()
