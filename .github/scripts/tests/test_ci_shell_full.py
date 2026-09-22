# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract checks for the trace-clock candidate and full-suite comparison tools."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
REPO = SCRIPTS.parents[1]
TEST_FILE = "src/hyperloom/inference_optimizer/tests/test_aiperf_client_sh.py"
BASELINE = "b8761298c0413b08937554077ad567a764a4e1dc"
sys.path.insert(0, str(SCRIPTS))


def clock_functions(source: str) -> dict:
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_virtual_trace_clock_env", "_fast_trace_poll_env"}
    ]
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=functions, type_ignores=[]), TEST_FILE, "exec"), namespace)
    return namespace


class TestClockCandidate(unittest.TestCase):
    def test_helper_does_not_enable_client_only_lifecycle(self):
        functions = clock_functions((REPO / TEST_FILE).read_text(encoding="utf-8"))
        self.assertIn("_virtual_trace_clock_env", functions)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = functions["_virtual_trace_clock_env"](root)
            self.assertEqual(set(env), {"BASH_ENV", "AGENTX_TEST_TRACE_CLOCK"})
            self.assertEqual(float(Path(env["AGENTX_TEST_TRACE_CLOCK"]).read_text()), 0)
            script = Path(env["BASH_ENV"]).read_text()
            self.assertIn("FUNCNAME", script)
            self.assertIn("_wait_for_trace_flush", script)
            prefix = script.split("sleep()", 1)[0]
            self.assertIn('if [ "${0##*/}" = aiperf_client.sh ]; then', prefix)
            self.assertNotIn("kill()", script)
            self.assertNotIn("fuser", script)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Linux Bash contract runs on hosted runner")
    def test_trace_sleep_is_virtual_but_cleanup_and_child_sleep_are_delegated(self):
        functions = clock_functions((REPO / TEST_FILE).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bindir = root / "bin"
            bindir.mkdir()
            marker = root / "sleeps.txt"
            fake_sleep = bindir / "sleep"
            fake_sleep.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$SLEEP_MARKER"\n', encoding="utf-8")
            fake_sleep.chmod(0o755)
            env = functions["_virtual_trace_clock_env"](root)
            child = root / "aiperf"
            child.write_text("#!/bin/bash\nsleep 6\nsleep 300\n", encoding="utf-8")
            script = root / "aiperf_client.sh"
            script.write_text(
                "_wait_for_trace_flush() { sleep 10; sleep 5; }\n"
                "cleanup() { sleep 2; }\n"
                '_wait_for_trace_flush\ncleanup\nbash "$CHILD_SCRIPT"\n',
                encoding="utf-8",
            )
            environment = dict(os.environ, **env, SLEEP_MARKER=str(marker), CHILD_SCRIPT=str(child))
            environment["PATH"] = str(bindir) + os.pathsep + os.environ["PATH"]
            result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(float(Path(env["AGENTX_TEST_TRACE_CLOCK"]).read_text()), 15)
            self.assertEqual(marker.read_text().splitlines(), ["2", "6", "300"])

    @unittest.skipUnless(
        sys.platform == "linux" and shutil.which("bash"), "Linux PID/EOF regression runs on hosted runner"
    )
    def test_old_wrapper_leaks_pipe_but_scoped_override_preserves_background_pid(self):
        old_source = subprocess.run(
            ["git", "-C", str(REPO), "show", f"c24b171c4fffb2c714714b9e1758d91eff4a769b:{TEST_FILE}"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
        new_source = (REPO / TEST_FILE).read_text(encoding="utf-8")

        def probe(source):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                env = dict(os.environ, **clock_functions(source)["_virtual_trace_clock_env"](root))
                pid_file = root / "background.pid"
                command_type = root / "sleep-type.txt"
                child = root / "fake_builtin.sh"
                child.write_text(
                    'type -t sleep > "$TYPE_FILE"\n'
                    'sleep 8 &\nprintf "%s\\n" "$!" > "$PID_FILE.tmp"\nmv "$PID_FILE.tmp" "$PID_FILE"\n',
                    encoding="utf-8",
                )
                env.update(PID_FILE=str(pid_file), TYPE_FILE=str(command_type))
                process = subprocess.Popen(
                    ["bash", str(child)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    deadline = time.monotonic() + 2
                    while not pid_file.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(pid_file.exists(), "Background process did not publish its PID")
                    pid = int(pid_file.read_text())
                    while time.monotonic() < deadline:
                        comm = Path(f"/proc/{pid}/comm").read_text().strip()
                        children_path = Path(f"/proc/{pid}/task/{pid}/children")
                        children = children_path.read_text().strip().split()
                        if comm == "sleep" or children:
                            break
                        time.sleep(0.01)
                    else:
                        self.fail("Background process never reached native sleep or its wrapper child")
                    group = os.getpgid(pid)
                    self.assertEqual(group, process.pid, "Probe child escaped its isolated process group")
                    self.assertNotEqual(group, pid, "Probe PID unexpectedly leads the process group")
                    self.assertEqual(process.wait(timeout=2), 0, "Launcher did not exit before EOF measurement")
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        os.kill(pid, signal.SIGTERM)
                    try:
                        process.communicate(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                    return {
                        "type": command_type.read_text().strip(),
                        "comm": comm,
                        "children": len(children),
                        "eof_timeout": timed_out,
                    }
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate(timeout=2)

        old, new = probe(old_source), probe(new_source)
        print("Lifecycle regression: " + json.dumps({"old": old, "new": new}, sort_keys=True), flush=True)
        self.assertEqual(old["type"], "function")
        self.assertTrue(old["eof_timeout"], "Old PID/EOF failure did not reproduce; stop before full suite")
        self.assertGreater(old["children"], 0)
        self.assertEqual(new["type"], "file")
        self.assertEqual(new["comm"], "sleep")
        self.assertFalse(new["eof_timeout"])

    def test_original_tests_parameters_and_assertions_are_preserved(self):
        baseline = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{BASELINE}:{TEST_FILE}"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
        candidate = (REPO / TEST_FILE).read_text(encoding="utf-8")

        def tests(source):
            return {
                node.name: node
                for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
            }

        before, after = tests(baseline), tests(candidate)
        self.assertEqual(set(before), set(after))
        for name, original in before.items():
            with self.subTest(test=name):
                current = after[name]
                self.assertEqual(
                    [ast.dump(n) for n in original.decorator_list], [ast.dump(n) for n in current.decorator_list]
                )
                assertions = [ast.dump(n) for n in ast.walk(current) if isinstance(n, ast.Assert)]
                for assertion in (n for n in ast.walk(original) if isinstance(n, ast.Assert)):
                    self.assertIn(ast.dump(assertion), assertions)


def stream_records(nodeids, outcomes=None, workers=2, collect_only=False):
    outcomes = outcomes or {nodeid: "pass" for nodeid in nodeids}
    records = [{"event": "session_start", "schema_version": 1, "collectonly": collect_only, "workers": workers}]
    for worker in [f"gw{i}" for i in range(workers)] if workers else [None]:
        records.append({"event": "collection", "worker_id": worker, "nodeids": nodeids})
    if not collect_only:
        for nodeid in nodeids:
            result = outcomes[nodeid]
            phases = ["setup", "teardown"] if result == "skip" else ["setup", "call", "teardown"]
            for phase in phases:
                outcome = "skipped" if result == "skip" and phase == "setup" else "passed"
                xfail = result == "xfail" and phase == "call"
                records.append(
                    {
                        "event": "report",
                        "nodeid": nodeid,
                        "when": phase,
                        "outcome": "skipped" if xfail else outcome,
                        "wasxfail": "known issue" if xfail else None,
                        "skipreason": "optional platform" if result == "skip" and phase == "setup" else None,
                        "longrepr": "known issue" if xfail else None,
                        "duration": 0.01,
                    }
                )
    for index in range(workers):
        records.append({"event": "worker_down", "worker_id": f"gw{index}", "error": None})
    records.append({"event": "session_finish", "exitstatus": 0, "testscollected": len(nodeids)})
    return records


class TestFullComparison(unittest.TestCase):
    def parse(self, records, nodeids=None, workers=2, collect_only=False):
        import ci_shell_full as full

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            return full.validate_stream(path, nodeids, workers, collect_only)

    def test_pass_skip_and_xfail_are_preserved(self):
        nodes = ["test.py::test_pass", "test.py::test_skip", "test.py::test_xfail"]
        outcomes = dict(zip(nodes, ["pass", "skip", "xfail"]))
        result = self.parse(stream_records(nodes, outcomes), nodes)
        self.assertEqual(result["outcomes"], outcomes)
        self.assertTrue(result["passed"])

    def test_incomplete_duplicate_and_crashed_streams_are_rejected(self):
        nodes = ["test.py::test_pass"]
        records = stream_records(nodes)
        variants = [records[:-1], records + [records[-1]], records[:3] + records[4:]]
        duplicate = list(records)
        duplicate.insert(4, records[3])
        variants.append(duplicate)
        crashed = json.loads(json.dumps(records))
        crashed[-2]["error"] = "worker crashed"
        variants.append(crashed)
        collection = json.loads(json.dumps(records))
        collection[2]["nodeids"] = ["test.py::other"]
        variants.append(collection)
        for invalid in variants:
            with self.subTest(records=invalid), self.assertRaises(ValueError):
                self.parse(invalid, nodes)

    def test_collect_only_records_catalog_without_claiming_execution(self):
        nodes = ["test.py::test_one", "test.py::test_two"]
        result = self.parse(stream_records(nodes, workers=0, collect_only=True), nodes, workers=0, collect_only=True)
        self.assertEqual(result["nodeids"], nodes)
        self.assertFalse(result["outcomes"])

    def test_plugin_records_real_synthetic_pytest_outcomes(self):
        import ci_shell_full as full

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = root / "subject"
            subject.mkdir()
            (subject / "test_sample.py").write_text(
                "import pytest\n"
                "def test_pass(): assert True\n"
                "@pytest.mark.skip(reason='optional platform')\n"
                "def test_skip(): assert False\n"
                "@pytest.mark.xfail(reason='known issue')\n"
                "def test_xfail(): assert False\n",
                encoding="utf-8",
            )
            events = root / "events.jsonl"
            env = dict(
                os.environ,
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                PYTHONPATH=str(SCRIPTS),
                CI_SHELL_FULL_EVENTS=str(events),
                PYTHONDONTWRITEBYTECODE="1",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "ci_shell_full_plugin",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    str(subject),
                ],
                cwd=subject,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            parsed = full.validate_stream(events, workers=0)
            self.assertEqual(
                parsed["outcomes"],
                {
                    "test_sample.py::test_pass": "pass",
                    "test_sample.py::test_skip": "skip",
                    "test_sample.py::test_xfail": "xfail",
                },
            )
            self.assertTrue(parsed["passed"])

    def test_six_shards_must_cover_catalog_exactly_once(self):
        import ci_shell_full as full

        nodes = [f"test.py::test_{i}" for i in range(6)]
        shards = [self.parse(stream_records([node]), [node]) for node in nodes]
        merged = full.merge_outcomes(shards, nodes)
        self.assertEqual(set(merged), set(nodes))
        for invalid in (shards[:-1], shards[:-1] + [shards[0]]):
            with self.subTest(shards=invalid), self.assertRaises(ValueError):
                full.merge_outcomes(invalid, nodes)

    def test_rejected_output_path_never_writes_failure_into_repository(self):
        import ci_shell_full as full

        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"RUNNER_TEMP": temporary}), patch.object(full, "write_json") as write:
                result = full.main(["prepare", "--output", str(REPO), "--seed", str(Path(temporary) / "seed")])
            self.assertEqual(result, 1)
            write.assert_not_called()

    def test_coverage_drop_and_denominator_changes_are_visible(self):
        import ci_shell_full as full

        baseline = {
            "files": {
                "src/example.py": {
                    "executed_lines": [1, 2],
                    "missing_lines": [3],
                    "excluded_lines": [],
                    "summary": {"num_statements": 3, "covered_lines": 2},
                }
            },
            "totals": {"num_statements": 3, "covered_lines": 2},
        }
        equal = full.compare_coverage(baseline, baseline)
        self.assertTrue(equal["valid"])
        inconsistent = json.loads(json.dumps(baseline))
        inconsistent["files"]["src/example.py"]["summary"]["covered_lines"] = 99
        self.assertFalse(full.compare_coverage(baseline, inconsistent)["valid"])
        candidate = json.loads(json.dumps(baseline))
        candidate["files"]["src/example.py"]["executed_lines"] = [1]
        candidate["files"]["src/example.py"]["missing_lines"] = [2, 3]
        candidate["files"]["src/example.py"]["summary"]["covered_lines"] = 1
        candidate["totals"]["covered_lines"] = 1
        self.assertTrue(full.compare_coverage(baseline, candidate)["valid"])
        self.assertFalse(full.compare_coverage(baseline, candidate)["no_drop"])
        candidate = json.loads(json.dumps(baseline))
        candidate["files"]["src/example.py"]["summary"]["num_statements"] = 4
        self.assertFalse(full.compare_coverage(baseline, candidate)["valid"])


if __name__ == "__main__":
    unittest.main()
