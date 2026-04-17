"""Unit tests for IPC — JSONL read/write, cursor-based reads, atomic appends."""

import json
import tempfile
from pathlib import Path

from marathon_harness import ipc


def test_init_ipc_files():
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)
        km_dir = Path(td) / "kernel_manager"
        assert km_dir.exists()
        assert (km_dir / "work_queue.jsonl").exists()
        assert (km_dir / "results.jsonl").exists()
        assert (km_dir / "event_log.jsonl").exists()
        assert (km_dir / "findings.jsonl").exists()


def test_write_and_read_work_queue():
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)
        entry = {"id": "wq_1", "kernel_name": "attn", "strategy": "oob-rewrite"}
        ipc.write_work_queue_entry(td, entry)

        entries = ipc.read_work_queue_all(td)
        assert len(entries) == 1
        assert entries[0]["id"] == "wq_1"

        # Write another
        ipc.write_work_queue_entry(td, {"id": "wq_2", "kernel_name": "mlp"})
        entries = ipc.read_work_queue_all(td)
        assert len(entries) == 2


def test_cursor_based_event_reading():
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)

        # Write 3 events
        for i in range(3):
            ipc.write_event(td, {
                "id": f"evt_{i}", "source": "test",
                "type": "crash", "severity": "error",
            })

        # Read all
        all_events = ipc.read_new_events(td)
        assert len(all_events) == 3

        # Read after cursor
        after_1 = ipc.read_new_events(td, after_id="evt_1")
        assert len(after_1) == 1
        assert after_1[0]["id"] == "evt_2"

        # Read after last → empty
        after_last = ipc.read_new_events(td, after_id="evt_2")
        assert len(after_last) == 0


def test_cursor_based_finding_reading():
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)

        for i in range(3):
            ipc.write_finding(td, {
                "event_id": f"finding_{i}",
                "classification": "register_spill",
                "root_cause": f"test cause {i}",
            })

        all_findings = ipc.read_new_findings(td)
        assert len(all_findings) == 3

        after_1 = ipc.read_new_findings(td, after_event_id="finding_1")
        assert len(after_1) == 1
        assert after_1[0]["event_id"] == "finding_2"


def test_write_result_and_read():
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)

        ipc.write_result(td, {
            "id": "res_1", "status": "merge-ready",
            "micro_speedup": 1.15,
        })

        results = ipc.read_new_results(td)
        assert len(results) == 1
        assert results[0]["status"] == "merge-ready"
