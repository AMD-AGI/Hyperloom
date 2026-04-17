"""Robustness tests for P0/P1 fixes: GPU lock, state lock, IPC cursor desync,
merge-ready atomicity, list capping, and process cleanup patterns."""

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest

from marathon_harness import ipc
from marathon_harness.state import MarathonState, StateLock



# ---------------------------------------------------------------------------
# IPC cursor desync: after_id not found should return all entries
# ---------------------------------------------------------------------------

def test_ipc_cursor_desync_returns_all():
    """When after_id doesn't exist (e.g. file rotated), returns all entries."""
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)
        for i in range(3):
            ipc.write_result(td, {"id": f"res_{i}", "status": "ok"})

        results = ipc.read_new_results(td, after_id="nonexistent_cursor")
        assert len(results) == 3, "Should return all entries on cursor desync"


def test_ipc_cursor_desync_events():
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)
        ipc.write_event(td, {"id": "e1", "source": "t", "type": "crash", "severity": "error"})
        ipc.write_event(td, {"id": "e2", "source": "t", "type": "crash", "severity": "error"})

        events = ipc.read_new_events(td, after_id="stale_id")
        assert len(events) == 2


def test_ipc_cursor_desync_insights():
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)
        ipc.write_insight(td, {"id": "ins_1", "type": "pattern-discovery", "body": "test"})
        ipc.write_insight(td, {"id": "ins_2", "type": "pattern-discovery", "body": "test2"})

        insights = ipc.read_new_insights(td, after_id="gone")
        assert len(insights) == 2


def test_ipc_cursor_desync_findings():
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)
        ipc.write_finding(td, {"event_id": "f1", "classification": "register_spill"})
        ipc.write_finding(td, {"event_id": "f2", "classification": "hardware"})

        findings = ipc.read_new_findings(td, after_event_id="deleted_event")
        assert len(findings) == 2


# ---------------------------------------------------------------------------
# Merge-ready atomicity: .ready marker
# ---------------------------------------------------------------------------

def test_merge_ready_has_ready_marker():
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)
        d = ipc.write_merge_ready(td, "task_42", {"apply_instructions": ["patch -p1"]},
                                  artifacts={"fix.patch": "diff content"})
        assert (d / ".ready").exists()
        assert (d / "metadata.json").exists()
        assert (d / "fix.patch").exists()


def test_merge_ready_reader_requires_ready():
    with tempfile.TemporaryDirectory() as td:
        ipc.init_ipc_files(td)
        d = Path(td) / "kernel_manager" / "merge_ready" / "task_99"
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({"apply_instructions": []}))

        meta = ipc.read_merge_ready_metadata(td, "task_99")
        assert meta is None, "Should not read merge-ready without .ready marker"

        (d / ".ready").write_text("")
        meta = ipc.read_merge_ready_metadata(td, "task_99")
        assert meta is not None
        assert "apply_instructions" in meta


# ---------------------------------------------------------------------------
# State list capping
# ---------------------------------------------------------------------------

def test_state_caps_lists_on_save():
    with tempfile.TemporaryDirectory() as td:
        st = MarathonState(session_dir=td, model_name="test")
        for i in range(300):
            st.completed_actions.append({"id": f"a_{i}", "action": "test"})
        for i in range(200):
            st.crash_log.append(f"crash {i}")

        st.save()
        assert len(st.completed_actions) <= 200
        assert len(st.crash_log) <= 100


def test_state_caps_dicts_on_save():
    with tempfile.TemporaryDirectory() as td:
        st = MarathonState(session_dir=td, model_name="test")
        for i in range(600):
            st.visit_map[f"key_{i}"] = i

        st.save()
        assert len(st.visit_map) <= 500


# ---------------------------------------------------------------------------
# StateLock serializes mutations
# ---------------------------------------------------------------------------

def test_state_lock_serializes():
    async def _run():
        slock = StateLock()
        counter = {"value": 0}

        async def increment():
            async with slock.mutate():
                v = counter["value"]
                await asyncio.sleep(0.01)
                counter["value"] = v + 1

        await asyncio.gather(*[increment() for _ in range(10)])
        assert counter["value"] == 10, "Lock should serialize all mutations"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# GPU lock: stale flag instead of force-release
# ---------------------------------------------------------------------------

def test_gpu_lock_stale_flag():
    async def _run():
        from marathon_harness.gpu_lock import GpuLock

        lock = GpuLock()
        acquired = asyncio.Event()
        done = asyncio.Event()

        async def holder():
            async with lock.acquire("compile", "test-holder"):
                acquired.set()
                await done.wait()

        task = asyncio.create_task(holder())
        await acquired.wait()

        assert lock._lock.locked()
        lock._force_release()
        assert lock.is_stale
        assert lock._lock.locked(), "Lock should still be held — not released from outside"

        done.set()
        await task
        assert not lock._lock.locked()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# GPU lock: wait_or_defer actually blocks after patience exhausted
# ---------------------------------------------------------------------------

def test_gpu_lock_wait_or_defer_blocks():
    async def _run():
        from marathon_harness.gpu_lock import GpuLock, DEFER_PATIENCE

        lock = GpuLock()

        async def short_holder():
            async with lock.acquire("compile", "holder"):
                await asyncio.sleep(0.2)

        holder_task = asyncio.create_task(short_holder())
        await asyncio.sleep(0.05)

        for _ in range(DEFER_PATIENCE):
            lock._defer_counts["waiter"] = lock._defer_counts.get("waiter", 0) + 1

        result = await lock.wait_or_defer("local-test", "waiter",
                                           quick_timeout_s=0.05, full_timeout_s=5)
        assert result is True, "Should have blocked and gotten the lock after holder released"
        assert lock._defer_counts.get("waiter", 0) == 0, "Deferrals should be reset"
        await holder_task

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Checkpoint symlink atomicity
# ---------------------------------------------------------------------------

def test_checkpoint_symlink_atomic():
    with tempfile.TemporaryDirectory() as td:
        st = MarathonState(session_dir=td, model_name="test")
        st.start_time = time.time()

        p1 = st.checkpoint("first")
        latest = Path(td) / "checkpoints" / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == p1.resolve()

        time.sleep(1.1)
        p2 = st.checkpoint("second")
        assert latest.is_symlink()
        assert latest.resolve() == p2.resolve()
