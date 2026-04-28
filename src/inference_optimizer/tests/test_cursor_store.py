"""Tests for orchestrator/cursor_store.py."""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.cursor_store import CursorStore


@pytest.mark.asyncio
async def test_load_missing_agent_returns_empty(db):
    store = CursorStore(db)
    cursor = await store.load("executor")
    assert cursor.last_processed_seq == 0
    assert cursor.last_processed_msg_id == ""


@pytest.mark.asyncio
async def test_advance_creates_row(db):
    store = CursorStore(db)
    await store.advance("executor", seq=5, msg_id="m5")
    cursor = await store.load("executor")
    assert cursor.last_processed_seq == 5
    assert cursor.last_processed_msg_id == "m5"


@pytest.mark.asyncio
async def test_advance_upserts(db):
    store = CursorStore(db)
    await store.advance("executor", seq=5, msg_id="m5")
    await store.advance("executor", seq=10, msg_id="m10")
    cursor = await store.load("executor")
    assert cursor.last_processed_seq == 10
    assert cursor.last_processed_msg_id == "m10"


@pytest.mark.asyncio
async def test_cursor_never_goes_backwards(db):
    store = CursorStore(db)
    await store.advance("executor", seq=10, msg_id="m10")
    result = await store.advance("executor", seq=3, msg_id="m3")
    assert result.last_processed_seq == 10
    cursor = await store.load("executor")
    assert cursor.last_processed_seq == 10
    assert cursor.last_processed_msg_id == "m10"


@pytest.mark.asyncio
async def test_idempotency_check(db):
    store = CursorStore(db)
    await store.advance("executor", seq=7, msg_id="m7")
    assert await store.is_already_processed("executor", 7) is True
    assert await store.is_already_processed("executor", 8) is False


@pytest.mark.asyncio
async def test_all_returns_each_agent(db):
    store = CursorStore(db)
    await store.advance("executor", seq=1, msg_id="a")
    await store.advance("critic", seq=2, msg_id="b")
    await store.advance("sage", seq=3, msg_id="c")
    everyone = await store.all()
    assert set(everyone) == {"executor", "critic", "sage"}
    assert everyone["sage"].last_processed_seq == 3
