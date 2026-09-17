"""Tests for the fixed-path token ledger shared by a rewrite and the forge-loop it nests."""

import json
import multiprocessing
from pathlib import Path

from kernelforge.tracker import (
    LEDGER_FILENAME,
    UsageAccumulator,
    UsageLedgerFile,
    read_usage_ledger,
)


def _priced(**counters):
    """One producer's running totals, fully priced by the provider."""
    return {
        "input_tokens": counters.get("input_tokens", 0),
        "output_tokens": counters.get("output_tokens", 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_cost_usd": counters.get("total_cost_usd", 0.0),
        "cost_available": True,
        "cost_source": "provider",
        "calls": counters.get("calls", 0),
    }


def test_ledger_lands_at_one_fixed_name_for_every_producer(tmp_path):
    UsageLedgerFile(tmp_path).publish(_priced(input_tokens=10, calls=1, total_cost_usd=0.5))

    assert (tmp_path / LEDGER_FILENAME).is_file()
    assert read_usage_ledger(tmp_path)["input_tokens"] == 10


def test_ledger_exists_before_the_first_call(tmp_path):
    """A caller must find the file even when the run has not billed anything yet."""
    ledger = UsageLedgerFile(tmp_path)
    ledger.publish(UsageAccumulator().totals())

    recorded = read_usage_ledger(tmp_path)
    assert recorded["calls"] == 0
    assert recorded["cost_source"] == "unavailable"


def test_ledger_adds_each_producer_share_without_overwriting_the_other(tmp_path):
    """A rewrite and its nested forge-loop write one file; neither may erase the other's spend."""
    rewrite = UsageLedgerFile(tmp_path)
    forge_loop = UsageLedgerFile(tmp_path)

    rewrite.publish(_priced(input_tokens=10, calls=1, total_cost_usd=0.1))
    forge_loop.publish(_priced(input_tokens=100, calls=2, total_cost_usd=1.0))
    # Each producer republishes its own running totals, not an increment.
    rewrite.publish(_priced(input_tokens=25, calls=2, total_cost_usd=0.3))
    forge_loop.publish(_priced(input_tokens=400, calls=5, total_cost_usd=4.0))

    recorded = read_usage_ledger(tmp_path)
    assert recorded["input_tokens"] == 425
    assert recorded["calls"] == 7
    assert recorded["total_cost_usd"] == 4.3
    assert recorded["cost_available"] is True
    assert recorded["cost_source"] == "provider"


def test_ledger_republish_without_new_spend_changes_nothing(tmp_path):
    ledger = UsageLedgerFile(tmp_path)
    ledger.publish(_priced(input_tokens=10, calls=1, total_cost_usd=0.1))
    ledger.publish(_priced(input_tokens=10, calls=1, total_cost_usd=0.1))

    assert read_usage_ledger(tmp_path)["calls"] == 1


def test_ledger_degrades_provenance_when_one_producer_is_unpriced(tmp_path):
    """One contributor without provider pricing makes the shared total unbillable, and it never recovers."""
    priced = UsageLedgerFile(tmp_path)
    unpriced = UsageLedgerFile(tmp_path)

    priced.publish(_priced(input_tokens=10, calls=1, total_cost_usd=0.1))
    unpriced.publish(
        {
            "input_tokens": 5,
            "calls": 1,
            "total_cost_usd": 0.0,
            "cost_available": False,
            "cost_source": "unavailable",
        }
    )
    priced.publish(_priced(input_tokens=20, calls=2, total_cost_usd=0.2))

    recorded = read_usage_ledger(tmp_path)
    assert recorded["input_tokens"] == 25
    assert recorded["calls"] == 3
    assert recorded["cost_available"] is False
    assert recorded["cost_source"] == "partial"


def test_ledger_restarts_from_this_producer_when_the_file_is_corrupt(tmp_path):
    (tmp_path / LEDGER_FILENAME).write_text("{ not json")

    UsageLedgerFile(tmp_path).publish(_priced(input_tokens=7, calls=1, total_cost_usd=0.07))

    assert read_usage_ledger(tmp_path)["input_tokens"] == 7


def test_ledger_publish_never_raises_on_an_unwritable_directory(tmp_path):
    unwritable = tmp_path / "denied"
    unwritable.mkdir()
    unwritable.chmod(0o500)
    try:
        UsageLedgerFile(unwritable).publish(_priced(input_tokens=1, calls=1))
    finally:
        unwritable.chmod(0o700)


def test_accumulator_publishes_on_every_counted_call(tmp_path):
    """The ledger must survive a kill between two calls, so it cannot wait for a coarse checkpoint."""
    ledger = UsageLedgerFile(tmp_path)
    usage = UsageAccumulator(on_update=ledger.publish)

    usage.add_usage({"input_tokens": 10, "output_tokens": 1}, total_cost_usd=0.1)
    assert read_usage_ledger(tmp_path)["calls"] == 1

    usage.add_usage({"input_tokens": 20, "output_tokens": 2}, total_cost_usd=0.2)
    recorded = read_usage_ledger(tmp_path)
    assert recorded["calls"] == 2
    assert recorded["input_tokens"] == 30
    assert recorded["total_cost_usd"] == 0.3


def test_accumulator_survives_a_failing_sink():
    def explode(_totals):
        raise RuntimeError("disk gone")

    usage = UsageAccumulator(on_update=explode)

    assert usage.add_usage({"input_tokens": 5}, total_cost_usd=0.1) is True
    assert usage.totals()["calls"] == 1


def _publish_repeatedly(args):
    """Run one producer's whole publish sequence in its own process."""
    directory, calls = args
    ledger = UsageLedgerFile(directory)
    for call in range(1, calls + 1):
        ledger.publish(_priced(input_tokens=call, calls=call, total_cost_usd=call / 100))


def test_ledger_loses_no_update_across_processes(tmp_path):
    """The rewrite and its nested loop publish concurrently; the lock is what keeps both shares."""
    context = multiprocessing.get_context("spawn")
    with context.Pool(2) as pool:
        pool.map(_publish_repeatedly, [(str(tmp_path), 20), (str(tmp_path), 20)])

    recorded = json.loads(Path(tmp_path, LEDGER_FILENAME).read_text())
    assert recorded["calls"] == 40
    assert recorded["input_tokens"] == 40
