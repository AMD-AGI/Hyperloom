# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which kind of device fault a crashed run hit, told apart from the timeout it causes downstream."""

from __future__ import annotations

from hyperloom.orchestrator.measurement import runtime_findings as rf

# Lines as a real GLM-5.3 server.log carries them: eight queue exceptions, then a watchdog ten minutes later.
_QUEUE_EXCEPTION = (
    "Warning: Queue error: HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception."
)
_WATCHDOG = (
    "  what():  [Rank 4] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=304735, "
    "OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600014 milliseconds before timing out."
)
_MEMORY_FAULT = (
    "Memory access fault by GPU node-4 (Agent handle: 0x5d0b40) on address 0x7f9c00000000. Reason: Page not present"
)


def _rule(rule_id: str) -> rf._LineRule:
    return next(r for r in rf._LINE_RULES if r.rule_id == rule_id)


def _match(rule_id: str, line: str):
    rule = _rule(rule_id)
    for pattern in rule.patterns:
        found = pattern.search(line)
        if found:
            return rule.subject(found)
    return None


def _detected(*rule_ids: str) -> list[dict]:
    return [{"rule_id": rule_id, "status": rf.DETECTED} for rule_id in rule_ids]


def test_a_queue_exception_is_recognised_and_names_the_status():
    assert _match("device.queue_exception", _QUEUE_EXCEPTION) == "HSA_STATUS_ERROR_EXCEPTION"


def test_a_memory_fault_is_recognised_and_names_the_node():
    assert _match("device.memory_fault", _MEMORY_FAULT) == "4"


def test_the_watchdog_timeout_is_not_mistaken_for_either_fault():
    """It is the loudest line in the log and the last; it is also downstream of both."""
    assert _match("device.queue_exception", _WATCHDOG) is None
    assert _match("device.memory_fault", _WATCHDOG) is None


def test_both_fault_rules_are_correctness_not_performance():
    assert rf._CATEGORIES["device.queue_exception"] == rf.CORRECTNESS
    assert rf._CATEGORIES["device.memory_fault"] == rf.CORRECTNESS


def test_a_queue_exception_without_a_memory_fault_reads_as_a_trap():
    (line,) = rf._device_fault_reading(_detected("device.queue_exception"))

    assert "trapped rather than ran off its allocation" in line
    assert "device-side assert" in line
    assert "non-finite value" in line
    assert "not a communication problem" in line


def test_a_queue_exception_with_a_memory_fault_reads_as_a_bad_address():
    (line,) = rf._device_fault_reading(_detected("device.queue_exception", "device.memory_fault"))

    assert "touched an address it should not have" in line
    assert "device-side assert" not in line


def test_both_readings_name_the_collective_timeout_as_downstream():
    trap = rf._device_fault_reading(_detected("device.queue_exception"))[0]
    fault = rf._device_fault_reading(_detected("device.queue_exception", "device.memory_fault"))[0]

    assert "downstream of the dead queue" in trap
    assert "downstream of the dead queue" in fault


def test_a_run_that_did_not_fault_gets_no_reading():
    assert rf._device_fault_reading([]) == []
    assert rf._device_fault_reading(_detected("device.memory_fault")) == []
    assert rf._device_fault_reading([{"rule_id": "device.queue_exception", "status": rf.NOT_DETECTED}]) == []


def test_the_fault_rules_apply_to_every_framework():
    """A trapped kernel is a property of the device, not of what dispatched to it."""
    for rule_id in ("device.queue_exception", "device.memory_fault"):
        assert _rule(rule_id).frameworks is None
        assert rule_id in rf._applicable_rules("sglang")
        assert rule_id in rf._applicable_rules("vllm")
