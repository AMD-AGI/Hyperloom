# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the host-side rewrite-evidence probe and its aggregator.

Two halves:

* the probe itself (``assets/host_probe/hl_host_probe.py``), exercised directly
  against synthetic call sites so the wrappers, the stack attribution and the
  strict/loose fingerprint split are covered without torch or a benchmark;
* the orchestrator-side aggregator, exercised against hand-written per-rank
  reports so each taxonomy classification is pinned independently.

The strict/loose split gets the most attention because it is what tells a
memoization candidate apart from a loop-hoist enabler, and getting that backwards
costs a whole optimization bundle: the enabler measures flat on its own, so a
greedy accept/reject loop discards it and the dependent memoizations then never
hit.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors import _framework_rewrite_evidence as evidence


def _load_probe_module():
    """Import the bundled probe as a standalone module.

    The probe deliberately lives outside the package tree (it is copied onto a
    benchmark process's ``PYTHONPATH``, not imported as
    ``hyperloom.…``), so tests load it by path.

    Returns:
        The imported ``hl_host_probe`` module.
    """
    path = evidence.probe_asset_dir() / "hl_host_probe.py"
    spec = importlib.util.spec_from_file_location("hl_host_probe_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def probe_module():
    """Provide the probe module, cleaned out of ``sys.modules`` afterwards."""
    module = _load_probe_module()
    yield module
    sys.modules.pop("hl_host_probe_under_test", None)


# --------------------------------------------------------------------------
# probe assets
# --------------------------------------------------------------------------


def test_probe_assets_are_bundled():
    """Both probe files ship, since a missing shim silently disables the probe."""
    asset_dir = evidence.probe_asset_dir()
    assert (asset_dir / "hl_host_probe.py").is_file()
    assert (asset_dir / "sitecustomize.py").is_file()


def test_probe_is_inert_without_env(probe_module, monkeypatch):
    """No env, no probe: a PYTHONPATH prefix alone must not change behaviour."""
    monkeypatch.delenv("HYPERLOOM_HOST_PROBE", raising=False)
    assert probe_module.install_from_env() is None


def test_probe_requires_an_output_dir(probe_module, monkeypatch):
    """Enabled but with nowhere to report is treated as disabled, not as an error."""
    monkeypatch.setenv("HYPERLOOM_HOST_PROBE", "1")
    monkeypatch.delenv("HYPERLOOM_HOST_PROBE_DIR", raising=False)
    assert probe_module.install_from_env() is None


def test_repeat_rate_is_zero_without_samples(probe_module):
    """A site with no fingerprints reports no repeats rather than dividing by zero."""
    assert probe_module._repeat_rate(0, 0) == 0.0
    assert probe_module._repeat_rate(10, 0) == 0.0
    assert probe_module._repeat_rate(10, 10) == 0.0
    assert probe_module._repeat_rate(10, 1) == 0.9


def test_strict_fingerprint_separates_tensor_identity(probe_module, tmp_path):
    """Strict keys on tensor identity; loose keys only on shape/dtype/device.

    This is the pairing that distinguishes the two candidate classes, so it is
    asserted on the fingerprint function directly rather than inferred from a
    classification downstream.
    """
    probe = probe_module.HostProbe(out_dir=str(tmp_path), roots=())

    class _FakeTensor:
        def __init__(self, ptr: int) -> None:
            self.shape = (2, 3)
            self.dtype = "float32"
            self.device = "cuda:0"
            self._version = 0
            self._ptr = ptr

        def data_ptr(self) -> int:
            return self._ptr

    same_value_new_buffer_a = _FakeTensor(0x1000)
    same_value_new_buffer_b = _FakeTensor(0x2000)

    assert probe._fingerprint(same_value_new_buffer_a, strict=False) == probe._fingerprint(
        same_value_new_buffer_b, strict=False
    )
    assert probe._fingerprint(same_value_new_buffer_a, strict=True) != probe._fingerprint(
        same_value_new_buffer_b, strict=True
    )


def test_a_live_tensor_keeps_its_generation_across_eviction(probe_module, tmp_path):
    """Trimming the table must not turn a memoize candidate into a hoist candidate.

    The table is bounded, and a long rollout allocates enough intermediates to
    reach that bound repeatedly. If the bound is enforced by clearing the table,
    a tensor that is still alive and still being passed loses its generation and
    reads as new. Strict repeats then vanish while loose repeats survive, which
    is exactly the signature the classifier reads as "hoist first, memoizing
    would never hit" -- on a site where memoizing was the right answer.
    """

    class _Obj:
        __slots__ = ("__weakref__",)

    probe = probe_module.HostProbe(out_dir=str(tmp_path), roots=())
    survivor = _Obj()
    first = probe._object_generation(survivor)

    # Churn well past the cap with objects that die immediately, as a rollout does.
    for _ in range(probe_module._MAX_TRACKED_OBJECTS * 3):
        probe._object_generation(_Obj())

    assert probe._object_generation(survivor) == first, "a live, repeatedly-passed object was forgotten"
    assert len(probe._object_generations) <= probe_module._MAX_TRACKED_OBJECTS, "the bound stopped holding"


def test_a_repeatedly_passed_tensor_outlives_live_pressure(probe_module, tmp_path):
    """Reclaiming dead entries is not enough on its own.

    When the table fills with objects that are all still alive, something live
    has to go. Dropping by insertion order alone would evict the loop-invariant
    tensor first -- it was inserted before the loop started -- which is the
    worst possible choice. Being passed again has to count as recency.
    """

    class _Obj:
        __slots__ = ("__weakref__",)

    probe = probe_module.HostProbe(out_dir=str(tmp_path), roots=())
    invariant = _Obj()
    first = probe._object_generation(invariant)
    held = []
    for i in range(probe_module._MAX_TRACKED_OBJECTS * 2):
        obj = _Obj()
        held.append(obj)  # keep it alive so the dead-entry sweep cannot reclaim it
        probe._object_generation(obj)
        if i % 8 == 0:
            probe._object_generation(invariant)  # the loop passes it every iteration

    assert probe._object_generation(invariant) == first, "the loop-invariant object was evicted"


def test_eviction_still_gives_a_new_generation_to_a_new_object(probe_module, tmp_path):
    """Keeping live objects must not go so far as to recycle their tokens."""

    class _Obj:
        __slots__ = ("__weakref__",)

    probe = probe_module.HostProbe(out_dir=str(tmp_path), roots=())
    seen = {probe._object_generation(_Obj()) for _ in range(probe_module._MAX_TRACKED_OBJECTS * 2)}
    assert len(seen) == probe_module._MAX_TRACKED_OBJECTS * 2, "two distinct objects shared a generation"


def test_strict_fingerprint_survives_allocator_address_reuse(probe_module, tmp_path):
    """A recycled allocation must not read as "the same tensor as last time".

    Under a caching allocator a tensor allocated and freed inside a loop gets the
    same address back next iteration. Keying strict identity on ``data_ptr`` would
    report that freshly built tensor as an argument repeat, which inverts the
    memoize/hoist distinction: every hoist candidate would be reclassified as a
    memoization that then never hits at runtime. Verified against real torch on a
    device before being pinned here.
    """
    probe = probe_module.HostProbe(out_dir=str(tmp_path), roots=())
    recycled_address = 0x7000

    class _Recycled:
        def __init__(self) -> None:
            self.shape = (8,)
            self.dtype = "float32"
            self.device = "cuda:0"
            self._version = 0

        def data_ptr(self) -> int:
            return recycled_address

    first = _Recycled()
    first_fingerprint = probe._fingerprint(first, strict=True)
    # Same object again: a genuine repeat.
    assert probe._fingerprint(first, strict=True) == first_fingerprint
    # A distinct object at the same address is not.
    del first
    second = _Recycled()
    assert probe._fingerprint(second, strict=True) != first_fingerprint
    # Both still look alike to the loose fingerprint, which is the whole point.
    assert probe._fingerprint(second, strict=False) == probe._fingerprint(_Recycled(), strict=False)


def test_strict_fingerprint_falls_back_for_a_weakref_hostile_object(probe_module, tmp_path):
    """An object that cannot be weak-referenced still fingerprints, via its address."""
    probe = probe_module.HostProbe(out_dir=str(tmp_path), roots=())

    class _NoWeakref:
        __slots__ = ("shape", "dtype", "device", "_version")

        def __init__(self) -> None:
            self.shape = (2,)
            self.dtype = "float16"
            self.device = "cpu"
            self._version = 0

        def data_ptr(self) -> int:
            return 0x99

    fingerprint = probe._fingerprint(_NoWeakref(), strict=True)
    assert fingerprint[0] == "T"
    assert -0x99 in fingerprint


def test_fingerprint_tolerates_a_storageless_tensor(probe_module, tmp_path):
    """A tensor whose ``data_ptr`` raises still fingerprints (meta/fake tensors)."""
    probe = probe_module.HostProbe(out_dir=str(tmp_path), roots=())

    class _MetaTensor:
        shape = (4,)
        dtype = "bfloat16"
        device = "meta"

        def data_ptr(self):  # noqa: ANN201
            raise RuntimeError("meta tensor has no storage")

    fingerprint = probe._fingerprint(_MetaTensor(), strict=True)
    assert fingerprint[0] == "T"
    assert -1 in fingerprint


def test_fingerprint_summarises_a_wide_container(probe_module, tmp_path):
    """A container past the width cap degrades to (type, length), not element-wise."""
    probe = probe_module.HostProbe(out_dir=str(tmp_path), roots=())
    wide = list(range(64))
    assert probe._fingerprint(wide, strict=True) == ("list", 64)


def test_deep_probe_counts_calls_and_argument_repeats(probe_module, tmp_path):
    """The tier-2 hook attributes calls to framework roots and rates arg repeats."""
    framework_root = tmp_path / "fake_framework"
    framework_root.mkdir()
    module_path = framework_root / "pipeline.py"
    module_path.write_text(
        "def invariant(value):\n"
        "    return value * 2\n"
        "\n"
        "def varying(value):\n"
        "    return value + 1\n"
        "\n"
        "def run(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += invariant(7)\n"
        "        total += varying(i)\n"
        "    return total\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("fake_framework_pipeline", module_path)
    assert spec is not None and spec.loader is not None
    target = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target)

    probe = probe_module.HostProbe(
        out_dir=str(tmp_path / "out"),
        roots=(str(framework_root) + "/",),
        deep=True,
    )
    probe._install_tier2()
    try:
        target.run(50)
    finally:
        probe.uninstall()

    report = probe.report()
    by_name = {row["function"].rsplit(":", 1)[-1]: row for row in report["framework_calls"]}
    assert by_name["invariant"]["count"] == 50
    assert by_name["varying"]["count"] == 50
    # Constant argument: every sampled call repeats the previous identity.
    assert by_name["invariant"]["strict_repeat_rate"] > 0.9
    # Loop counter: every call is distinct, so neither rate fires.
    assert by_name["varying"]["strict_repeat_rate"] == 0.0
    assert report["tier2_enabled"] is True
    # Tier 2 timestamps its rows so the aggregator can tell a constructor that
    # runs during model construction from a function that runs every step.
    for row in report["framework_calls"]:
        assert row["first_s"] >= 0.0
        assert row["last_s"] >= row["first_s"]


def test_probe_wraps_the_implicit_conversion_dunders(probe_module, tmp_path):
    """``if scalar_tensor == 0`` syncs, and the probe has to count it as one.

    A device-to-host read does not have to be spelled ``.item()``. The implicit
    forms are the ones a reviewer misses, because nothing at the call site looks
    like a transfer, so wrapping only the named methods reports a clean host-sync
    picture on a hot path that is in fact stalling every iteration.
    """

    class _FakeTensor:
        def __float__(self) -> float:
            return 1.0

        def __int__(self) -> int:
            return 1

        def __bool__(self) -> bool:
            return True

        def __index__(self) -> int:
            return 0

    probe = probe_module.HostProbe(out_dir=str(tmp_path), roots=())
    for attr in ("__float__", "__int__", "__bool__", "__index__"):
        probe._wrap(_FakeTensor, attr, f"torch.Tensor.{attr}")

    value = _FakeTensor()
    if value:  # __bool__
        float(value)
        int(value)
    _ = [7, 8][value]  # __index__

    recorded = {row["api"]: row["count"] for row in probe.report()["host_calls"]}
    assert recorded == {
        "torch.Tensor.__bool__": 1,
        "torch.Tensor.__float__": 1,
        "torch.Tensor.__int__": 1,
        "torch.Tensor.__index__": 1,
    }
    # And the aggregator has to call them what they are.
    assert all(api in evidence._HOST_SYNC_APIS for api in recorded)


def test_aten_level_scalar_conversion_is_a_declared_blind_spot():
    """Pin the limit the evidence notes claim, so a torch change cannot silently void it.

    ``torch.full((n,), t)`` with a 0-dim device tensor reads device memory, but the
    conversion happens in the C++ argument parser without calling any Python
    method, so no monkeypatch can see it. The evidence document tells the
    specialist to read the source for this class of sync; if torch ever routes it
    through ``__float__`` that advice becomes stale and the probe gains coverage.
    """
    torch = pytest.importorskip("torch")
    seen: list[str] = []
    originals = {name: getattr(torch.Tensor, name) for name in ("item", "__float__", "__index__")}

    def _spy(name, original):
        def wrapper(self, *args, **kwargs):
            seen.append(name)
            return original(self, *args, **kwargs)

        return wrapper

    for name, original in originals.items():
        setattr(torch.Tensor, name, _spy(name, original))
    try:
        torch.full((4,), torch.tensor(3.0))
        assert seen == [], f"torch now routes the scalar conversion through Python: {seen}"
        # The explicit form is visible, which is what makes the contrast meaningful.
        float(torch.tensor(3.0))
        assert "__float__" in seen
    finally:
        for name, original in originals.items():
            setattr(torch.Tensor, name, original)


def test_deep_probe_declines_to_displace_an_existing_profiler(probe_module, tmp_path):
    """Tier 2 backs off rather than evicting cProfile or a with_stack profiler."""

    def _other_hook(_frame, _event, _arg):  # noqa: ANN001, ANN202
        return None

    probe = probe_module.HostProbe(out_dir=str(tmp_path), roots=(), deep=True)
    sys.setprofile(_other_hook)
    try:
        probe._install_tier2()
    finally:
        sys.setprofile(None)
    assert probe._deep_installed is False
    assert any("already in use" in note for note in probe._notes)


def test_probe_writes_one_report_per_process(probe_module, tmp_path, monkeypatch):
    """The report lands under the name the aggregator globs for, tagged by rank and pid."""
    monkeypatch.setenv("RANK", "3")
    probe = probe_module.HostProbe(out_dir=str(tmp_path / "probe"), roots=())
    written = probe.write_report()
    name = Path(written).name
    assert name.startswith("hl_host_probe_rank3_pid")
    assert name.endswith(".json")
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    assert payload["schema"].startswith("hyperloom.host_probe/")
    assert payload["rank"] == 3
    # Idempotent: atexit may fire after an explicit call.
    assert probe.write_report() == ""


# --------------------------------------------------------------------------
# aggregator
# --------------------------------------------------------------------------


def _report(**overrides: Any) -> dict[str, Any]:
    """Build a minimal well-formed per-rank probe report.

    Args:
        **overrides: Fields to replace in the base report.

    Returns:
        The report dict.
    """
    base: dict[str, Any] = {
        "schema": "hyperloom.host_probe/1",
        "rank": 0,
        "world_size": 1,
        "pid": 1,
        "wall_seconds": 100.0,
        "roots": ["/src/hyvideo/"],
        "roots_unset": False,
        "tier1_enabled": True,
        "tier2_enabled": False,
        "host_calls": [],
        "framework_calls": [],
        "truncated": {"host_calls": False, "framework_calls": False},
        "notes": [],
    }
    base.update(overrides)
    return base


def test_empty_input_says_the_probe_did_not_run():
    """No reports is reported as such, not as "nothing to optimize"."""
    document = evidence.build_evidence([])
    assert document["ranks_merged"] == 0
    assert document["candidates"] == []
    assert any("not enabled" in note for note in document["notes"])


def test_object_collective_becomes_a_host_round_trip_candidate():
    """An object collective on the hot path classifies as a host round-trip."""
    document = evidence.build_evidence(
        [
            _report(
                host_calls=[
                    {
                        "api": "torch.distributed.all_gather_object",
                        "site": "utils/communications.py:60:_all_to_all_4D",
                        "count": 30720,
                        "wall_s": 41.2,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": ["models/attention.py:412:sequence_parallel_attention_vision"],
                    }
                ]
            )
        ]
    )
    assert len(document["candidates"]) == 1
    candidate = document["candidates"][0]
    assert candidate["category"] == evidence.CATEGORY_HOST_ROUND_TRIP
    assert candidate["taxonomy"] == "c"
    assert candidate["wall_pct"] == pytest.approx(41.2, abs=0.1)
    assert "round-trip" in candidate["signal"]
    assert "rendezvous" in candidate["suggested_rewrite"]


def test_startup_only_site_is_below_the_reporting_floor():
    """A site called a handful of times is start-up work, not the hot path."""
    document = evidence.build_evidence(
        [
            _report(
                host_calls=[
                    {
                        "api": "torch.distributed.all_gather_object",
                        "site": "utils/setup.py:10:init",
                        "count": evidence.MIN_HOST_CALLS - 1,
                        "wall_s": 0.5,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                    }
                ]
            )
        ]
    )
    assert document["candidates"] == []


def test_host_to_device_copies_become_a_residency_candidate():
    """Repeated H2D copies from a CPU source classify as keep-device-resident."""
    document = evidence.build_evidence(
        [
            _report(
                host_calls=[
                    {
                        "api": "torch.Tensor.to",
                        "site": "prope/camera_rope.py:88:apply_rotary_emb",
                        "count": 4000,
                        "wall_s": 12.0,
                        "bytes": 4 * 1024 * 1024 * 1024,
                        "shape_sigs": ["(1, 6240, 128)@torch.float32"],
                        "callers": [],
                    }
                ]
            )
        ]
    )
    candidate = document["candidates"][0]
    assert candidate["category"] == evidence.CATEGORY_DEVICE_RESIDENT
    assert candidate["taxonomy"] == "f"
    assert candidate["mib_per_rank"] == pytest.approx(4096.0, abs=1.0)


def test_adjacent_same_shape_collectives_become_a_fusion_candidate():
    """Several lines in one function moving one shape is the fusion signal.

    A collective wrapped in a framework helper attributes to a single line inside
    that helper however often it runs, so the enclosing call lines are the only
    thing that separates three adjacent q/k/v exchanges (fusable) from one
    exchange in a loop (not fusable).
    """
    document = evidence.build_evidence(
        [
            _report(
                host_calls=[
                    {
                        "api": "torch.distributed.all_to_all_single",
                        "site": "utils/communications.py:75:_all_to_all_4D",
                        "count": 3000,
                        "wall_s": 30.0,
                        "bytes": 0,
                        "shape_sigs": ["(2, 2, 6240, 128)@torch.bfloat16"],
                        "callers": [
                            "models/attention.py:412:sequence_parallel_attention_vision",
                            "models/attention.py:413:sequence_parallel_attention_vision",
                            "models/attention.py:414:sequence_parallel_attention_vision",
                        ],
                    }
                ]
            )
        ]
    )
    fusion = [c for c in document["candidates"] if c["category"] == evidence.CATEGORY_FUSE_COLLECTIVES]
    assert len(fusion) == 1
    assert fusion[0]["taxonomy"] == "d"
    assert len(fusion[0]["call_lines"]) == 3
    assert fusion[0]["site"] == "models/attention.py:sequence_parallel_attention_vision"


def test_one_collective_in_a_loop_is_not_a_fusion_candidate():
    """A single call line, however hot, offers nothing to fuse with."""
    document = evidence.build_evidence(
        [
            _report(
                host_calls=[
                    {
                        "api": "torch.distributed.all_to_all_single",
                        "site": "utils/communications.py:75:_all_to_all_4D",
                        "count": 3000,
                        "wall_s": 30.0,
                        "bytes": 0,
                        "shape_sigs": ["(2, 2, 6240, 128)@torch.bfloat16"],
                        "callers": ["models/attention.py:412:sequence_parallel_attention_vision"],
                    }
                ]
            )
        ]
    )
    assert not [c for c in document["candidates"] if c["category"] == evidence.CATEGORY_FUSE_COLLECTIVES]


def test_mixed_payload_shapes_are_not_a_fusion_candidate():
    """Fusion needs one identical payload shape; mixed shapes cannot be packed."""
    document = evidence.build_evidence(
        [
            _report(
                host_calls=[
                    {
                        "api": "torch.distributed.all_to_all_single",
                        "site": "utils/communications.py:75:_all_to_all_4D",
                        "count": 3000,
                        "wall_s": 30.0,
                        "bytes": 0,
                        "shape_sigs": ["(2, 2, 6240, 128)@torch.bfloat16", "(2, 2, 512, 128)@torch.bfloat16"],
                        "callers": [
                            "models/attention.py:412:sequence_parallel_attention_vision",
                            "models/attention.py:413:sequence_parallel_attention_vision",
                        ],
                    }
                ]
            )
        ]
    )
    assert not [c for c in document["candidates"] if c["category"] == evidence.CATEGORY_FUSE_COLLECTIVES]


def test_repeated_identical_arguments_become_a_memoize_candidate():
    """A high strict-repeat rate means the process recomputes what it already had."""
    document = evidence.build_evidence(
        [
            _report(
                framework_calls=[
                    {
                        "function": "prope/camera_rope.py:210:_prepare_apply_fns_all_dim",
                        "count": 2000,
                        "wall_s": 33.0,
                        "arg_samples": 256,
                        "strict_distinct": 4,
                        "loose_distinct": 4,
                        "strict_repeat_rate": 0.984,
                        "loose_repeat_rate": 0.984,
                    }
                ]
            )
        ]
    )
    candidate = document["candidates"][0]
    assert candidate["category"] == evidence.CATEGORY_MEMOIZE
    assert candidate["taxonomy"] == "a"
    assert not candidate.get("enabler")
    assert document["deep_probe_ran"] is True
    # Purity is a precondition the probe cannot check, and a real run showed
    # coarse module forwards topping the list at a 96% repeat rate while being
    # impossible to memoize. The caveat has to travel with the candidate.
    assert "PURE function" in candidate["signal"]


def test_setup_phase_work_is_marked_and_demoted():
    """One-time weight loading must not outrank per-step work.

    The numbers here are the real ones from an 8-rank scriptable run, because the
    obvious version of this discriminator failed on exactly this shape. Weight
    loading took 580s of a 644s process, so the generation phase spanned under 10%
    of wall clock; a "must span enough of the run" floor therefore marked *every*
    genuine finding as set-up. What separates them is whether a site was still
    being called when the process finished: set-up stops at 402s / 438s / 547s,
    while the per-step collective runs to 642s of 644s.
    """
    document = evidence.build_evidence(
        [
            _report(
                wall_seconds=644.6,
                host_calls=[
                    {
                        "api": "torch.Tensor.to",
                        "site": "pipelines/pipeline.py:2106:create_pipeline",
                        "count": 218,
                        "wall_s": 29.7,
                        "bytes": 5 * 1024**3,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 402.7,
                        "last_s": 432.5,
                    },
                    {
                        "api": "torch.Tensor.to",
                        "site": "text_encoders/__init__.py:123:load_text_encoder",
                        "count": 340,
                        "wall_s": 3.0,
                        "bytes": 13 * 1024**3,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 544.3,
                        "last_s": 547.7,
                    },
                    {
                        "api": "torch.distributed.all_gather_object",
                        "site": "utils/communications.py:53:_all_to_all_4D",
                        "count": 26892,
                        "wall_s": 26.5,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 580.6,
                        "last_s": 642.0,
                    },
                ],
            )
        ]
    )
    ranked = document["candidates"]
    # The per-step finding outranks the set-up burst that spent more wall time.
    assert ranked[0]["site"] == "utils/communications.py:53:_all_to_all_4D"
    assert not ranked[0].get("setup_phase")
    # It is the hottest site, so it anchors the hot loop at its own first call.
    assert ranked[0]["hot_loop_start_s"] == pytest.approx(580.6, abs=0.01)
    setup = [row for row in ranked if row.get("setup_phase")]
    assert len(setup) == 2
    assert all(row["last_call_s"] < row["hot_loop_start_s"] for row in setup)
    assert all("one-time set-up" in row["signal"] for row in setup)


def test_a_constructor_that_stops_being_called_is_demoted_like_setup_host_work():
    """Tier-2 candidates need the same set-up discriminator the tier-1 sites got.

    Model construction calls each module's ``__init__`` once per sub-module, which
    looks exactly like a repeated argument identity: on a real deep run two
    constructors ranked inside the top 40 as memoization candidates. Nothing can
    be memoized there — the calls stop before the first denoising step — so the
    timestamps have to travel with the framework rows too, and the anchor has to be
    shared with the tier-1 table rather than derived per table.

    The low-frequency rows are not padding: a real deep leg carries 49 host sites
    and 272 framework sites, so the median call count sits near 24. A three-row
    fixture would put the median up next to the hottest site and trip the
    flat-distribution guard, testing the guard instead of the discriminator.
    """
    document = evidence.build_evidence(
        [
            _report(
                wall_seconds=644.6,
                host_calls=[
                    {
                        "api": "torch.distributed.all_gather_object",
                        "site": "utils/communications.py:53:_all_to_all_4D",
                        "count": 26892,
                        "wall_s": 26.5,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 580.6,
                        "last_s": 642.0,
                    },
                    {
                        "api": "torch.Tensor.tolist",
                        "site": "commons/parallel_states.py:62:sp_rank",
                        "count": 24,
                        "wall_s": 0.01,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 581.0,
                        "last_s": 640.0,
                    },
                    {
                        "api": "torch.Tensor.item",
                        "site": "pipelines/pipeline.py:1240:_ar_rollout_inner",
                        "count": 16,
                        "wall_s": 0.01,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 582.0,
                        "last_s": 639.0,
                    },
                ],
                framework_calls=[
                    {
                        "function": "models/transformers/modulate_layers.py:27:__init__",
                        "count": 1200,
                        "wall_s": 40.0,
                        "arg_samples": 256,
                        "strict_distinct": 4,
                        "loose_distinct": 4,
                        "strict_repeat_rate": 0.98,
                        "loose_repeat_rate": 0.98,
                        "first_s": 12.0,
                        "last_s": 61.0,
                    },
                    {
                        "function": "prope/camera_rope.py:210:_prepare_apply_fns_all_dim",
                        "count": 2000,
                        "wall_s": 33.0,
                        "arg_samples": 256,
                        "strict_distinct": 4,
                        "loose_distinct": 4,
                        "strict_repeat_rate": 0.98,
                        "loose_repeat_rate": 0.98,
                        "first_s": 581.0,
                        "last_s": 641.0,
                    },
                ],
            )
        ]
    )
    ranked = document["candidates"]
    by_site = {row["site"]: row for row in ranked}
    constructor = by_site["models/transformers/modulate_layers.py:27:__init__"]
    per_step = by_site["prope/camera_rope.py:210:_prepare_apply_fns_all_dim"]
    # The constructor burned more wall time, so only the timestamps can demote it.
    assert constructor.get("setup_phase") is True
    assert "not steady-state per-step work" in constructor["signal"]
    assert "model construction" in constructor["signal"]
    assert not per_step.get("setup_phase")
    assert per_step["rank"] < constructor["rank"]
    # The anchor came from the tier-1 table, which is the point of deriving it
    # across both: a tier-2 row cannot see the collective that marks the loop.
    assert constructor["hot_loop_start_s"] == pytest.approx(580.6, abs=0.01)


def test_a_long_tail_after_the_hot_loop_does_not_demote_the_hot_loop():
    """A benchmark leg does not end when its hot loop ends, and the split must survive that.

    These are the real numbers from the first live orchestrator leg, where the
    previous discriminator marked all 20 candidates as set-up work — including the
    object collective that ranks first on every other measurement. It anchored on
    the latest call across all sites, and a ``barrier`` called **5 times** spanned
    518s to 1393s while the denoising loop finished at 995s. 995/1393 = 0.714, below
    the threshold, so the hot path was labelled one-time set-up and the specialist
    would have been told in writing that rewriting the biggest lever on the workload
    cannot change steady-state throughput.

    Set-up is not "stops early in absolute terms" — with a long weight load the hot
    loop starts late, and with a long tail it ends early. It is "finished before the
    hot loop started", and the hottest site by call count is what marks that start.
    """
    document = evidence.build_evidence(
        [
            _report(
                wall_seconds=2476.65,
                host_calls=[
                    {
                        "api": "torch.Tensor.cpu",
                        "site": "hyvideo/utils/communications.py:53:_all_to_all_4D",
                        "count": 2393792,
                        "wall_s": 60.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 160.4,
                        "last_s": 995.3,
                    },
                    {
                        # Five calls, spanning most of the leg: phase barriers.
                        "api": "torch.distributed.barrier",
                        "site": "mypkg/pipelines/pipeline.py:1900:__call__",
                        "count": 5,
                        "wall_s": 120.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 518.6,
                        "last_s": 1393.5,
                    },
                    {
                        # Weight loading: more wall time than the hot-path finding,
                        # which is why it has to be demoted rather than out-ranked.
                        "api": "torch.Tensor.to",
                        "site": "mypkg/pipelines/pipeline.py:2106:create_pipeline",
                        "count": 218,
                        "wall_s": 29.7,
                        "bytes": 5 * 1024**3,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 8.2,
                        "last_s": 74.5,
                    },
                ],
            )
        ]
    )
    by_site = {row["site"]: row for row in document["candidates"]}
    hot = by_site["hyvideo/utils/communications.py:53:_all_to_all_4D"]
    setup = by_site["mypkg/pipelines/pipeline.py:2106:create_pipeline"]

    # The hot loop itself, which the tail must not demote.
    assert not hot.get("setup_phase")
    assert hot["rank"] == 1
    assert hot["hot_loop_start_s"] == pytest.approx(160.4, abs=0.01)
    # Weight loading finished before the hot loop began, so it is demoted despite
    # spending more wall time (29.7s) than the finding that now outranks it.
    assert setup.get("setup_phase") is True
    assert "before the hot loop" in setup["signal"]
    # `barrier` is in no category, so it never becomes a candidate — which is what
    # made this bug so easy to miss. It still lands in the merged table, and under
    # the old anchor five calls were enough to redefine "the end of activity" and
    # demote everything else.
    assert "mypkg/pipelines/pipeline.py:1900:__call__" not in by_site


def test_a_flat_call_distribution_demotes_nothing():
    """With no dominant call site there is no hot loop to anchor on, so demote nothing.

    The anchor assumes the hottest site sits in the innermost loop, which holds when
    the loop product dominates: measured ratios of hottest-to-median call count were
    2689x, 2808x and 99741x on three real runs. A workload without that structure
    must fail conservatively — a wrong "this is set-up" note is worse than no note,
    because it tells the specialist to skip a site rather than to think about it.
    """
    document = evidence.build_evidence(
        [
            _report(
                wall_seconds=100.0,
                host_calls=[
                    {
                        "api": "torch.Tensor.cpu",
                        "site": f"framework/mod.py:{line}:fn",
                        "count": count,
                        "wall_s": 1.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 50.0,
                        "last_s": 51.0,
                    }
                    for line, count in ((10, 100), (20, 90), (30, 80), (40, 70))
                ],
            )
        ]
    )
    assert not [row for row in document["candidates"] if row.get("setup_phase")]
    assert any("no dominant call site" in note for note in document["notes"])


def test_mostly_uncalled_sites_do_not_anchor_a_hot_loop():
    """A zero median is not a "typical" call count to measure dominance against.

    Substituting 1 for it makes any called site look infinitely dominant, so a
    distribution that carries no loop at all still anchors one, and every site
    that finished early is then labelled set-up on an invented reference.
    """
    table = {
        f"k{i}": {"first_s": 10.0 + i, "last_s": 11.0 + i, "count_per_rank": count}
        for i, count in enumerate((1000, 900, 0, 0, 0, 0, 0))
    }
    assert evidence._hot_loop_start(table) is None
    # The same shape with a real median still anchors, so this is not a blanket refusal.
    called = {
        f"k{i}": {"first_s": 10.0 + i, "last_s": 11.0 + i, "count_per_rank": count}
        for i, count in enumerate((100000, 5, 4, 3, 2, 1, 1))
    }
    assert evidence._hot_loop_start(called) == 10.0


def test_a_collective_fused_only_during_setup_is_demoted():
    """Fusion was the one category that never checked the hot-loop anchor.

    Collectives issued while the model is being built are genuinely fusable and
    fusing them buys nothing, so left un-demoted they outrank steady-state finds
    in a list capped at MAX_CANDIDATES.
    """
    document = evidence.build_evidence(
        [
            _report(
                wall_seconds=100.0,
                host_calls=[
                    # The loop: dominant call count, anchors hot_loop_start.
                    {
                        "api": "torch.Tensor.cpu",
                        "site": "framework/loop.py:1:step",
                        "count": 500000,
                        "wall_s": 20.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                        "first_s": 30.0,
                        "last_s": 99.0,
                    },
                    # Adjacent same-shape collectives, all finished before it began.
                    {
                        "api": "torch.distributed.all_gather",
                        "site": "framework/build.py:5:construct",
                        "count": 6,
                        "wall_s": 2.0,
                        "bytes": 1024,
                        "shape_sigs": ["f32[8]"],
                        "callers": [
                            "framework/build.py:10:construct",
                            "framework/build.py:11:construct",
                            "framework/build.py:12:construct",
                        ],
                        "first_s": 1.0,
                        "last_s": 2.0,
                    },
                ],
            )
        ]
    )
    fusion = [row for row in document["candidates"] if row["category"] == evidence.CATEGORY_FUSE_COLLECTIVES]
    assert fusion, "the fusion candidate disappeared"
    assert fusion[0].get("setup_phase") is True


def test_a_report_without_timestamps_is_not_assumed_to_be_setup():
    """An older probe report has no timing, so nothing may be demoted on absent evidence."""
    document = evidence.build_evidence(
        [
            _report(
                host_calls=[
                    {
                        "api": "torch.distributed.all_gather_object",
                        "site": "c.py:1:f",
                        "count": 5000,
                        "wall_s": 10.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                    }
                ]
            )
        ]
    )
    candidate = document["candidates"][0]
    assert not candidate.get("setup_phase")
    assert "last_call_frac" not in candidate


def test_probe_reports_are_keyed_by_pid_not_only_rank(probe_module, tmp_path, monkeypatch):
    """Two processes sharing a RANK must not overwrite each other's report.

    Observed on a real 8-rank run: a 0.8-second helper process inherited RANK=0,
    installed its own probe, and its exit handler overwrote rank 0's complete
    report with an empty one — silently losing an eighth of the evidence.
    """
    monkeypatch.setenv("RANK", "0")
    out = tmp_path / "probe"
    first = probe_module.HostProbe(out_dir=str(out), roots=()).write_report()
    monkeypatch.setattr(probe_module.os, "getpid", lambda: 999999)
    second = probe_module.HostProbe(out_dir=str(out), roots=()).write_report()
    assert first != second
    assert len(list(out.glob(evidence.PROBE_FILE_GLOB))) == 2
    assert Path(second).name == "hl_host_probe_rank0_pid999999.json"


def test_the_aggregator_reads_every_pid_report(tmp_path):
    """The glob has to keep matching the pid-suffixed name."""
    (tmp_path / "hl_host_probe_rank0_pid100.json").write_text(json.dumps(_report(rank=0)), encoding="utf-8")
    (tmp_path / "hl_host_probe_rank0_pid200.json").write_text(json.dumps(_report(rank=0)), encoding="utf-8")
    assert len(evidence.read_probe_reports(tmp_path)) == 2


def test_reallocated_invariant_arguments_become_a_hoist_enabler():
    """High loose repeats with low strict repeats is the loop-hoist signature.

    The arguments are logically the same value rebuilt every iteration, so a
    cache keyed on tensor identity can never hit. The candidate must be marked as
    an enabler, because measured on its own it shows no gain and would otherwise
    be discarded along with everything it unlocks.
    """
    document = evidence.build_evidence(
        [
            _report(
                framework_calls=[
                    {
                        "function": "mypkg/transformer.py:301:_expand_geometry",
                        "count": 1000,
                        "wall_s": 20.0,
                        "arg_samples": 256,
                        "strict_distinct": 256,
                        "loose_distinct": 2,
                        "strict_repeat_rate": 0.0,
                        "loose_repeat_rate": 0.992,
                    }
                ]
            )
        ]
    )
    candidate = document["candidates"][0]
    assert candidate["category"] == evidence.CATEGORY_HOIST
    assert candidate["taxonomy"] == "b"
    assert candidate["enabler"] is True
    assert "would never hit" in candidate["signal"]
    assert "enabler" in candidate["suggested_rewrite"]


def test_a_recomputed_constant_is_a_memoize_candidate_not_a_hoist_one():
    """A high strict rate wins over a high loose rate: nothing needs hoisting.

    Both rates are high whenever the same object arrives repeatedly, so the
    classifier has to prefer the strict verdict or every memoization candidate
    would be misfiled as an enabler.
    """
    document = evidence.build_evidence(
        [
            _report(
                framework_calls=[
                    {
                        "function": "prope/camera_rope.py:210:_prepare_apply_fns_all_dim",
                        "count": 2000,
                        "wall_s": 33.0,
                        "arg_samples": 256,
                        "strict_distinct": 4,
                        "loose_distinct": 1,
                        "strict_repeat_rate": 0.984,
                        "loose_repeat_rate": 0.996,
                    }
                ]
            )
        ]
    )
    assert document["candidates"][0]["category"] == evidence.CATEGORY_MEMOIZE


def test_a_mixed_site_is_a_hoist_candidate_but_not_a_pure_enabler():
    """Partly-stable arguments mean part of the win is already reachable.

    Calling such a site a pure enabler would overstate the dependency, which
    matters because the enabler flag is what buys a candidate an exemption from
    being judged on its standalone gain.
    """
    document = evidence.build_evidence(
        [
            _report(
                framework_calls=[
                    {
                        "function": "mypkg/transformer.py:301:_expand_geometry",
                        "count": 1600,
                        "wall_s": 20.0,
                        "arg_samples": 160,
                        "strict_distinct": 81,
                        "loose_distinct": 2,
                        "strict_repeat_rate": 0.49,
                        "loose_repeat_rate": 0.99,
                    }
                ]
            )
        ]
    )
    candidate = document["candidates"][0]
    assert candidate["category"] == evidence.CATEGORY_HOIST
    assert candidate["enabler"] is False


def test_hoist_signal_states_its_own_limitation():
    """The candidate says the premise needs confirming against the source.

    "Loose repeat" means shape/dtype/device matched, not that the values were
    equal — the probe never reads tensor contents, because a device-to-host read
    per sampled call would inject the very stalls it measures. Presenting the
    inference as proof would send a specialist to rewrite working code.
    """
    document = evidence.build_evidence(
        [
            _report(
                framework_calls=[
                    {
                        "function": "m.py:1:f",
                        "count": 500,
                        "wall_s": 5.0,
                        "arg_samples": 256,
                        "strict_distinct": 256,
                        "loose_distinct": 1,
                        "strict_repeat_rate": 0.0,
                        "loose_repeat_rate": 0.996,
                    }
                ]
            )
        ]
    )
    signal = document["candidates"][0]["signal"]
    assert "Confirm against the source" in signal
    assert "does not read tensor contents" in signal


def test_a_genuinely_varying_function_is_not_a_candidate():
    """Neither rate firing means the work is real, not redundant."""
    document = evidence.build_evidence(
        [
            _report(
                framework_calls=[
                    {
                        "function": "models/attention.py:40:forward",
                        "count": 5000,
                        "wall_s": 90.0,
                        "arg_samples": 256,
                        "strict_distinct": 256,
                        "loose_distinct": 256,
                        "strict_repeat_rate": 0.0,
                        "loose_repeat_rate": 0.0,
                    }
                ]
            )
        ]
    )
    assert document["candidates"] == []


def test_missing_deep_probe_is_called_out():
    """A tier-1-only report says so, so absent categories are not read as absent wins."""
    document = evidence.build_evidence([_report()])
    assert document["deep_probe_ran"] is False
    assert any("deep probe did not run" in note for note in document["notes"])


def test_multi_rank_counts_are_normalised_per_rank():
    """Merged counts stay comparable to one run regardless of world size."""
    reports = [
        _report(
            rank=rank,
            host_calls=[
                {
                    "api": "torch.Tensor.item",
                    "site": "utils/flash_attn_no_pad.py:40:unpad",
                    "count": 1000,
                    "wall_s": 4.0,
                    "bytes": 0,
                    "shape_sigs": [],
                    "callers": [],
                }
            ],
        )
        for rank in range(8)
    ]
    document = evidence.build_evidence(reports)
    assert document["ranks_merged"] == 8
    candidate = document["candidates"][0]
    assert candidate["category"] == evidence.CATEGORY_HOST_SYNC
    assert candidate["count_per_rank"] == 1000
    assert candidate["wall_s_per_rank"] == pytest.approx(4.0)


def test_candidates_are_ranked_by_measured_cost():
    """The costliest candidate ranks first; the specialist reads a bounded list."""
    document = evidence.build_evidence(
        [
            _report(
                host_calls=[
                    {
                        "api": "torch.Tensor.item",
                        "site": "a.py:1:cheap",
                        "count": 500,
                        "wall_s": 1.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                    },
                    {
                        "api": "torch.distributed.all_gather_object",
                        "site": "b.py:1:expensive",
                        "count": 500,
                        "wall_s": 50.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                    },
                ]
            )
        ]
    )
    assert [c["rank"] for c in document["candidates"]] == [1, 2]
    assert document["candidates"][0]["site"] == "b.py:1:expensive"


def test_unreadable_report_is_skipped_not_fatal(tmp_path):
    """A truncated report from a killed rank must not lose the other ranks."""
    (tmp_path / "hl_host_probe_rank0.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "hl_host_probe_rank1.json").write_text(json.dumps(_report(rank=1)), encoding="utf-8")
    reports = evidence.read_probe_reports(tmp_path)
    assert [r["rank"] for r in reports] == [1]


def test_foreign_json_is_rejected(tmp_path):
    """A file matching the glob but not the schema is not treated as evidence."""
    (tmp_path / "hl_host_probe_rank0.json").write_text(json.dumps({"schema": "something/1"}), encoding="utf-8")
    assert evidence.read_probe_reports(tmp_path) == []


def test_aggregate_writes_the_document(tmp_path):
    """The merged document lands at the requested path with its schema."""
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "hl_host_probe_rank0.json").write_text(json.dumps(_report()), encoding="utf-8")
    out = tmp_path / evidence.EVIDENCE_FILENAME
    document = evidence.aggregate_probe_dir(probe_dir, out)
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == evidence.SCHEMA
    assert document["ranks_merged"] == 1


def test_prompt_summary_flags_enablers_and_is_empty_without_candidates():
    """The prompt block names the enabler contract; no candidates renders nothing."""
    assert evidence.summarize_for_prompt({"candidates": []}) == ""
    document = evidence.build_evidence(
        [
            _report(
                framework_calls=[
                    {
                        "function": "m.py:1:f",
                        "count": 1000,
                        "wall_s": 20.0,
                        "arg_samples": 256,
                        "strict_distinct": 256,
                        "loose_distinct": 2,
                        "strict_repeat_rate": 0.0,
                        "loose_repeat_rate": 0.99,
                    }
                ]
            )
        ]
    )
    text = evidence.summarize_for_prompt(document)
    assert "hoist_loop_invariant" in text
    assert "enabler" in text
    assert "enables" in text


# --------------------------------------------------------------------------
# probe env contract
# --------------------------------------------------------------------------


def test_probe_env_carries_dir_roots_and_deep_flag(tmp_path):
    """The env the benchmark process needs, without PYTHONPATH."""
    env = evidence.build_probe_env(
        probe_dir=tmp_path,
        source_roots=["/src/hyvideo", "  ", "/opt/rocm"],
        deep=True,
    )
    assert env["HYPERLOOM_HOST_PROBE"] == "1"
    assert env["HYPERLOOM_HOST_PROBE_DIR"] == str(tmp_path)
    assert env["HYPERLOOM_HOST_PROBE_ROOTS"] == "/src/hyvideo:/opt/rocm"
    assert env["HYPERLOOM_HOST_PROBE_DEEP"] == "1"
    # PYTHONPATH must be prepended to the config's own value, never replaced.
    assert "PYTHONPATH" not in env


def test_evidence_reaches_shared_state_from_a_composite_result():
    """The evidence is useless if the path never lands on SharedState.

    On a live session the probe produced 29 classified candidates and the
    specialist still reported "No host-side profiling evidence was available this
    round" — it had been left to guess landing points from source. The path was
    promoted only by the ``profile`` writeback, while the evidence is produced
    inside the composite ``roofline`` action, whose own promotion path never
    looked for it. Every measurement in the pipeline was collected and then
    dropped one step before the consumer.
    """
    from types import SimpleNamespace

    state = SimpleNamespace(last_framework_rewrite_evidence="")
    promoted = evidence.promote_evidence_path(
        state,
        {"framework_rewrite_evidence": "/runs/roofline/abc/framework_rewrite_evidence.json"},
    )
    assert promoted == "/runs/roofline/abc/framework_rewrite_evidence.json"
    assert state.last_framework_rewrite_evidence == promoted


def test_promoting_evidence_never_clears_a_path_already_on_record():
    """A later result without evidence must not erase the evidence we have.

    The deep leg and the cheap leg do not both produce a document, and a run whose
    probe was disabled produces none at all; treating that as "forget what you
    measured" would silently return the specialist to guessing.
    """
    from types import SimpleNamespace

    state = SimpleNamespace(last_framework_rewrite_evidence="/kept/evidence.json")
    assert evidence.promote_evidence_path(state, {}) == ""
    assert state.last_framework_rewrite_evidence == "/kept/evidence.json"
    assert evidence.promote_evidence_path(state, {"framework_rewrite_evidence": "  "}) == ""
    assert state.last_framework_rewrite_evidence == "/kept/evidence.json"


def test_probe_env_drops_a_bare_site_packages_root(tmp_path):
    """A site-packages root attributes call sites to torch, which is never a rewrite target.

    PolicyGate's allowlist legitimately contains ``dist-packages`` so a patch against
    an installed package such as sglang or vllm can land. Reusing it verbatim for
    call-site attribution is wrong: torch lives there too, so on the first live leg
    six of the top ten candidates pointed inside
    ``torch/distributed/distributed_c10d.py`` — code the specialist must not touch,
    and which displaced the framework call sites that actually reach those
    collectives. A specific package directory stays; the root it sits in does not.
    """
    env = evidence.build_probe_env(
        probe_dir=tmp_path,
        source_roots=[
            "/usr/local/lib/python3.12/dist-packages/",
            "/usr/lib/python3/dist-packages",
            "/opt/venv/lib/python3.12/site-packages/",
            "/usr/local/lib/python3.12/dist-packages/sglang/",
            "/opt/framework-checkout/",
            "/opt/rocm/",
        ],
    )
    kept = env["HYPERLOOM_HOST_PROBE_ROOTS"].split(os.pathsep)
    assert "/usr/local/lib/python3.12/dist-packages/sglang/" in kept
    assert "/opt/framework-checkout/" in kept
    assert "/opt/rocm/" in kept
    assert not [r for r in kept if r.rstrip("/").endswith(("dist-packages", "site-packages"))]


def test_probe_env_keeps_the_probe_off_when_every_root_is_too_wide(tmp_path):
    """Dropping every root must not silently leave the probe attributing to torch.

    With no usable root the probe falls back to the innermost frame, which is what
    produced the torch-internal candidates in the first place. Saying so in the
    report is the difference between "the framework has no host-side findings" and
    "nothing told the probe where the framework is".
    """
    env = evidence.build_probe_env(
        probe_dir=tmp_path,
        source_roots=["/usr/local/lib/python3.12/dist-packages/"],
    )
    assert "HYPERLOOM_HOST_PROBE_ROOTS" not in env


def test_probe_env_omits_deep_and_roots_when_not_asked(tmp_path):
    """Tier 2 and root attribution are both opt-in."""
    env = evidence.build_probe_env(probe_dir=tmp_path, source_roots=[])
    assert "HYPERLOOM_HOST_PROBE_DEEP" not in env
    assert "HYPERLOOM_HOST_PROBE_ROOTS" not in env


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_probe_can_be_switched_off(monkeypatch, value):
    """An operator chasing a profiler anomaly can take the probe out entirely."""
    monkeypatch.setenv(evidence.ENABLE_ENV, value)
    assert evidence.probe_enabled() is False


@pytest.mark.parametrize("value", ["", "1", "true", "anything-else"])
def test_probe_is_on_by_default(monkeypatch, value):
    """Tier 1 is cheap enough to be the default for the profile action."""
    monkeypatch.setenv(evidence.ENABLE_ENV, value)
    assert evidence.probe_enabled() is True


def test_deep_probe_is_off_by_default(monkeypatch):
    """Tier 2 skews a co-collected torch trace, so it must be requested."""
    monkeypatch.delenv(evidence.DEEP_ENV, raising=False)
    assert evidence.deep_probe_enabled() is False
    monkeypatch.setenv(evidence.DEEP_ENV, "1")
    assert evidence.deep_probe_enabled() is True


# --------------------------------------------------------------------------
# profile executor wiring
# --------------------------------------------------------------------------


def _write_profile_config(path: Path, envs: dict[str, Any] | None = None) -> None:
    """Write a minimal materialized profile YAML.

    Args:
        path: Destination YAML path.
        envs: Contents of ``benchmark.envs``.
    """
    import yaml

    path.write_text(
        yaml.safe_dump({"benchmark": {"framework": "custom", "envs": dict(envs or {})}}),
        encoding="utf-8",
    )


def test_probe_injection_prepends_to_an_existing_pythonpath(tmp_path, monkeypatch):
    """The probe dir is prepended, never substituted for the framework's own path.

    Replacing ``PYTHONPATH`` would break imports in any framework that sets it,
    which is why the injection edits the materialized YAML instead of going
    through ``extra_envs`` (which overrides).
    """
    import yaml

    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

    monkeypatch.delenv(evidence.ENABLE_ENV, raising=False)
    monkeypatch.delenv(evidence.DEEP_ENV, raising=False)
    config = tmp_path / "profile.yaml"
    _write_profile_config(config, {"PYTHONPATH": "/opt/framework/lib", "TP": 8})

    probe_dir = ProfileExecutor()._inject_host_probe(config, tmp_path / "ws")

    envs = yaml.safe_load(config.read_text(encoding="utf-8"))["benchmark"]["envs"]
    entries = envs["PYTHONPATH"].split(":")
    assert entries[0] == str(evidence.probe_asset_dir())
    assert "/opt/framework/lib" in entries
    assert envs["TP"] == 8
    assert envs["HYPERLOOM_HOST_PROBE"] == "1"
    assert envs["HYPERLOOM_HOST_PROBE_DIR"] == probe_dir
    assert "HYPERLOOM_HOST_PROBE_DEEP" not in envs
    assert Path(probe_dir).is_dir()


def test_probe_injection_is_idempotent(tmp_path, monkeypatch):
    """Re-arming does not stack duplicate PYTHONPATH entries."""
    import yaml

    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

    monkeypatch.delenv(evidence.ENABLE_ENV, raising=False)
    config = tmp_path / "profile.yaml"
    _write_profile_config(config)
    executor = ProfileExecutor()
    executor._inject_host_probe(config, tmp_path / "ws")
    executor._inject_host_probe(config, tmp_path / "ws")
    entries = yaml.safe_load(config.read_text(encoding="utf-8"))["benchmark"]["envs"]["PYTHONPATH"].split(":")
    assert entries.count(str(evidence.probe_asset_dir())) == 1


def test_probe_injection_respects_the_off_switch(tmp_path, monkeypatch):
    """With the probe switched off the config is left untouched."""
    import yaml

    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

    monkeypatch.setenv(evidence.ENABLE_ENV, "0")
    config = tmp_path / "profile.yaml"
    _write_profile_config(config, {"TP": 8})
    assert ProfileExecutor()._inject_host_probe(config, tmp_path / "ws") == ""
    envs = yaml.safe_load(config.read_text(encoding="utf-8"))["benchmark"]["envs"]
    assert envs == {"TP": 8}


def test_evidence_collection_annotates_the_profile_result(tmp_path, monkeypatch):
    """A successful aggregation surfaces the document path on the result."""
    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

    workspace = tmp_path / "ws"
    probe_dir = workspace / evidence.PROBE_SUBDIR
    probe_dir.mkdir(parents=True)
    (probe_dir / "hl_host_probe_rank0.json").write_text(
        json.dumps(
            _report(
                host_calls=[
                    {
                        "api": "torch.distributed.all_gather_object",
                        "site": "utils/communications.py:60:_all_to_all_4D",
                        "count": 5000,
                        "wall_s": 10.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    executor = ProfileExecutor()
    executor._host_probe_dir = str(probe_dir)
    result: dict[str, Any] = {}
    executor._collect_rewrite_evidence(result)
    assert result["framework_rewrite_candidate_count"] == 1
    assert Path(result["framework_rewrite_evidence"]) == workspace / evidence.EVIDENCE_FILENAME


def test_evidence_collection_is_silent_without_candidates(tmp_path):
    """No candidates means no result key, so downstream cannot read an empty doc."""
    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

    probe_dir = tmp_path / "ws" / evidence.PROBE_SUBDIR
    probe_dir.mkdir(parents=True)
    (probe_dir / "hl_host_probe_rank0.json").write_text(json.dumps(_report()), encoding="utf-8")
    executor = ProfileExecutor()
    executor._host_probe_dir = str(probe_dir)
    result: dict[str, Any] = {}
    executor._collect_rewrite_evidence(result)
    assert "framework_rewrite_evidence" not in result


def test_evidence_collection_reports_that_it_was_unarmed():
    """An unarmed leg publishes no document, and says so rather than staying mute.

    A bare empty result is indistinguishable from a probe that ran and found
    nothing, which is the difference between "look elsewhere" and "the
    instrument is broken".
    """
    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

    result: dict[str, Any] = {}
    ProfileExecutor()._collect_rewrite_evidence(result)
    assert result == {"framework_rewrite_evidence_status": "probe_not_armed"}


def test_evidence_collection_reports_an_aggregation_failure(tmp_path, monkeypatch):
    """A crash while merging the reports must not read as 'nothing to rewrite'."""
    from hyperloom.orchestrator.actions.executors import _framework_rewrite_evidence as _ev
    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

    def _boom(*_args, **_kwargs):
        raise RuntimeError("truncated report")

    monkeypatch.setattr(_ev, "aggregate_probe_dir", _boom)
    executor = ProfileExecutor()
    executor._host_probe_dir = str(tmp_path / "host_probe")
    result: dict[str, Any] = {}
    executor._collect_rewrite_evidence(result)
    assert result["framework_rewrite_evidence_status"].startswith("aggregation_failed")
    assert "truncated report" in result["framework_rewrite_evidence_status"]
    assert "framework_rewrite_evidence" not in result


# --------------------------------------------------------------------------
# gap composition
# --------------------------------------------------------------------------


def test_gap_carries_the_host_side_bottleneck(tmp_path):
    """Host-side evidence contributes its own keyword to the framework arm's gap.

    Without this the gap can only ever name a device-side kernel, so an arm
    dispatched to fix a collective rendezvous would be steered at attention.
    """
    from hyperloom.orchestrator.actions.executors._framework_gap_composer import compose_gap

    document = evidence.build_evidence(
        [
            _report(
                host_calls=[
                    {
                        "api": "torch.distributed.all_gather_object",
                        "site": "utils/communications.py:60:_all_to_all_4D",
                        "count": 5000,
                        "wall_s": 40.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                    }
                ]
            )
        ]
    )
    path = tmp_path / evidence.EVIDENCE_FILENAME
    path.write_text(json.dumps(document), encoding="utf-8")

    gap, keywords = compose_gap(
        framework="custom",
        gpu_type="mi355x",
        precision="bf16",
        rewrite_evidence_path=path,
    )
    assert "collective_rendezvous" in gap
    assert "collective_rendezvous" in keywords


def test_gap_is_unchanged_without_evidence():
    """Omitting the evidence path reproduces the previous gap exactly."""
    from hyperloom.orchestrator.actions.executors._framework_gap_composer import compose_gap

    assert compose_gap(framework="sglang", gpu_type="mi300x", precision="fp8") == compose_gap(
        framework="sglang",
        gpu_type="mi300x",
        precision="fp8",
        rewrite_evidence_path=None,
    )


def test_gap_tolerates_a_missing_evidence_file(tmp_path):
    """A stale path degrades to the manifest-only gap rather than raising."""
    from hyperloom.orchestrator.actions.executors._framework_gap_composer import compose_gap

    gap, keywords = compose_gap(framework="custom", rewrite_evidence_path=tmp_path / "absent.json")
    assert gap == "improve custom throughput"
    assert keywords == ["custom"]


def test_site_stats_record_caps_shape_sigs_and_callers_while_tallies_grow(probe_module):
    """Capped sets must stop growing while count/wall/nbytes and first/last keep moving."""
    stats = probe_module._SiteStats()
    width = probe_module._MAX_CONTAINER_WIDTH
    callers_cap = probe_module._MAX_CALLERS_PER_SITE
    for i in range(width + 3):
        # Start at 1.0 (not 0.0) so first_s == 1.0 is distinguishable from the
        # degenerate sentinel value (-1.0 → latched 0.0 when first call passes
        # at_s=0.0).  Both the sentinel-init mutation and the always-overwrite
        # mutation now turn this test red.
        stats.record(elapsed=0.1, nbytes=10, shape_sig=f"s{i}", caller=f"c{i}", at_s=float(i) + 1.0)

    assert stats.count == width + 3
    assert stats.wall_s == pytest.approx(0.1 * (width + 3))
    assert stats.nbytes == 10 * (width + 3)
    assert len(stats.shape_sigs) == width
    assert len(stats.callers) == callers_cap
    assert stats.first_s == 1.0
    assert stats.last_s == float(width + 2) + 1.0

    stats.record(elapsed=0.5, nbytes=7, shape_sig="overflow-shape", caller="overflow-caller", at_s=99.0)
    assert stats.count == width + 4
    assert stats.wall_s == pytest.approx(0.1 * (width + 3) + 0.5)
    assert stats.nbytes == 10 * (width + 3) + 7
    assert "overflow-shape" not in stats.shape_sigs
    assert "overflow-caller" not in stats.callers
    assert stats.last_s == 99.0
    assert stats.first_s == 1.0


def test_site_stats_record_skips_empty_shape_and_caller(probe_module):
    """Falsy shape_sig/caller must not occupy a cap slot."""
    stats = probe_module._SiteStats()
    stats.record(elapsed=0.1, nbytes=10, shape_sig="", caller="", at_s=1.0)
    assert stats.count == 1
    assert "" not in stats.shape_sigs
    assert "" not in stats.callers
    assert stats.first_s == 1.0
