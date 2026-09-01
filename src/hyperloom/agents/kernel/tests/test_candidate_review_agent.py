###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Guards on the agent review of the deterministic kernel-candidate table.

The review exists because the deterministic tiers fail by being confidently
wrong, and it is given a tool-enabled session to check their work. That freedom
is what these tests bound: the session proposes, and everything it proposes is
either verified or dropped before it reaches a candidate row.

Three properties carry the weight, and each has a concrete failure behind it:
a measured field overwritten by a model would corrupt the impact ranking and
the tuning harness that are computed from it; an invented path would hand a
backend the wrong file to rewrite, which is the failure the whole pipeline
exists to prevent; and a session that edited the framework tree would leave the
benchmark that follows measuring an unrecorded change.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

# One import style per module: CodeQL flags mixing ``import x`` with
# ``from x import y``, and reaching the module object is what the sandbox tests
# need in order to substitute its turn runner.
import hyperloom.common.codex_session as codex_session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _candidate_review_agent as cra  # noqa: E402


def _candidate(**overrides) -> dict:
    """A finalized candidate row shaped the way the pipeline emits one."""
    row = {
        "kernel_id": "k001",
        "name": "_gqa_sparse_decode_kernel",
        "gpu_pct": 9.47,
        "duration_us": 1234.0,
        "call_count": 1710,
        "shapes": ["(8192,1024) bf16"],
        "raw_arg_spec": {"0": "tensor"},
        "source_file": "",
        "source_resolution_method": "name_grep",
    }
    row.update(overrides)
    return row


@pytest.fixture
def tree(tmp_path: Path):
    """A framework root holding one real kernel source."""
    root = tmp_path / "vllm"
    (root / "ops").mkdir(parents=True)
    defines = root / "ops" / "sparse_attn.py"
    defines.write_text("def _gqa_sparse_decode_kernel(): pass\n", encoding="utf-8")
    return root, defines


# --- measured fields are evidence, not suggestions --------------------------


class TestImmutableFields:
    def test_trace_measurements_are_declared_immutable(self):
        """Event durations off the trace; also the dispatch floor's input."""
        for field in ("gpu_pct", "duration_us", "call_count"):
            assert field in cra.IMMUTABLE_FIELDS

    def test_join_keys_are_declared_immutable(self):
        """Revising these detaches the row from its ledger and from the CSVs."""
        for field in ("kernel_id", "name", "device_kernel_name"):
            assert field in cra.IMMUTABLE_FIELDS

    def test_judgement_fields_stay_revisable(self):
        """Locking these would leave the review nothing to correct."""
        for field in (
            "source_file",
            "reusable_native_kernel",
            "skip_reason",
            "benchmark_files",
            "shapes",
            "input_dtypes",
        ):
            assert field not in cra.IMMUTABLE_FIELDS

    def test_alternate_shape_representations_are_derived_not_revisable(self):
        """Accepting these alongside shapes is how the three drift apart."""
        for field in ("input_shapes", "invocation_cases", "raw_arg_spec"):
            assert field in cra.DERIVED_SHAPE_FIELDS
            assert field not in cra.IMMUTABLE_FIELDS

    def test_a_revision_cannot_overwrite_what_the_trace_measured(self, tree):
        """The impact ranking and the closing gain figure are computed from these.

        A plausible-looking edit here is indistinguishable from data, so it is
        dropped and reported rather than trusted.
        """
        root, defines = tree
        row = _candidate()
        notes = cra.apply_revisions(
            [row],
            [
                {
                    "kernel_id": "k001",
                    "action": "rewrite",
                    "source_file": str(defines),
                    "gpu_pct": 99.0,
                    "duration_us": 1.0,
                    "name": "something_else",
                }
            ],
            framework_roots=(str(root),),
        )
        assert row["gpu_pct"] == 9.47
        assert row["duration_us"] == 1234.0
        assert row["name"] == "_gqa_sparse_decode_kernel"
        assert any("ignored measured field" in note for note in notes)
        # The judgement half of the same revision still lands.
        assert row["source_file"] == str(defines)

    def test_a_veto_without_a_reason_is_the_input_echoed_back(self, tree):
        """Every unresolved row in the table carries ``reusable_native_kernel``
        false, and a session correcting such a row has been observed returning
        that value while its own prose argued the file is editable.

        Honouring it refused four kernels the review had just located, 16% of
        GPU time, in the run that found this. Nothing separates the echo from an
        intended refusal except the reason the prompt asks for alongside it.
        """
        root, defines = tree
        row = _candidate(source_file="", reusable_native_kernel=False)
        notes = cra.apply_revisions(
            [row],
            [
                {
                    "kernel_id": "k001",
                    "action": "rewrite",
                    "source_file": str(defines),
                    "reusable_native_kernel": False,
                    "skip_reason": "",
                    "reason": "opened it; defines the kernel at line 73",
                }
            ],
            framework_roots=(str(root),),
        )
        assert "review_reusable_hint" not in row
        assert any("veto ignored" in note for note in notes)
        # The half of the revision that was meant still lands.
        assert row["source_file"] == str(defines)

    def test_a_veto_with_a_reason_is_still_honoured(self, tree):
        """The refusal itself stays available; only the silent one is dropped."""
        root, defines = tree
        row = _candidate(source_file="", reusable_native_kernel=False)
        cra.apply_revisions(
            [row],
            [
                {
                    "kernel_id": "k001",
                    "action": "rewrite",
                    "source_file": str(defines),
                    "reusable_native_kernel": False,
                    "skip_reason": "vendor template; only its launch config is tunable",
                }
            ],
            framework_roots=(str(root),),
        )
        assert row["review_reusable_hint"] is False
        assert row["review_skip_reason"] == "vendor template; only its launch config is tunable"

    def test_a_derived_representation_is_dropped_with_a_note(self, tree):
        """Silently ignoring a field the prompt discusses is how drift hides."""
        root, defines = tree
        row = _candidate(invocation_cases=[{"operation": "real"}])
        notes = cra.apply_revisions(
            [row],
            [
                {
                    "kernel_id": "k001",
                    "action": "rewrite",
                    "source_file": str(defines),
                    "invocation_cases": [{"operation": "invented"}],
                    "raw_arg_spec": {"0": "invented"},
                }
            ],
            framework_roots=(str(root),),
        )
        assert row["invocation_cases"] == [{"operation": "real"}]
        assert any("ignored derived field" in note for note in notes)


# --- operand dims the trace never recorded ---------------------------------


class TestShapeProposals:
    """A graph replay records no arguments, so the hottest kernels of a
    captured model arrive with no shape. Left empty, the tuning backend picks
    its own without any view of the serving configuration.
    """

    def test_dims_are_staged_for_the_deterministic_pass_not_written(self, tree):
        """Same split as the routability hint: stamping stays the only writer."""
        root, defines = tree
        row = _candidate(shapes=[], source_file=str(defines))
        cra.apply_revisions(
            [row],
            [
                {
                    "kernel_id": "k001",
                    "action": "keep",
                    "shapes": ["(8192,6144) bf16", "(6144,1536) fp4"],
                    "input_dtypes": ["bf16", "fp4"],
                    "shape_provenance": cra.REVIEW_BACKFILL_PROVENANCE,
                }
            ],
            framework_roots=(str(root),),
        )
        assert row["shapes"] == []
        assert row["review_shapes"] == ["(8192,6144) bf16", "(6144,1536) fp4"]
        assert row["review_input_dtypes"] == ["bf16", "fp4"]
        assert row["review_shape_provenance"] == cra.REVIEW_BACKFILL_PROVENANCE

    def test_a_confirmed_path_still_carries_its_dims(self, tree):
        """The rows most needing dims are the ones already resolved correctly.

        A rewrite naming the path the row already holds is not a correction, but
        dropping the whole revision there would discard the shapes proposed with
        it -- which is every kernel the deterministic tiers got right.
        """
        root, defines = tree
        row = _candidate(shapes=[], source_file=str(defines))
        cra.apply_revisions(
            [row],
            [
                {
                    "kernel_id": "k001",
                    "action": "rewrite",
                    "source_file": str(defines),
                    "shapes": ["(64,9216) bf16"],
                }
            ],
            framework_roots=(str(root),),
        )
        assert row["review_shapes"] == ["(64,9216) bf16"]
        # The path did not move, so nothing was recorded as a correction.
        assert "previous_source_file" not in row
        assert row["source_resolution_method"] == "name_grep"

    def test_a_derivation_cannot_be_claimed_as_a_measurement(self, tree):
        """Provenance is the only thing separating a recovered shape from a
        computed one when a tuned kernel later fails to move throughput.
        """
        root, defines = tree
        row = _candidate(shapes=[], source_file=str(defines))
        cra.apply_revisions(
            [row],
            [
                {
                    "kernel_id": "k001",
                    "action": "keep",
                    "shapes": ["(1,1) fp32"],
                    "shape_provenance": "torch_trace",
                }
            ],
            framework_roots=(str(root),),
        )
        assert row["review_shape_provenance"] == cra.REVIEW_DERIVED_PROVENANCE

    def test_an_unlabelled_derivation_is_not_promoted(self, tree):
        root, defines = tree
        row = _candidate(shapes=[], source_file=str(defines))
        cra.apply_revisions(
            [row],
            [{"kernel_id": "k001", "action": "keep", "shapes": ["(8,8) bf16"]}],
            framework_roots=(str(root),),
        )
        assert row["review_shape_provenance"] == cra.REVIEW_DERIVED_PROVENANCE

    def test_an_empty_proposal_is_reported_rather_than_staged(self, tree):
        """Clearing dims is not a correction the review has any use for."""
        root, defines = tree
        row = _candidate(shapes=[], source_file=str(defines))
        notes = cra.apply_revisions(
            [row],
            [{"kernel_id": "k001", "action": "keep", "shapes": []}],
            framework_roots=(str(root),),
        )
        assert "review_shapes" not in row
        assert any("empty shapes proposal" in note for note in notes)

    def test_an_unmentioned_row_keeps_its_dims(self, tree):
        root, _ = tree
        row = _candidate()
        before = dict(row)
        cra.apply_revisions([row], [], framework_roots=(str(root),))
        assert row == before

    def test_review_provenance_is_dispatchable(self):
        """Dims the gate rejects are worse than none: an empty shape has an
        override, an untrusted provenance does not.
        """
        from hyperloom.common.kernel_shape_contract import DISPATCHABLE_SHAPE_PROVENANCE

        for provenance in cra.REVIEW_SHAPE_PROVENANCE:
            assert provenance in DISPATCHABLE_SHAPE_PROVENANCE


# --- a path is taken only when it can be verified ---------------------------


class TestApplyRevisions:
    def test_rewrite_to_a_verified_path_records_what_it_replaced(self, tree):
        root, defines = tree
        row = _candidate(source_file="/gone/wrong.py", source_resolution_method="name_grep")
        notes = cra.apply_revisions(
            [row],
            [{"kernel_id": "k001", "action": "rewrite", "source_file": str(defines), "reason": "defines it"}],
            framework_roots=(str(root),),
        )
        assert row["source_file"] == str(defines)
        assert row["previous_source_file"] == "/gone/wrong.py"
        assert row["previous_method"] == "name_grep"
        assert row["source_resolution_method"] == "llm_review"
        assert row["review_reason"] == "defines it"
        assert notes == [f"k001: /gone/wrong.py -> {defines}"]

    def test_an_invented_path_is_refused(self, tree):
        """Existence is the floor. Without it a backend rewrites a fiction."""
        root, _ = tree
        row = _candidate(source_file="/repo/current.py")
        notes = cra.apply_revisions(
            [row],
            [{"kernel_id": "k001", "action": "rewrite", "source_file": str(root / "ops/invented.py")}],
            framework_roots=(str(root),),
        )
        assert row["source_file"] == "/repo/current.py"
        assert any("rejected unverifiable path" in note for note in notes)

    def test_a_real_path_outside_every_root_is_refused(self, tree, tmp_path):
        """Being openable is not enough; it must be framework source."""
        root, _ = tree
        outside = tmp_path / "elsewhere.py"
        outside.write_text("x = 1\n", encoding="utf-8")
        row = _candidate(source_file="/repo/current.py")
        notes = cra.apply_revisions(
            [row],
            [{"kernel_id": "k001", "action": "rewrite", "source_file": str(outside)}],
            framework_roots=(str(root),),
        )
        assert row["source_file"] == "/repo/current.py"
        assert any("rejected unverifiable path" in note for note in notes)

    def test_rewrite_without_a_path_is_ignored(self, tree):
        root, _ = tree
        row = _candidate(source_file="/repo/current.py")
        notes = cra.apply_revisions(
            [row],
            [{"kernel_id": "k001", "action": "rewrite"}],
            framework_roots=(str(root),),
        )
        assert row["source_file"] == "/repo/current.py"
        assert any("rewrite without a path" in note for note in notes)

    def test_keep_changes_nothing(self, tree):
        root, _ = tree
        row = _candidate(source_file="/repo/current.py")
        before = dict(row)
        assert cra.apply_revisions([row], [{"kernel_id": "k001", "action": "keep"}], framework_roots=(str(root),)) == []
        assert row == before

    @pytest.mark.parametrize("action", ["unresolve", "drop"])
    def test_unresolve_and_drop_clear_the_source(self, tree, action):
        """Both mean "do not send a backend here"; an empty source says so."""
        root, _ = tree
        row = _candidate(source_file="/repo/wrong.py", source_line=42, source_function="launch")
        cra.apply_revisions(
            [row],
            [{"kernel_id": "k001", "action": action, "reason": "dispatch wrapper"}],
            framework_roots=(str(root),),
        )
        assert row["source_file"] == ""
        assert row["previous_source_file"] == "/repo/wrong.py"
        assert row["review_action"] == action
        assert "source_line" not in row and "source_function" not in row

    def test_an_unmentioned_candidate_is_left_alone(self, tree):
        """Silence is not a verdict; only named rows move."""
        root, _ = tree
        rows = [_candidate(), _candidate(kernel_id="k002", source_file="/repo/other.py")]
        cra.apply_revisions(rows, [{"kernel_id": "k001", "action": "unresolve"}], framework_roots=(str(root),))
        assert rows[1]["source_file"] == "/repo/other.py"
        assert "review_action" not in rows[1]

    def test_unknown_id_and_action_are_reported_not_applied(self, tree):
        root, _ = tree
        row = _candidate()
        notes = cra.apply_revisions(
            [row],
            [
                {"kernel_id": "k999", "action": "unresolve"},
                {"kernel_id": "k001", "action": "teleport"},
            ],
            framework_roots=(str(root),),
        )
        assert any("unknown kernel_id" in note for note in notes)
        assert any("unknown action" in note for note in notes)
        assert "review_action" not in row

    def test_an_authoritative_resolution_is_not_overridable(self, tree):
        """The active finder demangles the symbol the binary actually exports.

        Reading the same tree cannot beat knowing that, so a curated resolution
        is protected from a session that merely looked at the source.
        """
        root, defines = tree
        row = _candidate(source_file="/curated/truth.py", source_resolution_method="active_finder")
        notes = cra.apply_revisions(
            [row],
            [{"kernel_id": "k001", "action": "rewrite", "source_file": str(defines)}],
            framework_roots=(str(root),),
            protected_ids={"k001"},
        )
        assert row["source_file"] == "/curated/truth.py"
        assert any("resolved by an authoritative tier" in note for note in notes)

    def test_a_protected_candidate_may_still_be_kept(self, tree):
        """Protection blocks changes, not agreement."""
        root, _ = tree
        row = _candidate(source_file="/curated/truth.py")
        notes = cra.apply_revisions(
            [row],
            [{"kernel_id": "k001", "action": "keep"}],
            framework_roots=(str(root),),
            protected_ids={"k001"},
        )
        assert notes == []

    def test_routability_hints_are_recorded_for_the_caller_to_weigh(self, tree):
        """The gate stays deterministic; the session only leaves a hint."""
        root, defines = tree
        row = _candidate()
        cra.apply_revisions(
            [row],
            [
                {
                    "kernel_id": "k001",
                    "action": "rewrite",
                    "source_file": str(defines),
                    "reusable_native_kernel": False,
                    "skip_reason": "dispatch wrapper",
                }
            ],
            framework_roots=(str(root),),
        )
        assert row["review_reusable_hint"] is False
        assert row["review_skip_reason"] == "dispatch wrapper"
        # Never written directly -- classify_patchability owns this field.
        assert "reusable_native_kernel" not in row


# --- harnesses the session claims must be openable, and in the tree ---------


class TestVerifiedHarnesses:
    def test_absent_paths_are_dropped(self, tree):
        root, defines = tree
        assert cra._verified_harnesses(["/gone/test_pa.py", str(defines)], (str(root),)) == [str(defines)]

    def test_a_real_path_outside_every_root_is_refused(self, tree, tmp_path):
        """Existence is not containment, and this list is what a backend runs.

        Checking ``isfile`` alone accepted any readable path on the host, so a
        proposal could point the measurement at a file outside the tree under
        optimization -- and this list is the only channel by which anything the
        session says about harnesses reaches a backend.
        """
        root, _ = tree
        elsewhere = tmp_path / "outside" / "bench.py"
        elsewhere.parent.mkdir(parents=True)
        elsewhere.write_text("def bench(): pass\n", encoding="utf-8")
        assert elsewhere.is_file()
        assert cra._verified_harnesses([str(elsewhere)], (str(root),)) == []

    def test_an_explicit_empty_list_is_honoured(self, tree):
        """ "This kernel has no harness" is an answer worth keeping."""
        root, _ = tree
        assert cra._verified_harnesses([], (str(root),)) == []

    @pytest.mark.parametrize("proposed", [None, "not-a-list", 42])
    def test_no_proposal_leaves_the_field_alone(self, proposed, tree):
        root, _ = tree
        assert cra._verified_harnesses(proposed, (str(root),)) is None

    def test_a_verified_list_reaches_the_candidate(self, tree):
        root, defines = tree
        row = _candidate()
        notes = cra.apply_revisions(
            [row],
            [
                {
                    "kernel_id": "k001",
                    "action": "rewrite",
                    "source_file": str(defines),
                    "benchmark_files": ["/gone/bench.py", str(defines)],
                }
            ],
            framework_roots=(str(root),),
        )
        assert row["review_benchmark_files"] == [str(defines)]
        assert any("benchmark_files -> 1 verified path" in note for note in notes)


# --- the answer is a file, so a half-written one is not mistaken for one ----


class TestLoadRevisions:
    def test_a_missing_file_is_reported_as_such(self, tmp_path):
        revisions, error = cra.load_revisions(tmp_path / "nope.json")
        assert revisions == [] and "not written" in error

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("{not json", "not valid JSON"),
            ("[1, 2]", "not a JSON object"),
            ('{"other": []}', "no 'revisions' list"),
        ],
    )
    def test_an_unusable_body_names_what_was_wrong(self, tmp_path, body, expected):
        path = tmp_path / cra.REVISIONS_FILENAME
        path.write_text(body, encoding="utf-8")
        revisions, error = cra.load_revisions(path)
        assert revisions == [] and expected in error

    def test_non_dict_entries_are_dropped(self, tmp_path):
        path = tmp_path / cra.REVISIONS_FILENAME
        path.write_text(json.dumps({"revisions": [{"kernel_id": "k001"}, "junk"]}), encoding="utf-8")
        revisions, error = cra.load_revisions(path)
        assert error == "" and revisions == [{"kernel_id": "k001"}]


# --- the prompt hands over paths, never contents ----------------------------


class TestBuildReviewPrompt:
    def test_it_offers_paths_rather_than_pre_loaded_source(self, tmp_path, tree):
        """Pre-loading would bound the review by what was guessed relevant."""
        root, defines = tree
        prompt = cra.build_review_prompt(
            run_dir=tmp_path,
            raw_candidates_path=tmp_path / cra.RAW_CANDIDATES_FILENAME,
            revisions_path=tmp_path / cra.REVISIONS_FILENAME,
            reference_paths={"tracelens report": "/run/analysis.md"},
            framework_roots=(str(root),),
        )
        assert str(tmp_path / cra.RAW_CANDIDATES_FILENAME) in prompt
        assert "/run/analysis.md" in prompt
        assert str(root) in prompt
        # The body of a framework file is never shipped by the prompt itself.
        assert defines.read_text(encoding="utf-8") not in prompt

    def test_it_states_the_actions_and_the_measured_field_ban(self, tmp_path):
        prompt = cra.build_review_prompt(
            run_dir=tmp_path,
            raw_candidates_path=tmp_path / "raw.json",
            revisions_path=tmp_path / "rev.json",
            reference_paths={},
            framework_roots=(),
        )
        for token in ("keep", "rewrite", "unresolve", "drop", "gpu_pct", "benchmark_files"):
            assert token in prompt

    def test_it_asks_for_dims_and_for_how_they_were_obtained(self, tmp_path):
        """An unstated derivation is no more reviewable than the backend's own
        guess, which is what the dims are there to replace.
        """
        prompt = cra.build_review_prompt(
            run_dir=tmp_path,
            raw_candidates_path=tmp_path / "raw.json",
            revisions_path=tmp_path / "rev.json",
            reference_paths={},
            framework_roots=(),
        )
        assert "shapes" in prompt
        assert cra.REVIEW_BACKFILL_PROVENANCE in prompt
        assert cra.REVIEW_DERIVED_PROVENANCE in prompt
        assert "State where the dims came from in reason" in prompt

    def test_it_tells_the_session_not_to_echo_the_routability_field(self, tmp_path):
        """The table it audits carries the field, so "revisable" reads as
        "return it". Saying so cost four located kernels in one run.
        """
        prompt = cra.build_review_prompt(
            run_dir=tmp_path,
            raw_candidates_path=tmp_path / "raw.json",
            revisions_path=tmp_path / "rev.json",
            reference_paths={},
            framework_roots=(),
        )
        assert "Do not copy reusable_native_kernel back" in prompt
        assert "a false with no" in prompt and "skip_reason is ignored" in prompt

    def test_it_does_not_send_the_session_after_tracelens_internals(self, tmp_path):
        """``analysis.md`` is TraceLens' only supported output; the rest of that
        directory is internal and may be deleted.

        Nothing is really given up by staying inside the contract: for every
        operator the sidecars describe, ``analysis.md`` carries the same dims and
        launcher in its own table, and for a graph-launched operator neither has
        anything.
        """
        prompt = cra.build_review_prompt(
            run_dir=tmp_path,
            raw_candidates_path=tmp_path / "raw.json",
            revisions_path=tmp_path / "rev.json",
            reference_paths={},
            framework_roots=(),
        )
        for internal in ("category_data", "priority_data", "perf_report_csvs"):
            assert internal not in prompt
        assert "analysis.md is TraceLens' only supported output" in prompt

    def test_it_does_not_offer_a_field_the_stamping_pass_recomputes(self, tmp_path):
        """Inviting a revision that is then silently overwritten spends the
        session's effort on nothing and hides the overwrite from the audit.
        """
        prompt = cra.build_review_prompt(
            run_dir=tmp_path,
            raw_candidates_path=tmp_path / "raw.json",
            revisions_path=tmp_path / "rev.json",
            reference_paths={},
            framework_roots=(),
        )
        assert "recommended_backends" not in prompt

    def test_write_scope_is_stated(self, tmp_path):
        prompt = cra.build_review_prompt(
            run_dir=tmp_path,
            raw_candidates_path=tmp_path / "raw.json",
            revisions_path=tmp_path / "rev.json",
            reference_paths={},
            framework_roots=(),
        )
        assert f"Write nothing outside {tmp_path}" in prompt


# --- a mandatory pass that must never take the run down with it -------------


class TestRunCandidateReview:
    def _args(self, tmp_path: Path) -> dict:
        return {
            "run_dir": tmp_path,
            "raw_candidates_path": tmp_path / cra.RAW_CANDIDATES_FILENAME,
            "reference_paths": {},
            "framework_roots": (),
        }

    def test_the_written_file_is_what_marks_success(self, tmp_path):
        """Artifact presence decides, as it does for the TraceLens runner.

        A provider can report an error after the answer already landed.
        """

        def _runner(*, prompt, run_dir, model, timeout_sec):
            (run_dir / cra.REVISIONS_FILENAME).write_text(
                json.dumps({"revisions": [{"kernel_id": "k001", "action": "keep"}]}),
                encoding="utf-8",
            )
            return "transport reset after the write"

        out = cra.run_candidate_review(**self._args(tmp_path), session_runner=_runner)
        assert out.ok and out.status == "completed"
        assert out.revisions == [{"kernel_id": "k001", "action": "keep"}]

    def test_a_transient_failure_is_retried(self, tmp_path):
        """The pass is mandatory; a gateway hiccup should not skip the audit."""
        calls = {"n": 0}

        def _runner(*, prompt, run_dir, model, timeout_sec):
            calls["n"] += 1
            if calls["n"] == 1:
                return "gateway 502"
            (run_dir / cra.REVISIONS_FILENAME).write_text(json.dumps({"revisions": []}), encoding="utf-8")
            return ""

        out = cra.run_candidate_review(**self._args(tmp_path), session_runner=_runner)
        assert calls["n"] == 2 and out.ok

    def test_a_definitive_failure_is_reported_not_raised(self, tmp_path):
        """Losing the audit costs candidates; raising would cost the run."""
        out = cra.run_candidate_review(
            **self._args(tmp_path),
            attempts=2,
            session_runner=lambda **_kwargs: "gateway down",
        )
        assert not out.ok and out.status == "failed"
        assert "gateway down" in out.detail

    def test_a_raising_session_is_contained(self, tmp_path):
        def _boom(**_kwargs):
            raise RuntimeError("https://secret.example/?token=leak")

        out = cra.run_candidate_review(**self._args(tmp_path), attempts=1, session_runner=_boom)
        assert not out.ok
        assert "leak" not in out.detail and "RuntimeError" in out.detail

    def test_a_stale_answer_cannot_be_mistaken_for_a_fresh_one(self, tmp_path):
        """Each attempt clears the file first, so silence never reads as success."""
        (tmp_path / cra.REVISIONS_FILENAME).write_text(
            json.dumps({"revisions": [{"kernel_id": "stale", "action": "drop"}]}),
            encoding="utf-8",
        )
        out = cra.run_candidate_review(**self._args(tmp_path), attempts=1, session_runner=lambda **_kwargs: "no answer")
        assert not out.ok

    def test_an_async_caller_awaits_the_coroutine(self, tmp_path):
        """Both backends are async SDKs; a caller owning a loop awaits directly."""

        async def _runner(*, prompt, run_dir, model, timeout_sec):
            (run_dir / cra.REVISIONS_FILENAME).write_text(
                json.dumps({"revisions": [{"kernel_id": "k001", "action": "keep"}]}),
                encoding="utf-8",
            )
            return ""

        out = asyncio.run(cra.run_candidate_review_async(**self._args(tmp_path), session_runner=_runner))
        assert out.ok and out.revisions == [{"kernel_id": "k001", "action": "keep"}]

    def test_the_sync_wrapper_says_what_to_await_instead(self, tmp_path):
        """It owns the loop, so inside one it names the coroutine rather than
        failing deep in ``asyncio.run``."""

        async def _from_a_loop():
            with pytest.raises(RuntimeError, match="run_candidate_review_async"):
                cra.run_candidate_review(**self._args(tmp_path), session_runner=lambda **_kwargs: "")

        asyncio.run(_from_a_loop())


# --- tool scope: the session investigates, it does not patch ----------------


class TestToolScope:
    def test_reading_and_searching_are_allowed(self):
        for tool in ("Read", "Grep", "Glob"):
            assert tool in cra.ALLOWED_TOOLS

    def test_editing_the_framework_tree_is_not(self):
        """The tree here is the code under optimization; the agent proposes."""
        assert "Edit" not in cra.ALLOWED_TOOLS
        assert "Edit" in cra._DENIED_TOOLS

    def test_sub_agents_and_the_web_are_denied(self):
        """Turns and cost stay bounded, and the answers are all local."""
        for tool in ("Task", "WebFetch", "WebSearch"):
            assert tool not in cra.ALLOWED_TOOLS
            assert tool in cra._DENIED_TOOLS

    def test_the_writing_tool_is_not_pre_approved(self):
        """A tool in ``allowed_tools`` skips the permission callback entirely.

        Leaving it there is what made the run-directory boundary a sentence in
        the prompt rather than something the session runs into.
        """
        assert "Write" not in cra.ALLOWED_TOOLS
        assert "Write" not in cra._DENIED_TOOLS
        assert "Write" in cra._GATED_TOOLS

    def test_no_shell_is_granted(self):
        """A shell cannot be confined to the run directory, so it is not given.

        The boundary was previously enforced for ``Write`` and only *detected*
        for ``Bash``, and detection covered the candidate files and the names in
        their directories -- so a sibling helper edited in place, or any file
        under a directory no candidate named, drifted nothing. Demangling was
        the one job that wanted a shell and the host does it now.
        """
        for tool in ("Bash", "BashOutput", "KillShell"):
            assert tool not in cra.ALLOWED_TOOLS
            assert tool not in cra._GATED_TOOLS
            assert tool in cra._DENIED_TOOLS

    def test_only_the_answer_channel_can_write(self):
        """Every granted tool is read-only except the one that reports."""
        assert cra._GATED_TOOLS == ("Write",)


class TestWriteBoundaryGuard:
    """The Claude backend's counterpart to Codex's ``writable_roots``."""

    @staticmethod
    def _decide(run_dir: Path, tool: str, tool_input: dict):
        guard = cra._write_boundary_guard(run_dir)
        return asyncio.run(guard(tool, tool_input, None))

    def test_a_write_inside_the_run_directory_is_allowed(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        decision = self._decide(run_dir, "Write", {"file_path": str(run_dir / cra.REVISIONS_FILENAME)})
        assert decision.behavior == "allow"

    def test_a_write_outside_the_run_directory_is_refused(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        outside = tmp_path / "vllm" / "ops" / "sparse_attn.py"
        decision = self._decide(run_dir, "Write", {"file_path": str(outside)})
        assert decision.behavior == "deny"
        assert str(run_dir) in decision.message

    def test_traversal_out_of_the_run_directory_is_refused(self, tmp_path):
        """``..`` must not spell around the boundary."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        decision = self._decide(run_dir, "Write", {"file_path": "../vllm/ops/attn.py"})
        assert decision.behavior == "deny"

    def test_a_symlink_pointing_out_is_refused(self, tmp_path):
        """Both sides are resolved, so the link target decides, not its name."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        target_dir = tmp_path / "vllm"
        target_dir.mkdir()
        (run_dir / "escape").symlink_to(target_dir, target_is_directory=True)
        decision = self._decide(run_dir, "Write", {"file_path": str(run_dir / "escape" / "x.py")})
        assert decision.behavior == "deny"

    def test_a_write_with_no_recognisable_path_is_refused(self, tmp_path):
        """An unrecognised input spelling must not become a bypass."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        decision = self._decide(run_dir, "Write", {"contents": "x"})
        assert decision.behavior == "deny"

    def test_a_shell_command_is_refused(self, tmp_path):
        """No command string is inspected, because no shell is granted at all.

        Guessing from the command whether it writes is the guard that fails on
        the first construction nobody predicted -- ``python3 -c`` defeats any
        allowlist -- so the capability goes instead of the guess.
        """
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for command in ("c++filt _ZN5aiter4moeE", "cp a.py /tmp/b.py"):
            decision = self._decide(run_dir, "Bash", {"command": command})
            assert decision.behavior == "deny", command

    def test_an_unknown_tool_is_refused_not_allowed(self, tmp_path):
        """The tool set is the SDK's to grow.

        A callback whose fallthrough is "allow" hands every tool added by a
        later SDK the freedom this stage withholds, silently, on an upgrade
        nobody reviewed.
        """
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for tool in ("Edit", "NotebookEdit", "SomeFutureFileMover"):
            decision = self._decide(run_dir, tool, {"file_path": str(run_dir / "x.py")})
            assert decision.behavior == "deny", tool

    def test_the_read_only_tools_agree_with_the_allowlist(self, tmp_path):
        """They are pre-approved and should not arrive, but must not contradict."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        for tool in cra.ALLOWED_TOOLS:
            assert self._decide(run_dir, tool, {}).behavior == "allow", tool


class TestTheCodexBackendIsContainedToo:
    """The backend is chosen by credentials, so the weaker one must still hold.

    Codex has no tool allowlist and no ``can_use_tool``; its containment is the
    OS sandbox. ``writable_roots`` *widens* ``workspace-write`` rather than
    imposing it, and under a deployment-level ``bypass`` it goes unread -- which
    left this stage with no boundary at all on the backend an operator never
    chose, while the docstrings claimed both were confined.
    """

    def _captured(self, monkeypatch, tmp_path) -> dict:
        captured: dict = {}

        async def _fake_turn(**kwargs):
            captured.update(kwargs)
            return type("R", (), {"error": ""})()

        monkeypatch.setattr(codex_session, "run_codex_turn", _fake_turn)
        asyncio.run(
            cra._run_codex_session(
                "prompt",
                run_dir=tmp_path,
                model="gpt-5",
                timeout_sec=900.0,
            )
        )
        return captured

    def test_the_sandbox_mode_is_pinned_not_inherited(self, monkeypatch, tmp_path):
        captured = self._captured(monkeypatch, tmp_path)
        assert captured["sandbox_mode"] == "workspace-write"
        assert captured["writable_roots"] == (tmp_path,)

    def test_a_bypass_deployment_does_not_reach_the_review(self, monkeypatch, tmp_path):
        """Stating the mode outranks the environment, so bypass cannot apply."""
        monkeypatch.setenv("HYPERLOOM_CODEX_SANDBOX_MODE", "bypass")
        captured = self._captured(monkeypatch, tmp_path)
        resolved = codex_session.resolve_codex_sandbox_mode(sandbox_mode=captured["sandbox_mode"])
        assert resolved != "bypass"

    def test_the_containment_in_force_is_recorded(self, monkeypatch, tmp_path):
        """Which backend ran is otherwise invisible to the operator."""
        seen: list[str] = []

        async def _fake_turn(**_kwargs):
            return type("R", (), {"error": ""})()

        monkeypatch.setattr(codex_session, "run_codex_turn", _fake_turn)
        asyncio.run(
            cra._run_codex_session(
                "prompt",
                run_dir=tmp_path,
                model="gpt-5",
                timeout_sec=900.0,
                log=seen.append,
            )
        )
        assert any("sandbox_mode=workspace-write" in line for line in seen)


class TestTheBoundaryIsNotTradedForCompatibility:
    def test_an_sdk_without_can_use_tool_gets_no_session(self, tmp_path, monkeypatch):
        """Widening allowed_tools to keep going is a silent, host-shaped hole.

        The callback is the whole boundary -- nothing sits behind it -- so an
        SDK that rejects it gets no session. Refusing is a reported outcome; the
        stage is advisory, so the deterministic table still stands.
        """
        import claude_agent_sdk as sdk

        class _OptionsWithoutGuard:
            def __init__(self, **kwargs):
                if "can_use_tool" in kwargs:
                    raise TypeError("unexpected keyword argument 'can_use_tool'")

        monkeypatch.setattr(sdk, "ClaudeAgentOptions", _OptionsWithoutGuard)
        error = asyncio.run(
            cra._run_claude_session(
                "prompt",
                run_dir=tmp_path,
                model="m",
                timeout_sec=1.0,
                log=None,
            )
        )
        assert "can_use_tool" in error
        assert "unconfined" in error

    def test_an_sdk_without_cwd_still_gets_the_guard(self, tmp_path, monkeypatch):
        """``cwd`` orders relative paths and carries none of the boundary."""
        import claude_agent_sdk as sdk

        captured: dict = {}

        class _OptionsWithoutCwd:
            def __init__(self, **kwargs):
                if "cwd" in kwargs:
                    raise TypeError("unexpected keyword argument 'cwd'")
                captured.update(kwargs)

        monkeypatch.setattr(sdk, "ClaudeAgentOptions", _OptionsWithoutCwd)
        monkeypatch.setattr(sdk, "query", lambda **_kw: iter(()))
        asyncio.run(cra._run_claude_session("prompt", run_dir=tmp_path, model="m", timeout_sec=1.0, log=None))
        assert captured.get("can_use_tool") is not None
        assert list(captured["allowed_tools"]) == list(cra.ALLOWED_TOOLS)
        for tool in cra._GATED_TOOLS:
            assert tool not in captured["allowed_tools"]


# --- the stage boundary in tracelens_analysis -------------------------------

import argparse  # noqa: E402

import tracelens_analysis as tla  # noqa: E402


class TestReviewStageBoundary:
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(model_name="m", framework="vllm", source_root=None)

    def test_an_unexpected_fault_costs_the_audit_not_the_run(self, tmp_path, monkeypatch):
        """The stage sits at the end of an analysis a benchmark paid for.

        Nothing inside it may propagate: the deterministic table is still
        usable, and killing a multi-hour run over an advisory pass trades a
        small loss for a total one.
        """

        def _boom(*_args, **_kwargs):
            raise RuntimeError("unforeseen")

        monkeypatch.setattr(tla, "_run_candidate_review_stage", _boom)
        warnings: list[dict] = []
        out = tla.run_candidate_review_stage(tmp_path, candidates=[], args=self._args(), trace_health_warnings=warnings)
        assert out == {}
        assert warnings[0]["code"] == "candidate_review_failed"
        assert warnings[0]["severity"] == "error"
        assert warnings[0]["detail"] == "RuntimeError"

    def test_a_clean_stage_passes_its_artifacts_through(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            tla,
            "_run_candidate_review_stage",
            lambda *_a, **_kw: {"kernel_candidates_raw": "/x/raw.json"},
        )
        assert tla.run_candidate_review_stage(tmp_path, candidates=[], args=self._args()) == {
            "kernel_candidates_raw": "/x/raw.json"
        }

    def test_the_session_is_only_pointed_at_supported_outputs(self, tmp_path, monkeypatch):
        """The reference list is what the session is invited to read.

        TraceLens supports ``analysis.md`` and nothing else in that directory,
        so naming a sidecar there both breaks the contract and points the
        session at a file that may not exist.
        """
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return cra.ReviewOutcome(status="failed", detail="not run")

        monkeypatch.setattr(cra, "run_candidate_review", _capture)
        tla._run_candidate_review_stage(tmp_path, candidates=[], args=self._args())

        offered = " ".join(captured["reference_paths"].values())
        for internal in ("category_data", "priority_data", "perf_report_csvs"):
            assert internal not in offered
        assert "analysis.md" in offered

    def test_the_public_audit_is_rebuilt_from_the_reviewed_table(self, tmp_path, monkeypatch):
        """The audit was written during finalize, before this stage ran.

        So a review that moved a ``source_file`` left the artifact a human reads
        naming the old path while kernel_candidates.json named the new one --
        two answers to "where does this kernel live", and the public one was
        the stale one.
        """
        pkg = tmp_path / "aiter" / "ops"
        pkg.mkdir(parents=True)
        wrong = pkg / "caller.py"
        wrong.write_text("from .real import k\n", encoding="utf-8")
        right = pkg / "real.py"
        right.write_text("def k(): pass\n", encoding="utf-8")
        candidate = {
            "kernel_id": "k001",
            "name": "k",
            "gpu_pct": 9.0,
            "source_file": str(wrong),
            "source_resolution_method": "name_grep",
        }

        # Pre-review audit, exactly as finalize leaves it.
        audit_path = tmp_path / tla._SOURCE_RESOLUTION_NAME
        tla.write_source_resolution_artifact([candidate], audit_path, framework="vllm", model_name="m")
        assert json.loads(audit_path.read_text(encoding="utf-8"))["entries"][0]["source_file"] == str(wrong)

        # A revised path is only taken when it resolves under a known root.
        monkeypatch.setattr(tla, "kernel_search_roots", lambda: (str(tmp_path / "aiter"),))
        monkeypatch.setattr(
            cra,
            "run_candidate_review",
            lambda **_kw: cra.ReviewOutcome(
                status="completed",
                revisions=[
                    {
                        "kernel_id": "k001",
                        "action": "rewrite",
                        "source_file": str(right),
                        "reason": "caller.py imports it; real.py defines it",
                    }
                ],
            ),
        )
        artifacts = tla._run_candidate_review_stage(
            tmp_path,
            candidates=[candidate],
            args=self._args(),
            trace_health_warnings=[],
        )

        assert candidate["source_file"] == str(right)
        assert artifacts["kernel_source_resolution"] == str(audit_path)
        entry = json.loads(audit_path.read_text(encoding="utf-8"))["entries"][0]
        assert entry["source_file"] == str(right)
        # And the displaced path is recorded rather than dropped. The audit read
        # ``source_resolution_previous_file``, which nothing writes, so this
        # column was empty for the one case it exists to record.
        assert entry["previous_source_file"] == str(wrong)
        assert entry["previous_method"] == "name_grep"

    def test_the_session_is_handed_the_demangled_symbol(self, tmp_path, monkeypatch):
        """Demangling was the one job that wanted a shell.

        A shell cannot be confined to the run directory, so the host demangles
        before the session starts and the session keeps a read-only tool
        surface. Written onto a copy: the reviewed table has to stay diffable
        against the deterministic one.
        """
        candidate = {
            "kernel_id": "k001",
            "name": "aiter_op",
            "device_kernel_name": "hipModuleLaunchKernel->_ZN5aiter18add_rmsnorm_kernelE",
        }
        monkeypatch.setattr(
            cra,
            "run_candidate_review",
            lambda **_kw: cra.ReviewOutcome(status="failed", detail="not run"),
        )
        artifacts = tla._run_candidate_review_stage(
            tmp_path,
            candidates=[candidate],
            args=self._args(),
            trace_health_warnings=[],
        )

        raw = json.loads(Path(artifacts["kernel_candidates_raw"]).read_text(encoding="utf-8"))
        assert raw["hot_kernels"][0]["device_kernel_name_demangled"] == "aiter::add_rmsnorm_kernel"
        assert "device_kernel_name_demangled" not in candidate

    def test_a_symbol_that_was_never_mangled_grows_no_field(self, tmp_path, monkeypatch):
        """A field restating its neighbour is noise in every row that has one."""
        candidate = {"kernel_id": "k001", "name": "k", "device_kernel_name": "_gqa_sparse_fwd_kernel"}
        monkeypatch.setattr(
            cra,
            "run_candidate_review",
            lambda **_kw: cra.ReviewOutcome(status="failed", detail="not run"),
        )
        artifacts = tla._run_candidate_review_stage(
            tmp_path,
            candidates=[candidate],
            args=self._args(),
            trace_health_warnings=[],
        )
        raw = json.loads(Path(artifacts["kernel_candidates_raw"]).read_text(encoding="utf-8"))
        assert "device_kernel_name_demangled" not in raw["hot_kernels"][0]

    def test_dims_land_even_though_the_path_did_not_move(self, tmp_path, monkeypatch):
        """The re-derivation is skipped for rows that did not change, and for a
        long time a path was the only thing that could change.

        Operand dims are most often supplied for a kernel the deterministic
        tiers already located, so a check keyed on the path alone drops exactly
        the proposals that were hardest to get. Two production analyses staged
        `review_backfill` dims on `keep` revisions and shipped a table with none.
        """
        source = tmp_path / "sparse_attn.py"
        source.write_text("def k(): pass\n", encoding="utf-8")
        candidate = {
            "kernel_id": "k004",
            "name": "_gqa_sparse_fwd_kernel",
            "source_file": str(source),
            "source_type": "python",
            "shapes": [],
        }
        monkeypatch.setattr(tla, "_reusable_roots", lambda: (str(tmp_path).lower(),))
        monkeypatch.setattr(
            cra,
            "run_candidate_review",
            lambda **_kw: cra.ReviewOutcome(
                status="completed",
                revisions=[
                    {
                        "kernel_id": "k004",
                        "action": "keep",
                        "shapes": ["(8192,8,128) bf16", "(8192,1,128) bf16"],
                        "shape_provenance": cra.REVIEW_BACKFILL_PROVENANCE,
                    }
                ],
            ),
        )

        tla._run_candidate_review_stage(tmp_path, candidates=[candidate], args=self._args())

        assert candidate["shapes"] == ["(8192,8,128) bf16", "(8192,1,128) bf16"]
        assert candidate["shape_provenance"] == cra.REVIEW_BACKFILL_PROVENANCE


class TestReviewLeavesSourceJudgmentsAlone:
    """A revision that moved nothing must not clear what nothing invalidated.

    Eighteen of the nineteen source-derived keys have no producer in this pass:
    the finder that filled them read the demangled device symbol and the
    binary's exports, and restamping cannot rebuild that from a path. Clearing
    them is right when the candidate names a different source than the values
    describe, and a silent, permanent loss otherwise -- ``kernel_kind`` and
    ``prebuilt_binary`` are what tell ``classify_patchability`` a kernel is
    prebuilt assembly, and ``source_framework`` is read before ``source_file``
    when the backend picks a kernel_backend.
    """

    FINDER_KEYS = (
        "kernel_kind",
        "prebuilt_binary",
        "source_framework",
        "op_to_source_status",
        "op_to_source_patchable",
        "source_resolution_confidence",
        "launcher_source_file",
    )

    def _candidate(self, source):
        item = {
            "name": "k",
            "source_file": str(source),
            "source_type": "python",
            "shapes": [],
            "review_shapes": ["(8192,6144) bf16"],
        }
        item.update(
            {
                "kernel_kind": "prebuilt_assembly",
                "prebuilt_binary": True,
                "source_framework": "aiter",
                "op_to_source_status": "resolved",
                "op_to_source_patchable": False,
                "source_resolution_confidence": 0.95,
                "launcher_source_file": "/pkg/launcher.py",
            }
        )
        return item

    def test_a_shapes_only_revision_keeps_them(self, tmp_path):
        source = tmp_path / "k.py"
        source.write_text("def k(): pass\n", encoding="utf-8")
        item = self._candidate(source)

        tla._accept_review_proposals(item)

        assert item["shapes"] == ["(8192,6144) bf16"], "the proposal is still taken"
        for key in self.FINDER_KEYS:
            assert item.get(key) is not None, f"{key} was cleared with nothing to rebuild it"
        assert item["kernel_kind"] == "prebuilt_assembly"
        assert item["prebuilt_binary"] is True

    def test_a_moved_path_still_clears_them(self, tmp_path):
        """The old values describe a source the candidate no longer names."""
        source = tmp_path / "k.py"
        source.write_text("def k(): pass\n", encoding="utf-8")
        item = self._candidate(source)
        item["source_file"] = str(tmp_path / "other.py")

        tla._rederive_after_review(item)

        for key in self.FINDER_KEYS:
            assert item.get(key) is None, f"{key} describes the path that was replaced"


class TestRederiveAfterReview:
    def test_a_veto_is_honoured(self, tmp_path, monkeypatch):
        """The session may refuse a kernel it knows is not worth a session.

        The deterministic gate has to pass first, or the veto is moot -- this
        pins that a kernel the rules accept can still be turned down, and that
        the session's reason is the one reported.
        """
        root = tmp_path / "vllm"
        root.mkdir()
        source = root / "k.py"
        source.write_text("def k(): pass\n", encoding="utf-8")
        monkeypatch.setattr(tla, "_reusable_roots", lambda: (str(root).lower(),))

        accepted = {"name": "k", "source_file": str(source), "source_type": "python"}
        tla._rederive_after_review(accepted)
        assert accepted["reusable_native_kernel"] is True, "gate must accept it first"

        vetoed = {
            "name": "k",
            "source_file": str(source),
            "source_type": "python",
            "review_reusable_hint": False,
            "review_skip_reason": "dispatch wrapper only",
        }
        tla._rederive_after_review(vetoed)
        assert vetoed["reusable_native_kernel"] is False
        assert vetoed["skip_reason"] == "dispatch wrapper only"

    def test_a_promotion_is_not(self, tmp_path):
        """A hint cannot talk the gate into dispatching what it rejected.

        classify_patchability stays the one gate; letting a permissive hint
        through would give it a second, model-written one.
        """
        item = {"name": "k", "source_file": "", "review_reusable_hint": True}
        tla._rederive_after_review(item)
        assert item["reusable_native_kernel"] is False

    def test_a_verified_harness_list_survives_restamping(self, tmp_path):
        """Stamping recomputes benchmark_files from the coarse marker table.

        A session that went and looked has the better answer.
        """
        source = tmp_path / "attention.py"
        source.write_text("def paged_attention(): pass\n", encoding="utf-8")
        item = {
            "name": "kernel_paged_attention_2d",
            "source_file": str(source),
            "review_benchmark_files": [str(source)],
        }
        tla._rederive_after_review(item)
        assert item["benchmark_files"] == [str(source)]

    def test_an_all_missing_harness_proposal_leaves_the_curated_one(self, tmp_path):
        """An empty verified list says the proposal was wrong, not the table.

        ``_verified_harnesses`` returns ``[]`` when the session named harnesses
        and none of them exist. Adopting that would empty ``benchmark_files``,
        which is what ``has_benchmark`` and the invocation spec's benchmark
        evidence are built from -- so a review that guessed badly would strip a
        runnable harness the deterministic table had already found.
        """
        harness = tmp_path / "aiter" / "op_tests" / "test_rmsnorm2d.py"
        harness.parent.mkdir(parents=True)
        harness.write_text("def test_rmsnorm2d(): pass\n", encoding="utf-8")
        monkey_roots = (str(tmp_path / "aiter"),)
        original_roots = tla.kernel_search_roots
        tla.kernel_search_roots = lambda: monkey_roots
        tla._harness_search_bases.cache_clear()
        try:
            name = "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256EEEv"
            source = "/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu"
            deterministic = {"name": name, "source_file": source}
            tla._accept_review_proposals(deterministic)
            assert deterministic["benchmark_files"] == [str(harness)]

            reviewed = {
                "name": name,
                "source_file": source,
                "review_benchmark_files": [],
            }
            tla._accept_review_proposals(reviewed)
            assert reviewed["benchmark_files"] == [str(harness)]
        finally:
            tla.kernel_search_roots = original_roots
            tla._harness_search_bases.cache_clear()


class TestAdoptReviewedShapes:
    def test_dims_are_taken_where_the_trace_recorded_none(self, tmp_path):
        """Empty dims are not a neutral state: the backend then picks its own
        without any view of the serving configuration.
        """
        source = tmp_path / "k.py"
        source.write_text("def k(): pass\n", encoding="utf-8")
        item = {
            "name": "k",
            "source_file": str(source),
            "shapes": [],
            "review_shapes": ["(8192,6144) bf16"],
            "review_input_dtypes": ["bf16"],
            "review_shape_provenance": "review_backfill",
        }
        tla._rederive_after_review(item)
        assert item["shapes"] == ["(8192,6144) bf16"]
        assert item["input_dtypes"] == ["bf16"]
        assert item["shape_provenance"] == "review_backfill"

    def test_a_recorded_shape_outranks_a_reviewed_one(self, tmp_path):
        """Nothing downstream re-measures the reviewed dims, so a real
        measurement is never given up for them.
        """
        item = {
            "name": "k",
            "source_file": "",
            "shapes": ["(1,1) fp32"],
            "shape_provenance": "torch_trace",
            "review_shapes": ["(9,9) bf16"],
            "review_shape_provenance": "review_derived",
        }
        tla._adopt_reviewed_shapes(item)
        assert item["shapes"] == ["(1,1) fp32"]
        assert item["shape_provenance"] == "torch_trace"

    def test_only_the_representation_that_gets_rebuilt_is_dropped(self, tmp_path):
        """A harness built from a mix of old and new dims benchmarks the wrong
        thing, so a representation of the replaced dims must not survive -- but
        only where something rebuilds it.

        ``input_shapes`` is refilled from ``shapes`` by
        ``enrich_candidates_with_runtime_metadata``, which runs after this stage,
        so clearing it is safe. ``invocation_cases`` and ``raw_arg_spec`` have no
        producer after finalize and cannot be re-derived from a list of dims:
        they carry an ordered scalar argument list, and a CSV row can supply one
        while yielding no tensor operands at all -- which is exactly a row whose
        ``shapes`` is empty, so exactly the row this function fires on. Clearing
        them there dropped the scalar signature and collapsed multi-case task
        groups, so a stage that exists to add operand evidence removed some.
        """
        item = {
            "name": "k",
            "source_file": "",
            "shapes": [],
            "input_shapes": [["stale"]],
            "invocation_cases": [{"operation": "scalar_only"}],
            "raw_arg_spec": {"0": "int 8"},
            "_input_shapes_synthetic": True,
            "review_shapes": ["(8192,6144) bf16"],
        }
        tla._adopt_reviewed_shapes(item)

        assert "input_shapes" not in item
        assert "_input_shapes_synthetic" not in item
        assert item["invocation_cases"] == [{"operation": "scalar_only"}]
        assert item["raw_arg_spec"] == {"0": "int 8"}
        assert item["shapes"] == ["(8192,6144) bf16"]

    def test_an_unlabelled_adoption_still_records_a_provenance(self, tmp_path):
        """The dispatch gate reads this field; leaving it blank reads as
        measured, which is the one thing it must not say.
        """
        item = {"name": "k", "source_file": "", "shapes": [], "review_shapes": ["(8,8) bf16"]}
        tla._adopt_reviewed_shapes(item)
        assert item["shape_provenance"] == "review_derived"
