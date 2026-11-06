# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from importlib import resources
from typing import Any

import pytest
from hyperloom.measure.tests.bundle_factory import REQUEST_ID, point_document, point_yaml, spec_document, spec_json
from jsonschema import Draft202012Validator

from hyperloom.measure import spec as spec_module
from hyperloom.measure.docs import DocumentError
from hyperloom.measure.long_context.plan import LongContextPlan
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.spec import (
    MeasurementSpec,
    blob_input_path,
    canonical_for_fingerprint,
    fingerprint,
    spec_sha256,
)

VERSION = "1.1.3+gtest"
SCHEMA = json.loads(resources.files("hyperloom.contracts").joinpath("measurement_spec.v1.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def _fp(document: dict[str, Any], **kwargs: Any) -> str:
    return fingerprint(
        MeasurementSpec.from_document(document), hyperloom_version=kwargs.pop("version", VERSION), **kwargs
    )


def _edit(change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    document = copy.deepcopy(spec_document())
    change(document)
    return document


def _yaml(document: dict[str, Any], text: str) -> None:
    document["config"] = [point_document(text)]


def test_a_sheet_round_trips_through_bytes_and_keeps_its_hash() -> None:
    spec = MeasurementSpec.from_document(spec_document())
    again = MeasurementSpec.from_bytes(spec.to_bytes())
    assert again == spec and again.to_bytes() == spec.to_bytes()
    assert spec.sha256() == spec_sha256(spec.to_bytes()) == hashlib.sha256(spec.to_bytes()).hexdigest()


def test_the_sheet_hash_changes_with_any_changed_byte() -> None:
    base = MeasurementSpec.from_document(spec_document()).sha256()
    assert MeasurementSpec.from_document(_edit(lambda d: d.update(request_id="other"))).sha256() != base


def test_the_valid_sheet_passes_the_json_schema() -> None:
    assert list(VALIDATOR.iter_errors(spec_document())) == []


NEGATIVES: dict[str, Callable[[dict[str, Any]], None]] = {
    "unknown top-level key": lambda d: d.update(surprise=1),
    "missing section": lambda d: d.pop("outputs"),
    "other schema version": lambda d: d.update(schema="hyperloom.measurement-spec/v2"),
    "unknown kind": lambda d: d.update(kind="bench"),
    "bool as gpu_count": lambda d: d["hardware"].update(gpu_count=True),
    "zero gpus": lambda d: d["hardware"].update(gpu_count=0),
    "priority out of range": lambda d: d["meta"].update(priority=5),
    "confirm without retest_of": lambda d: d["meta"].update(purpose="confirm"),
    "bad image digest": lambda d: d["image"].update(digest="sha256:xyz"),
    "short commit": lambda d: d["tools"]["magpie"].update(commit="abc"),
    "absolute entry path": lambda d: d["environment"]["roots"][0]["entries"][0].update(path="/etc/x"),
    "dotdot entry path": lambda d: d["environment"]["roots"][0]["entries"][0].update(path="a/../b"),
    "upsert without blob": lambda d: d["environment"]["roots"][0]["entries"][0].pop("blob"),
    "delete with blob": lambda d: d["environment"]["roots"][0]["entries"][1].update(blob="sha256:" + "0" * 64),
    "bad mode": lambda d: d["environment"]["roots"][0]["entries"][0].update(mode="rw"),
    "bad env name": lambda d: d["process_env"].update({"A-B": "x"}),
    "platform variable outside the list": lambda d: d["platform_env"].update({"LD_PRELOAD": "x"}),
    "duplicate unset": lambda d: d.update(unset_envs=["A", "A"]),
    "bad attachment kind": lambda d: d["attachments"][0].update(kind="socket"),
    "bad attachment id": lambda d: d["attachments"][0].update(id="../x"),
    "model hash not sha256": lambda d: d["model_hashes"].update({"config.json": "xyz"}),
    "absolute result_dir": lambda d: d["outputs"].update(result_dir="/tmp/out"),
    "zero measure rounds": lambda d: d["procedure"]["rounds"].update(measure=0),
    "unknown prep": lambda d: d["procedure"].update(prep="sweep"),
    "accuracy kind without accuracy": lambda d: d.update(kind="accuracy"),
}


@pytest.mark.parametrize("name", sorted(NEGATIVES))
def test_a_sheet_that_breaks_the_contract_is_refused_by_the_reader_and_by_the_schema(name: str) -> None:
    document = _edit(NEGATIVES[name])
    with pytest.raises(DocumentError):
        MeasurementSpec.from_document(document)
    assert list(VALIDATOR.iter_errors(document)) != [], "the schema accepts what the reader refuses"


def test_a_config_whose_hash_does_not_match_its_text_is_refused() -> None:
    document = _edit(lambda d: d["config"][0].update(yaml=d["config"][0]["yaml"] + "# edited\n"))
    with pytest.raises(DocumentError, match="does not match the YAML text"):
        MeasurementSpec.from_document(document)


_KEY, _MANIFEST = "1" * 64, "2" * 64


def _built(d: dict[str, Any]) -> None:
    d["environment"]["roots"][0]["kind"] = "runtime"
    d["environment"]["runtime"] = None
    d["environment"]["builds"] = [
        {"root_id": d["environment"]["roots"][0]["id"], "key": _KEY, "manifest_sha256": _MANIFEST}
    ]


def test_a_runtime_root_names_its_ready_build_and_the_build_is_in_the_fingerprint() -> None:
    document = _edit(_built)
    spec = MeasurementSpec.from_document(document)

    assert [(b.key, b.manifest_sha256) for b in spec.environment.builds] == [(_KEY, _MANIFEST)]
    assert MeasurementSpec.from_bytes(spec.to_bytes()) == spec
    assert list(VALIDATOR.iter_errors(document)) == []
    other = _edit(lambda d: (_built(d), d["environment"]["builds"][0].update(manifest_sha256="3" * 64)))
    assert _fp(document) != _fp(other), "another manifest is another measurement"
    assert canonical_for_fingerprint(spec)["environment"]["builds"][0]["key"] == _KEY


def test_a_sheet_with_no_build_leaves_the_key_out_so_its_bytes_and_fingerprint_are_unchanged() -> None:
    spec = MeasurementSpec.from_document(spec_document())

    assert "builds" not in spec.to_document()["environment"]
    assert spec.to_document()["environment"] == spec_document()["environment"]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda d: (_built(d), d["environment"].pop("builds")), "each runtime root needs exactly one build"),
        (lambda d: (_built(d), d["environment"]["builds"][0].update(root_id="nowhere")), "each build names a root"),
        (
            lambda d: (_built(d), d["environment"]["builds"].append(dict(d["environment"]["builds"][0]))),
            "each runtime root needs exactly one build",
        ),
        (lambda d: (_built(d), d["environment"]["builds"][0].update(key="k")), "builds"),
        (lambda d: d["environment"].update(builds=[]), "an empty list"),
        (lambda d: (_built(d), d["environment"]["builds"][0].update(extra=1)), "extra"),
    ],
    ids=["runtime-root-without-build", "build-of-no-root", "two-builds", "bad-key", "empty", "unknown-field"],
)
def test_a_build_that_does_not_name_exactly_one_runtime_root_is_refused(change: Any, reason: str) -> None:
    document = _edit(change)

    with pytest.raises(DocumentError, match=reason):
        MeasurementSpec.from_document(document)


def test_a_package_root_may_name_the_build_of_its_jit_modules() -> None:
    """aiter's package root names the CPU build of the modules its patch touches; which package may have
    one is the runner's to say."""
    document = _edit(lambda d: (_built(d), d["environment"]["roots"][0].update(kind="package")))

    spec = MeasurementSpec.from_document(document)

    assert [b.root_id for b in spec.environment.builds] == [spec.environment.roots[0].id]


def test_duplicate_json_keys_and_non_finite_numbers_are_refused() -> None:
    text = spec_json(spec_document()).decode()
    with pytest.raises(DocumentError, match="duplicate"):
        MeasurementSpec.from_bytes(text.replace('"kind": "test"', '"kind": "test", "kind": "profile"').encode())
    with pytest.raises(DocumentError, match="NaN"):
        MeasurementSpec.from_bytes(text.replace('"deadline_unix": 1790000000.0', '"deadline_unix": NaN').encode())
    with pytest.raises(DocumentError, match="not valid"):
        MeasurementSpec.from_bytes(b"\xff\xfe")
    with pytest.raises(DocumentError, match="top level"):
        MeasurementSpec.from_bytes(b"[]")


def test_the_fingerprint_is_sixteen_hex_digits_and_stable() -> None:
    first = _fp(spec_document())
    assert len(first) == 16 and int(first, 16) >= 0
    assert _fp(copy.deepcopy(spec_document())) == first


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(request_id="a-completely-different-id"),
        lambda d: d["meta"].update(purpose="confirm", retest_of="r0", branch_id="b9", priority=0, parent_commit="zz"),
        lambda d: d["procedure"].update(deadline_unix=1.0),
        lambda d: _yaml(d, d["config"][0]["yaml"].replace("/shared/u1/sess-a/pids", "/shared/u2/sess-b/pids")),
        lambda d: d["environment"].update(blob_root="/shared/u2/other/blobs"),
        lambda d: d["environment"]["runtime"].update(snapshot="/shared/u2/other/rt"),
    ],
    ids=["request_id", "meta", "deadline", "yaml path field", "blob_root", "snapshot location"],
)
def test_the_fingerprint_ignores_what_is_not_measurement_content(change) -> None:
    assert _fp(_edit(change)) == _fp(spec_document())


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["hardware"].update(gpu_count=2),
        lambda d: d["image"].update(digest="sha256:" + "99" * 32),
        lambda d: d["tools"]["magpie"].update(commit="c" * 40),
        lambda d: d["process_env"].update(AITER_LOG_LEVEL="DEBUG"),
        lambda d: d["platform_env"].update(NCCL_IB_SL="4"),
        lambda d: d["unset_envs"].append("OTHER"),
        lambda d: d["procedure"]["rounds"].update(measure=4),
        lambda d: d["procedure"].update(accuracy=True),
        lambda d: d["environment"]["roots"][0]["entries"][0].update(blob="sha256:" + "88" * 32),
        lambda d: d["attachments"][0].update(manifest_sha256="aa" * 32),
        lambda d: d["model_hashes"].update({"config.json": "bb" * 32}),
        lambda d: _yaml(d, d["config"][0]["yaml"].replace("port: 8000", "port: 8001")),
    ],
    ids=[
        "gpu_count",
        "image",
        "tool commit",
        "process_env",
        "platform_env",
        "unset_envs",
        "rounds",
        "accuracy",
        "overlay blob",
        "attachment hash",
        "model hash",
        "yaml value",
    ],
)
def test_the_fingerprint_changes_with_measurement_content(change) -> None:
    assert _fp(_edit(change)) != _fp(spec_document())


def test_the_runner_version_is_part_of_the_fingerprint() -> None:
    assert _fp(spec_document(), version="1.1.4") != _fp(spec_document())
    with pytest.raises(DocumentError, match="version"):
        _fp(spec_document(), version="")


def test_session_directory_prefixes_are_replaced_so_two_sessions_agree() -> None:
    def with_session(root: str) -> dict[str, Any]:
        def change(d: dict[str, Any]) -> None:
            _yaml(d, point_yaml(conc=4) + f"extra_path: {root}/kernels\n")
            d["process_env"]["PYTHONPATH"] = f"{root}/overlay:/opt/x"

        return _edit(change)

    one = _fp(with_session("/shared/u1/sess-a"), session_dir="/shared/u1/sess-a")
    two = _fp(with_session("/shared/u2/sess-b"), session_dir="/shared/u2/sess-b")
    assert one == two
    assert one != _fp(with_session("/shared/u1/sess-a"))  # without the session directory the paths differ


def test_the_canonical_form_keeps_blob_and_attachment_hashes_but_no_locations() -> None:
    canonical = canonical_for_fingerprint(MeasurementSpec.from_document(spec_document()))
    text = json.dumps(canonical)
    assert "blob_root" not in text and "snapshot" not in text and "sess-a" not in text
    assert "55" * 32 in text and "44" * 32 in text


def test_a_yaml_that_does_not_parse_is_refused_when_fingerprinting() -> None:
    document = _edit(lambda d: _yaml(d, "a: [unclosed\n"))
    with pytest.raises(DocumentError, match="not valid YAML"):
        _fp(document)


@pytest.mark.parametrize(
    "text",
    ["model: m\nwhen: 2026-01-01\n", "model: m\nratio: .nan\n", "model: m\nseen: !!set {a: null}\n"],
    ids=["a-date", "a-non-finite-float", "a-set"],
)
def test_a_yaml_value_json_cannot_hold_is_refused_when_fingerprinting(text: str) -> None:
    with pytest.raises(DocumentError, match="JSON cannot hold"):
        _fp(_edit(lambda d: _yaml(d, text)))


def test_a_config_whose_yaml_is_not_unicode_text_is_refused_by_the_reader() -> None:
    document = _edit(lambda d: d["config"][0].update(yaml="model: \ud800\n"))
    raw = json.dumps(document).encode()
    assert b"\\ud800" in raw, "the JSON text carries the lone surrogate as an escape"
    with pytest.raises(DocumentError, match=r"spec\.config\[0\]\.yaml"):
        MeasurementSpec.from_bytes(raw)
    with pytest.raises(DocumentError, match=r"spec\.config\[0\]\.yaml"):  # a sheet built in code, not read from JSON
        MeasurementSpec.from_document(document)


#: YAML values the loader cannot load, or loads to a string UTF-8 cannot encode.
_UNLOADABLE = {
    "an-integer-past-the-digit-limit": "9" * 4301,
    "nesting-past-the-loader-stack": "[" * 20_000 + "]" * 20_000,
    "a-day-no-month-has": "2026-02-30",
    "a-lone-surrogate": '"\\ud800"',
}


@pytest.mark.parametrize("name", sorted(_UNLOADABLE))
def test_a_yaml_the_loader_cannot_load_or_utf8_cannot_encode_is_refused_when_fingerprinting(name: str) -> None:
    with pytest.raises(DocumentError, match="spec.config.yaml"):
        _fp(_edit(lambda d: _yaml(d, f"model: {_UNLOADABLE[name]}\n")))


@pytest.mark.parametrize("name", sorted(_UNLOADABLE))
def test_a_sweep_point_the_loader_cannot_load_or_utf8_cannot_encode_is_refused_by_the_reader(name: str) -> None:
    point = point_yaml(conc=8) + f"extra: {_UNLOADABLE[name]}\n"
    with pytest.raises(DocumentError, match="spec.config.yaml"):
        MeasurementSpec.from_document(_sweep(point_yaml(conc=4), point))


def _anchor_chain(levels: int) -> str:
    """One line per level, each anchor a list of the one before: the last nests ``levels`` lists in a short text."""
    return "a0: &a0 [x]\n" + "".join(f"a{i}: &a{i} [*a{i - 1}]\n" for i in range(1, levels))


@pytest.mark.parametrize("levels", [600, 1000])
def test_a_yaml_nested_deep_by_a_chain_of_anchors_is_refused_when_fingerprinting(levels: int) -> None:
    with pytest.raises(DocumentError, match="levels deep"):
        _fp(_edit(lambda d: _yaml(d, _anchor_chain(levels))))


@pytest.mark.parametrize("levels", [600, 1000])
def test_a_sweep_point_nested_deep_by_a_chain_of_anchors_is_refused_by_the_reader(levels: int) -> None:
    chain = _anchor_chain(levels)  # in both points, so comparing them goes all the way down
    with pytest.raises(DocumentError, match="levels deep"):
        MeasurementSpec.from_document(_sweep(point_yaml(conc=4) + chain, point_yaml(conc=8) + chain))


_BOUND = spec_module._MAX_YAML_DEPTH


def _pair_chain(tag: str, first: str = "[x]") -> str:
    """A chain of ``!!omap`` or ``!!pairs`` links after ``first``, each a list of pairs: two levels per link. With the
    mapping at the top and ``first`` one list, it nests exactly as deep as the bound."""
    links = (_BOUND - 2) // 2
    return f"a0: &a0 {first}\n" + "".join(f"a{i}: &a{i} {tag} [{{k: *a{i - 1}}}]\n" for i in range(1, links + 1))


def _lists(levels: int) -> str:
    """Lists nested ``levels`` deep, written out, with no anchor."""
    return "k: " + "[" * levels + "x" + "]" * levels + "\n"


# The mapping at the top is a level too.
@pytest.mark.parametrize(
    "text",
    [_anchor_chain(_BOUND - 1), _pair_chain("!!omap"), _pair_chain("!!pairs"), _lists(_BOUND - 1)],
    ids=["a-chain-of-anchors", "a-chain-of-omap", "a-chain-of-pairs", "lists-written-out"],
)
def test_a_yaml_as_deep_as_the_bound_is_fingerprinted(text: str) -> None:
    assert _BOUND % 2 == 0, "the chains of pairs reach the bound exactly"
    _fp(_edit(lambda d: _yaml(d, text)))


@pytest.mark.parametrize(
    "text",
    [
        _anchor_chain(_BOUND),
        _anchor_chain(_BOUND - 1) + f"again: [*a{_BOUND - 2}]\n",
        "a: &a [x, *a]\n",
        _pair_chain("!!omap", first="[[x]]"),
        _pair_chain("!!pairs", first="[[x]]"),
        _lists(_BOUND),
    ],
    ids=[
        "a-longer-chain",
        "an-alias-a-level-deeper-than-where-its-anchor-was-measured",
        "an-anchor-in-its-own-value",
        "a-chain-of-omap-on-a-deeper-first-link",
        "a-chain-of-pairs-on-a-deeper-first-link",
        "lists-written-out",
    ],
)
def test_a_yaml_a_level_deeper_than_the_bound_is_refused(text: str) -> None:
    with pytest.raises(DocumentError, match="levels deep"):
        _fp(_edit(lambda d: _yaml(d, text)))


@pytest.mark.parametrize(
    "value",
    [10**400, "1" * 5000, "1e999", "[" * 100_000 + "]" * 100_000],
    ids=["an-integer-no-float-holds", "an-integer-past-the-digit-limit", "infinity", "nesting-past-the-stack"],
)
def test_a_number_or_nesting_no_reader_holds_is_a_document_error(value) -> None:
    raw = json.dumps(spec_document()).replace('"timeout_sec": 3600', f'"timeout_sec": {value}').encode()
    assert b"3600" not in raw
    with pytest.raises(DocumentError):
        MeasurementSpec.from_bytes(raw)


def _sweep(*points: str) -> dict[str, Any]:
    def change(d: dict[str, Any]) -> None:
        d["procedure"]["prep"] = "conc_sweep"
        d["config"] = [point_document(text) for text in points]

    return _edit(change)


def test_a_conc_sweep_arm_carries_one_point_per_level_that_differ_only_in_the_level() -> None:
    document = _sweep(point_yaml(conc=64), point_yaml(conc=32), point_yaml(conc=8))
    spec = MeasurementSpec.from_document(document)
    assert [p.yaml for p in spec.config] == [p["yaml"] for p in document["config"]]  # order is kept
    assert list(VALIDATOR.iter_errors(document)) == []
    assert _fp(document) != _fp(_sweep(point_yaml(conc=8), point_yaml(conc=32), point_yaml(conc=64)))


@pytest.mark.parametrize(
    ("document", "message", "schema_rejects"),
    [
        (lambda: _sweep(point_yaml(conc=4), point_yaml(conc=8, port=8001)), "more than the concurrency level", False),
        (lambda: _sweep(point_yaml(conc=4), point_yaml(conc=4)), "appears twice", False),
        (_sweep, "at least one", True),
        (lambda: _edit(lambda d: d["config"].append(point_document(point_yaml(conc=8)))), "exactly one", True),
        (lambda: _edit(lambda d: d.update(config=[])), "exactly one", True),
    ],
    ids=["server args differ", "duplicate point", "no point", "grid with two points", "grid with none"],
)
def test_the_points_of_a_sheet_are_held_to_the_prep(document, message: str, schema_rejects: bool) -> None:
    doc = document()
    with pytest.raises(DocumentError, match=message):
        MeasurementSpec.from_document(doc)
    # Comparing YAML content is a rule only the reader states; counts the schema states as well.
    assert (list(VALIDATOR.iter_errors(doc)) != []) == schema_rejects


def test_a_blob_lives_at_in_blobs_under_its_bare_hex_hash() -> None:
    assert blob_input_path("sha256:" + "ab" * 32) == "/in/blobs/" + "ab" * 32
    for bad in ("ab" * 32, "sha256:xyz", "sha256:" + "AB" * 32, ""):
        with pytest.raises(DocumentError, match="not a blob reference"):
            blob_input_path(bad)


def test_the_request_id_in_the_sheet_is_the_one_given() -> None:
    assert MeasurementSpec.from_document(spec_document()).request_id == REQUEST_ID


def test_a_sheet_written_while_the_decode_pass_existed_still_parses_and_the_flag_is_dropped() -> None:
    old = copy.deepcopy(spec_document())
    old["procedure"]["decode_pass"] = True
    assert list(VALIDATOR.iter_errors(old)) == []

    spec = MeasurementSpec.from_document(old)

    assert "decode_pass" not in spec.to_document()["procedure"]
    assert spec.to_document() == MeasurementSpec.from_document(spec_document()).to_document()
    assert not hasattr(spec.procedure, "decode_pass")


def _short_prompts(**changes: Any) -> dict[str, Any]:
    plan = {
        "prompt_set": "short-prompts-0123456789abcdef",
        "prompts_sha256": "a" * 64,
        "max_new_tokens": 32,
        "phase_cap_sec": 1200.0,
        "trigger": ["purpose final", "arg --attention-backend aiter"],
        "recipe": "sha256:" + "1" * 64,
        "parent_recipe": "sha256:" + "2" * 64,
    }
    return {**plan, **changes}


def test_a_sheet_that_asks_for_the_short_prompt_check_carries_the_plan_and_the_schema_allows_it() -> None:
    document = _edit(lambda d: d["procedure"].update(short_prompts=_short_prompts()))

    spec = MeasurementSpec.from_document(document)

    assert spec.procedure.short_prompts is not None and spec.procedure.short_prompts.max_new_tokens == 32
    assert list(VALIDATOR.iter_errors(document)) == []
    assert MeasurementSpec.from_bytes(spec.to_bytes()) == spec


def test_a_sheet_that_asks_for_none_has_no_such_key_so_its_bytes_and_hash_are_unchanged() -> None:
    spec = MeasurementSpec.from_document(spec_document())

    assert spec.procedure.short_prompts is None and "short_prompts" not in spec.to_document()["procedure"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"max_new_tokens": 0}, "must be at least 1"),
        ({"processes": 6}, "unknown key 'processes'"),
        ({"prompts_sha256": "nope"}, "does not match"),
        ({"prompt_set": "Bad Key"}, "does not match"),
        ({"phase_cap_sec": "slow"}, "expected a number"),
        ({"trigger": [""]}, "non-empty string"),
        ({"extra": 1}, "unknown key 'extra'"),
    ],
)
def test_a_short_prompt_plan_that_breaks_its_contract_is_refused_by_the_reader_and_the_schema(change, message) -> None:
    document = _edit(lambda d: d["procedure"].update(short_prompts=_short_prompts(**change)))

    with pytest.raises(DocumentError, match=message):
        MeasurementSpec.from_document(document)
    assert list(VALIDATOR.iter_errors(document)) != []


def test_the_fingerprint_changes_with_the_short_prompt_plan_because_the_measurement_then_is_another() -> None:
    base = _fp(spec_document())

    assert _fp(_edit(lambda d: d["procedure"].update(short_prompts=_short_prompts()))) != base


def _long_context(**changes: Any) -> dict[str, Any]:
    plan = {
        "prompt_set": "long-context-0123456789abcdef",
        "prompts_sha256": "a" * 64,
        "length": 131072,
        "max_new_tokens": 128,
        "concurrency": 4,
        "phase_cap_sec": 2400.0,
        "trigger": ["request long_context_check"],
    }
    return {**plan, **changes}


def test_a_sheet_that_asks_for_the_long_context_check_carries_the_plan_and_the_schema_allows_it() -> None:
    document = _edit(lambda d: d["procedure"].update(long_context=_long_context()))

    spec = MeasurementSpec.from_document(document)

    assert spec.procedure.long_context is not None and spec.procedure.long_context.length == 131072
    assert spec.procedure.short_prompts is None
    assert list(VALIDATOR.iter_errors(document)) == []
    assert MeasurementSpec.from_bytes(spec.to_bytes()) == spec


def test_a_sheet_that_asks_for_no_long_context_check_has_no_such_key_so_its_bytes_and_hash_are_unchanged() -> None:
    spec = MeasurementSpec.from_document(spec_document())

    assert spec.procedure.long_context is None and "long_context" not in spec.to_document()["procedure"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"length": 10}, "must be at least 1024"),
        ({"max_new_tokens": 0}, "must be at least 1"),
        ({"concurrency": 0}, "must be at least 1"),
        ({"prompts_sha256": "nope"}, "does not match"),
        ({"prompt_set": "Bad Key"}, "does not match"),
        ({"phase_cap_sec": "slow"}, "expected a number"),
        ({"trigger": [""]}, "non-empty string"),
        ({"extra": 1}, "unknown key 'extra'"),
    ],
)
def test_a_long_context_plan_that_breaks_its_contract_is_refused_by_the_reader_and_the_schema(change, message) -> None:
    document = _edit(lambda d: d["procedure"].update(long_context=_long_context(**change)))

    with pytest.raises(DocumentError, match=message):
        MeasurementSpec.from_document(document)
    assert list(VALIDATOR.iter_errors(document)) != []


def test_a_max_new_tokens_beyond_what_the_headroom_covers_is_refused_by_the_schema_too() -> None:
    document = _edit(lambda d: d["procedure"].update(long_context=_long_context(max_new_tokens=129)))

    assert list(VALIDATOR.iter_errors(document)) != []
    with pytest.raises(DocumentError, match="max_new_tokens must be 1 to 128"):
        MeasurementSpec.from_document(document)


def test_the_fingerprint_changes_with_the_long_context_plan_because_the_measurement_then_is_another() -> None:
    base = _fp(spec_document())

    assert _fp(_edit(lambda d: d["procedure"].update(long_context=_long_context()))) != base


@pytest.mark.parametrize(
    "make",
    [
        lambda cap: ShortPromptPlan("short-prompts-test", "a" * 64, phase_cap_sec=cap),
        lambda cap: LongContextPlan("long-context-test", "a" * 64, 4096, phase_cap_sec=cap),
    ],
    ids=["short-prompts", "long-context"],
)
def test_a_plan_is_built_only_with_a_phase_cap_its_reader_accepts_so_what_it_writes_reads_back(make) -> None:
    with pytest.raises(DocumentError, match="phase_cap_sec"):
        make(0.5)
    plan = make(1.0)
    assert type(plan).parse(plan.to_document(), "plan") == plan
