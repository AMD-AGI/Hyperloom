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

from hyperloom.measure.docs import DocumentError
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


def test_a_runtime_root_needs_a_runtime_snapshot() -> None:
    def change(d: dict[str, Any]) -> None:
        d["environment"]["roots"][0]["kind"] = "runtime"
        d["environment"]["runtime"] = None

    with pytest.raises(DocumentError, match="runtime snapshot"):
        MeasurementSpec.from_document(_edit(change))


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
        (lambda: _sweep(), "at least one", True),
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
