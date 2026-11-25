# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manifest layering, variable expansion and fingerprint (4.2.2, 4.2.3)."""

from __future__ import annotations

import json

import pytest

from hyperloom.plugs import ManifestError, ManifestSources, builtin_registry, resolve_manifest
from hyperloom.plugs.manifest import (
    SCHEMA,
    SocketSetting,
    apply_layer,
    builtin_profile_names,
    cli_document,
    expand_manifest,
    expand_variables,
    fingerprint,
)


def _doc(**sockets):
    return {"schema": SCHEMA, "sockets": sockets}


def _write(tmp_path, name, doc):
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _base(**settings):
    return {k: SocketSetting(v[0], v[1], "builtin") for k, v in settings.items()}


def test_builtin_profiles_are_local_hosted_gpu_and_hosted_cpu() -> None:
    assert builtin_profile_names() == ("hosted-cpu", "hosted-gpu", "local")


def test_the_builtin_local_profile_is_always_the_base() -> None:
    manifest = resolve_manifest(builtin_registry())
    assert manifest.profile == "local"
    assert {k: (s.impl, s.layer) for k, s in manifest.sockets.items()} == {"events": ("local", "builtin")}
    assert manifest.is_builtin_default


def test_a_profile_layer_overrides_the_builtin_layer() -> None:
    manifest = resolve_manifest(builtin_registry(), ManifestSources(profile="hosted-cpu"))
    assert manifest.profile == "hosted-cpu"
    assert manifest.sockets["events"].impl == "hosted"
    assert manifest.sockets["events"].layer == "profile"
    assert not manifest.is_builtin_default


def test_an_unknown_profile_names_the_available_ones() -> None:
    with pytest.raises(ManifestError, match="unknown profile 'nope'; available: hosted-cpu, hosted-gpu, local"):
        resolve_manifest(builtin_registry(), ManifestSources(profile="nope"))


def test_a_profile_name_cannot_walk_out_of_the_profile_directory() -> None:
    with pytest.raises(ManifestError, match="unknown profile"):
        resolve_manifest(builtin_registry(), ManifestSources(profile="../manifest"))


def test_layers_apply_in_order_profile_manifest_overrides_cli(tmp_path) -> None:
    manifest_file = _write(tmp_path, "m.json", _doc(events={"impl": "from-manifest"}))
    overrides = _write(tmp_path, "o.json", _doc(events={"impl": "from-overrides"}))
    registry = builtin_registry()
    sources = ManifestSources(profile="hosted-cpu", manifest_path=manifest_file, overrides_path=overrides)
    assert resolve_manifest(registry, sources).sockets["events"].layer == "overrides"
    assert resolve_manifest(registry, sources).sockets["events"].impl == "from-overrides"
    cli = ManifestSources(**{**sources.__dict__, "cli_plugs": ("events=from-cli",)})
    assert resolve_manifest(registry, cli).sockets["events"].impl == "from-cli"
    assert resolve_manifest(registry, cli).sockets["events"].layer == "cli"


def test_a_resumed_file_is_the_profile_layer_and_an_explicit_profile_still_overrides_it() -> None:
    resumed = {"schema": SCHEMA, "profile": "custom", "sockets": {"events": {"impl": "from-file"}}}
    registry = builtin_registry()
    assert resolve_manifest(registry, ManifestSources(resumed=resumed)).sockets["events"].impl == "from-file"
    explicit = ManifestSources(resumed=resumed, profile="hosted-cpu")
    assert resolve_manifest(registry, explicit).sockets["events"].impl == "hosted"


def test_rule_1_later_layers_override_per_socket() -> None:
    merged = apply_layer(_base(events=("a", {})), _doc(events={"impl": "b"}), "cli", builtin_registry())
    assert merged["events"].impl == "b"


def test_rule_2_a_changed_impl_drops_the_earlier_config() -> None:
    base = _base(events=("a", {"dir": "/x", "keep": 1}))
    merged = apply_layer(base, _doc(events={"impl": "b", "config": {"new": 2}}), "cli", builtin_registry())
    assert merged["events"].config == {"new": 2}


def test_rule_3_an_unchanged_impl_merges_config_recursively() -> None:
    base = _base(events=("a", {"k": 1, "nested": {"x": 1, "y": 2}, "items": [1, 2], "gone": "v"}))
    patch = {"nested": {"y": 3, "z": 4}, "items": [9], "gone": None}
    merged = apply_layer(base, _doc(events={"config": patch}), "cli", builtin_registry())
    assert merged["events"].config == {"k": 1, "nested": {"x": 1, "y": 3, "z": 4}, "items": [9]}


def test_rule_3_naming_the_same_impl_again_keeps_the_config() -> None:
    base = _base(events=("a", {"k": 1}))
    merged = apply_layer(base, _doc(events={"impl": "a", "config": {"j": 2}}), "cli", builtin_registry())
    assert merged["events"].config == {"k": 1, "j": 2}


def test_rule_3_a_layer_does_not_mutate_the_one_below() -> None:
    base = _base(events=("a", {"nested": {"x": 1}}))
    apply_layer(base, _doc(events={"config": {"nested": {"x": 2}}}), "cli", builtin_registry())
    assert base["events"].config == {"nested": {"x": 1}}


def test_rule_4_variables_expand_with_and_without_defaults() -> None:
    env = {"A": "alpha"}
    assert expand_variables({"p": "${A}/x", "q": "${B:-fallback}", "r": ["${A}"], "n": 3}, env, "w") == {
        "p": "alpha/x",
        "q": "fallback",
        "r": ["alpha"],
        "n": 3,
    }


def test_rule_4_a_missing_variable_without_a_default_is_an_error() -> None:
    with pytest.raises(ManifestError, match=r"socket events\.dir: environment variable HYPERLOOM_NOPE is not set"):
        expand_manifest(
            resolve_manifest(builtin_registry(), ManifestSources(cli_configs=("events.dir=${HYPERLOOM_NOPE}",))),
            {},
        )


def test_rule_4_an_empty_default_is_a_default() -> None:
    assert expand_variables("${B:-}", {}, "w") == ""


def test_rule_5_an_undefined_socket_is_an_error_in_every_layer(tmp_path) -> None:
    registry = builtin_registry()
    with pytest.raises(ManifestError, match="cli: unknown socket 'warp'"):
        resolve_manifest(registry, ManifestSources(cli_plugs=("warp=x",)))
    bad = _write(tmp_path, "m.json", _doc(warp={"impl": "x"}))
    with pytest.raises(ManifestError, match="manifest: unknown socket 'warp'"):
        resolve_manifest(registry, ManifestSources(manifest_path=bad))


def test_a_socket_with_no_impl_anywhere_is_an_error() -> None:
    with pytest.raises(ManifestError, match="socket 'sandbox' sets no impl and no earlier layer chose one"):
        resolve_manifest(builtin_registry(), ManifestSources(cli_configs=("sandbox.x=1",)))


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ([], "a manifest must be a mapping"),
        ({"schema": "other/v9", "sockets": {}}, "schema must be 'hyperloom.plugs/v1'"),
        ({"schema": SCHEMA, "sockets": []}, "sockets must be a mapping"),
        (_doc(events={"implementation": "x"}), r"unknown keys \['implementation'\]"),
        (_doc(events={"impl": ""}), "impl must be a non-empty string"),
        (_doc(events={"config": []}), "config must be a mapping"),
    ],
)
def test_a_malformed_manifest_is_refused_for_its_own_reason(tmp_path, doc, message) -> None:
    with pytest.raises(ManifestError, match=message):
        resolve_manifest(builtin_registry(), ManifestSources(manifest_path=_write(tmp_path, "m.json", doc)))


def test_an_unreadable_or_invalid_manifest_file_is_refused(tmp_path) -> None:
    registry = builtin_registry()
    with pytest.raises(ManifestError, match="cannot read manifest"):
        resolve_manifest(registry, ManifestSources(manifest_path=tmp_path / "absent.json"))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="is not valid JSON"):
        resolve_manifest(registry, ManifestSources(manifest_path=broken))


def test_a_yaml_manifest_is_refused_as_invalid_json(tmp_path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("schema: hyperloom.plugs/v1\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="is not valid JSON"):
        resolve_manifest(builtin_registry(), ManifestSources(manifest_path=path))


def test_a_manifest_without_a_profile_is_called_custom(tmp_path) -> None:
    path = _write(tmp_path, "m.json", _doc(events={"impl": "x"}))
    assert resolve_manifest(builtin_registry(), ManifestSources(manifest_path=path)).profile == "custom"


def test_cli_values_parse_as_json_when_they_can_and_as_text_otherwise() -> None:
    doc = cli_document(["events=hosted"], ["events.n=3", "events.flag=true", "events.name=abc", "events.none=null"])
    assert doc["sockets"]["events"] == {
        "impl": "hosted",
        "config": {"n": 3, "flag": True, "name": "abc", "none": None},
    }


@pytest.mark.parametrize("bad", ["events", "=x", "events="])
def test_a_malformed_plug_flag_is_refused(bad) -> None:
    with pytest.raises(ManifestError, match="--plug expects SOCKET=IMPL"):
        cli_document([bad], [])


@pytest.mark.parametrize("bad", ["events=1", "events.=1", ".k=1", "events.k"])
def test_a_malformed_plug_config_flag_is_refused(bad) -> None:
    with pytest.raises(ManifestError, match="--plug-config expects SOCKET.KEY=VALUE"):
        cli_document([], [bad])


def test_cli_impl_and_config_in_one_layer_keep_the_config_with_the_new_impl() -> None:
    sources = ManifestSources(profile="hosted-cpu", cli_plugs=("events=local",), cli_configs=("events.k=1",))
    events = resolve_manifest(builtin_registry(), sources).sockets["events"]
    assert (events.impl, events.config) == ("local", {"k": 1})


def test_fingerprint_is_sixteen_hex_digits_and_independent_of_key_order() -> None:
    a = fingerprint("p", {"s": {"impl": "x", "origin": "builtin", "config": {"a": 1, "b": 2}}})
    b = fingerprint("p", {"s": {"config": {"b": 2, "a": 1}, "origin": "builtin", "impl": "x"}})
    assert a == b and len(a) == 16 and int(a, 16) >= 0


@pytest.mark.parametrize(
    "changed",
    [
        ("q", {"s": {"impl": "x", "origin": "builtin", "config": {}}}),
        ("p", {"s": {"impl": "y", "origin": "builtin", "config": {}}}),
        ("p", {"s": {"impl": "x", "origin": "pkg==1", "config": {}}}),
        ("p", {"s": {"impl": "x", "origin": "builtin", "config": {"k": 1}}}),
    ],
)
def test_fingerprint_covers_profile_impl_origin_and_config(changed) -> None:
    base = fingerprint("p", {"s": {"impl": "x", "origin": "builtin", "config": {}}})
    assert fingerprint(*changed) != base
