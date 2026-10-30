# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rule pack is data a session reads, and a pack that is wrong is refused instead of read as "no rules".

What is pinned: the reference pack passes its own examples and holds exactly the facts the code held as constants before
the pack existed; a pack that fails an example, lacks a field, repeats a key or lacks an entry the code reads is refused
by name; the file a session reads is the one ``HYPERLOOM_RULE_PACK`` names; the recorded identity follows the bytes; and
the default pack is declared as package data, because an installed package has no other copy.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from collections import Counter
from collections.abc import Callable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import pytest
import yaml

import hyperloom
from hyperloom.common import outside_facts
from hyperloom.common.outside_facts.rule_pack import (
    DEFAULT_PACK_PATH,
    PACK_ENV,
    Kind,
    RulePackError,
    load_pack,
    rule_pack_identity,
)

try:  # tomllib is stdlib from 3.11; the ``ci`` extra pins tomli for 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on py3.10
    import tomli as tomllib  # type: ignore[no-redef]

_AITER_TUNED_HIT = r"""(?x)
shape (?:\s+is)? \s+ M:\d+, \s+ N:\d+, \s+ K:\d+ ,?
(?: \s+ dtype='(?P<dtype>[^']*)' \s+ otype='(?P<outdtype>[^']*)'
    \s+ bias=(?P<bias>True|False), \s+ scaleAB=(?P<scaleAB>True|False)
    , \s+ bpreshuffle=(?P<bpreshuffle>True|False) )?
\s+ found \s+ padded_M: \s+ (?P<M>\d+), \s+ N:(?P<N>\d+), \s+ K:(?P<K>\d+)
\s+ is \s+ tuned \s+ on \s+ cu_num \s+ = \s+ (?P<cu_num>\d+) \s+ in \s+ (?P<table>[^\s,!]+)
(?: \s* , \s+ libtype \s+ is \s+ (?P<libtype>\w+) )?
(?: \s* , \s+ kernel \s+ name \s+ is \s+ (?P<kernelName>[^\s!]*) )?"""

_SGLANG_CUDA_GRAPH_CAPTURE_END = r"""(?ix)
Capture \s+ (?: [a-z]+ \s+ )*? cuda \s+ graph \s+ end \. \s+ (?: Time \s+ )? elapsed (?: : \s* | = \s*)
(\d+(?:\.\d+)?) \s* s\b"""

_SGLANG_CUDA_GRAPH_DECODE_BS = r"""(?x)
(?: 'cuda_graph_config': \s* \{ \s* 'decode': \s* \{ [^{}]*? 'bs'
  | \bcuda_graph_bs\b'? )
\s* [:=] \s* \[ (?P<sizes>\d+ (?: \s*,\s* \d+ )* ) \]"""

_SHORT_PROMPT_TOUCH = r"""(?ix) ^ (?:
    arg \s+ --(?: block-size | max-num-batched-tokens | num-continuous-decode-steps ) \b
  | arg \s+ --(?! mm- ) [a-z0-9-]*? (?<![a-z0-9])
               (?: atten | attn | f?mha | mla | kv | prefill | radix | prefix-cach | page | hicache
                 | hierarchical-cache | hisparse | nsa | dsa | kda | swa | sliding-window | rotary | rope )
  | env \s+ \S*? (?: ATTN | ATTENTION | FLASH | MHA | MLA | KV | PREFILL | RADIX ) \S*? =
  | env \s+ (?: SGLANG_USE_AITER | VLLM_ROCM_USE_AITER | VLLM_USE_AITER ) =
  | (?: arg | env ) \s .* vectorized_5d
  | ref \s+ (?: aiter | sglang | vllm | flash-attention | flash_attn | triton ) \b
  | patch-file \s+ \S+ \s+ (?! \S*? vision ) \S*? (?<![a-z0-9])
               (?: atten | attn | f?mha | mla | flash(?!infer) | kv | prefill | radix[_-]?(?:cache|tree)
                 | mem_cache | paged | pa(?=[_./-]) | asm_pa | rotary | rope | forward_batch_info | schedule_batch
                 | model_runner | cuda_graph | cache_kernels | block_pool | models/ )
  | patch-file \s+ \S+ \s+ (?! \S*? (?: vision | moe | gemm(?!a) ) ) \S*? (?<![a-z0-9]) decode(?!r)
  | kernel \s+ (?: fusion | rewrite ) \s+ \S*? (?<![a-z0-9])
               (?: atten | attn | f?mha | mla | flash | kv | prefill | paged | pa(?=[_./-]) | rotary | rope )
  | image \s+ changed
  | unexpressed \s
)"""

#: Every value ``outside_facts`` held as a literal before the pack existed. Written out, not derived from the pack or
#: from ``outside_facts``, so that a value that moved with the pack would fail here.
_LITERALS: dict[str, str | list[str] | dict[str, str]] = {
    "AIPERF_ERROR_CLASS_KEY": "type",
    "AIPERF_ERROR_KEY": "error",
    "AIPERF_ERROR_OUTPUT_LENGTH_METRIC": "error_osl",
    "AIPERF_EXPORT_DIR": "aiperf_artifacts",
    "AIPERF_EXPORT_FILE": "profile_export.jsonl",
    "AIPERF_INVALID_REPLY_ERROR": "InvalidInferenceResultError",
    "AIPERF_METADATA_KEY": "metadata",
    "AIPERF_METRICS_KEY": "metrics",
    "AIPERF_OUTPUT_LENGTH_FILE": "src/aiperf/metrics/types/output_sequence_length_metric.py",
    "AIPERF_PARSER_FILE": "src/aiperf/records/inference_result_parser.py",
    "AIPERF_PHASE_KEY": "benchmark_phase",
    "AIPERF_PROFILING_PHASE": "profiling",
    "AIPERF_SUBMODULE": "utils/aiperf",
    "AITER_DEVELOP_MODE": "develop",
    "AITER_INSTALL_MODE_FILE": "install_mode",
    "AITER_LOG_TUNED_CONFIG_ENV": "AITER_LOG_TUNED_CONFIG",
    "AITER_MERGED_VIEW_DIR": "/tmp/aiter_configs",
    "AITER_META_DIR_ENV": "AITER_META_DIR",
    "AITER_META_MARKER_DIR": "csrc",
    "AITER_META_WHEEL_DIR": "aiter_meta",
    "AITER_TUNED_HIT": _AITER_TUNED_HIT,
    "AITER_TUNED_HIT_MARKER": r"\btuned(?: on| at|,)|found (?:padded_M|lookup M):",
    "ARGPARSE_INVALID_CHOICE": "invalid choice",
    "ARGPARSE_UNRECOGNIZED_ARGUMENTS_RE": "unrecognized arguments?",
    "ASM_LIBTYPE": "asm",
    "CONTEXT_LENGTH_FLAGS": ["--context-length", "--max-model-len"],
    "ENV_CONC": "CONC",
    "ENV_EP": "EP",
    "ENV_ISL": "ISL",
    "ENV_MAGPIE_EVAL_LIMIT": "MAGPIE_EVAL_LIMIT",
    "ENV_MAGPIE_EVAL_TASKS": "MAGPIE_EVAL_TASKS",
    "ENV_MAGPIE_EVAL_TOKENIZED_REQUESTS": "MAGPIE_EVAL_TOKENIZED_REQUESTS",
    "ENV_MAX_MODEL_LEN": "MAX_MODEL_LEN",
    "ENV_NUM_PROMPTS": "NUM_PROMPTS",
    "ENV_NUM_WARMUPS": "NUM_WARMUPS",
    "ENV_OSL": "OSL",
    "ENV_RANDOM_RANGE_RATIO": "RANDOM_RANGE_RATIO",
    "ENV_RUN_EVAL": "RUN_EVAL",
    "ENV_TP": "TP",
    "FORGE_REL_NORM_ERR_COLUMN": "relNormErr",
    "HIP_ERROR_PREFIX": "hipError",
    "DRM_CLIENT_ID_KEY": "drm-client-id",
    "DRM_FD_DIR": "fd",
    "DRM_FDINFO_DIR": "fdinfo",
    "DRM_RENDER_NODE_PREFIX": "renderD",
    "DRM_VRAM_KEY": "drm-memory-vram",
    "HIP_GRAPH_PACKET_CAPTURE_ENV": "DEBUG_CLR_GRAPH_PACKET_CAPTURE",
    "HIP_GRAPH_PACKET_CAPTURE_NODE_BY_NODE": "0",
    "HIP_NO_BINARY_FOR_GPU": "hipErrorNoBinaryForGpu",
    "HIP_NO_KERNEL_IMAGE": "no kernel image is available",
    "HIP_UNDEFINED_SYMBOL": "undefined symbol",
    "HSA_STATUS_ERROR": "HSA_STATUS_ERROR",
    "INITIAL_PID_NAMESPACE": "pid:[4026531836]",
    "KFD_NODES": "topology/nodes",
    "KFD_NODE_GPU_ID": "gpu_id",
    "KFD_NODE_PROPERTIES": "properties",
    "KFD_PROCESSES": "proc",
    "KFD_QUEUES": "queues",
    "KFD_QUEUE_GPU_ID": "gpuid",
    "KFD_RENDER_MINOR_KEY": "drm_render_minor",
    "KFD_ROOT": "/sys/class/kfd/kfd",
    "KFD_VRAM_PREFIX": "vram_",
    "PID_NAMESPACE_LINK": "self/ns/pid",
    "PY_IMPORT_ERROR": "ImportError",
    "PY_MODULE_NOT_FOUND_ERROR": "ModuleNotFoundError",
    "ROCR_VISIBLE_DEVICES_ENV": "ROCR_VISIBLE_DEVICES",
    "SGLANG_CUDA_GRAPH_CAPTURE_END": _SGLANG_CUDA_GRAPH_CAPTURE_END,
    "SGLANG_CUDA_GRAPH_DECODE_BS": _SGLANG_CUDA_GRAPH_DECODE_BS,
    "SGLANG_IMAGE_AITER_CONFIGS": "/sgl-workspace/aiter/aiter/configs",
    "SHORT_PROMPT_CHECK": {
        "max_new_tokens": "32",
        "phase_cap_sec": "1200",
        "always_purposes": "baseline final",
    },
    "SHORT_PROMPT_TOUCH": _SHORT_PROMPT_TOUCH,
    "SGLANG_IMAGE_WORKSPACE": "/sgl-workspace",
    "TRANSFORMERS_DOES_NOT_RECOGNIZE": "Transformers does not recognize",
    "TRANSFORMERS_UNKNOWN_ARCHITECTURE": "does not recognize this architecture",
    "TUNABLEOP_ENABLED_ENV": "PYTORCH_TUNABLEOP_ENABLED",
    "TUNABLEOP_TABLE_ENV": "PYTORCH_TUNABLEOP_FILENAME",
    "TUNABLEOP_TUNING_ENV": "PYTORCH_TUNABLEOP_TUNING",
    "TUNED_LIBTYPE_COLUMN": "libtype",
    "TUNED_SPLITK_COLUMN": "splitK",
    "UPSTREAM_PR_REPOS": ["sgl-project/sglang", "vllm-project/vllm", "ROCm/aiter", "pytorch/pytorch"],
}

Facts = dict[str, Any]


def _variant(tmp_path: Path, change: Callable[[Facts], None], version: str | None = None) -> Path:
    """A copy of the reference pack with ``change`` applied to its entries."""
    document = yaml.safe_load(DEFAULT_PACK_PATH.read_text(encoding="utf-8"))
    change(document["facts"])
    if version is not None:
        document["version"] = version
    path = tmp_path / "pack.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _drop(entry: str, field: str) -> Callable[[Facts], None]:
    return lambda facts: facts[entry].pop(field)


def _set(entry: str, field: str, value: object) -> Callable[[Facts], None]:
    return lambda facts: facts[entry].__setitem__(field, value)


def _forget_role(facts: Facts) -> None:
    del facts["pytorch_tunableop_env"]["names"]["tuning"]
    del facts["pytorch_tunableop_env"]["image_check"]


def _rename_sglang_flag(facts: Facts) -> None:
    """The flag a changed fact names, and the help its image check is pinned to, which shows the renamed flag."""
    entry = facts["context_length_flags"]
    entry["names"]["sglang"] = "--ctx"
    example = entry["image_check"][0]["examples"]["pass"][0]
    example["help"] = example["help"].replace("--context-length", "--ctx")


def _forget_entry(facts: Facts) -> None:
    del facts["pytorch_tunableop_env"]


def _unmatched_no_match(facts: Facts) -> None:
    facts["hsa_status_error"]["examples"]["no_match"].append("HSA_STATUS_ERROR_OUT_OF_RESOURCES")


def _fresh(code: str, pack: str | None) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a new interpreter that reads the pack ``pack`` names (the reference pack when ``None``)."""
    env = {"PYTHONPATH": os.pathsep.join(str(Path(entry).resolve()) for entry in sys.path if entry)}
    if pack is not None:
        env[PACK_ENV] = pack
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd="/", env=env, check=False
    )


def test_the_reference_pack_loads_and_passes_its_own_examples() -> None:
    pack = load_pack(DEFAULT_PACK_PATH)
    assert pack.version == str(yaml.safe_load(DEFAULT_PACK_PATH.read_text(encoding="utf-8"))["version"])
    assert Counter(entry.kind for entry in pack.entries.values()) == {
        Kind.NAMES: 20,
        Kind.LOG_REGEX: 15,
        Kind.ENV_DEFAULT: 1,
        Kind.COMPONENT_LAYOUT: 9,
    }


def test_every_outside_fact_is_the_literal_it_was_before_the_pack() -> None:
    held = [name for name in outside_facts.__all__ if not callable(getattr(outside_facts, name))]
    assert sorted(held) == sorted(_LITERALS)
    for name, literal in _LITERALS.items():
        value = getattr(outside_facts, name)
        assert (list(value) if isinstance(value, tuple) else value) == literal, name


def test_every_entry_of_the_reference_pack_is_read_by_outside_facts() -> None:
    tree = ast.parse(Path(outside_facts.__file__).read_text(encoding="utf-8"))
    read = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "_PACK"
    }
    assert read == set(load_pack(DEFAULT_PACK_PATH).entries)


@pytest.mark.parametrize(
    ("change", "entry", "reason"),
    [
        (_set("hip_no_kernel_image", "pattern", "no kernel image is unavailable"), "hip_no_kernel_image", "must-match"),
        (_unmatched_no_match, "hsa_status_error", "must-not-match"),
        (_set("hip_error_prefix", "pattern", "hipError("), "hip_error_prefix", "does not compile"),
        (_set("pytorch_tunableop_env", "examples", []), "pytorch_tunableop_env", "at least one example"),
        (_set("hip_graph_packet_capture", "value", 0), "hip_graph_packet_capture", "non-empty string"),
        (_drop("aiter_merged_view", "reason"), "aiter_merged_view", "holds exactly"),
        (_set("aiter_merged_view", "colour", "red"), "aiter_merged_view", "holds exactly"),
        (
            _set("component_layout_triton", "examples", {"place": ["torch/x.py -> package:x.py"], "refuse": ["a.so"]}),
            "component_layout_triton",
            "is not placed as it says",
        ),
        (
            _set(
                "component_layout_triton",
                "examples",
                {"place": ["python/triton/a.py -> package:a.py"], "refuse": ["python/triton/a.py"]},
            ),
            "component_layout_triton",
            "must be refused",
        ),
        (
            _set("component_layout_triton", "sections", [{"prefix": "python/triton", "base": "elsewhere"}]),
            "component_layout_triton",
            "is not one of",
        ),
        (
            _set("component_layout_triton", "sections", [{"prefix": "../x", "base": "package"}]),
            "component_layout_triton",
            "plain relative path",
        ),
        (
            _set(
                "component_layout_triton",
                "sections",
                [{"prefix": "a", "base": "package"}, {"prefix": "a", "base": "package"}],
            ),
            "component_layout_triton",
            "same prefix",
        ),
        (_drop("component_layout_torch", "refuse"), "component_layout_torch", "holds exactly"),
        (_set("aiter_merged_view", "kind", "arg_value"), "aiter_merged_view", "reserved"),
        (_set("aiter_merged_view", "kind", "predicate"), "aiter_merged_view", "is not one of"),
    ],
    ids=[
        "failed-must-match",
        "failed-must-not-match",
        "pattern-does-not-compile",
        "no-examples",
        "unquoted-number",
        "lacks-reason",
        "unknown-field",
        "layout-example-misplaced",
        "layout-refuse-example-taken",
        "layout-unknown-base",
        "layout-escaping-prefix",
        "layout-repeated-prefix",
        "layout-lacks-a-field",
        "reserved-kind",
        "unknown-kind",
    ],
)
def test_a_pack_with_a_wrong_entry_is_refused_naming_the_entry(
    tmp_path: Path, change: Callable[[Facts], None], entry: str, reason: str
) -> None:
    with pytest.raises(RulePackError) as refused:
        load_pack(_variant(tmp_path, change))
    assert entry in str(refused.value)
    assert reason in str(refused.value)
    assert str(tmp_path / "pack.yaml") in str(refused.value)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "exactly the keys"),
        ("version: [", "not one valid YAML document"),
        ("version: '1'\nfacts: {}\n---\nversion: '2'\n", "not one valid YAML document"),
        ("version: '1'\nversion: '2'\nfacts: {a: 1}\n", "written twice"),
        ("version: '1'\nfacts:\n  a: {kind: names}\n  a: {kind: names}\n", "written twice"),
        (
            "version: '1'\nfacts:\n  hip_error_prefix: {kind: names, kind: log_regex}\n",
            "facts.hip_error_prefix.*written twice",
        ),
        (
            "version: '1'\nfacts:\n  a: &base {kind: names}\n  b:\n    <<: *base\n    kind: log_regex\n",
            "facts.b uses a YAML merge key",
        ),
        ("version: '1'\nfacts: {a: &x 1, b: *x}\n", "alias"),
        ("version: '1'\nfacts: &loop {a: *loop}\n", "alias"),
        ("version: '1'\nfacts:\n  a:\n    <<: {kind: names}\n", "facts.a uses a YAML merge key"),
        ("version: '1'\nfacts: {}\n", "non-empty mapping"),
        ("version: 1\nfacts: {a: 1}\n", "version must be a non-empty string"),
        ("version: '1'\nfacts: {a: 1}\nextra: 1\n", "exactly the keys"),
        ("version: '1'\nfacts: {a: 1}\n", "must be a mapping"),
    ],
)
def test_a_malformed_pack_is_refused(tmp_path: Path, text: str, reason: str) -> None:
    path = tmp_path / "pack.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RulePackError, match=reason):
        load_pack(path)


def test_a_missing_pack_file_is_refused_naming_it(tmp_path: Path) -> None:
    with pytest.raises(RulePackError, match="absent.yaml.*cannot be read"):
        load_pack(tmp_path / "absent.yaml")


_SHOW = textwrap.dedent(
    """
    import json
    from hyperloom.common import outside_facts
    from hyperloom.common.outside_facts import rule_pack

    print(json.dumps({"flags": outside_facts.CONTEXT_LENGTH_FLAGS, "identity": rule_pack.rule_pack_identity()}))
    """
)


def test_a_session_reads_the_pack_its_input_names_and_a_changed_fact_needs_no_code_change(tmp_path: Path) -> None:
    pack = _variant(tmp_path, _rename_sglang_flag, version="2")
    done = _fresh(_SHOW, str(pack))
    assert done.returncode == 0, done.stderr[-2000:]
    shown = json.loads(done.stdout)
    assert shown["flags"] == ["--ctx", "--max-model-len"]
    assert shown["identity"] == {"version": "2", "digest": "sha256:" + hashlib.sha256(pack.read_bytes()).hexdigest()}


@pytest.mark.parametrize(
    ("pack", "entry", "reason"),
    [
        ("{tmp}/absent.yaml", "absent.yaml", "cannot be read"),
        ("", PACK_ENV, "set but empty"),
        (_forget_entry, "pytorch_tunableop_env", "has no entry"),
        (_forget_role, "pytorch_tunableop_env", "no name for the role"),
        (_unmatched_no_match, "hsa_status_error", "must-not-match"),
    ],
    ids=["missing-file", "empty-input", "lacks-an-entry-the-code-reads", "lacks-a-role", "failing-example"],
)
def test_a_session_whose_pack_cannot_be_used_stops_at_import_naming_the_entry(
    tmp_path: Path, pack: object, entry: str, reason: str
) -> None:
    named = _variant(tmp_path, pack) if callable(pack) else str(pack).format(tmp=tmp_path)
    done = _fresh("import hyperloom.common.outside_facts", str(named))
    assert done.returncode != 0
    assert "RulePackError" in done.stderr
    assert entry in done.stderr
    assert reason in done.stderr


def test_the_recorded_identity_is_the_digest_of_the_bytes_read_and_is_stable(tmp_path: Path) -> None:
    expected = {
        "version": load_pack(DEFAULT_PACK_PATH).version,
        "digest": "sha256:" + hashlib.sha256(DEFAULT_PACK_PATH.read_bytes()).hexdigest(),
    }
    assert rule_pack_identity() == expected
    assert rule_pack_identity() == rule_pack_identity()
    elsewhere = _fresh(_SHOW, None)
    assert elsewhere.returncode == 0, elsewhere.stderr[-2000:]
    assert json.loads(elsewhere.stdout)["identity"] == expected

    edited = tmp_path / "edited.yaml"
    edited.write_bytes(DEFAULT_PACK_PATH.read_bytes() + b"# a comment changes the bytes, not the facts\n")
    assert load_pack(edited).digest != expected["digest"]
    assert load_pack(edited).version == expected["version"]


def test_the_default_pack_is_declared_as_package_data() -> None:
    root = Path(hyperloom.__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        pytest.skip("the declaration is read from the source checkout")
    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"][
        "package-data"
    ]
    relative = DEFAULT_PACK_PATH.resolve().relative_to(Path(hyperloom.__file__).resolve().parent)
    assert any(fnmatchcase(relative.as_posix(), pattern) for pattern in declared["hyperloom"])
