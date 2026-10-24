# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Each fact about an outside project is stated once, in ``hyperloom.common.outside_facts``, and every reader reads it.

Three checks, so that a fact cannot go back to being a copy:

* a fresh interpreter changes every owned string before any reader is imported, and each reader must show the changed
  value (this tells *which* module a reader takes its fact from, not only that it agrees today);
* no non-docstring string of our layers repeats an owned value (the checklist is the owner's own ``__all__``);
* the two copies that cannot import the owner, a standalone sandbox hook and the profile YAML files, hold its value.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

import hyperloom
import hyperloom_multibranch
from hyperloom.common import outside_facts
from hyperloom.measure import aiter_merge_hook

#: Owned names whose value is a bare ``0`` or a shell-level phrase rather than a name: too common to scan for. So are
#: the keys of aiperf's per-request export that are ordinary words (``error``, ``type``, ``metadata``, ``metrics``).
_NOT_SCANNED = frozenset(
    {
        "HIP_GRAPH_PACKET_CAPTURE_NODE_BY_NODE",
        "KFD_NODES",
        "KFD_NODE_GPU_ID",
        "KFD_PROCESSES",
        "KFD_QUEUES",
        "KFD_QUEUE_GPU_ID",
        "KFD_VRAM_PREFIX",
        "PID_NAMESPACE_LINK",
        "AIPERF_ERROR_CLASS_KEY",
        "AIPERF_ERROR_KEY",
        "AIPERF_METADATA_KEY",
        "AIPERF_METRICS_KEY",
        "AIPERF_PROFILING_PHASE",
    }
)
#: The one source file allowed to hold an owned value as a literal: it runs inside a sandbox with the standard library only.
_STANDALONE = {"hyperloom/measure/aiter_merge_hook.py": {"AITER_MERGED_VIEW_DIR"}}
#: The module that binds the facts to their names: the one place that holds the values.
_OWNER = Path(outside_facts.__file__).resolve()
_ROOTS = (
    Path(hyperloom.__file__).parent / "measure",
    Path(hyperloom.__file__).parent / "common",
    Path(hyperloom.__file__).parent / "sandbox",
    Path(hyperloom_multibranch.__file__).parent / "toolbox",
)
_PROFILES = Path(hyperloom.__file__).parent / "inference_optimizer" / "assets" / "configs"
#: The facts that are sentences or regular-expression source: found inside longer strings, not only as a whole string.
#: (``ImportError`` and the ``hipError`` prefix are left out: Python code and other HIP tokens contain them.)
_FRAGMENTS = frozenset(
    {
        "HIP_UNDEFINED_SYMBOL",
        "HIP_NO_KERNEL_IMAGE",
        "HIP_NO_BINARY_FOR_GPU",
        "HSA_STATUS_ERROR",
        "ARGPARSE_UNRECOGNIZED_ARGUMENTS_RE",
        "ARGPARSE_INVALID_CHOICE",
        "TRANSFORMERS_UNKNOWN_ARCHITECTURE",
        "TRANSFORMERS_DOES_NOT_RECOGNIZE",
    }
)


def _owned() -> dict[str, str]:
    owned: dict[str, str] = {}
    for name in outside_facts.__all__:
        value = getattr(outside_facts, name)
        if isinstance(value, str) and name not in _NOT_SCANNED:
            owned[name] = value
        elif isinstance(value, tuple):
            owned.update({f"{name}[{index}]": item for index, item in enumerate(value)})
    return owned


def _strings_outside_docstrings(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


def test_no_reader_repeats_an_owned_value() -> None:
    owned = _owned()
    assert len(owned) > 30, "the scan has its checklist"
    found = []
    for root in _ROOTS:
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root.parent.parent).as_posix()
            if "/tests/" in relative or path.name.startswith("test_") or path.resolve() == _OWNER:
                continue
            for line, text in _strings_outside_docstrings(path):
                for name, value in owned.items():
                    if name.split("[")[0] in _STANDALONE.get(relative, ()):
                        continue
                    fragment = name.split("[")[0] in _FRAGMENTS
                    if text == value or (fragment and value in text):
                        found.append(f"{relative}:{line} repeats {name}")
    assert not found, "\n".join(found)


_PROBE = textwrap.dedent(
    """
    import json
    import hyperloom.common.outside_facts as o

    for name in o.__all__:
        value = getattr(o, name)
        if isinstance(value, str):
            setattr(o, name, name)
        elif isinstance(value, tuple):
            setattr(o, name, tuple(f"{name}_{i}" for i in range(len(value))))

    from hyperloom.common import env_safety, failure_signature, visible_devices
    from hyperloom.measure import compile as compiled
    from hyperloom.measure import agentx_requests, eval_context, farm, hosted_spec, local, log_causes, points, surface
    from hyperloom.measure.framework import base as framework_base
    from hyperloom.measure.runner import run
    from hyperloom.sandbox import card_occupancy, templates
    from hyperloom_multibranch.toolbox import acceptance, hosted_executor, mcp_tools
    from hyperloom_multibranch.toolbox.tools import kernel_tune, tracelens_report, tune_rows

    out = {
        "signature": [pattern.pattern for rule in failure_signature._RULES for pattern in rule.patterns],
        "causes": [value.pattern for name, value in vars(log_causes).items() if hasattr(value, "pattern")],
        "surface": [surface.SURFACE_ENV_KEYS, surface.POINT_ENV_KEYS, surface.REQUEST_ENV_KEYS],
        "points": [points._PROFILE_PINNED, points._PROFILE_SURFACE_EXEMPT],
        "local": [sorted(local._BOOT_CLIENT_ENVS), sorted(local._MEASURED_ENVS)],
        "compile": [compiled.IMAGE_SOURCE_ROOTS, sorted(compiled.PER_DEVICE_TABLE_ENVS)],
        "configs": framework_base.DEFAULT_CONFIGS,
        "context": [eval_context.EVAL_CONTEXT_ENV, eval_context.CONTEXT_FLAGS],
        "reason": acceptance.ONLINE_TUNING_REASON,
        "tool": mcp_tools._PUT_KERNEL_ARTIFACT["properties"]["env"]["description"],
        "hints": [tracelens_report.SPLIT_HINTS, tracelens_report.WORKLOAD_HINTS],
        "blocked": sorted(env_safety.BLOCKED_EXTERNAL_ENV_NAMES),
        "reserved": sorted(templates.RESERVED_ENV_NAMES),
        "rocr": visible_devices.ROCR_LEVEL_VARS[0],
        "gpus": hosted_spec.gpu_count_of({"benchmark": {"envs": {"ENV_TP": "4"}}}),
        "concurrency": hosted_executor._concurrency_in("benchmark: {envs: {ENV_CONC: 7}}"),
        "visible": run._visible({"ROCR_VISIBLE_DEVICES_ENV": "0,2"}),
        "export": list(agentx_requests._EXPORT_PATTERNS),
        "kfd": [
            str(card_occupancy.KFD_PATH),
            card_occupancy.KFD_PROCESSES,
            card_occupancy.KFD_NODES,
            card_occupancy.KFD_NODE_GPU_ID,
            card_occupancy.KFD_QUEUES,
            card_occupancy.KFD_QUEUE_GPU_ID,
            card_occupancy.KFD_VRAM_PREFIX,
            card_occupancy.PID_NAMESPACE_LINK,
            card_occupancy.INITIAL_PID_NAMESPACE,
        ],
        "meta": [
            farm.AITER_META_DIR_ENV,
            farm.AITER_META_MARKER_DIR,
            farm.AITER_META_WHEEL_DIR,
            farm.AITER_INSTALL_MODE_FILE,
            farm.AITER_DEVELOP_MODE,
        ],
        "tuner": kernel_tune.AITER_META_MARKER_DIR,
        "relnorm": tune_rows._rel_norm(
            {
                "TUNED_LIBTYPE_COLUMN": "ASM_LIBTYPE",
                "TUNED_SPLITK_COLUMN": "6",
                "FORGE_REL_NORM_ERR_COLUMN": "0.005",
            }
        ),
    }
    print(json.dumps(out))
    """
)


@pytest.fixture(scope="module")
def probe() -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        cwd="/",
        env={"PYTHONPATH": os.pathsep.join(str(Path(entry).resolve()) for entry in sys.path if entry)},
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.splitlines()[-1])


def test_log_markers_are_read_from_the_owner_by_both_log_readers(probe: dict[str, Any]) -> None:
    """D6, D11: the signature table (``common``) and the cause readers (``measure``) take one set of markers."""
    shared = [
        "HIP_UNDEFINED_SYMBOL",
        "HIP_NO_KERNEL_IMAGE",
        "HIP_NO_BINARY_FOR_GPU",
        "HSA_STATUS_ERROR",
        "HIP_ERROR_PREFIX",
        "PY_IMPORT_ERROR",
        "PY_MODULE_NOT_FOUND_ERROR",
        "ARGPARSE_UNRECOGNIZED_ARGUMENTS_RE",
        "ARGPARSE_INVALID_CHOICE",
    ]
    for table in ("signature", "causes"):
        text = "\n".join(probe[table])
        assert [name for name in shared if name not in text] == [], table
    assert "TRANSFORMERS_UNKNOWN_ARCHITECTURE" in "\n".join(probe["signature"])
    assert "TRANSFORMERS_DOES_NOT_RECOGNIZE" in "\n".join(probe["signature"])


def test_image_layout_is_read_from_the_owner(probe: dict[str, Any]) -> None:
    """D3: the image's workspace and aiter's default tables."""
    assert probe["configs"] == "SGLANG_IMAGE_AITER_CONFIGS"
    assert probe["compile"][0][0] == "SGLANG_IMAGE_WORKSPACE"
    assert outside_facts.SGLANG_IMAGE_AITER_CONFIGS.startswith(outside_facts.SGLANG_IMAGE_WORKSPACE + "/")


def test_context_flags_are_read_from_the_owner(probe: dict[str, Any]) -> None:
    """D4: the flags that set the served context, and the point variable the evaluation's length comes from."""
    assert probe["context"] == ["ENV_MAX_MODEL_LEN", ["CONTEXT_LENGTH_FLAGS_0", "CONTEXT_LENGTH_FLAGS_1"]]


def test_tunableop_names_are_read_from_the_owner(probe: dict[str, Any]) -> None:
    """D8: the refusal's text, the tool's description and the compiler's per-device list name TunableOp's variables."""
    for text in (probe["reason"], probe["tool"]):
        for name in ("TUNABLEOP_ENABLED_ENV", "TUNABLEOP_TUNING_ENV", "TUNABLEOP_TABLE_ENV"):
            assert name in text, name
    assert probe["compile"][1] == ["TUNABLEOP_TABLE_ENV"]


def test_inferencex_env_names_are_read_from_the_owner(probe: dict[str, Any]) -> None:
    """D9: surface, points, the local backend, the splitter hints, the block list and the two small readers."""
    surface = probe["surface"]
    assert surface[0] == [
        "ENV_ISL",
        "ENV_OSL",
        "ENV_RANDOM_RANGE_RATIO",
        "ENV_MAGPIE_EVAL_TASKS",
        "ENV_MAGPIE_EVAL_LIMIT",
        "ENV_MAGPIE_EVAL_TOKENIZED_REQUESTS",
    ]
    assert surface[1:] == [["ENV_NUM_PROMPTS", "ENV_NUM_WARMUPS"], ["ENV_CONC", "ENV_RUN_EVAL"]]
    assert probe["points"] == [
        ["ENV_TP", "ENV_EP", "ENV_ISL", "ENV_RANDOM_RANGE_RATIO", "ENV_MAX_MODEL_LEN"],
        ["envs.ENV_OSL"],
    ]
    assert probe["local"] == [["ENV_NUM_PROMPTS", "ENV_RUN_EVAL"], ["ENV_RUN_EVAL"]]
    assert [hint[2] for hint in probe["hints"][1]] == ["ENV_CONC", "ENV_OSL", "ENV_RANDOM_RANGE_RATIO", "ENV_ISL"]
    assert {"ENV_TP", "ENV_EP", "ENV_CONC", "ENV_ISL", "ENV_OSL", "ENV_MAX_MODEL_LEN", "ENV_RUN_EVAL"} <= set(
        probe["blocked"]
    )
    assert probe["gpus"] == 4
    assert probe["concurrency"] == 7


def test_rocr_mask_name_is_read_from_the_owner(probe: dict[str, Any]) -> None:
    """D10: the sandbox's reserved names, the mask chain and the runner's reader."""
    assert "ROCR_VISIBLE_DEVICES_ENV" in probe["reserved"]
    assert probe["rocr"] == "ROCR_VISIBLE_DEVICES_ENV"
    assert probe["visible"] == [0, 2]


def test_the_kernels_process_list_is_read_where_the_owner_says_it_is(probe: dict[str, Any]) -> None:
    """The card-occupancy probe finds the GPU processes by the owner's names and tells its namespace by the owner's link."""
    assert probe["kfd"] == [
        "KFD_ROOT",
        "KFD_PROCESSES",
        "KFD_NODES",
        "KFD_NODE_GPU_ID",
        "KFD_QUEUES",
        "KFD_QUEUE_GPU_ID",
        "KFD_VRAM_PREFIX",
        "PID_NAMESPACE_LINK",
        "INITIAL_PID_NAMESPACE",
    ]


def test_aiperf_export_names_are_read_from_the_owner(probe: dict[str, Any]) -> None:
    """The per-request export of an AgentX round is looked for where the owner says the replay client writes it."""
    assert probe["export"] == [
        "AIPERF_EXPORT_DIR/AIPERF_EXPORT_FILE",
        "*/AIPERF_EXPORT_DIR/AIPERF_EXPORT_FILE",
    ]


def test_aiter_meta_directory_names_are_read_from_the_owner(probe: dict[str, Any]) -> None:
    """The farm names the variable, the directories and the install mode it resolves, and the tuner's refusal the marker."""
    assert probe["meta"] == [
        "AITER_META_DIR_ENV",
        "AITER_META_MARKER_DIR",
        "AITER_META_WHEEL_DIR",
        "AITER_INSTALL_MODE_FILE",
        "AITER_DEVELOP_MODE",
    ]
    assert probe["tuner"] == "AITER_META_MARKER_DIR"


def test_the_columns_that_pick_out_the_rows_judged_by_their_relative_norm_error_are_read_from_the_owner(
    probe: dict[str, Any],
) -> None:
    """A row is one of them by the libtype and splitK columns and the libtype the owner names, its figure is in the column it names."""
    assert probe["relnorm"] == 0.005


def test_standalone_hook_holds_the_owners_merge_directory() -> None:
    """D2: the hook runs in a sandbox with the standard library only, so it cannot import the owner; it must agree."""
    assert aiter_merge_hook._HARD_CODED == outside_facts.AITER_MERGED_VIEW_DIR


def _entries(node: Any, key: str) -> list[Any]:
    found = []
    if isinstance(node, dict):
        found += [node[key]] if key in node else []
        for child in node.values():
            found += _entries(child, key)
    elif isinstance(node, list):
        for child in node:
            found += _entries(child, key)
    return found


def test_every_profile_yaml_copy_holds_the_owners_replay_setting() -> None:
    """D7: the profile files are data, so they stay literal; each copy of the HIP graph-replay setting is the owner's."""
    holders = {}
    for path in sorted(_PROFILES.glob("profile_*.yaml")):
        values = _entries(yaml.safe_load(path.read_text(encoding="utf-8")), outside_facts.HIP_GRAPH_PACKET_CAPTURE_ENV)
        if values:
            holders[path.name] = values
    assert {"profile_sglang.yaml", "profile_atom.yaml", "profile_vllm.yaml"} <= set(holders)
    assert {
        name: values
        for name, values in holders.items()
        if values != [outside_facts.HIP_GRAPH_PACKET_CAPTURE_NODE_BY_NODE]
    } == {}
