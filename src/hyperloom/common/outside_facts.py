# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Facts about projects outside this repository that more than one of our readers depends on, each stated once.

aiter, SGLang, vLLM, PyTorch, the HIP runtime, argparse, Transformers and InferenceX decide these; we only repeat them. A
fact is one name here, and every reader in ``hyperloom.measure``, ``hyperloom.common``, ``hyperloom.sandbox`` and
``hyperloom_multibranch.toolbox`` imports that name, so a change in the outside project is made in one place. The module
lives in ``hyperloom.common`` because that is the one layer all of those readers may import (``common`` imports nothing
else of the project).

Two kinds of copy cannot import it and are pinned to it by a test instead: a YAML data file (``profile_*.yaml``), and a
file that is copied into a sandbox and runs there with the standard library only (``aiter_merge_hook``). The names are
the entries a pack of outside facts (board #110) will later back; until then they are plain constants and this module
loads nothing.
"""

from __future__ import annotations

__all__ = [
    "AITER_MERGED_VIEW_DIR",
    "ARGPARSE_INVALID_CHOICE",
    "ARGPARSE_UNRECOGNIZED_ARGUMENTS_RE",
    "CONTEXT_LENGTH_FLAGS",
    "ENV_CONC",
    "ENV_EP",
    "ENV_ISL",
    "ENV_MAGPIE_EVAL_LIMIT",
    "ENV_MAGPIE_EVAL_TASKS",
    "ENV_MAGPIE_EVAL_TOKENIZED_REQUESTS",
    "ENV_MAX_MODEL_LEN",
    "ENV_NUM_PROMPTS",
    "ENV_NUM_WARMUPS",
    "ENV_OSL",
    "ENV_RANDOM_RANGE_RATIO",
    "ENV_RUN_EVAL",
    "ENV_TP",
    "HIP_ERROR_PREFIX",
    "HIP_GRAPH_PACKET_CAPTURE_ENV",
    "HIP_GRAPH_PACKET_CAPTURE_NODE_BY_NODE",
    "HIP_NO_BINARY_FOR_GPU",
    "HIP_NO_KERNEL_IMAGE",
    "HIP_UNDEFINED_SYMBOL",
    "HSA_STATUS_ERROR",
    "PY_IMPORT_ERROR",
    "PY_MODULE_NOT_FOUND_ERROR",
    "ROCR_VISIBLE_DEVICES_ENV",
    "SGLANG_IMAGE_AITER_CONFIGS",
    "SGLANG_IMAGE_WORKSPACE",
    "TRANSFORMERS_DOES_NOT_RECOGNIZE",
    "TRANSFORMERS_UNKNOWN_ARCHITECTURE",
    "TUNABLEOP_ENABLED_ENV",
    "TUNABLEOP_TABLE_ENV",
    "TUNABLEOP_TUNING_ENV",
    "spaced",
]

# --- aiter ------------------------------------------------------------------------------------------------------------
#: Where aiter's ``AITER_CONFIG.update_config_files`` writes the merged view of a variable that lists several tables: a
#: literal in aiter's function body that no variable and no ``TMPDIR`` moves.
AITER_MERGED_VIEW_DIR = "/tmp/aiter_configs"

# --- the SGLang serving image's layout --------------------------------------------------------------------------------
#: The directory the SGLang image keeps its source checkouts (aiter, sglang, vllm, flydsl) in.
SGLANG_IMAGE_WORKSPACE = "/sgl-workspace"
#: The directory of aiter's default tuned tables in that image.
SGLANG_IMAGE_AITER_CONFIGS = f"{SGLANG_IMAGE_WORKSPACE}/aiter/aiter/configs"

# --- serving flags ----------------------------------------------------------------------------------------------------
#: The server flags that set the context length a server serves, by the framework that takes them
#: (``--context-length`` on SGLang, ``--max-model-len`` on vLLM).
CONTEXT_LENGTH_FLAGS = ("--context-length", "--max-model-len")

# --- HIP runtime and ROCm log markers ---------------------------------------------------------------------------------
#: What a missing compiled kernel prints: an extension that references a symbol nothing defines, a GPU binary built for
#: another device, an HSA status.
HIP_UNDEFINED_SYMBOL = "undefined symbol"
HIP_NO_KERNEL_IMAGE = "no kernel image is available"
HIP_NO_BINARY_FOR_GPU = "hipErrorNoBinaryForGpu"
HSA_STATUS_ERROR = "HSA_STATUS_ERROR"
#: Every HIP call that fails names its error with this prefix (``hipErrorIllegalAddress``, ...).
HIP_ERROR_PREFIX = "hipError"
#: The HIP runtime setting that replays a captured graph node by node (value ``0``) instead of from its recorded packets,
#: so the ROCm profiler records every replay's kernels. Set by the profile configurations (``profile_*.yaml``).
HIP_GRAPH_PACKET_CAPTURE_ENV = "DEBUG_CLR_GRAPH_PACKET_CAPTURE"
HIP_GRAPH_PACKET_CAPTURE_NODE_BY_NODE = "0"
#: The ROCr device mask a sandbox's cards are leased through; it renumbers the visible devices ``0..N-1``.
ROCR_VISIBLE_DEVICES_ENV = "ROCR_VISIBLE_DEVICES"

# --- Python, argparse and Transformers messages -----------------------------------------------------------------------
#: The exceptions a failed import raises.
PY_IMPORT_ERROR = "ImportError"
PY_MODULE_NOT_FOUND_ERROR = "ModuleNotFoundError"
#: What argparse prints when it refuses a command line (``argparse.py``). The first is regular-expression source (the
#: singular and the plural both occur), the second a plain phrase.
ARGPARSE_UNRECOGNIZED_ARGUMENTS_RE = "unrecognized arguments?"
ARGPARSE_INVALID_CHOICE = "invalid choice"
#: What Transformers raises for a checkpoint whose ``model_type`` or architecture it does not know.
TRANSFORMERS_UNKNOWN_ARCHITECTURE = "does not recognize this architecture"
TRANSFORMERS_DOES_NOT_RECOGNIZE = "Transformers does not recognize"

# --- PyTorch TunableOp ------------------------------------------------------------------------------------------------
#: TunableOp tunes a GEMM shape the first time a server runs it when ``ENABLED`` is ``1`` and ``TUNING`` is not ``0``; the
#: results file is named by ``FILENAME``.
TUNABLEOP_ENABLED_ENV = "PYTORCH_TUNABLEOP_ENABLED"
TUNABLEOP_TUNING_ENV = "PYTORCH_TUNABLEOP_TUNING"
TUNABLEOP_TABLE_ENV = "PYTORCH_TUNABLEOP_FILENAME"

# --- InferenceX / Magpie benchmark environment ------------------------------------------------------------------------
#: The names of the workload, the request and the evaluation in ``benchmark.envs``, as InferenceX's scripts read them.
ENV_ISL = "ISL"
ENV_OSL = "OSL"
ENV_RANDOM_RANGE_RATIO = "RANDOM_RANGE_RATIO"
ENV_CONC = "CONC"
ENV_RUN_EVAL = "RUN_EVAL"
ENV_NUM_PROMPTS = "NUM_PROMPTS"
ENV_NUM_WARMUPS = "NUM_WARMUPS"
ENV_MAX_MODEL_LEN = "MAX_MODEL_LEN"
ENV_TP = "TP"
ENV_EP = "EP"
ENV_MAGPIE_EVAL_TASKS = "MAGPIE_EVAL_TASKS"
ENV_MAGPIE_EVAL_LIMIT = "MAGPIE_EVAL_LIMIT"
ENV_MAGPIE_EVAL_TOKENIZED_REQUESTS = "MAGPIE_EVAL_TOKENIZED_REQUESTS"


def spaced(phrase: str) -> str:
    """Regular-expression source for ``phrase`` in which each run of whitespace matches any run of whitespace."""
    return r"\s+".join(phrase.split())
