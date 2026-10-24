# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceX's reference launches for the session's model become leads that say what they were written for."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_multibranch.notebook.knowledge.reference_leads import reference_leads
from hyperloom_multibranch.notebook.tests.factories import profile

REVISION = "3d5581562f643f9bdeb8410cd924e2c70906c966"
SCRIPTS = "benchmarks/single_node/fixed_seq_len"
MATRIX = """\
dsr1-fp8-mi355x-vllm:
  image: rocm/vllm:v0.11
  model: deepseek-ai/DeepSeek-R1
  model-prefix: dsr1
  runner: mi355x
  precision: fp8
  framework: vllm
  multinode: false
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 1024
      search-space:
      - { tp: 8, conc-start: 4, conc-end: 64 }
    - isl: 8192
      osl: 1024
      search-space:
      - { tp: 8, conc-start: 4, conc-end: 64 }

dsr1-fp8-mi300x-sglang-mtp:
  image: lmsysorg/sglang:v0.5.12-rocm700-mi30x
  model: deepseek-ai/DeepSeek-R1
  model-prefix: dsr1
  runner: cluster:mi300x-amds
  precision: fp8
  framework: sglang
  multinode: false
  scenarios:
    fixed-seq-len:
    - isl: 8192
      osl: 1024
      search-space:
      - { tp: 4, conc-start: 4, conc-end: 32, spec-decoding: mtp }

dsr1-fp8-mi355x-sglang-disagg:
  image: lmsysorg/sglang:v0.5.12-rocm700-mi35x
  model: deepseek-ai/DeepSeek-R1
  model-prefix: dsr1
  runner: mi355x
  precision: fp8
  framework: sglang
  multinode: true
  scenarios: {}

qwen3.5-fp8-mi355x-sglang:
  image: lmsysorg/sglang:v0.5.12-rocm700-mi35x
  model: Qwen/Qwen3.5-397B-A17B-FP8
  model-prefix: qwen3.5
  runner: mi355x
  precision: fp8
  framework: sglang
  multinode: false
  scenarios:
    fixed-seq-len:
    - isl: 1024
      osl: 1024
      search-space:
      - { tp: 8, conc-start: 4, conc-end: 64 }
"""
AGENTIC = """\
dsr1-fp8-mi355x-sglang-agentic:
  image: lmsysorg/sglang:v0.5.16-rocm720-mi35x
  model: deepseek-ai/DeepSeek-R1
  model-prefix: dsr1
  runner: cluster:mi355x-amds
  precision: fp8
  framework: sglang
  multinode: false
  scenarios:
    agentic-coding:
    - dram-utilization: 0.80
      search-space:
      - { tp: 2, ep: 2, conc-list: [1, 4, 6] }
      - { tp: 4, ep: 1, conc-list: [1, 8, 16, 48] }
"""


ORDERED = """\
far-gpu:
  model: deepseek-ai/DeepSeek-R1
  model-prefix: dsr1
  runner: mi325x
  precision: fp8
  framework: vllm
  scenarios:
    fixed-seq-len:
    - { isl: 1024, osl: 1024, search-space: [{ tp: 8, conc-start: 4, conc-end: 64 }] }
other-sweep:
  model: deepseek-ai/DeepSeek-R1
  model-prefix: dsr1
  runner: mi355x
  precision: fp8
  framework: vllm
  scenarios:
    fixed-seq-len:
    - { isl: 8192, osl: 1024, search-space: [{ tp: 4, conc-start: 4, conc-end: 32 }] }
"""


def _checkout(root: Path, matrix: str, *scripts: str) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "amd-master.yaml").write_text(matrix, encoding="utf-8")
    (root / SCRIPTS).mkdir(parents=True)
    for name in scripts:
        (root / SCRIPTS / name).write_text("#!/bin/bash\n", encoding="utf-8")
    return root


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    return _checkout(tmp_path, MATRIX, "dsr1_fp8_mi355x_vllm.sh", "dsr1_fp8_mi300x_mtp.sh", "qwen3.5_fp8_mi355x.sh")


def test_the_single_node_entries_of_the_sessions_model_are_leads_its_own_setup_first(checkout: Path) -> None:
    exact, other = reference_leads(checkout, profile(), REVISION)

    assert (exact.config, exact.script, exact.differing) == (
        "dsr1-fp8-mi355x-vllm",
        f"{SCRIPTS}/dsr1_fp8_mi355x_vllm.sh",
        {},
    )
    assert exact.workload == "ISL 1024/OSL 1024, ISL 8192/OSL 1024 at TP 8, conc 4-64"
    assert (other.config, other.script) == ("dsr1-fp8-mi300x-sglang-mtp", f"{SCRIPTS}/dsr1_fp8_mi300x_mtp.sh")
    assert other.differing == {
        "gpu": ("mi300x", "mi355x"),
        "framework": ("sglang", "vllm"),
        "workload": ("ISL 8192/OSL 1024", "ISL 1024/OSL 1024"),
        "tp": ("4", "8"),
        "conc": ("4-32", "64"),
    }


def test_an_entry_on_the_sessions_gpu_framework_and_precision_comes_before_one_on_other_hardware(
    tmp_path: Path,
) -> None:
    leads = reference_leads(_checkout(tmp_path, ORDERED), profile())

    assert [(lead.config, tuple(lead.differing)) for lead in leads] == [
        ("other-sweep", ("workload", "tp", "conc")),
        ("far-gpu", ("gpu",)),
    ]


def test_a_lead_names_its_script_what_it_was_written_for_and_that_it_was_not_measured_here(checkout: Path) -> None:
    exact, other = reference_leads(checkout, profile(), REVISION)

    assert exact.describe() == (
        f"InferenceX dsr1-fp8-mi355x-vllm @ 3d5581562: `inferencex/{SCRIPTS}/dsr1_fp8_mi355x_vllm.sh`, written for "
        "vllm fp8 on mi355x, image `rocm/vllm:v0.11`, ISL 1024/OSL 1024, ISL 8192/OSL 1024 at TP 8, conc 4-64; "
        "not measured here"
    )
    assert ", gpu mi300x (here mi355x), framework sglang (here vllm)," in other.describe()
    assert other.render().startswith("- InferenceX dsr1-fp8-mi300x-sglang-mtp")


def test_no_lead_without_an_entry_for_the_model_and_an_unreadable_matrix_is_left_out(
    checkout: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    (checkout / "configs" / "nvidia-master.yaml").write_text("{ not yaml", encoding="utf-8")

    assert reference_leads(checkout, profile(model="meta-llama/Llama-3.1-8B")) == ()
    assert [lead.config for lead in reference_leads(checkout, profile(), limit=1)] == ["dsr1-fp8-mi355x-vllm"]
    assert reference_leads(tmp_path_factory.mktemp("not-a-checkout"), profile()) == ()


def test_an_agentx_session_reads_the_agentic_scenario_and_a_lead_says_when_its_script_is_missing(
    tmp_path: Path,
) -> None:
    agentx = profile(isl=None, osl=None, framework="sglang", tp=4, conc=16)

    [lead] = reference_leads(_checkout(tmp_path, AGENTIC), agentx)

    assert (lead.script, lead.gpu, lead.workload, lead.differing) == ("", "mi355x", "agentic at TP 2, 4, conc 1-48", {})
    assert "no launch script in the copy, written for sglang fp8 on mi355x" in lead.describe()
