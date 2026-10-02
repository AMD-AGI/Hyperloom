# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Synthetic profiling and eval patches work with the pinned infx project layout."""

from __future__ import annotations

import subprocess

import pytest

from hyperloom.inference_optimizer.tests import test_inferencex_anchor_contract as contract
from hyperloom.orchestrator.actions.executors import _inferencex_patcher as patcher


_PROFILE_CLIENT = """def profile_request():
    return dict(
            extra_body={
                "num_steps": 1,
                "merge_profiles": True,
                "profile_by_stage": True,
            },
    )
"""
_EVAL_LIBRARY = """stage_result() {
    local stem=results extension=.json eval_conc=8 target suffix
        target="./${stem}_conc${eval_conc}${extension}"
        suffix=2
        while [ -e "$target" ]; do
            target="./${stem}_conc${eval_conc}_${suffix}${extension}"
            suffix=$((suffix + 1))
        done
    printf 'result' > "$target"
}
clean_eval() {
    local results_dir=cleanup
    export EVAL_RESULT_DIR="$results_dir"
}
run_eval() {
    local results_dir=accuracy
    # Read by append_lm_eval_summary.
    export EVAL_RESULT_DIR="$results_dir"
}
run_benchmark() {
    local max_concurrency=8 num_prompts
        num_prompts="$max_concurrency"
    printf '%s' "$num_prompts"
}
"""


def _modern_tree(tmp_path, nested):
    project = tmp_path / "inferencex-e2e" if nested else tmp_path
    paths = {
        "benchmarks/benchmark_lib.sh": _EVAL_LIBRARY,
        "infx/bench_serving/benchmark_serving.py": _PROFILE_CLIENT,
        "infx/evals/patches/lm_eval_sitecustomize.py": "# upstream eval hooks\n",
    }
    for name, content in paths.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return project


@pytest.mark.parametrize("nested", [False, True])
def test_modern_profile_settings_reach_request_body_and_keep_legacy_defaults(tmp_path, monkeypatch, nested):
    project = _modern_tree(tmp_path, nested)
    client = project / "infx/bench_serving/benchmark_serving.py"
    assert patcher.benchmark_serving_path_in(tmp_path) == client
    assert patcher.ensure_benchmark_serving_patched(tmp_path)
    once = client.read_text()
    assert patcher.ensure_benchmark_serving_patched(project)
    assert client.read_text() == once
    namespace = {}
    exec(compile(once, str(client), "exec"), namespace)
    monkeypatch.delenv("PROFILE_EXTRA_BODY", raising=False)
    assert namespace["profile_request"]()["extra_body"] == {
        "num_steps": 1,
        "merge_profiles": True,
        "profile_by_stage": True,
    }
    monkeypatch.setenv("PROFILE_EXTRA_BODY", '{"num_steps": 20, "start_step": 5}')
    assert namespace["profile_request"]()["extra_body"] == {"num_steps": 20, "start_step": 5}


@pytest.mark.parametrize("nested", [False, True])
def test_modern_eval_artifacts_and_collision_suffixes_stay_in_result_dir(tmp_path, nested):
    project = _modern_tree(tmp_path, nested)
    assert patcher.ensure_benchmark_lib_eval_dest_patched(tmp_path)
    assert patcher.ensure_benchmark_lib_eval_start_patched(tmp_path)
    assert patcher.ensure_benchmark_lib_patched(tmp_path)
    library = project / "benchmarks/benchmark_lib.sh"
    output = tmp_path / "run outputs"
    output.mkdir()
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; export RESULT_DIR="$2"; stage_result; stage_result; clean_eval; run_eval; NUM_PROMPTS=23 run_benchmark',
            "bash",
            str(library),
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert sorted(path.name for path in output.iterdir()) == ["results_conc8.json", "results_conc8_2.json"]
    assert not (tmp_path / "results_conc8.json").exists()
    assert result.stderr.splitlines() == ["HYPERLOOM_EVAL_START"]
    assert result.stdout == "23"
    assert patcher.ensure_benchmark_lib_eval_dest_patched(project)
    assert patcher.ensure_benchmark_lib_eval_start_patched(project)


@pytest.mark.parametrize("nested", [False, True])
def test_modern_probe_targets_active_module_and_all_anchors_verify(tmp_path, nested):
    project = _modern_tree(tmp_path, nested)
    target = project / "infx/evals/patches/lm_eval_sitecustomize.py"
    statuses = patcher.verify_patch_anchors(tmp_path)
    assert len(statuses) == 4
    assert all(status.hits == 1 and status.ok for status in statuses)
    assert patcher.failed_patch_anchors_in(tmp_path) == []
    assert patcher.eval_probe_targets_exist(tmp_path)
    assert patcher.ensure_eval_probe_patched(tmp_path)
    assert patcher.ensure_eval_unbound_outputs_patched(tmp_path)
    code = target.read_text()
    assert "HYPERLOOM_EVAL_PROBE" in code
    assert "HYPERLOOM_EVAL_UNBOUND_OUTPUTS" in code
    compile(code, str(target), "exec")
    assert patcher.ensure_eval_probe_patched(project)
    assert target.read_text() == code


def test_contract_local_refresh_reads_committed_bytes_not_modified_checkout(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    target = tmp_path / "file.txt"
    target.write_text("committed\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "file.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    target.write_text("changed\n")
    assert contract.fetch_pinned_file("file.txt", revision, tmp_path) == "committed\n"
    assert contract.fetch_pinned_file("missing", revision, tmp_path) is None


def test_contract_refuses_missing_current_anchor(monkeypatch):
    monkeypatch.setattr(contract, "fetch_pinned_file", lambda *_args: "incompatible upstream content\n")
    with pytest.raises(RuntimeError, match="anchor num_prompts matched 0"):
        contract.build_record("a" * 40)
