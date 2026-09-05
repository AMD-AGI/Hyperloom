# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Boot-observation layer: digest stability, stream precedence, path normalisation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hyperloom.common.bringup import (
    BootObservation,
    LadderStage,
    TerminalFrame,
    failure_digest,
    normalise_file_rel,
    redact,
    render_excerpt,
)
from hyperloom.orchestrator.bringup import ladder, trees


# --- digest stability ------------------------------------------------------


def _crash_log(*, root: str, pid: int, stamp: str, line: int) -> str:
    """Build the same crash as seen on one host, with host-specific noise."""
    return (
        f"[{stamp}] [pid={pid}] server starting\n"
        "Traceback (most recent call last):\n"
        f'  File "{root}/engine/model_runner.py", line {line}, in load_model\n'
        "    self._load()\n"
        f"RuntimeError: could not allocate 4096 bytes at 0x7f1a2b3c for {root}/weights\n"
    )


def _observe(*, root: str, pid: int, stamp: str, line: int) -> BootObservation:
    """Classify one host's rendering of the shared crash."""
    return ladder.classify(
        server_log=_crash_log(root=root, pid=pid, stamp=stamp, line=line),
        server_elapsed_sec=float(pid % 7),
        trees=[root],
    )


def test_failure_digest_is_stable_across_runs() -> None:
    left = _observe(root="/opt/venv/lib/python3.10/site-packages/vllm", pid=4211, stamp="2026-01-02T03:04:05", line=884)
    right = _observe(root="/sgl-workspace/vllm", pid=90177, stamp="2026-08-19T22:10:00", line=884)

    assert failure_digest(left) == failure_digest(right)
    assert left.terminal_frame is not None
    assert left.terminal_frame.file_rel == "engine/model_runner.py"


def test_failure_digest_separates_different_failures() -> None:
    base = _observe(root="/sgl-workspace/vllm", pid=1, stamp="s", line=884)
    other = ladder.classify(
        server_log=(
            "Traceback (most recent call last):\n"
            '  File "/sgl-workspace/vllm/engine/model_runner.py", line 884, in load_model\n'
            "KeyError: 'model.layers.0.mlp.down_proj.weight'\n"
        ),
        server_elapsed_sec=1.0,
        trees=["/sgl-workspace/vllm"],
    )
    assert failure_digest(base) != failure_digest(other)


def test_unmatched_failure_still_digests() -> None:
    observation = ladder.classify(
        server_log=(
            "Traceback (most recent call last):\n"
            '  File "/sgl-workspace/vllm/srt/entry.py", line 12, in boot\n'
            "ZeroDivisionError: division by zero\n"
        ),
        server_elapsed_sec=2.0,
        trees=["/sgl-workspace/vllm"],
    )
    assert observation.matched_marker == ""
    assert observation.stage_failed is not None
    assert len(failure_digest(observation)) == 64


def test_digest_ignores_stage_reached_noise() -> None:
    frame = TerminalFrame(exc_type="ValueError", module="a.b", file_rel="a/b.py", line=3)
    left = BootObservation(
        producer="p",
        stage_reached=LadderStage.IMPORT,
        stage_failed=LadderStage.ENGINE_INIT,
        terminal_frame=frame,
        server_elapsed_sec=1.0,
    )
    right = BootObservation(
        producer="other",
        stage_reached=LadderStage.WEIGHTS_LOADED,
        stage_failed=LadderStage.ENGINE_INIT,
        terminal_frame=frame,
        server_elapsed_sec=999.0,
        evidence_ref="/some/session/server.log",
    )
    assert failure_digest(left) == failure_digest(right)


# --- argparse family, from the server log, with no new rule ----------------

_ARGV_ERROR = "python3 -m sglang.launch_server: error: unrecognized arguments: --enable-torch-compile\n"
_INVALID_CHOICE = (
    "launch_server: error: argument --attention-backend: invalid choice: 'aiter' (choose from 'flashinfer', 'triton')\n"
)


def test_argv_failure_classifies_from_server_log() -> None:
    observation = ladder.classify(
        server_log=_ARGV_ERROR,
        server_elapsed_sec=0.4,
        wrapper_stderr="server exited with code 2\n",
        trees=[],
    )
    assert observation.stage_failed is LadderStage.ARGV_PARSE
    assert observation.stage_reached is LadderStage.ARGV_PARSE
    assert observation.matched_marker == "serve_flag"
    assert observation.evidence_ref == ladder.SERVER_LOG
    assert "--enable-torch-compile" in (observation.excerpt.text if observation.excerpt else "")


def test_invalid_choice_classifies_from_server_log() -> None:
    observation = ladder.classify(
        server_log=_INVALID_CHOICE,
        server_elapsed_sec=0.4,
        wrapper_stderr="server exited with code 2\n",
        trees=[],
    )
    assert observation.stage_failed is LadderStage.ARGV_PARSE
    assert observation.matched_marker == "serve_flag"


def test_wrapper_stream_alone_never_reaches_the_argv_rule() -> None:
    """The wrapper reports the child's death, not its cause."""
    observation = ladder.classify(
        server_log="",
        server_elapsed_sec=0.4,
        wrapper_stderr="ERROR launcher: server process exited with code 2 after 0.4s\n",
        trees=[],
    )
    assert observation.matched_marker != "serve_flag"


def test_progress_witness_advances_the_failure_stage() -> None:
    observation = ladder.classify(
        server_log=(
            "server_args=ServerArgs(model_path='m')\n"
            "Loading weights took 40.2s\n"
            "KV cache is allocated\n"
            "Capture cuda graph begin\n"
            "RuntimeError: no kernel image is available for execution on the device\n"
        ),
        server_elapsed_sec=61.0,
        trees=[],
    )
    assert observation.stage_reached is LadderStage.GRAPH_CAPTURE
    assert observation.stage_failed is LadderStage.HTTP_READY
    assert observation.progress_witness is not None
    assert observation.progress_witness["WEIGHTS_LOADED"] == "loading weights took"


def test_resource_constraint_is_marked_an_env_fault() -> None:
    observation = ladder.classify(
        server_log="torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB\n",
        server_elapsed_sec=30.0,
        trees=[],
    )
    assert observation.env_fault == "resource_constraint"


def test_clean_boot_has_no_failed_stage() -> None:
    observation = ladder.classify(
        server_log="server_args=ServerArgs(model_path='m')\nApplication startup complete.\nUvicorn running on 0.0.0.0\n",
        server_elapsed_sec=55.0,
        trees=[],
    )
    assert observation.stage_failed is None
    assert observation.stage_reached is LadderStage.HTTP_READY
    assert observation.booted


def test_observation_round_trips_through_json_types() -> None:
    observation = ladder.classify(
        server_log=_ARGV_ERROR,
        server_elapsed_sec=0.4,
        trees=[],
    )
    assert BootObservation.from_dict(observation.to_dict()) == observation
    assert ladder.observation_summary(observation)["failure_digest"] == failure_digest(observation)


# --- excerpt + redaction ---------------------------------------------------


def test_render_excerpt_is_anchored_not_a_tail_slice() -> None:
    text = "prefix\n" + "MATCH here\n" + ("filler line\n" * 400)
    anchor = text.index("MATCH here")
    excerpt = render_excerpt(text, anchor=anchor, width=120, stream="server_log")
    assert "MATCH here" in excerpt.text
    assert excerpt.byte_end - excerpt.byte_start <= 120
    assert excerpt.stream == "server_log"


def test_excerpt_redacts_the_session_root() -> None:
    root = "/data/sessions/model/20260101T000000-abcd1234"
    observation = ladder.classify(
        server_log=f"RuntimeError: cannot open {root}/runs/baseline/server.log\n",
        server_elapsed_sec=1.0,
        trees=[],
        session_root=root,
    )
    assert observation.excerpt is not None
    assert root not in observation.excerpt.text
    assert "<session>" in observation.excerpt.text


def test_redact_prefers_the_longer_root() -> None:
    out = redact("/a/b/c/file", roots=("/a", "/a/b/c"))
    assert out == "<session>/file"


# --- file normalisation ----------------------------------------------------


def test_normalise_file_rel_strips_the_longest_root() -> None:
    roots = ("/opt/venv/lib/python3.10/site-packages", "/opt/venv/lib/python3.10/site-packages/vllm")
    assert normalise_file_rel("/opt/venv/lib/python3.10/site-packages/vllm/engine/core.py", roots) == "engine/core.py"


def test_normalise_file_rel_marks_frames_outside_every_root() -> None:
    assert (
        normalise_file_rel("/usr/lib/python3.10/asyncio/events.py", ("/sgl-workspace/vllm",)) == "<external>/events.py"
    )
    assert normalise_file_rel("", ("/sgl-workspace/vllm",)) == ""


# --- tree identity ---------------------------------------------------------


def test_resolve_trees_marks_a_git_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "vllm"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True, capture_output=True)

    resolved = trees.resolve_trees([str(checkout)])

    assert len(resolved) == 1
    assert resolved[0].vcs == trees.VCS_GIT
    assert Path(resolved[0].root) == checkout.resolve()
    assert resolved[0].tree_id.startswith("vllm-")


def test_resolve_trees_does_not_borrow_an_enclosing_repo(tmp_path: Path) -> None:
    """A package nested in an unrelated checkout is its own root, not that repo."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    package = tmp_path / "lib" / "python3.10" / "site-packages" / "sglang"
    package.mkdir(parents=True)

    resolved = trees.resolve_trees([str(package)])

    assert len(resolved) == 1
    assert resolved[0].vcs == trees.VCS_NONE
    assert Path(resolved[0].root) == package
    assert resolved[0].package_dirs == (str(package),)


def test_resolve_trees_skips_absent_roots_and_groups_by_root(tmp_path: Path) -> None:
    package = tmp_path / "sglang"
    package.mkdir()

    resolved = trees.resolve_trees([str(package), str(package) + "/", str(tmp_path / "absent")])

    assert len(resolved) == 1
    assert resolved[0].package_dirs == (str(package),)


def test_trees_round_trip_through_the_session_artifact(tmp_path: Path) -> None:
    package = tmp_path / "sglang"
    package.mkdir()
    pinned = trees.resolve_trees([str(package)])

    written = trees.write_trees(pinned, session_dir=tmp_path)

    assert written == tmp_path / "reports" / "bringup" / "trees.json"
    assert trees.read_trees(tmp_path) == pinned


def test_tree_roots_orders_longest_first(tmp_path: Path) -> None:
    identity = trees.TreeIdentity(
        tree_id="t",
        root="/a",
        package_dirs=("/a/pkg/deeper", "/a/pkg"),
        vcs=trees.VCS_NONE,
    )
    assert trees.tree_roots([identity]) == ("/a/pkg/deeper", "/a/pkg", "/a")
