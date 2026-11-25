# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist reads of the model files a launch names, and what a confined one reaches with it."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.measure.tests.bundle_factory import base_launch
from hyperloom.orchestrator.specialists.launch_env import LaunchPlace, specialist_confinement
from hyperloom.orchestrator.specialists import model_dirs
from hyperloom.orchestrator.specialists.model_dirs import launch_model_reads, session_reads
from hyperloom.orchestrator.state.base_launch import freeze_base
from hyperloom_common.bwrap import BwrapPolicy
from hyperloom_common.proctree import REAP_MARK_ENV


def _file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")
    return path


def _link(path: Path, target: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)
    return path


class Models:
    """A served model whose one file is a link into a store, a draft beside other builds of it, and a second draft that
    is itself a link."""

    def __init__(self, root: Path) -> None:
        self.store = root / "store" / "blobs"
        self.served = root / "models" / "target"
        _link(self.served / "target-Q4_K_M.gguf", _file(self.store / "abc"))
        self.drafts = root / "drafts"
        self.draft = _file(self.drafts / "draft-Q4_0.gguf")
        _file(self.drafts / "draft-Q8_0.gguf")
        self.heads = root / "heads"
        self.head = _link(root / "links" / "head.gguf", _file(self.heads / "head-f16.gguf"))


def test_a_checkout_built_server_s_launch_names_where_its_model_leads_and_where_its_auxiliary_models_lie(
    tmp_path: Path,
) -> None:
    models = Models(tmp_path)
    args = f"--jinja -md {models.draft} -ngl 99 --model-draft={models.head} --spec-draft-model relative/draft.gguf"

    dirs = launch_model_reads("llamacpp", str(models.served), args, None)

    assert dirs == (models.store, models.drafts, models.head.parent, models.heads)
    elsewhere = "-ngl 99 --model /elsewhere/x.gguf"
    assert launch_model_reads("llamacpp", str(models.served), elsewhere, None) == (models.store,)


class Loose:
    """A data root that holds the session beside a served model's file and a draft, and a home that holds a second
    draft beside its logins; each launch flag names one file loose there and one in a directory of its own."""

    def __init__(self, root: Path) -> None:
        self.data, self.home = root / "work", root / "home"
        self.session = self.data / "sessions" / "s1"
        self.session.mkdir(parents=True)
        self.served = self.data / "models" / "target"
        self.weights = _file(self.data / "target-Q4_K_M.gguf")
        _link(self.served / "target-Q4_K_M.gguf", self.weights)
        self.draft = _file(self.data / "draft-Q4_0.gguf")
        self.head = _file(self.home / "head-f16.gguf")
        self.kept = _file(self.home / "models" / "drafts" / "draft-Q8_0.gguf")
        self.logins = [_file(self.home / ".ssh" / "id_rsa"), _file(self.home / ".netrc"), _file(self.home / "notes")]
        self.args = f"-md {self.draft} --model-draft {self.head} --spec-draft-model {self.kept}"


def _session(root: Path, args: tuple[str, ...]) -> Path:
    session = root / "session"
    session.mkdir()
    freeze_base(session, replace(base_launch(), args=args))
    return session


def test_a_session_s_specialist_reads_its_model_and_what_its_frozen_base_launch_names(tmp_path: Path) -> None:
    models = Models(tmp_path)
    session = _session(tmp_path, ("-ngl", "99", "-md", str(models.draft)))
    bare = tmp_path / "bare"
    bare.mkdir()

    task = {"framework": "llamacpp", "model_path": str(models.served)}

    assert session_reads(session, task) == (models.served, models.store, models.drafts)
    for no_base in (bare, None):
        assert session_reads(no_base, task) == (models.served, models.store)
    assert session_reads(session, {**task, "framework": "sglang"}) == (models.served,)
    assert session_reads(None, None) == (Path(""),), "a task that names no model"


def test_only_a_checkout_built_server_s_specialist_has_the_frozen_base_launch_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = Models(tmp_path)
    session = _session(tmp_path, ("-ngl", "99", "-md", str(models.draft)))
    read: list[Path] = []
    monkeypatch.setattr(model_dirs, "frozen_base", lambda at: read.append(at))

    for framework in ("sglang", "vllm", "custom", ""):
        assert session_reads(session, {"framework": framework, "model_path": str(models.served)}) == (models.served,)
    assert read == []
    session_reads(session, {"framework": "llamacpp", "model_path": str(models.served)})
    assert read == [session]


def shown(policy: BwrapPolicy, path: Path) -> bool:
    """Whether the sandbox of ``policy`` shows ``path``: a bind at it or above it, and no mask over it."""
    bound = [Path(bind) for bind in (*policy.read_only, *policy.writable)]
    masked = [Path(mask) for mask in policy.masked]
    return any(bind == path or bind in path.parents for bind in bound) and not any(
        mask == path or mask in path.parents for mask in masked
    )


def test_a_confined_specialist_is_shown_neither_its_session_nor_a_home_whole_for_a_model_file_loose_in_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loose = Loose(tmp_path)
    monkeypatch.setenv("HOME", str(loose.home))
    freeze_base(loose.session, replace(base_launch(), args=tuple(loose.args.split())))
    workspace = loose.session / "specialists" / "t1"
    (workspace / "worktree").mkdir(parents=True)
    rounds = _file(loose.session / "runs" / "round_1" / "server.log")
    elsewhere = _file(loose.data / "sessions" / "s2" / "runs" / "round_1" / "server.log")
    readable = session_reads(loose.session, {"framework": "llamacpp", "model_path": str(loose.served)})
    place = LaunchPlace(workspace, workspace / "worktree", readable=readable)

    env = {"PATH": "/usr/bin:/bin", "HOME": str(loose.home), REAP_MARK_ENV: "mark-1"}
    policy = specialist_confinement(place, sys.executable, env).policy()

    assert policy.problems((0, 11, 0)) == []
    for reached in (workspace, loose.weights, loose.draft, loose.head, loose.kept):
        assert shown(policy, reached), reached
    for hidden in (rounds, elsewhere, loose.session / "base_launch.json", *loose.logins):
        assert not shown(policy, hidden), hidden
    other = tmp_path / "sessions" / "s2"
    assert launch_model_reads("llamacpp", str(loose.served), loose.args, other)[:2] == (loose.data, loose.head)
