"""N23 — ``--resume`` is N17-layout-aware.

The May 2026 Qwen1.5-7B 10h silent stall traced back to operators
pointing ``$USER_DATA_PATH`` at a per-session subdir (so
``USER_DATA_PATH/runtime/kernel-agent.env.sh`` didn't exist —> N24
hard-fail catches that). N23 is the other half of the fix: ``--resume``
itself must understand the N17 layout — workspace is the parent of
``<model>/<UTC ts>/`` per-session subdirs — and must:

* leave $USER_DATA_PATH alone (workspace level) so runtime/ resolves;
* pick the LATEST per-session subdir under workspace_root/<model>/<ts>/
  when no --resume-from is given;
* accept --resume-from as an explicit override (must be under
  workspace_root, must exist);
* pin INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR to the resolved
  subdir BEFORE any state load, so subprocesses inherit it.

These tests exercise the path-resolution layer
(``inference_optimizer.paths.find_latest_per_session_dir``) that the
``--resume`` block now delegates to. The pin / sys.exit behaviour of
``cli.py`` itself is exercised end-to-end by the existing
``test_p1_4_resume.py`` flow once the per-session pin is in env.
"""

from __future__ import annotations

import pytest

from inference_optimizer import paths


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Pin USER_DATA_PATH to a fresh tmp workspace per test."""
    monkeypatch.setenv(paths.ENV_USER_DATA_PATH, str(tmp_path))
    monkeypatch.delenv(paths.ENV_CURRENT_SESSION_DIR, raising=False)
    monkeypatch.delenv(paths.ENV_SESSION_LAYOUT, raising=False)


def test_resume_picks_latest_subdir_after_two_launches(tmp_path):
    """Two consecutive make_session_dir() calls -> the second is the
    resume target; find_latest_per_session_dir() must agree."""
    sd1 = paths.make_session_dir(model_name="DeepSeek-R1-0528")
    # find_latest mid-run sees only sd1
    assert paths.find_latest_per_session_dir() == sd1
    assert paths.find_latest_per_session_dir(model_name="DeepSeek-R1-0528") == sd1

    # Second launch (different ts; we name it manually to bypass the
    # tz-locked %S resolution clash that would otherwise force a sleep).
    later_ts = "29990101T000000Z"
    sd2 = tmp_path / "DeepSeek-R1-0528" / later_ts
    sd2.mkdir(parents=True)

    assert paths.find_latest_per_session_dir() == sd2
    assert paths.find_latest_per_session_dir(model_name="DeepSeek-R1-0528") == sd2


def test_resume_does_not_mutate_user_data_path(tmp_path):
    """The contract is: USER_DATA_PATH stays workspace-rooted across
    a resume; only INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR moves."""
    sd = paths.make_session_dir(model_name="Qwen3-32B")
    import os as _os
    assert _os.environ[paths.ENV_USER_DATA_PATH] == str(tmp_path)
    assert _os.environ[paths.ENV_CURRENT_SESSION_DIR] == str(sd)
    # The per-session subdir is NOT a parent of USER_DATA_PATH (would
    # be inverted nesting, which is the bug we're guarding against).
    assert paths.workspace_root() == tmp_path
    assert tmp_path in sd.parents


def test_resume_falls_back_to_flat_when_no_per_session_subdir(tmp_path):
    """An empty workspace (legacy / brand-new install) returns None;
    the cli --resume block falls back to workspace_root itself
    (validated separately in test_p1_4_resume.py)."""
    assert paths.find_latest_per_session_dir() is None


def test_resume_from_explicit_path_must_be_under_workspace_root(
    tmp_path, monkeypatch,
):
    """A --resume-from outside USER_DATA_PATH is an operator error; the
    contract enforced by cli.py is to refuse it. We assert the
    invariant: any valid --resume-from must be a descendant of
    workspace_root."""
    sd = paths.make_session_dir(model_name="Qwen3-32B")
    assert tmp_path.resolve() in sd.resolve().parents

    foreign = tmp_path.parent / "stranger_workspace" / "sess"
    foreign.mkdir(parents=True)
    # Sanity: the foreign path is NOT under workspace_root.
    try:
        foreign.resolve().relative_to(tmp_path.resolve())
        assert False, "foreign path should not be under workspace_root"
    except ValueError:
        pass


def test_latest_picks_across_models_when_model_name_omitted(tmp_path):
    """A workspace with mixed-model launches -> latest ts across all
    models wins when no model_name filter is given."""
    (tmp_path / "ModelA").mkdir()
    (tmp_path / "ModelA" / "20260101T000000Z").mkdir()
    (tmp_path / "ModelB").mkdir()
    (tmp_path / "ModelB" / "20260520T000000Z").mkdir()
    (tmp_path / "ModelC").mkdir()
    (tmp_path / "ModelC" / "20260315T000000Z").mkdir()

    picked = paths.find_latest_per_session_dir()
    assert picked is not None
    assert picked.parent.name == "ModelB"
    assert picked.name == "20260520T000000Z"


def test_workspace_shared_dirs_never_picked_as_session(tmp_path):
    """runtime/ + logs/ are workspace-shared (per N17 split) — even
    if they accidentally contain a ts-shaped child, they must never be
    returned as a resume target."""
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "20990101T000000Z").mkdir()  # decoy
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "20990101T000000Z").mkdir()  # decoy
    (tmp_path / "RealModel").mkdir()
    (tmp_path / "RealModel" / "20260518T100000Z").mkdir()

    picked = paths.find_latest_per_session_dir()
    assert picked is not None
    assert picked.parent.name == "RealModel"
    assert "runtime" not in str(picked)
    assert "logs" not in str(picked)
