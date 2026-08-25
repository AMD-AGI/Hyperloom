# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cross-repo patch grounding, artifact stacking, and accumulated-config replay.

Regression fixtures come from sessions 101858 (two advanced rounds whose envs a
later KEEP erased) and 101901 (an artifact-only repair whose replay script came
out empty).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hyperloom.inference_optimizer.reference_script import render_reference_script
from hyperloom.orchestrator.phases._enablement_artifacts import write_setting_script
from hyperloom.orchestrator.specialists import patch_safety as _ps
from hyperloom.orchestrator.specialists.runner import _sibling_checkouts
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound


def _git(*args: str) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True)


def _checkout(root: Path, filename: str) -> Path:
    """Create a git checkout holding one committed file."""
    root.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", str(root))
    (root / filename).write_text("one\n", encoding="utf-8")
    _git("-C", str(root), "add", ".")
    _git("-C", str(root), "-c", "user.email=a@b", "-c", "user.name=x", "commit", "-qm", "init")
    return root


def _diff(path: str) -> str:
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-one\n+two\n"


# Cross-repo grounding


def test_patch_grounds_against_a_candidate_root(tmp_path):
    """The aiter worktree does not hold the target; the sglang checkout does."""
    aiter = _checkout(tmp_path / "aiter", "aiter_file.py")
    sglang = _checkout(tmp_path / "sglang", "sglang_file.py")

    res = _ps.ground_patch_text(_diff("sglang_file.py"), base_checkout=aiter, candidate_roots=(sglang,))
    assert res.verdict == _ps.GROUND_APPLIES


def test_patch_absent_from_every_root_is_dropped(tmp_path):
    aiter = _checkout(tmp_path / "aiter", "aiter_file.py")
    sglang = _checkout(tmp_path / "sglang", "sglang_file.py")

    res = _ps.ground_patch_text(_diff("ghost.py"), base_checkout=aiter, candidate_roots=(sglang,))
    assert res.verdict == _ps.GROUND_MISSING_TARGET
    assert res.is_garbage


def test_duplicate_matching_roots_are_rejected_as_ambiguous(tmp_path):
    base = _checkout(tmp_path / "base", "f.py")
    other = _checkout(tmp_path / "other", "f.py")
    (base / "f.py").write_text("drifted\n", encoding="utf-8")
    _git("-C", str(base), "-c", "user.email=a@b", "-c", "user.name=x", "commit", "-qam", "drift")

    res = _ps.ground_patch_text(_diff("f.py"), base_checkout=base, candidate_roots=(other,))
    assert res.verdict == _ps.GROUND_MISSING_TARGET
    assert res.detail.startswith("ambiguous_root:")
    assert res.is_garbage


def test_basename_does_not_create_false_ambiguity(tmp_path):
    """A deep sglang path must not match an aiter tree that holds only the same basename.

    python/sglang/srt/layers/utils.py stripped at high -p levels reduces to utils.py.
    That bare filename must not be accepted as a match, because any tree that happens
    to hold a top-level utils.py would then be considered a candidate, triggering
    false ambiguous_root and dropping the entire patch set.
    """
    sglang = tmp_path / "sglang"
    sglang.mkdir()
    _git("init", "-q", str(sglang))
    sglang.joinpath("python/sglang/srt/layers").mkdir(parents=True)
    sglang.joinpath("python/sglang/srt/layers/utils.py").write_text("one\n", encoding="utf-8")
    _git("-C", str(sglang), "add", ".")
    _git("-C", str(sglang), "-c", "user.email=a@b", "-c", "user.name=x", "commit", "-qm", "init")
    aiter = _checkout(tmp_path / "aiter", "utils.py")

    deep_diff = _diff("python/sglang/srt/layers/utils.py")
    assert _ps.patch_targets_missing(deep_diff, aiter) != [], "bare basename must not match"

    r = _ps.resolve_patch_apply_root((deep_diff,), explicit_root=None, candidate_roots=(sglang, aiter))
    assert r.root is not None, f"expected sglang, got reason={r.reason!r}"
    assert r.root.resolve() == sglang.resolve()


_CREATE_ONLY_DIFF = "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+new\n"


def test_pure_create_needs_a_root_the_caller_already_knows(tmp_path):
    """No pre-image can pick a candidate, so scanning them is refused."""
    root = _checkout(tmp_path / "root", "existing.py")

    rejected = _ps.resolve_patch_apply_root(
        (_CREATE_ONLY_DIFF,),
        explicit_root=None,
        candidate_roots=(root,),
    )
    accepted = _ps.resolve_patch_apply_root(
        (_CREATE_ONLY_DIFF,),
        explicit_root=root,
        candidate_roots=(),
    )

    assert rejected.root is None
    assert rejected.reason == "pure_create_requires_explicit_root"
    assert accepted.root == root.resolve()


def test_pure_create_lands_in_the_worktree_base(tmp_path):
    """Adding a source file is a normal repair; the base is not a guess."""
    base = _checkout(tmp_path / "base", "existing.py")
    other = _checkout(tmp_path / "other", "existing.py")
    patch = tmp_path / "create.patch"
    patch.write_text(_CREATE_ONLY_DIFF, encoding="utf-8")

    kept, dropped, grounding, spans_roots = _ps.vet_patches(
        [str(patch)],
        base_checkout=base,
        candidate_roots=(other,),
    )

    assert kept == [str(patch)]
    assert dropped == []
    assert not spans_roots


def test_all_preimages_must_share_one_unique_root(tmp_path):
    complete = _checkout(tmp_path / "complete", "first.py")
    (complete / "second.py").write_text("one\n", encoding="utf-8")
    _git("-C", str(complete), "add", ".")
    _git("-C", str(complete), "-c", "user.email=a@b", "-c", "user.name=x", "commit", "-qm", "second")
    partial = _checkout(tmp_path / "partial", "first.py")
    patches = (_diff("first.py"), _diff("second.py"))

    resolution = _ps.resolve_patch_apply_root(
        patches,
        explicit_root=None,
        candidate_roots=(partial, complete),
    )

    assert resolution.root == complete.resolve()


def test_vet_patches_rescues_a_cross_repo_patch(tmp_path):
    aiter = _checkout(tmp_path / "aiter", "aiter_file.py")
    sglang = _checkout(tmp_path / "sglang", "sglang_file.py")
    patch = tmp_path / "fix.patch"
    patch.write_text(_diff("sglang_file.py"), encoding="utf-8")

    kept, dropped, grounding, spans_roots = _ps.vet_patches([str(patch)], base_checkout=aiter, candidate_roots=(sglang,))
    assert kept == [str(patch)]
    assert dropped == []
    assert grounding[str(patch)] == _ps.GROUND_APPLIES
    assert not spans_roots


def test_vet_patches_resolves_the_complete_set_once(tmp_path):
    """A set where all targets exist in a single root is kept with no multi-root signal."""
    complete = _checkout(tmp_path / "complete", "first_unique.py")
    (complete / "second_unique.py").write_text("one\n", encoding="utf-8")
    _git("-C", str(complete), "add", ".")
    _git("-C", str(complete), "-c", "user.email=a@b", "-c", "user.name=x", "commit", "-qm", "second")
    patches = []
    for name in ("first_unique.py", "second_unique.py"):
        patch = tmp_path / f"{name}.patch"
        patch.write_text(_diff(name), encoding="utf-8")
        patches.append(str(patch))

    kept, dropped, grounding, spans_roots = _ps.vet_patches(
        patches,
        base_checkout=complete,
        candidate_roots=(),
    )

    assert kept == patches
    assert dropped == []
    assert set(grounding.values()) == {_ps.GROUND_APPLIES}
    assert not spans_roots


def test_vet_patches_keeps_a_set_split_across_roots_and_reports_it(tmp_path):
    """Each patch grounds against its own root; both are kept and spans_roots is True."""
    first = _checkout(tmp_path / "first", "first.py")
    second = _checkout(tmp_path / "second", "second.py")
    patches = []
    for name in ("first.py", "second.py"):
        patch = tmp_path / f"{name}.patch"
        patch.write_text(_diff(name), encoding="utf-8")
        patches.append(str(patch))

    kept, dropped, grounding, spans_roots = _ps.vet_patches(
        patches,
        base_checkout=first,
        candidate_roots=(second,),
    )

    assert set(kept) == set(patches)
    assert dropped == []
    assert spans_roots


def test_vet_patches_absent_from_every_root_is_dropped(tmp_path):
    """A patch whose target exists in no root is still dropped."""
    first = _checkout(tmp_path / "first", "first.py")
    second = _checkout(tmp_path / "second", "second.py")
    ghost = tmp_path / "ghost.patch"
    ghost.write_text(_diff("ghost.py"), encoding="utf-8")

    kept, dropped, grounding, spans_roots = _ps.vet_patches(
        [str(ghost)],
        base_checkout=first,
        candidate_roots=(second,),
    )

    assert kept == []
    assert len(dropped) == 1
    assert grounding[str(ghost)] == _ps.GROUND_MISSING_TARGET
    assert not spans_roots


def test_sibling_checkouts_excludes_the_base_but_keeps_plain_trees(tmp_path):
    """A pip-installed framework is not a repo and still holds the targets."""
    base = _checkout(tmp_path / "base", "f.py")
    other = _checkout(tmp_path / "other", "g.py")
    plain = tmp_path / "plain"
    plain.mkdir()
    missing = tmp_path / "missing"

    siblings = _sibling_checkouts((str(base), str(other), str(plain), str(missing)), base)
    assert siblings == (other, plain)


def test_patch_grounds_against_a_non_git_tree(tmp_path):
    """Target matching stats files and git apply --check runs outside a repo."""
    base = _checkout(tmp_path / "base", "base_only.py")
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "installed.py").write_text("one\n", encoding="utf-8")
    patch = tmp_path / "fix.patch"
    patch.write_text(_diff("installed.py"), encoding="utf-8")

    kept, dropped, grounding, spans_roots = _ps.vet_patches([str(patch)], base_checkout=base, candidate_roots=(plain,))

    assert kept == [str(patch)]
    assert grounding[str(patch)] == _ps.GROUND_APPLIES
    assert not spans_roots


def test_no_tree_to_ground_against_keeps_the_patch(tmp_path):
    """Absence of a checkout is not evidence the patch is wrong."""
    patch = tmp_path / "fix.patch"
    patch.write_text(_diff("anything.py"), encoding="utf-8")

    kept, dropped, grounding, spans_roots = _ps.vet_patches([str(patch)], base_checkout=None, candidate_roots=())

    assert kept == [str(patch)]
    assert dropped == []
    assert grounding[str(patch)] == _ps.GROUND_UNCHECKED
    assert not spans_roots


def test_no_tree_to_ground_against_keeps_a_create_only_patch(tmp_path):
    patch = tmp_path / "create.patch"
    patch.write_text(_CREATE_ONLY_DIFF, encoding="utf-8")

    kept, dropped, grounding, spans_roots = _ps.vet_patches([str(patch)], base_checkout=None, candidate_roots=())

    assert kept == [str(patch)]
    assert grounding[str(patch)] == _ps.GROUND_UNCHECKED
    assert not spans_roots


def test_no_candidate_roots_is_distinct_from_a_miss(tmp_path):
    """The two must not collapse: only one of them indicts the patch."""
    root = _checkout(tmp_path / "root", "present.py")

    absent = _ps.resolve_patch_apply_root((_diff("ghost.py"),), explicit_root=None, candidate_roots=(root,))
    nothing = _ps.resolve_patch_apply_root((_diff("ghost.py"),), explicit_root=None, candidate_roots=())

    assert absent.reason == "no_matching_root"
    assert nothing.reason == "no_candidate_roots"


def test_create_only_reason_does_not_depend_on_the_candidates(tmp_path):
    """A create-only set has no pre-image, so candidates cannot speak to it.

    Letting the candidate list rename this reason made the outcome depend on
    which framework trees happen to exist on the host: a clean CI box, where the
    allowlist names roots that are all absent, disagreed with a dev box where one
    of them was real.
    """
    real = _checkout(tmp_path / "root", "present.py")
    absent = tmp_path / "absent"

    for candidates in ((real,), (absent,), ()):
        resolution = _ps.resolve_patch_apply_root((_CREATE_ONLY_DIFF,), explicit_root=None, candidate_roots=candidates)
        assert resolution.reason == "pure_create_requires_explicit_root", candidates


# Artifact stacking


def test_kept_artifacts_reach_the_replay_script(tmp_path):
    source = tmp_path / "server_args.py"
    source.write_text("# fixed\n", encoding="utf-8")
    backup = tmp_path / "server_args.py.bak"
    backup.write_text("# original\n", encoding="utf-8")

    enablement = EnablementRound()
    enablement.kept_artifacts = [
        {
            "target": "/sgl-workspace/sglang/python/sglang/srt/server_args.py",
            "source": str(source),
            "backup": str(backup),
        }
    ]

    write_setting_script(tmp_path, enablement, "sglang", model="/models/M")
    artifacts = tmp_path / "reports" / "enablement" / "artifacts"
    assert (artifacts / "001_server_args.py").read_text() == "# fixed\n"
    assert (artifacts / "001_server_args.py.orig").read_text() == "# original\n"


def test_artifact_only_repair_renders_a_replay_script(tmp_path):
    """101901 shipped two artifacts and no diff; its script had no install lines."""
    sources = []
    for name in ("server_args.py", "quark_w4a4_mxfp4_moe.py"):
        src = tmp_path / name
        src.write_text(f"# {name}\n", encoding="utf-8")
        sources.append(src)

    enablement = EnablementRound()
    enablement.kept_artifacts = [
        {"target": f"/sgl-workspace/sglang/python/sglang/srt/{s.name}", "source": str(s)} for s in sources
    ]

    write_setting_script(tmp_path, enablement, "sglang", model="/models/GLM-5.2-MXFP4", tp=8)
    text = (tmp_path / "reports" / "enablement" / "enablement_setting.sh").read_text()
    assert "enablement fix replay script" in text
    assert text.count("install -D") == 2


def test_script_without_artifacts_stays_a_launch_recipe():
    text = render_reference_script(framework="sglang", server_args="")
    assert "current best launch recipe" in text
    assert "install -D" not in text


def test_generated_artifact_script_installs_and_launches(tmp_path):
    source = tmp_path / "patched.py"
    source.write_text("# patched\n", encoding="utf-8")
    target = tmp_path / "tree" / "pkg" / "mod.py"

    enablement = EnablementRound()
    enablement.kept_artifacts = [{"target": str(target), "source": str(source)}]
    rel = write_setting_script(tmp_path, enablement, "sglang", model="/models/M")

    proc = subprocess.run(
        ["bash", "-c", f'python3(){{ echo LAUNCHED; }}; source "{tmp_path / rel}"'],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "LAUNCHED" in proc.stdout
    assert target.read_text() == "# patched\n"


# Accumulated config


def test_kept_round_keeps_the_envs_earlier_advances_accepted(tmp_path):
    """101858: two advances accepted three envs; the KEEP bench must carry them.

    The KEEP leg is benched with ``base_extra_envs``, so its effective_config
    is the whole stack and writing it back over ``accepted_config`` is lossless.
    """
    enablement = EnablementRound()
    enablement.accepted_config = {
        "extra_envs": {
            "SGLANG_DSV4_MHC_PREWARM": "0",
            "SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D": "1",
            "SGLANG_USE_AITER": "1",
        },
        "extra_server_args": "",
    }
    base_extra_envs = dict(enablement.accepted_config["extra_envs"])

    # The KEEP round proposes no env of its own; the variant still launches with
    # the accumulated three, so effective_config comes back holding all of them.
    enablement.accepted_config = {"extra_envs": base_extra_envs, "extra_server_args": ""}

    write_setting_script(tmp_path, enablement, "sglang", model="/models/DeepSeek-V4-Pro", tp=8)
    text = (tmp_path / "reports" / "enablement" / "enablement_setting.sh").read_text()
    assert "export SGLANG_DSV4_MHC_PREWARM=0" in text
    assert "export SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1" in text
    assert "export SGLANG_USE_AITER=1" in text


def test_replay_script_is_valid_bash(tmp_path):
    source = tmp_path / "art.py"
    source.write_text("# artifact\n", encoding="utf-8")

    enablement = EnablementRound()
    enablement.accepted_config = {"extra_envs": {"SGLANG_USE_AITER": "1"}, "extra_server_args": "--tp 8"}
    enablement.kept_artifacts = [{"target": "/sgl-workspace/sglang/art.py", "source": str(source)}]

    rel = write_setting_script(tmp_path, enablement, "sglang", model="/models/M", tp=8)
    proc = subprocess.run(["bash", "-n", str(tmp_path / rel)], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()


# State migration


def test_state_without_the_new_fields_still_loads():
    enablement = EnablementRound.from_dict({"kept_patches": ["/patch1"], "attempts": 3})
    assert enablement.kept_patches == ["/patch1"]
    assert enablement.kept_artifacts == []
    assert enablement.last_grounding_drop_reason == []
