# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Repo scope: discovery and authoring over the framework tree, not one file.

The behaviour under test is one property end to end: a fusion whose call sites
the operator never named must survive every stage that used to derive its file
set from ``source_file`` alone -- discovery, the shadow index, the campaign
invocation, the wiring gate, and the exported patch.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from kernelforge.fusion.anchor import AnchorReport, Slot, build_anchored_discovery_prompt
from kernelforge.fusion.campaign import build_campaign_program_md, build_forge_loop_command
from kernelforge.fusion.discover import parse_discovered_recipes, resolve_repo_file
from kernelforge.fusion.emit import export_artifacts
from kernelforge.fusion.models import Recipe
from kernelforge.fusion.shadow_repo import ensure_git_workspace
from kernelforge.fusion.validate import fused_symbol_invocation_evidence


def _repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A framework whose fusible chain is NOT in the model file.

    This is the dsv4 shape: the arch-class file delegates through one opaque call
    to a runtime in a different package, and the scan worth fusing is there.
    """
    root = tmp_path / "sglang"
    models = root / "python" / "sglang" / "srt" / "models"
    runtime = root / "python" / "sglang" / "kernels" / "ops"
    models.mkdir(parents=True)
    runtime.mkdir(parents=True)
    model_file = models / "deepseek_v4.py"
    model_file.write_text(
        "from sglang.kernels.ops.runtime import build_indices\n\n\ndef forward(x):\n    return build_indices(x)\n",
        encoding="utf-8",
    )
    runtime_file = runtime / "runtime.py"
    runtime_file.write_text(
        "import torch\nimport torch.nn.functional as F\n\n\n"
        "def build_indices(lengths):\n"
        "    return F.pad(torch.cumsum(lengths, dim=0), (1, 0))\n",
        encoding="utf-8",
    )
    return root, model_file, runtime_file


def _anchor_report() -> AnchorReport:
    return AnchorReport(
        name="init_lookback_scan_state_kernel",
        category="elementwise",
        occurrences=6,
        total_us=12.0,
        avg_us=2.0,
        share=0.01,
        signature="elementwise -> elementwise -> elementwise",
        consistency=1.0,
        before=Slot(category="elementwise", names=[("fill", 6)]),
        after=Slot(category="elementwise", names=[("scan", 6)]),
        span=[{"category": "elementwise", "name": "scan", "is_anchor": True}],
        patterns=[("elementwise -> elementwise -> elementwise", 6)],
    )


def _proposal(source_file: str, additional: list[str] | None = None) -> str:
    return json.dumps(
        [
            {
                "name": "prefill_indptr",
                "env_flag": "FUSED_INDPTR",
                "op_chain": "cumsum + pad",
                "ops": ["elementwise"],
                "source_file": source_file,
                "additional_files": additional or [],
                "fusion_math": "one scan instead of three launches",
                "eager_reference": "build_indices",
                "priority": 0.9,
            }
        ]
    )


def _parse(raw: str, *, root: Path, model_file: Path) -> list[Recipe]:
    return parse_discovered_recipes(
        raw,
        model_type="deepseek_v4",
        framework="sglang",
        source_file=str(model_file),
        shapes={},
        repo_scope=True,
        repo_root=str(root),
    )


class TestProposalResolution:
    """What a repo-scope proposal is allowed to name."""

    def test_the_call_site_may_be_a_file_the_prompt_never_showed(self, tmp_path):
        """The whole point: reach the runtime without anyone naming it."""
        root, model_file, runtime_file = _repo(tmp_path)

        recipes = _parse(_proposal(str(runtime_file)), root=root, model_file=model_file)

        assert [r.source_file for r in recipes] == [str(runtime_file)]

    def test_a_relative_path_resolves_against_the_repo_root(self, tmp_path):
        """An agent that answers in repo-relative paths is answering correctly."""
        root, model_file, runtime_file = _repo(tmp_path)
        relative = runtime_file.relative_to(root).as_posix()

        recipes = _parse(_proposal(relative), root=root, model_file=model_file)

        assert [r.source_file for r in recipes] == [str(runtime_file)]

    def test_extra_call_sites_are_carried_on_the_recipe(self, tmp_path):
        """A fusion spanning two files has to arrive downstream as two files."""
        root, model_file, runtime_file = _repo(tmp_path)

        recipes = _parse(
            _proposal(str(runtime_file), [str(model_file)]),
            root=root,
            model_file=model_file,
        )

        assert recipes[0].extra_files == [str(model_file)]
        assert recipes[0].edit_files == [str(runtime_file), str(model_file)]

    def test_an_invented_call_site_drops_the_proposal(self, tmp_path):
        """Silently retargeting a hallucinated path is the bug repo scope removes.

        Single-file discovery falls back to the model file here, which is how a
        run ends up authoring against a file the model never proposed.
        """
        root, model_file, _runtime = _repo(tmp_path)

        recipes = _parse(_proposal("python/sglang/nope.py"), root=root, model_file=model_file)

        assert recipes == []

    def test_a_path_outside_the_repo_is_refused(self, tmp_path):
        """The repository is the scope; an absolute path elsewhere is not in it."""
        root, _model_file, _runtime = _repo(tmp_path)
        outsider = tmp_path / "elsewhere.py"
        outsider.write_text("x = 1\n", encoding="utf-8")

        assert resolve_repo_file(str(outsider), str(root)) == ""

    def test_an_unresolvable_extra_file_does_not_sink_the_proposal(self, tmp_path):
        """One bad entry in a list is not a reason to lose a good call site."""
        root, model_file, runtime_file = _repo(tmp_path)

        recipes = _parse(
            _proposal(str(runtime_file), ["python/sglang/ghost.py"]),
            root=root,
            model_file=model_file,
        )

        assert recipes[0].source_file == str(runtime_file)
        assert recipes[0].extra_files == []


class TestPrompt:
    """What the repo-scope discovery prompt does and does not contain."""

    def test_it_embeds_no_source_and_names_the_root(self, tmp_path):
        """A 162 KB model file in the prompt is what repo scope replaces."""
        root, model_file, _runtime = _repo(tmp_path)

        prompt = build_anchored_discovery_prompt(
            model_type="deepseek_v4",
            framework="sglang",
            source_files=[str(model_file)],
            report=_anchor_report(),
            shapes={},
            repo_scope=True,
            repo_root=str(root),
        )

        assert "```python" not in prompt
        assert str(root) in prompt
        assert "additional_files" in prompt
        # The single-file rule must be gone, not merely contradicted later.
        assert "One patch, one file." not in prompt

    def test_the_single_file_prompt_is_unchanged(self, tmp_path):
        """Repo scope is opt-in; the default path still embeds and still fences."""
        _root, model_file, _runtime = _repo(tmp_path)

        prompt = build_anchored_discovery_prompt(
            model_type="deepseek_v4",
            framework="sglang",
            source_files=[str(model_file)],
            report=_anchor_report(),
            shapes={},
        )

        assert "```python" in prompt
        assert "One patch, one file." in prompt


class TestCampaignHandoff:
    """What the authoring campaign is told it may edit."""

    def test_every_call_site_reaches_the_loop(self, tmp_path):
        _root, model_file, runtime_file = _repo(tmp_path)
        recipe = _recipe(str(runtime_file), extra_files=[str(model_file)])

        cmd = build_forge_loop_command(
            recipe,
            workspace="/fw",
            driver_path="/out/driver.py",
            experiments_dir="/out/exp",
            result_json="/out/result.json",
            program_md_file="/out/program.md",
            fused_module="/fw/fused.py",
        )

        listed = cmd[cmd.index("--source-files") + 1].split(",")
        assert str(runtime_file) in listed
        assert str(model_file) in listed

    def test_the_program_md_admits_the_tracked_roots(self, tmp_path):
        """The old text promised only one file was tracked, which was false here."""
        root, model_file, runtime_file = _repo(tmp_path)
        recipe = _recipe(str(runtime_file), extra_files=[str(model_file)])

        text = build_campaign_program_md(
            recipe,
            harness_path="/out/harness.py",
            fused_module="/fw/fused.py",
            repo_scope=True,
            tracked_roots=[str(root / "python")],
        )

        assert str(root / "python") in text
        assert "Do NOT create any other new module." not in text


def _recipe(source_file: str, **over) -> Recipe:
    base = dict(
        pattern_id="prefill_indptr",
        description="Fuse the indptr scans.",
        env_flag="DSV4_FUSED_INDPTR",
        source_file=source_file,
        source_hints=["cumsum"],
        fusion_math="one scan",
        eager_reference_hint="build_indices",
        shapes={},
        matched_categories=["elementwise"],
        trigger_share=0.5,
    )
    base.update(over)
    return Recipe(**base)


class TestShadowIndex:
    """Which trees the loop can keep and revert."""

    def test_a_second_package_is_indexed(self, tmp_path):
        """An edit in an unindexed tree is neither keepable nor revertible."""
        root = tmp_path / "fw"
        first = root / "pkg_a" / "mod.py"
        second = root / "pkg_b" / "runtime.py"
        first.parent.mkdir(parents=True)
        second.parent.mkdir(parents=True)
        first.write_text("a = 1\n", encoding="utf-8")
        second.write_text("b = 2\n", encoding="utf-8")

        shadow = ensure_git_workspace(
            str(root),
            str(first),
            git_dir=str(tmp_path / "shadow.git"),
            scope_files=[str(second)],
        )

        assert shadow is not None
        try:
            tracked = subprocess.run(
                ["git", "ls-files"],
                cwd=shadow.root,
                capture_output=True,
                text=True,
                env={**os.environ, **shadow.env},
                check=True,
            ).stdout.split()
            assert "pkg_a/mod.py" in tracked
            assert "pkg_b/runtime.py" in tracked
        finally:
            shadow.dispose()


class TestWiringGate:
    """A fusion wired from the second file is still wired."""

    def test_a_reference_in_any_edited_file_counts(self, tmp_path):
        call_site = tmp_path / "runtime.py"
        model = tmp_path / "model.py"
        call_site.write_text("x = 1\n", encoding="utf-8")
        model.write_text(
            "from .m_fused_chain import run\n\n\ndef forward(t):\n    return run(t)\n",
            encoding="utf-8",
        )

        wiring = fused_symbol_invocation_evidence(str(call_site), [str(model)])

        assert wiring.verdict == "wired", wiring.reason

    def test_a_dead_import_anywhere_still_fails(self, tmp_path):
        call_site = tmp_path / "runtime.py"
        model = tmp_path / "model.py"
        call_site.write_text("x = 1\n", encoding="utf-8")
        model.write_text("from .m_fused_chain import run  # noqa: F401\n", encoding="utf-8")

        wiring = fused_symbol_invocation_evidence(str(call_site), [str(model)])

        assert wiring.verdict == "not_wired"
        assert "dead code" in wiring.reason


class TestExport:
    """The patch is the only thing that leaves this pipeline."""

    def test_an_extra_call_site_edit_reaches_the_patch(self, tmp_path):
        """Tracked-and-kept but not exported is the worst of the three states."""
        root, model_file, runtime_file = _repo(tmp_path)
        pristine = tmp_path / "pristine"
        for path in (model_file, runtime_file):
            dest = pristine / path.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        runtime_file.write_text(runtime_file.read_text(encoding="utf-8") + "# fused\n", encoding="utf-8")
        model_file.write_text(model_file.read_text(encoding="utf-8") + "# wired\n", encoding="utf-8")

        artifacts = export_artifacts(
            str(root),
            str(runtime_file),
            tmp_path / "out",
            pristine_dir=pristine,
            extra_files=[str(model_file)],
        )

        assert artifacts.patch
        patch = Path(artifacts.patch).read_text(encoding="utf-8")
        assert "kernels/ops/runtime.py" in patch
        assert "srt/models/deepseek_v4.py" in patch

    def test_the_git_path_sweeps_in_an_unnamed_edit_under_repo_scope(self, tmp_path):
        """Repo scope tells the author the tree is editable, so export cannot
        assume the edited set was known when the recipe was built."""
        root, model_file, runtime_file = _repo(tmp_path)
        _git_init(root)
        runtime_file.write_text(runtime_file.read_text(encoding="utf-8") + "# fused\n", encoding="utf-8")
        # Nobody listed this file on the recipe; the author decided to touch it.
        model_file.write_text(model_file.read_text(encoding="utf-8") + "# wired\n", encoding="utf-8")

        artifacts = export_artifacts(
            str(root),
            str(runtime_file),
            tmp_path / "out",
            repo_scope=True,
        )

        patch = Path(artifacts.patch).read_text(encoding="utf-8")
        assert "kernels/ops/runtime.py" in patch
        assert "srt/models/deepseek_v4.py" in patch

    def test_without_repo_scope_the_git_path_stays_scoped(self, tmp_path):
        """The default export must not start shipping unrelated dirty files."""
        root, model_file, runtime_file = _repo(tmp_path)
        _git_init(root)
        runtime_file.write_text(runtime_file.read_text(encoding="utf-8") + "# fused\n", encoding="utf-8")
        model_file.write_text(model_file.read_text(encoding="utf-8") + "# unrelated\n", encoding="utf-8")

        artifacts = export_artifacts(str(root), str(runtime_file), tmp_path / "out")

        patch = Path(artifacts.patch).read_text(encoding="utf-8")
        assert "kernels/ops/runtime.py" in patch
        assert "srt/models/deepseek_v4.py" not in patch


def _git_init(root: Path) -> None:
    """Commit ``root`` so the export takes its git path, as a real run does."""
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t"}
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "base", "--no-gpg-sign"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=env)
