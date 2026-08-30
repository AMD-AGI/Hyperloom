# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two pieces that let the forge-loop drive a fusion.

The loop keeps and reverts with git and scores from stdout, and a serving
framework offers neither: it is usually a pip install with no repository, and
the harness reports JSON. These tests cover the adapters for both.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from kernelforge.fusion import campaign as campaign_module
from kernelforge.fusion.campaign import (
    build_campaign_program_md,
    build_forge_loop_command,
    fused_module_path,
    run_recipe_campaign,
)
from kernelforge.fusion.driver_shim import render_driver
from kernelforge.fusion.models import Recipe
from kernelforge.fusion.shadow_repo import SHADOW_BRANCH, ensure_git_workspace


def _recipe(**over) -> Recipe:
    base = dict(
        pattern_id="residual_add_rmsnorm",
        description="Fold residual-add into RMSNorm.",
        env_flag="LFM2_FUSED_RESIDUAL",
        source_file="/sgl/models/lfm2.py",
        source_hints=["+ residual"],
        fusion_math="y, residual = norm(x + residual)",
        eager_reference_hint="Import RMSNorm; compare.",
        shapes={"hidden_size": 2048},
        matched_categories=["rmsnorm"],
        trigger_share=0.3,
    )
    base.update(over)
    return Recipe(**base)


def _framework_tree(tmp_path: Path) -> tuple[Path, Path]:
    """An installed framework package with a neighbouring wheel beside it."""
    install_root = tmp_path / "site-packages"
    package = install_root / "sglang" / "layers"
    package.mkdir(parents=True)
    (install_root / "sglang" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = package / "lfm2.py"
    source.write_text("def forward(x):\n    return x\n", encoding="utf-8")
    neighbour = install_root / "torch"
    neighbour.mkdir()
    (neighbour / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (neighbour / "big.so").write_bytes(b"\0" * 4096)
    return install_root, source


def _git_out(shadow, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=shadow.root,
        capture_output=True,
        text=True,
        env={**os.environ, **shadow.env},
        check=True,
    ).stdout


def _make_checkout(root: Path) -> str:
    """Turn ``root`` into the developer's own repository and return its HEAD."""
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "user base",
            "--no-gpg-sign",
        ],
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestShadowRepo:
    def test_the_framework_tree_receives_no_git_data(self, tmp_path):
        """The shadow leaves only a pointer file, no git objects, in the tree."""
        root, source = _framework_tree(tmp_path)

        shadow = ensure_git_workspace(str(root), str(source), git_dir=str(tmp_path / "out" / "shadow.git"))

        assert shadow is not None
        # The tree now has a .git POINTER FILE (one line pointing to shadow.git).
        # It is not a directory, so no git data is stored inside the tree itself.
        git_entry = root / ".git"
        assert git_entry.is_file(), "expected a .git pointer file, not a directory"
        assert not git_entry.is_dir()
        assert not (root / ".gitignore").exists()
        assert (tmp_path / "out" / "shadow.git").is_dir()
        # Tracked in OUR repository, which is the point: keep/revert can see it.
        assert "sglang/layers/lfm2.py" in _git_out(shadow, "ls-files")

    def test_pointer_file_is_removed_on_dispose(self, tmp_path):
        """dispose() leaves the framework tree exactly as it was found."""
        root, source = _framework_tree(tmp_path)
        shadow = ensure_git_workspace(str(root), str(source), git_dir=str(tmp_path / "out" / "shadow.git"))
        assert shadow is not None
        assert (root / ".git").is_file(), "pointer file should exist before dispose"

        shadow.dispose()

        assert not (root / ".git").exists(), "pointer file should be gone after dispose"
        assert not (tmp_path / "out" / "shadow.git").exists(), "git dir should be gone after dispose"

    def test_existing_git_dir_uses_env_fallback_and_keeps_history(self, tmp_path):
        """If the tree already has a .git (editable checkout), don't move its objects."""
        root, source = _framework_tree(tmp_path)
        user_head = _make_checkout(root)

        shadow = ensure_git_workspace(str(root), str(source), git_dir=str(tmp_path / "out" / "shadow.git"))
        # The env-fallback was used: the developer's .git is still a directory.
        assert shadow is not None
        assert (root / ".git").is_dir(), "developer .git must remain a directory"
        # Our shadow's env carries GIT_DIR so its git calls go to shadow.git.
        assert shadow.env.get("GIT_DIR", "").endswith("shadow.git")
        # dispose() must not touch the developer's .git.
        shadow.dispose()
        assert (root / ".git").is_dir(), "disposal must not touch a real repository"
        after = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        assert after.returncode == 0
        assert after.stdout.strip() == user_head, "developer's HEAD was altered"

    def test_only_the_framework_package_is_indexed(self, tmp_path):
        """A pip install sits beside gigabytes that are no campaign's business."""
        root, source = _framework_tree(tmp_path)

        shadow = ensure_git_workspace(str(root), str(source), git_dir=str(tmp_path / "out" / "shadow.git"))

        tracked = _git_out(shadow, "ls-files").split()
        assert "sglang/layers/lfm2.py" in tracked
        assert not any(path.startswith("torch") for path in tracked)
        # The exclude also keeps git from WALKING the neighbour, which is what
        # makes every later status cheap rather than merely correct.
        assert not any(
            path.startswith("torch") for path in _git_out(shadow, "ls-files", "--others", "--exclude-standard").split()
        )

    def test_diff_paths_stay_package_relative(self, tmp_path):
        """The knowledge base replays a patch against the package, not below it."""
        root, source = _framework_tree(tmp_path)
        shadow = ensure_git_workspace(str(root), str(source), git_dir=str(tmp_path / "out" / "shadow.git"))
        source.write_text("def forward(x):\n    return fused(x)\n", encoding="utf-8")

        diff = _git_out(shadow, "diff", "HEAD")

        assert "b/sglang/layers/lfm2.py" in diff

    def test_a_developer_checkout_keeps_its_own_history(self, tmp_path):
        """The loop's commits are its deliverable, but not into someone's repo."""
        root, source = _framework_tree(tmp_path)
        user_head = _make_checkout(root)
        # With an existing .git dir, ensure_git_workspace uses the env fallback.
        # The developer's history must be unaffected by commits in the shadow.

        shadow = ensure_git_workspace(str(root), str(source), git_dir=str(tmp_path / "out" / "shadow.git"))
        assert shadow is not None
        source.write_text("def forward(x):\n    return fused(x)\n", encoding="utf-8")
        _git_out(shadow, "add", "-u")
        _git_out(
            shadow,
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "keep",
            "--no-gpg-sign",
        )

        after = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert after == user_head, "a keep commit landed in the developer's repository"
        shadow.dispose()
        assert (root / ".git").is_dir(), "disposal must not touch a real repository"

    def test_the_fused_module_is_tracked_before_the_campaign_starts(self, tmp_path):
        """``git add -u`` can only ever commit what was tracked at the baseline."""
        root, source = _framework_tree(tmp_path)
        fused = source.parent / "lfm2_fused_residual.py"

        shadow = ensure_git_workspace(
            str(root),
            str(source),
            git_dir=str(tmp_path / "out" / "shadow.git"),
            extra_paths=(str(fused),),
        )

        assert fused.is_file() and fused.read_text(encoding="utf-8") == ""
        # The author fills it in; the loop's keep must carry it.
        fused.write_text("def fused(x):\n    return x\n", encoding="utf-8")
        _git_out(shadow, "add", "-u")
        _git_out(
            shadow,
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "keep",
            "--no-gpg-sign",
        )
        patch = _git_out(shadow, "diff", shadow.base_commit, "HEAD")
        assert "sglang/layers/lfm2_fused_residual.py" in patch
        assert "def fused(x):" in patch

    def test_the_baseline_is_not_left_on_a_trunk_branch(self, tmp_path):
        """``create_campaign_config`` refuses an unnamed, main or master branch.

        A freshly initialized repository is on exactly one of those, so without
        this every campaign would be rejected before its first iteration.
        """
        root, source = _framework_tree(tmp_path)

        shadow = ensure_git_workspace(str(root), str(source), git_dir=str(tmp_path / "out" / "shadow.git"))

        branch = _git_out(shadow, "branch", "--show-current").strip()
        assert branch and branch not in {"main", "master"}
        # The branch has to point AT the baseline, not at an unborn HEAD.
        assert _git_out(shadow, "rev-parse", "HEAD").strip() == shadow.base_commit

    def test_the_loop_accepts_the_workspace_the_shadow_hands_it(self, tmp_path, monkeypatch):
        """Run the loop's own campaign resolution against a real shadow tree.

        Every other test here mocks the subprocess away, so nothing else would
        notice that the loop rejects the workspace before its first iteration.
        """
        from kernelforge.loop.campaign_config import create_campaign_config

        root, source = _framework_tree(tmp_path)
        fused = source.parent / "lfm2_fused_residual.py"
        shadow = ensure_git_workspace(
            str(root),
            str(source),
            git_dir=str(tmp_path / "out" / "shadow.git"),
            extra_paths=(str(fused),),
        )
        driver = tmp_path / "driver.py"
        driver.write_text("print('wall_ms: 1.0')\n", encoding="utf-8")
        program = tmp_path / "program.md"
        program.write_text("# task\n", encoding="utf-8")
        for name, value in shadow.env.items():
            monkeypatch.setenv(name, value)

        campaign = create_campaign_config(
            workspace_dir=shadow.root,
            kernel=str(source),
            driver=str(driver),
            source_files=list(shadow.created_paths),
            program_md_file=str(program),
            git_branch=SHADOW_BRANCH,
            kernel_backend="fusion",
            task_type="repository",
            producer="fusion",
            operator_name="residual_add_rmsnorm",
            gpu_type="mi355x",
            gpu_target="gfx950",
        )

        assert campaign.producer == "fusion"
        # Package-relative, which is what makes a diff taken here replayable
        # against an install rather than against one directory inside it.
        assert campaign.kernel_path == "sglang/layers/lfm2.py"
        assert campaign.source_files == [
            "sglang/layers/lfm2.py",
            "sglang/layers/lfm2_fused_residual.py",
        ]

    def test_a_crashed_runs_leftover_does_not_become_the_baseline(self, tmp_path):
        """The baseline is the UNFUSED framework, whatever was on disk."""
        root, source = _framework_tree(tmp_path)
        leftover = source.parent / "lfm2_fused_residual.py"
        leftover.write_text("REJECTED = 1\n", encoding="utf-8")

        shadow = ensure_git_workspace(
            str(root),
            str(source),
            git_dir=str(tmp_path / "out" / "shadow.git"),
            extra_paths=(str(leftover),),
        )

        rel = leftover.relative_to(shadow.root).as_posix()
        assert leftover.read_text(encoding="utf-8") == ""
        assert _git_out(shadow, "show", f"{shadow.base_commit}:{rel}") == ""

    def test_a_flat_framework_still_tracks_its_placeholder(self, tmp_path):
        """A source directly under the export root has no package to admit.

        Each placeholder is named in the exclude too, or the whitelist would
        leave the one file the campaign has to keep untracked.
        """
        root = tmp_path / "framework"
        root.mkdir()
        source = root / "lfm2.py"
        source.write_text("def forward(x):\n    return x\n", encoding="utf-8")
        fused = root / "lfm2_fused_residual.py"

        shadow = ensure_git_workspace(
            str(root),
            str(source),
            git_dir=str(tmp_path / "out" / "shadow.git"),
            extra_paths=(str(fused),),
        )

        assert "lfm2_fused_residual.py" in _git_out(shadow, "ls-files").split()
        fused.write_text("FUSED = 1\n", encoding="utf-8")
        _git_out(shadow, "add", "-u")
        _git_out(
            shadow,
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "keep",
            "--no-gpg-sign",
        )
        assert "FUSED = 1" in _git_out(shadow, "diff", shadow.base_commit, "HEAD")

    def test_a_reset_undoes_the_previous_campaign_entirely(self, tmp_path):
        """The next recipe has to measure its baseline on unfused code."""
        root, source = _framework_tree(tmp_path)
        fused = source.parent / "lfm2_fused_residual.py"
        shadow = ensure_git_workspace(
            str(root),
            str(source),
            git_dir=str(tmp_path / "out" / "shadow.git"),
            extra_paths=(str(fused),),
        )
        source.write_text("def forward(x):\n    return fused(x)\n", encoding="utf-8")
        fused.write_text("def fused(x):\n    return x\n", encoding="utf-8")
        _git_out(shadow, "add", "-u")
        _git_out(
            shadow,
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "keep",
            "--no-gpg-sign",
        )
        stray = source.parent / "lfm2_fusion_helper.py"
        stray.write_text("junk\n", encoding="utf-8")

        assert shadow.reset_to_base() is True

        assert source.read_text(encoding="utf-8") == "def forward(x):\n    return x\n"
        assert fused.read_text(encoding="utf-8") == ""
        assert not stray.exists(), "an author-created stray must not reach the next recipe"

    def test_disposal_removes_an_unused_placeholder_but_not_an_authored_one(self, tmp_path):
        """The export still has to read a module the author actually wrote."""
        root, source = _framework_tree(tmp_path)
        unused = source.parent / "lfm2_fused_a.py"
        authored = source.parent / "lfm2_fused_b.py"
        shadow = ensure_git_workspace(
            str(root),
            str(source),
            git_dir=str(tmp_path / "out" / "shadow.git"),
            extra_paths=(str(unused), str(authored)),
        )
        authored.write_text("def fused(x):\n    return x\n", encoding="utf-8")

        shadow.dispose()

        assert not unused.exists()
        assert authored.is_file(), "the export has not read this yet"
        assert not (tmp_path / "out" / "shadow.git").exists()

    def test_a_source_outside_the_root_is_refused(self, tmp_path):
        """Nothing can be indexed for a file the export root does not contain."""
        root, _source = _framework_tree(tmp_path)
        stranger = tmp_path / "elsewhere.py"
        stranger.write_text("x = 1\n", encoding="utf-8")

        assert ensure_git_workspace(str(root), str(stranger), git_dir=str(tmp_path / "out" / "shadow.git")) is None


class TestDriverShim:
    def _run(
        self,
        tmp_path: Path,
        harness_report: dict,
        fused_module: str = "",
    ) -> subprocess.CompletedProcess:
        harness = tmp_path / "kernel_harness.py"
        harness.write_text("import json\nprint(json.dumps(%r))\n" % harness_report, encoding="utf-8")
        driver = tmp_path / "driver.py"
        driver.write_text(
            render_driver(
                str(harness),
                ("LFM2_FUSED_RESIDUAL",),
                report_log=str(tmp_path / "reports.jsonl"),
                case_id="decode",
                fused_module=fused_module,
            ),
            encoding="utf-8",
        )
        import sys

        return subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, cwd=tmp_path)

    def test_parity_and_timing_become_the_loop_contract(self, tmp_path):
        proc = self._run(
            tmp_path,
            {
                "compiled": True,
                "is_triton": True,
                "error": "",
                "parity": [{"snr_db": 41.5, "max_abs_err": 3e-05, "label": "T16"}],
                "eager_us": 120.0,
                "fused_us": 96.0,
                "skipped": False,
                "skip_reason": "",
            },
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "SNR: 41.50 dB" in proc.stdout
        assert "case_ms: decode 0.096000" in proc.stdout

    def test_the_worst_shape_decides_parity(self, tmp_path):
        proc = self._run(
            tmp_path,
            {
                "compiled": True,
                "is_triton": True,
                "error": "",
                "parity": [
                    {"snr_db": 55.0, "max_abs_err": 1e-06, "label": "a"},
                    {"snr_db": 22.0, "max_abs_err": 9e-03, "label": "b"},
                ],
                "eager_us": 100.0,
                "fused_us": 90.0,
                "skipped": False,
                "skip_reason": "",
            },
        )
        assert "SNR: 22.00 dB" in proc.stdout
        assert "max_diff: 9.000000e-03" in proc.stdout

    def test_a_compile_failure_fails_the_iteration(self, tmp_path):
        fused = tmp_path / "model_fused.py"
        fused.write_text("def fused(x):\n    return x\n", encoding="utf-8")
        proc = self._run(
            tmp_path,
            {
                "compiled": False,
                "is_triton": False,
                "error": "cuda_bf16.h not found",
                "parity": [],
                "eager_us": None,
                "fused_us": None,
                "skipped": False,
                "skip_reason": "",
            },
            fused_module=str(fused),
        )
        assert proc.returncode == 1
        assert "COMPILE FAILED: cuda_bf16.h not found" in proc.stdout

    def test_an_unfused_baseline_anchors_instead_of_failing(self, tmp_path):
        """The pristine bench runs before any kernel exists, so nothing compiled.

        Observed in production: the harness reported ``compiled: false`` with
        eager-vs-eager parity, the driver called it a crash, and the loop died
        with "mean case scoring requires pristine per-case timings" before its
        first iteration -- every fusion campaign failed the same way.
        """
        fused = tmp_path / "model_fused.py"
        fused.write_text("", encoding="utf-8")  # committed empty by the campaign
        proc = self._run(
            tmp_path,
            {
                "compiled": False,
                "is_triton": False,
                "error": "",
                "parity": [{"snr_db": 999.0, "max_abs_err": 0.0, "label": "T16 (baseline: eager vs eager)"}],
                "eager_us": 425.0,
                "fused_us": 426.0,
                "skipped": False,
                "skip_reason": "",
            },
            fused_module=str(fused),
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "case_ms: decode 0.426000" in proc.stdout
        assert "COMPILE FAILED" not in proc.stdout

    def test_a_missing_fused_module_still_anchors(self, tmp_path):
        """The placeholder is absent until ensure_git_workspace creates it."""
        proc = self._run(
            tmp_path,
            {
                "compiled": False,
                "is_triton": False,
                "error": "",
                "parity": [{"snr_db": 999.0, "max_abs_err": 0.0, "label": "T16"}],
                "eager_us": 425.0,
                "fused_us": 426.0,
                "skipped": False,
                "skip_reason": "",
            },
            fused_module=str(tmp_path / "never_created.py"),
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "case_ms: decode 0.426000" in proc.stdout

    def test_a_skipped_microbench_is_not_a_failure(self, tmp_path):
        proc = self._run(
            tmp_path,
            {
                "compiled": True,
                "is_triton": True,
                "error": "",
                "parity": [{"snr_db": 44.0, "max_abs_err": 1e-05, "label": "T16"}],
                "eager_us": 130.0,
                "fused_us": None,
                "skipped": True,
                "skip_reason": "Mamba backend cannot init on ROCm",
            },
        )
        assert proc.returncode == 0
        assert "SKIPPED: Mamba backend cannot init on ROCm" in proc.stdout
        # Reporting the eager time for both arms shows no speedup rather than an
        # error, which is what a missing microbench actually means.
        assert "case_ms: decode 0.130000" in proc.stdout


class TestFusedModulePath:
    def test_the_path_is_derived_from_the_recipe(self):
        path = fused_module_path(_recipe())
        assert path == "/sgl/models/lfm2_fused_residual_add_rmsnorm.py"

    def test_a_combined_recipe_still_yields_one_importable_name(self):
        """``_combined_recipe`` joins pattern ids with ``+``, which is not a name."""
        path = Path(fused_module_path(_recipe(pattern_id="a+b")))
        assert path.name == "lfm2_fused_a_b.py"
        assert path.stem.isidentifier()


class TestCampaignCommand:
    def test_the_loop_is_told_who_owns_what(self, tmp_path):
        cmd = build_forge_loop_command(
            _recipe(),
            workspace="/fw",
            driver_path="/out/driver.py",
            experiments_dir="/out/exp",
            result_json="/out/r.json",
            program_md_file="/out/p.md",
            gpu_target="gfx950",
            fused_module="/sgl/models/lfm2_fused_residual_add_rmsnorm.py",
        )
        assert "forge-loop" in cmd
        assert cmd[cmd.index("--kernel-backend") + 1] == "fusion"
        assert cmd[cmd.index("--task-type") + 1] == "repository"
        # The loop refuses an unnamed / main / master branch, and a shadow
        # repository is freshly initialized onto exactly one of those.
        assert cmd[cmd.index("--git-branch") + 1] == SHADOW_BRANCH
        assert SHADOW_BRANCH not in {"", "main", "master"}
        # Discovery and the harness are the pipeline's.
        assert "--no-prepare-task" in cmd
        # The fused module is an entry point too, or the loop orients on a model
        # file and never sees where the kernel actually lives.
        assert cmd[cmd.index("--source-files") + 1] == (
            "/sgl/models/lfm2.py,/sgl/models/lfm2_fused_residual_add_rmsnorm.py"
        )

    def test_the_campaign_pins_one_lane(self, tmp_path):
        """A lane measures a copy; fusion is measured through the real install.

        The loop's own default is above one, so this has to be stated. A lane
        edits a workspace copy while the benchmark and the serving gate import
        the framework from where it is installed, and the driver sits outside
        the workspace entirely, which a round refuses outright.
        """
        cmd = build_forge_loop_command(
            _recipe(),
            workspace="/fw",
            driver_path="/out/driver.py",
            experiments_dir="/out/exp",
            result_json="/out/r.json",
            program_md_file="/out/p.md",
        )
        assert cmd[cmd.index("--lanes") + 1] == "1"

    def test_fusion_records_go_to_their_own_producer_and_never_warm_start(self, tmp_path):
        cmd = build_forge_loop_command(
            _recipe(),
            workspace="/fw",
            driver_path="/out/driver.py",
            experiments_dir="/out/exp",
            result_json="/out/r.json",
            program_md_file="/out/p.md",
        )
        assert "--experience-kb" in cmd and "--no-experience-kb" not in cmd
        assert cmd[cmd.index("--producer") + 1] == "fusion"
        # Keyed on the chain: several chains share one model file, and the file
        # is all the loop could infer on its own.
        assert cmd[cmd.index("--operator-name") + 1] == "residual_add_rmsnorm"
        assert "--no-kb-warmstart" in cmd

    def test_the_callers_agent_runtime_reaches_the_process_that_edits(self, tmp_path):
        """The loop resolves its own runtime from Config defaults otherwise.

        A caller asking for a restricted sandbox would silently get the default
        ``bypass`` in the one process that writes to the framework.
        """
        cmd = build_forge_loop_command(
            _recipe(),
            workspace="/fw",
            driver_path="/out/driver.py",
            experiments_dir="/out/exp",
            result_json="/out/r.json",
            program_md_file="/out/p.md",
            agent_backend="codex",
            agent_sandbox_mode="workspace-write",
        )
        assert cmd[cmd.index("--agent-backend") + 1] == "codex"
        assert cmd[cmd.index("--agent-sandbox-mode") + 1] == "workspace-write"

    def test_the_task_document_names_the_only_writable_module(self, tmp_path):
        program = build_campaign_program_md(
            _recipe(),
            harness_path="/out/kernel_harness.py",
            fused_module="/sgl/models/lfm2_fused_residual_add_rmsnorm.py",
        )
        assert "/sgl/models/lfm2_fused_residual_add_rmsnorm.py" in program
        assert "Do NOT create any other new module." in program

    def test_the_task_document_hands_over_the_harness_read_only(self, tmp_path):
        """The implementer is measured by this file, so it may not author it.

        The document used to order it to write the harness, which the campaign
        had already written -- and which the in-session gate would have denied.
        """
        program = build_campaign_program_md(
            _recipe(),
            harness_path="/out/kernel_harness.py",
            fused_module="/sgl/models/lfm2_fused_residual_add_rmsnorm.py",
        )
        assert "/out/kernel_harness.py" in program
        assert "READ-ONLY" in program
        assert "Do NOT\nmodify or recreate it." in program
        assert "you must write" not in program.lower()

    def test_the_authoring_pass_gets_no_harness_section(self, tmp_path):
        """It states its own contract, which is to WRITE the file."""
        program = build_campaign_program_md(_recipe(), harness_path="")

        assert "harness" not in program.lower()

    def test_a_campaign_writes_its_driver_and_task_document(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            campaign_module.subprocess,
            "Popen",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no forge-loop here")),
        )
        outcome = run_recipe_campaign(
            _recipe(),
            workspace=str(tmp_path),
            harness_path=str(tmp_path / "kernel_harness.py"),
            output_dir=str(tmp_path),
            experience="earlier: parity failed at 12 dB",
        )
        written = {p.name for p in tmp_path.iterdir() if p.is_file()}
        assert "driver_residual_add_rmsnorm.py" in written
        assert "program_residual_add_rmsnorm.md" in written
        program = (tmp_path / "program_residual_add_rmsnorm.md").read_text()
        assert "LFM2_FUSED_RESIDUAL" in program
        assert "earlier: parity failed at 12 dB" in program
        assert outcome.result.kept is False
        assert "CAMPAIGN FAILED" in outcome.result.note

    def _campaign_with_result(self, tmp_path, monkeypatch, payload, reports=()):
        """Run a campaign whose forge-loop writes ``payload`` to --result-json.

        The payload keys are the forge-loop's own (see cli.py _build_result), not
        a shape invented here: an adapter tested against a fixture it also made up
        proves only that it is self-consistent. ``reports`` stands in for what the
        driver recorded while the loop ran.
        """
        result_json = tmp_path / "forge_loop_residual_add_rmsnorm.json"
        report_log = tmp_path / "harness_reports_residual_add_rmsnorm.jsonl"

        class _Proc:
            stdout = iter(["Experiment: exp-1\n"])
            returncode = 0

            def wait(self, timeout=None):
                return 0

        def fake_popen(*_a, **_k):
            result_json.write_text(json.dumps(payload), encoding="utf-8")
            report_log.write_text("".join(json.dumps(r) + "\n" for r in reports), encoding="utf-8")
            return _Proc()

        monkeypatch.setattr(campaign_module.subprocess, "Popen", fake_popen)
        return run_recipe_campaign(
            _recipe(),
            workspace=str(tmp_path),
            harness_path=str(tmp_path / "kernel_harness.py"),
            output_dir=str(tmp_path),
        )

    def test_a_campaign_that_beat_the_bar_is_kept(self, tmp_path, monkeypatch):
        outcome = self._campaign_with_result(
            tmp_path,
            monkeypatch,
            {
                "mean_case_speedup": 1.21,
                "total_speedup": 1.21,
                "baseline_ms": 100.0,
                "best_ms": 82.6,
                "improved": True,
                "best_iteration": 3,
                "best_commit": "c0ffee1",
                "experiment_id": "exp-1",
            },
        )

        assert outcome.result.kept is True
        assert outcome.result.correctness_passed is True
        assert outcome.result.kernel_speedup == 1.21
        assert outcome.experiment_id == "exp-1"

    def test_the_loops_pre_iteration_anchor_is_not_a_validated_candidate(self, tmp_path, monkeypatch):
        """A run that kept nothing still reports the 1.0 anchor it started from."""
        outcome = self._campaign_with_result(
            tmp_path,
            monkeypatch,
            {
                "mean_case_speedup": 1.0,
                "baseline_ms": 100.0,
                "best_ms": 100.0,
                "improved": False,
                "best_iteration": 0,
                "best_commit": "",
                "experiment_id": "exp-1",
            },
        )

        assert outcome.result.kept is False
        assert outcome.result.correctness_passed is False
        assert "no validated candidate" in outcome.result.note

    def test_a_campaign_below_the_fusion_bar_is_not_kept(self, tmp_path, monkeypatch):
        """The loop keeps on its own smaller margin; the fusion bar is 1.03."""
        outcome = self._campaign_with_result(
            tmp_path,
            monkeypatch,
            {
                "mean_case_speedup": 1.01,
                "baseline_ms": 100.0,
                "best_ms": 99.0,
                "improved": True,
                "best_iteration": 2,
                "experiment_id": "exp-1",
            },
        )

        assert outcome.result.kept is False
        assert outcome.result.kernel_speedup == 1.01

    def test_a_campaign_that_validated_nothing_reports_no_result(self, tmp_path, monkeypatch):
        outcome = self._campaign_with_result(
            tmp_path,
            monkeypatch,
            {
                "mean_case_speedup": None,
                "baseline_ms": 100.0,
                "best_ms": None,
                "improved": False,
                "best_iteration": 0,
                "experiment_id": "exp-1",
            },
        )

        assert outcome.result.kept is False
        assert outcome.result.correctness_passed is False
        assert outcome.result.kernel_speedup is None

    def test_parity_and_timings_come_from_the_report_behind_the_kept_candidate(self, tmp_path, monkeypatch):
        """The loop's result carries a speedup and nothing else."""
        outcome = self._campaign_with_result(
            tmp_path,
            monkeypatch,
            {
                "mean_case_speedup": 1.25,
                "baseline_ms": 0.120,
                "best_ms": 0.096,
                "improved": True,
                "best_iteration": 3,
                "experiment_id": "exp-1",
            },
            reports=[
                # An earlier, slower candidate the loop did not settle on.
                {
                    "compiled": True,
                    "skipped": False,
                    "eager_us": 120.0,
                    "fused_us": 111.0,
                    "parity": [{"snr_db": 44.0, "max_abs_err": 1e-05, "label": "a"}],
                },
                {
                    "compiled": True,
                    "skipped": False,
                    "eager_us": 120.0,
                    "fused_us": 96.0,
                    "parity": [
                        {"snr_db": 55.0, "max_abs_err": 1e-06, "label": "a"},
                        {"snr_db": 38.0, "max_abs_err": 4e-04, "label": "b"},
                    ],
                },
            ],
        )

        assert outcome.result.eager_us == 120.0
        assert outcome.result.fused_us == 96.0
        # The WORST shape decided correctness, so it is the one recorded.
        assert outcome.result.max_abs_err == 4e-04
        # Provenance differs on purpose: the speedup is the loop's mean over
        # repeated benchmarks, not fused_us/eager_us from this one report.
        assert outcome.result.kernel_speedup == 1.25
        assert outcome.result.rtol is None

    def test_a_crashed_campaign_does_not_report_the_previous_runs_keep(self, tmp_path, monkeypatch):
        """The result file outlives the run that wrote it."""
        stale = tmp_path / "forge_loop_residual_add_rmsnorm.json"
        stale.write_text(json.dumps({"mean_case_speedup": 1.4}), encoding="utf-8")

        class _Proc:
            stdout = iter(["boom\n"])

            def wait(self, timeout=None):
                return 1

        monkeypatch.setattr(campaign_module.subprocess, "Popen", lambda *a, **k: _Proc())
        outcome = run_recipe_campaign(
            _recipe(),
            workspace=str(tmp_path),
            harness_path=str(tmp_path / "kernel_harness.py"),
            output_dir=str(tmp_path),
        )

        assert outcome.result.kept is False
        assert outcome.result.kernel_speedup is None
        assert "exited 1" in outcome.result.note
        assert not stale.exists(), "the stale result must be gone before the run"

    def test_a_missing_report_log_degrades_rather_than_failing_the_recipe(self, tmp_path, monkeypatch):
        outcome = self._campaign_with_result(
            tmp_path,
            monkeypatch,
            {
                "mean_case_speedup": 1.21,
                "baseline_ms": 0.120,
                "best_ms": 0.099,
                "improved": True,
                "best_iteration": 2,
                "best_commit": "c0ffee1",
                "experiment_id": "exp-1",
            },
        )

        assert outcome.result.kept is True
        assert outcome.result.kernel_speedup == 1.21
        assert outcome.result.eager_us is None
        assert outcome.result.max_abs_err is None
