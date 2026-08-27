# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the review fixes: harness contract in the author prompt, the
combined (fuse-all) recipe, and the fusion-scoped emit patch."""

from __future__ import annotations

import re
import subprocess

from kernelforge.fusion.author import build_author_prompt, build_multi_author_prompt
from kernelforge.fusion.command import _combined_recipe, _safe_artifact_id
from kernelforge.fusion.emit import export_artifacts, restore_exported_changes
from kernelforge.fusion.models import Recipe


def _recipe(pattern_id="residual_add_rmsnorm", env_flag="LFM2_FUSED_RESIDUAL", **over) -> Recipe:
    base = dict(
        pattern_id=pattern_id,
        description="d",
        env_flag=env_flag,
        source_file="/sgl/models/lfm2.py",
        source_hints=["+ residual"],
        fusion_math="y=norm(x+r)",
        eager_reference_hint="import RMSNorm",
        shapes={"hidden_size": 2048, "T": 16},
        matched_categories=["rmsnorm"],
        trigger_share=0.3,
    )
    base.update(over)
    return Recipe(**base)


class TestHarnessContractInPrompt:
    def test_single_prompt_includes_harness_when_path_given(self):
        p = build_author_prompt(
            _recipe().to_dict(), framework="sglang", ab_hint="x", harness_path="/out/kernel_harness.py"
        )
        assert "/out/kernel_harness.py" in p
        assert '"compiled"' in p and '"parity"' in p and '"snr_db"' in p  # JSON contract
        assert "LFM2_FUSED_RESIDUAL" in p

    def test_single_prompt_omits_harness_without_path(self):
        p = build_author_prompt(_recipe().to_dict(), framework="sglang", ab_hint="x")
        assert "kernel_harness.py" not in p

    def test_multi_prompt_includes_harness_and_all_flags(self):
        rs = [_recipe().to_dict(), _recipe(pattern_id="swiglu_silu_mul", env_flag="LFM2_FUSED_SILU").to_dict()]
        p = build_multi_author_prompt(rs, framework="sglang", ab_hint="x", harness_path="/out/kernel_harness.py")
        assert "/out/kernel_harness.py" in p
        assert "LFM2_FUSED_RESIDUAL" in p and "LFM2_FUSED_SILU" in p


class TestCombinedRecipe:
    def test_folds_flags_and_ids(self):
        combined = _combined_recipe(
            [
                _recipe(),
                _recipe(pattern_id="swiglu_silu_mul", env_flag="LFM2_FUSED_SILU"),
            ]
        )
        assert combined.env_flag == "LFM2_FUSED_RESIDUAL LFM2_FUSED_SILU"
        assert combined.pattern_id == "residual_add_rmsnorm+swiglu_silu_mul"
        # validate_fn splits env_flag -> both flags toggled together.
        assert set(combined.env_flag.split()) == {"LFM2_FUSED_RESIDUAL", "LFM2_FUSED_SILU"}

    def test_dedupes_repeated_stable_flags(self):
        combined = _combined_recipe(
            [
                _recipe(pattern_id="normalize_qk", env_flag="ZAYA_FUSED_QK"),
                _recipe(pattern_id="grouped_qk", env_flag="ZAYA_FUSED_QK"),
                _recipe(pattern_id="residual", env_flag="ZAYA_FUSED_RESIDUAL"),
            ]
        )
        assert combined.env_flag == "ZAYA_FUSED_QK ZAYA_FUSED_RESIDUAL"


class TestArtifactId:
    """Artifact filenames are built from ``Recipe.pattern_id``, which is not a
    filesystem-safe string. LLM-proposed recipes are named ``llm:<pattern>`` and
    _combined_recipe joins them with ``+``, so a combined id both contains ``:``
    and grows without bound. NFS rejects ``:`` in a path component with EINVAL and
    every filesystem caps a component at NAME_MAX, so writing
    ``author_<pattern_id>.log`` raised OSError and failed the whole authoring
    attempt."""

    def test_combined_llm_id_is_reduced_to_safe_characters(self):
        combined = _combined_recipe(
            [
                _recipe(pattern_id="llm:qk_norm_rope", env_flag="Q_FUSED_QK"),
                _recipe(pattern_id="llm:add_rmsnorm_input", env_flag="Q_FUSED_ADD"),
                _recipe(pattern_id="llm:silu_mul_mlp", env_flag="Q_FUSED_SILU"),
            ]
        )

        safe = _safe_artifact_id(combined.pattern_id)

        assert ":" in combined.pattern_id, "precondition: raw id carries the colon"
        assert ":" not in safe
        assert "+" not in safe
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", safe)
        # Still recognizable: an artifact log must be attributable to its recipe.
        assert "qk_norm_rope" in safe

    def test_length_is_bounded_and_truncation_does_not_collide(self):
        # Combined ids share long prefixes, so plain truncation would alias them.
        common = "llm:" + "very_long_candidate_name_" * 8
        first = _safe_artifact_id(common + "alpha")
        second = _safe_artifact_id(common + "beta")

        assert len(first) <= 80 and len(second) <= 80
        assert first != second

    def test_rejects_path_traversal_and_separators(self):
        for raw in ("../../etc/passwd", "llm:a/b", "llm:..", "", "   "):
            safe = _safe_artifact_id(raw)
            assert "/" not in safe
            assert safe not in {"", ".", ".."}
            assert re.fullmatch(r"[A-Za-z0-9_.-]+", safe)

    def test_authoring_writes_artifacts_under_sanitized_names(self, tmp_path, monkeypatch):
        """Locks the call site, not just the helper: both the prompt dump and the
        author log must go through sanitization."""
        combined = _combined_recipe(
            [
                _recipe(pattern_id="llm:qk_norm_rope", env_flag="Q_FUSED_QK"),
                _recipe(pattern_id="llm:add_rmsnorm_input", env_flag="Q_FUSED_ADD"),
            ]
        )
        from kernelforge.fusion import campaign as campaign_module

        monkeypatch.setattr(
            campaign_module.subprocess,
            "Popen",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no forge-loop here")),
        )
        campaign_module.run_recipe_campaign(
            combined,
            workspace=str(tmp_path),
            harness_path=str(tmp_path / "kernel_harness.py"),
            output_dir=str(tmp_path),
        )

        written = [p.name for p in tmp_path.iterdir() if p.is_file()]
        assert written, "the campaign wrote no artifact"
        for name in written:
            assert ":" not in name
            assert "+" not in name
            assert len(name) <= 120


class TestScopedEmit:
    def test_patch_scoped_to_fusion_files_only(self, tmp_path):
        repo = tmp_path / "sglang"
        mdir = repo / "python" / "sglang" / "srt" / "models"
        mdir.mkdir(parents=True)
        (mdir / "lfm2.py").write_text("# eager\n", encoding="utf-8")
        (repo / "unrelated.py").write_text("# pre-existing\n", encoding="utf-8")
        for args in (
            ["init", "-q"],
            ["add", "-A"],
            ["-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-qm", "base"],
        ):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
        # Simulate a fusion: edit lfm2.py + add a new fused module + dirty an
        # UNRELATED tracked file (must NOT appear in the scoped patch).
        (mdir / "lfm2.py").write_text("# eager\n# fused edit\n", encoding="utf-8")
        (mdir / "lfm2_fused.py").write_text("# triton kernel\n", encoding="utf-8")
        (repo / "unrelated.py").write_text("# pre-existing\n# dirty\n", encoding="utf-8")

        out = tmp_path / "out"
        arts = export_artifacts(str(repo), str(mdir / "lfm2.py"), out)
        paths = {c["path"] for c in arts.changes}
        assert any(p.endswith("models/lfm2.py") for p in paths)
        assert any(p.endswith("models/lfm2_fused.py") for p in paths)
        assert not any("unrelated.py" in p for p in paths)  # scoped out
        patch_text = (out / "fusion.patch").read_text()
        assert "unrelated.py" not in patch_text
        cached = subprocess.run(
            ["git", "-C", str(repo), "diff", "--cached", "--name-only"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert cached == ""

        restore_exported_changes(str(repo), arts)
        assert (mdir / "lfm2.py").read_text(encoding="utf-8") == "# eager\n"
        assert not (mdir / "lfm2_fused.py").exists()
        assert (out / "fusion.patch").is_file()
