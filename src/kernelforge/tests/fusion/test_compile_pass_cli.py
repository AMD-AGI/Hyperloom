# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI-level contract for claiming a framework compile pass.

Covers the guarantees that cannot be shown by unit-testing text replacement: the
edited install is always restored, a patch is only exported when a same-shape
disabled/enabled A/B actually paid off, pre-existing edits are neither lost nor
smuggled into the patch, and the manifest says which of those happened.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from kernelforge.fusion.command import main
from kernelforge.fusion.vllm_passes import PassState

FLAG = "enable_qk_norm_rope_fusion"
CONFIG_BODY = (
    "@config\n"
    "class PassConfig:\n"
    "    fuse_norm_quant: bool = None  # type: ignore[assignment]\n"
    f"    {FLAG}: bool = None  # type: ignore[assignment]\n"
    '    """Enable fused Q/K RMSNorm + RoPE pass."""\n'
)
MODEL_BODY = (
    "class Qwen3Attention:\n"
    "    def forward(self, positions, hidden_states):\n"
    "        q_norm = self.q_norm(q)\n"
    "        k_norm = self.k_norm(k)\n"
    "        return self.rotary_emb(positions, q_norm, k_norm)\n"
)


def _trace(path, add_dur: int = 4):
    """Launch-bound decode trace whose rmsnorm+rope share triggers qk_norm_rope.

    A bigger ``add_dur`` additionally triggers the residual add+rmsnorm patterns,
    giving a run BOTH a compile_pass and authoring candidates.
    """
    events = []
    ts = 0
    for _ in range(4):
        events += [
            {"cat": "kernel", "name": "Cijk_gemm", "ts": ts, "dur": 30},
            {"cat": "kernel", "name": "rms_norm_kernel", "ts": ts + 4000, "dur": 14},
            {"cat": "kernel", "name": "rotary_embedding_kernel", "ts": ts + 8000, "dur": 9},
            {"cat": "kernel", "name": "vectorized_elementwise add", "ts": ts + 12000, "dur": add_dur},
        ]
        ts += 20000
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")


class Harness:
    """Fake vLLM install plus stubbed probe / serving, driven from one place."""

    def __init__(self, tmp_path):
        self.root = tmp_path / "fw"
        cfg_dir = self.root / "vllm" / "config"
        mdl_dir = self.root / "vllm" / "model_executor" / "models"
        cfg_dir.mkdir(parents=True)
        mdl_dir.mkdir(parents=True)
        for pkg in (self.root / "vllm", cfg_dir, self.root / "vllm" / "model_executor", mdl_dir):
            (pkg / "__init__.py").write_text("", encoding="utf-8")
        self.config_file = cfg_dir / "compilation.py"
        self.config_file.write_text(CONFIG_BODY, encoding="utf-8")
        (mdl_dir / "qwen3.py").write_text(MODEL_BODY, encoding="utf-8")

        self.model = tmp_path / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "hidden_size": 4096, "num_attention_heads": 32}), encoding="utf-8"
        )
        self.trace = tmp_path / "decode.trace.json"
        _trace(self.trace)
        self.out = tmp_path / "out"

        # Serving arms: baseline then enabled. Overridden per test.
        self.arms = [(True, 100.0), (True, 110.0)]
        self.activation = True
        self.enabled_after_edit = True
        self.arm_calls: list[dict] = []

    # -- stubs ------------------------------------------------------------
    def probe(self, flags, **kw):
        return {
            f: PassState(flag=f, present=True, enabled=False, source="default", config_file=str(self.config_file))
            for f in flags
        }

    def serving(self, model_path, env_flags, **kw):
        idx = len(self.arm_calls)
        self.arm_calls.append({"env_flags": dict(env_flags or {}), "kw": kw})
        ok, tok_s = self.arms[min(idx, len(self.arms) - 1)]
        metrics = kw.get("metrics")
        if metrics is not None and ok:
            metrics.update({"tok_s": tok_s, "output_tokens": 100, "seconds": 1.0})
            if idx > 0:
                metrics["pass_activated"] = self.activation
                metrics["activation_evidence"] = ["Fused QK Norm+RoPE on 1 sites"]
        return (ok, "ok" if ok else "server crashed at startup: boom")

    def verify(self, flag, **kw):
        return PassState(
            flag=flag,
            present=True,
            enabled=self.enabled_after_edit,
            source="default",
            config_file=str(self.config_file),
        )

    def install(self, monkeypatch):
        import kernelforge.fusion.command as cli
        import kernelforge.fusion.locate as locate

        rt = cli.TargetRuntime(
            framework="vllm", python="/fake/python", launcher_exe="/fake/vllm", require_root=str(self.root)
        )
        monkeypatch.setattr(locate, "probe_pass_states", self.probe)
        monkeypatch.setattr(locate, "resolve_target_runtime", lambda *a, **k: rt)
        monkeypatch.setattr(cli, "resolve_target_runtime", lambda *a, **k: rt)
        monkeypatch.setattr(cli, "serving_smoke", self.serving)
        monkeypatch.setattr(cli, "verify_pass_enabled", self.verify)
        return self

    def make_mixed(self):
        """Also surface an authoring candidate, so the run has BOTH kinds."""
        mdl = self.root / "vllm" / "model_executor" / "models" / "qwen3.py"
        mdl.write_text(
            MODEL_BODY + "        hidden_states = hidden_states + residual\n"
            "        x = self.input_layernorm(hidden_states)\n",
            encoding="utf-8",
        )
        _trace(self.trace, add_dur=20)
        return self

    def run(self, *extra):
        return CliRunner().invoke(
            main,
            [
                "--trace",
                str(self.trace),
                "--model-path",
                str(self.model),
                "--framework",
                "vllm",
                "--framework-root",
                str(self.root),
                "--output-dir",
                str(self.out),
                *extra,
            ],
        )

    # -- assertions -------------------------------------------------------
    @property
    def manifest(self):
        return json.loads((self.out / "fusion_manifest.json").read_text())

    def assert_config_restored(self):
        assert self.config_file.read_text(encoding="utf-8") == CONFIG_BODY, "the live framework file was left modified"


@pytest.fixture
def hz(tmp_path, monkeypatch):
    return Harness(tmp_path).install(monkeypatch)


def test_compile_pass_recipe_is_selected(hz):
    res = hz.run("--dry-run")
    assert res.exit_code == 0, res.output
    assert hz.manifest["fusion"]["candidate_kind"] == "compile_pass"


class TestKeepPath:
    def test_ab_win_exports_patch_and_restores_the_install(self, hz):
        res = hz.run()
        assert res.exit_code == 0, res.output
        cp = hz.manifest["compile_pass"]
        assert cp["kept"] is True and cp["validated"] is True
        assert cp["baseline_tok_s"] == 100.0 and cp["enabled_tok_s"] == 110.0
        assert cp["speedup"] == pytest.approx(1.1)
        assert cp["reverted"] is True
        # Patch shipped, and the install is byte-identical again.
        patch = (hz.out / "fusion.patch").read_text()
        assert f"-    {FLAG}: bool = None" in patch
        assert f"+    {FLAG}: bool = True" in patch
        hz.assert_config_restored()
        # CLI must report ITS verdict, not a null that reads as "never validated".
        echoed = json.loads(res.output.strip().splitlines()[-1])
        assert echoed["kept"] is True and echoed["speedup"] == pytest.approx(1.1)

    def test_enabled_arm_runs_with_debug_logging_for_activation_evidence(self, hz):
        hz.run()
        assert hz.arm_calls[0]["env_flags"] == {}
        assert hz.arm_calls[1]["env_flags"].get("VLLM_LOGGING_LEVEL") == "DEBUG"
        # Both arms must hit the SAME pinned launcher and request shape.
        assert {c["kw"]["launcher_exe"] for c in hz.arm_calls} == {"/fake/vllm"}
        assert {(c["kw"]["isl"], c["kw"]["osl"]) for c in hz.arm_calls} == {(512, 128)}


class TestRejectPaths:
    """Every rejection must restore the install and export nothing."""

    def _assert_rejected(self, hz, res, needle):
        assert res.exit_code == 0, res.output
        cp = hz.manifest["compile_pass"]
        assert cp["kept"] is False, cp
        assert needle in cp["note"], cp["note"]
        assert hz.manifest["artifacts"] is None
        assert not (hz.out / "fusion.patch").exists()
        hz.assert_config_restored()

    def test_no_op_change_is_rejected(self, hz):
        hz.arms = [(True, 100.0), (True, 100.4)]  # +0.4% < 3% target
        self._assert_rejected(hz, hz.run(), "not faster")

    def test_regression_is_rejected(self, hz):
        hz.arms = [(True, 100.0), (True, 88.0)]
        self._assert_rejected(hz, hz.run(), "not faster")

    def test_pass_that_matches_nothing_is_rejected(self, hz):
        hz.activation = False
        self._assert_rejected(hz, hz.run(), "matched NOTHING")

    def test_edit_that_does_not_change_resolved_config_is_rejected(self, hz):
        # The level-pinned case: the file changed, the runtime did not.
        hz.enabled_after_edit = False
        self._assert_rejected(hz, hz.run(), "would have no effect")

    def test_enabled_arm_crash_is_rejected(self, hz):
        hz.arms = [(True, 100.0), (False, 0.0)]
        self._assert_rejected(hz, hz.run(), "enabled arm failed")

    def test_baseline_arm_crash_is_rejected_before_any_edit(self, hz):
        hz.arms = [(False, 0.0)]
        self._assert_rejected(hz, hz.run(), "baseline (pass disabled) arm failed")

    def test_exception_mid_run_still_restores(self, hz, monkeypatch):
        import kernelforge.fusion.command as cli

        def boom(*a, **k):
            raise RuntimeError("export exploded")

        monkeypatch.setattr(cli, "export_artifacts", boom)
        res = hz.run()
        assert res.exit_code != 0
        hz.assert_config_restored()


class TestPreExistingEdits:
    def test_dirty_file_is_preserved_and_kept_out_of_the_patch(self, hz):
        dirty = CONFIG_BODY + "\n# operator's own local edit\n"
        hz.config_file.write_text(dirty, encoding="utf-8")
        res = hz.run()
        assert res.exit_code == 0, res.output
        assert hz.manifest["compile_pass"]["kept"] is True
        # Restored to the PRE-RUN bytes, not to some pristine upstream copy.
        assert hz.config_file.read_text(encoding="utf-8") == dirty
        patch = (hz.out / "fusion.patch").read_text()
        # It may appear as diff CONTEXT, but must never be claimed as part of the
        # change (that is what would smuggle an unrelated edit downstream).
        changed = [ln for ln in patch.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
        assert not any("operator's own local edit" in ln for ln in changed), changed
        assert f"+    {FLAG}: bool = True" in patch
        assert len(changed) == 2, changed  # exactly the one-line flip


class TestNoValidate:
    def test_no_validate_keeps_the_confirmed_edit_but_says_no_ab_ran(self, hz):
        res = hz.run("--no-validate")
        assert res.exit_code == 0, res.output
        cp = hz.manifest["compile_pass"]
        assert cp["kept"] is True and cp["validated"] is False
        assert cp["speedup"] is None and "NO serving A/B" in cp["note"]
        assert hz.arm_calls == []  # no server was booted
        hz.assert_config_restored()

    def test_no_validate_still_refuses_an_edit_with_no_effect(self, hz):
        hz.enabled_after_edit = None  # undecidable after the edit
        res = hz.run("--no-validate")
        assert res.exit_code == 0, res.output
        assert hz.manifest["compile_pass"]["kept"] is False
        hz.assert_config_restored()


class TestFuseAllCombination:
    def test_mixed_candidates_claim_the_compile_pass_first(self, hz):
        """One run cannot do both, and the flag is on by default.

        Refusing would fail a run the caller has no way to fix, so the cheaper
        claim goes first and the authored candidates wait for a later round.
        """
        hz.make_mixed()
        res = hz.run("--fuse-all-confirmed")
        assert res.exit_code == 0, res.output
        assert hz.manifest["compile_pass"]["flag"]
        # Deferred, not dropped: the manifest still lists what was located.
        kinds = {c.get("candidate_kind") for c in hz.manifest["fusion_candidates"]}
        assert kinds == {"compile_pass", "new_fusion"}
        hz.assert_config_restored()
