# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The CLI contract for fusing around a kernel the operator named."""

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

from kernelforge.fusion import command as cli_module
from kernelforge.fusion.command import main
from kernelforge.fusion.models import Diagnosis
from kernelforge.fusion.report import ANCHOR_REPORT_NAME, ANCHOR_RESOLVED_VERDICT

ANCHOR = (
    "void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda"
    "(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >"
    "(int, std::array<char*, 2ul>)"
)


def _write_trace(path, events):
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")


def _kernel(name, ts, dur=4.0):
    return {"cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"device": 0, "stream": 7}}


class Harness:
    """A launch-bound trace, a model config, and a model source, wired for the CLI."""

    def __init__(self, tmp_path):
        self.trace = tmp_path / "decode.trace.json"
        events = []
        for i in range(6):
            events += [
                _kernel(f"hgemm_bf16_SPK{i % 3}", i * 1000, 30.0),
                _kernel(ANCHOR, i * 1000 + 100, 4.0),
                _kernel("rms_norm_kernel", i * 1000 + 200, 14.0),
                _kernel("vectorized_elementwise CUDAFunctor_add", i * 1000 + 300, 10.0),
            ]
        _write_trace(self.trace, events)

        self.model = tmp_path / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(
            json.dumps({"model_type": "toylm", "hidden_size": 2048, "num_attention_heads": 16}),
            encoding="utf-8",
        )
        self.source = tmp_path / "toylm.py"
        self.source.write_text("class Attn:\n    def forward(self, x):\n        return x\n", encoding="utf-8")
        self.out = tmp_path / "out"

    def run(self, *extra):
        return CliRunner().invoke(
            main,
            [
                "--trace",
                str(self.trace),
                "--model-path",
                str(self.model),
                "--framework",
                "sglang",
                "--output-dir",
                str(self.out),
                *extra,
            ],
        )

    @property
    def manifest(self):
        return json.loads((self.out / "fusion_manifest.json").read_text())

    @property
    def anchor(self):
        return json.loads((self.out / ANCHOR_REPORT_NAME).read_text())


class TestDryRun:
    def test_resolution_is_published_without_reaching_an_agent(self, tmp_path, monkeypatch):
        hz = Harness(tmp_path)

        def refuse(*_args, **_kwargs):
            raise AssertionError("a dry run must not create an agent backend")

        monkeypatch.setattr(cli_module, "_create_agent_backend", refuse)

        res = hz.run("--dry-run", "--fuse-kernel", ANCHOR)

        assert res.exit_code == 0, res.output
        assert hz.anchor["occurrences"] == 6
        assert hz.anchor["signature"] == "gemm -> cast -> rmsnorm"
        assert hz.anchor["name"] == ANCHOR

    def test_a_resolved_anchor_is_not_a_no_opportunity_verdict(self, tmp_path):
        """Nothing was looked at yet, so the run has no opinion about what to fuse."""
        hz = Harness(tmp_path)

        res = hz.run("--dry-run", "--fuse-kernel", ANCHOR)

        assert res.exit_code == 0, res.output
        assert hz.manifest["verdict"] == ANCHOR_RESOLVED_VERDICT
        assert json.loads(res.output.strip().splitlines()[-1])["verdict"] == ANCHOR_RESOLVED_VERDICT

    def test_the_manifest_carries_the_anchor(self, tmp_path):
        hz = Harness(tmp_path)

        hz.run("--dry-run", "--fuse-kernel", ANCHOR)

        anchor = hz.manifest["anchor"]
        assert anchor["category"] == "cast"
        assert anchor["before"]["category"] == "gemm"
        assert anchor["after"]["category"] == "rmsnorm"

    def test_an_unnamed_run_carries_no_anchor(self, tmp_path):
        """The default path is untouched: no anchor key content, no anchor artifact."""
        hz = Harness(tmp_path)

        res = hz.run("--dry-run")

        assert res.exit_code == 0, res.output
        assert hz.manifest["anchor"] is None
        assert not (hz.out / ANCHOR_REPORT_NAME).exists()


class TestUsage:
    def test_a_name_absent_from_the_trace_is_a_usage_error(self, tmp_path):
        hz = Harness(tmp_path)

        res = hz.run("--dry-run", "--fuse-kernel", "kernel_that_never_ran")

        assert res.exit_code != 0
        assert "no kernel" in res.output

    def test_a_fragment_is_refused_with_the_full_name(self, tmp_path):
        hz = Harness(tmp_path)

        res = hz.run("--dry-run", "--fuse-kernel", "bfloat16tofloat32_copy_kernel_cuda")

        assert res.exit_code != 0
        assert "full name is required" in res.output

    def test_anchored_mode_without_a_kernel_is_refused(self, tmp_path):
        hz = Harness(tmp_path)

        res = hz.run("--dry-run", "--discover", "anchored")

        assert res.exit_code != 0
        assert "requires --fuse-kernel" in res.output

    def test_naming_a_kernel_selects_anchored_mode(self, tmp_path, monkeypatch):
        """--fuse-kernel implies the mode, so the two cannot disagree."""
        hz = Harness(tmp_path)
        seen: dict = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(cli_module, "discover_anchored_recipes", spy)
        monkeypatch.setattr(
            cli_module,
            "_create_agent_backend",
            lambda *_a: SimpleNamespace(name="codex", runtime=SimpleNamespace(model="m", sandbox_mode="bypass")),
        )
        monkeypatch.setattr(cli_module, "registered_agent_llm_fn", lambda *_a, **_k: lambda _p: "[]")
        monkeypatch.setattr(
            cli_module, "resolve_framework_source_file", lambda *a, **k: (str(hz.source), "path convention")
        )

        res = hz.run("--fuse-kernel", ANCHOR, "--no-author", "--no-validate", "--discover", "patterns")

        assert res.exit_code == 0, res.output
        assert seen["report"].name == ANCHOR


class TestDiscovery:
    def _stub_agent(self, monkeypatch, hz, payload):
        monkeypatch.setattr(
            cli_module,
            "_create_agent_backend",
            lambda *_a: SimpleNamespace(name="codex", runtime=SimpleNamespace(model="m", sandbox_mode="bypass")),
        )
        captured: dict = {}

        def fake_agent(_backend, **_kwargs):
            def _fn(prompt):
                captured["prompt"] = prompt
                return payload

            return _fn

        monkeypatch.setattr(cli_module, "registered_agent_llm_fn", fake_agent)
        monkeypatch.setattr(
            cli_module, "resolve_framework_source_file", lambda *a, **k: (str(hz.source), "path convention")
        )
        return captured

    def test_a_non_candidate_trace_still_runs_when_a_kernel_was_named(self, tmp_path, monkeypatch):
        """The operator naming a kernel overrides a trace-wide 'nothing to fuse' verdict."""
        hz = Harness(tmp_path)
        captured = self._stub_agent(monkeypatch, hz, "[]")
        monkeypatch.setattr(
            cli_module,
            "diagnose_trace",
            lambda *a, **k: Diagnosis(0.0, 0.9, [], 0.0, {}, False, "compute dominated"),
        )

        res = hz.run("--fuse-kernel", ANCHOR, "--no-author", "--no-validate")

        assert res.exit_code == 0, res.output
        assert hz.manifest["diagnosis"]["is_candidate"] is False
        assert "prompt" in captured, "discovery must run even on a non-candidate trace"

    def test_the_prompt_carries_the_untruncated_anchor(self, tmp_path, monkeypatch):
        hz = Harness(tmp_path)
        captured = self._stub_agent(monkeypatch, hz, "[]")

        hz.run("--fuse-kernel", ANCHOR, "--no-author", "--no-validate")

        assert len(ANCHOR) > 90
        assert ANCHOR in captured["prompt"]
        assert "gemm -> cast -> rmsnorm" in captured["prompt"]


def test_the_campaign_is_handed_a_concrete_agent_provider(tmp_path, monkeypatch):
    """forge-loop rejects 'auto'; forwarding this command's own spelling makes the two disagree."""
    hz = Harness(tmp_path)
    seen: dict = {}

    def spy(*_args, **kwargs):
        seen.update(kwargs)
        raise cli_module.FusionAbort("stop after the call is made")

    monkeypatch.setattr(
        cli_module,
        "_create_agent_backend",
        lambda *_a: SimpleNamespace(name="claude", runtime=SimpleNamespace(model="m", sandbox_mode="bypass")),
    )
    monkeypatch.setattr(cli_module, "registered_agent_llm_fn", lambda *_a, **_k: lambda _p: "[]")
    monkeypatch.setattr(
        cli_module, "resolve_framework_source_file", lambda *a, **k: (str(hz.source), "path convention")
    )
    monkeypatch.setattr(cli_module, "_resolve_agent_choice", lambda backend, model: ("claude", "m"))
    monkeypatch.setattr(cli_module, "_run_fusion_autoloop", spy)
    monkeypatch.setattr(cli_module, "build_recipes", lambda *a, **k: [_recipe(hz)])

    hz.run("--discover", "patterns")

    assert seen.get("agent_backend") == "claude", "the campaign must be handed a provider forge-loop accepts"


def _recipe(hz):
    from kernelforge.fusion.models import Recipe

    return Recipe(
        pattern_id="residual_add_rmsnorm",
        description="d",
        env_flag="TOYLM_FUSED",
        source_file=str(hz.source),
        source_hints=["forward"],
        fusion_math="m",
        eager_reference_hint="r",
        shapes={"hidden_size": 2048},
        matched_categories=["rmsnorm"],
        trigger_share=0.5,
    )
