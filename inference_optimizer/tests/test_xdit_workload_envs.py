# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""xDiT offline config materialization + default config resolution."""

from __future__ import annotations

import yaml

from inference_optimizer.orchestrator.action_executors._workload_envs import (
    default_baseline_config,
    materialize_config_with_envs,
)
from inference_optimizer.orchestrator.action_executors.profile import (
    _default_profile_config,
)


def _write_xdit_template(path):
    path.write_text(
        yaml.safe_dump(
            {"benchmark": {"framework": "xdit", "model": "", "run_mode": "local"}}
        ),
        encoding="utf-8",
    )


class TestDefaultConfigResolution:
    def test_baseline_config_resolves_xdit(self, monkeypatch):
        monkeypatch.setenv("FRAMEWORK", "xdit")
        cfg = default_baseline_config()
        assert cfg.name == "baseline_xdit.yaml"
        assert cfg.exists(), "shipped baseline_xdit.yaml must exist"

    def test_profile_config_resolves_xdit(self, monkeypatch):
        monkeypatch.setenv("FRAMEWORK", "xdit")
        cfg = _default_profile_config()
        assert cfg.name == "profile_xdit.yaml"
        assert cfg.exists(), "shipped profile_xdit.yaml must exist"

    def test_shipped_xdit_configs_are_offline(self):
        for name in ("baseline_xdit.yaml", "profile_xdit.yaml"):
            path = (
                default_baseline_config().parent / name
            )
            data = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmark"]
            assert data["framework"] == "xdit"
            assert data["run_mode"] == "local"


class TestXditMaterialization:
    def test_emits_run_cmd_and_drops_llm_block(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FRAMEWORK", "xdit")
        monkeypatch.setenv(
            "RUN_CMD",
            'xdit --model /models/flux --ulysses_degree 2 '
            '--num_inference_steps 50 --prompt "a cat"',
        )
        src = tmp_path / "cfg.yaml"
        _write_xdit_template(src)

        out = materialize_config_with_envs(
            src, tmp_path / "out", model_path="/models/flux", gpu_type="mi300x"
        )
        bench = yaml.safe_load(out.read_text())["benchmark"]

        assert bench["framework"] == "xdit"
        assert bench["run_mode"] == "local"
        assert bench["run_cmd"].startswith("xdit --model /models/flux")
        assert bench["runner_type"] == "mi300x"
        # No LLM serving knobs / InferenceX coupling leaked in.
        assert "envs" not in bench
        assert "inferencex_path" not in bench
        assert "benchmark_script" not in bench

    def test_model_path_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FRAMEWORK", "xdit")
        monkeypatch.setenv("RUN_CMD", "xdit --model /models/flux")
        src = tmp_path / "cfg.yaml"
        _write_xdit_template(src)
        out = materialize_config_with_envs(
            src, tmp_path / "out", model_path="/models/flux"
        )
        bench = yaml.safe_load(out.read_text())["benchmark"]
        assert bench["model"] == "/models/flux"

    def test_extra_envs_passed_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FRAMEWORK", "xdit")
        monkeypatch.setenv("RUN_CMD", "xdit --model /models/flux")
        src = tmp_path / "cfg.yaml"
        _write_xdit_template(src)
        out = materialize_config_with_envs(
            src,
            tmp_path / "out",
            model_path="/models/flux",
            extra_envs={"HF_HOME": "/hf"},
        )
        bench = yaml.safe_load(out.read_text())["benchmark"]
        assert bench["envs"]["HF_HOME"] == "/hf"

    def test_offline_branch_does_not_require_inferencex(self, tmp_path, monkeypatch):
        """Offline materialization must not set inferencex_path even if the env
        var is present (online-only concern)."""
        monkeypatch.setenv("FRAMEWORK", "xdit")
        monkeypatch.setenv("RUN_CMD", "xdit --model /models/flux")
        monkeypatch.setenv("INFERENCEX_PATH", "/somewhere/InferenceX")
        src = tmp_path / "cfg.yaml"
        _write_xdit_template(src)
        out = materialize_config_with_envs(
            src, tmp_path / "out", model_path="/models/flux"
        )
        bench = yaml.safe_load(out.read_text())["benchmark"]
        assert "inferencex_path" not in bench


class TestOnlineMaterializationUnaffected:
    def test_sglang_still_gets_llm_envs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FRAMEWORK", "sglang")
        monkeypatch.delenv("RUN_CMD", raising=False)
        src = tmp_path / "cfg.yaml"
        src.write_text(
            yaml.safe_dump(
                {"benchmark": {"framework": "sglang", "model": "/m", "envs": {}}}
            ),
            encoding="utf-8",
        )
        out = materialize_config_with_envs(
            src, tmp_path / "out", model_path="/m", gpu_type="mi300x"
        )
        bench = yaml.safe_load(out.read_text())["benchmark"]
        assert "envs" in bench
        assert bench["benchmark_script"] == "sglang_mi300x.sh"
