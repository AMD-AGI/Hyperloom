# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the framework capability registry (xDiT support)."""

from __future__ import annotations

import pytest

from inference_optimizer import framework_registry as fr


class TestSupportedFrameworks:
    def test_known_frameworks_registered(self):
        supported = fr.supported_frameworks()
        for fw in ("sglang", "vllm", "atom", "xdit"):
            assert fw in supported

    @pytest.mark.parametrize("fw", ["xdit", "XDIT", "xDiT", " xdit "])
    def test_is_supported_case_and_whitespace_insensitive(self, fw):
        assert fr.is_supported(fw) is True

    def test_unknown_not_supported(self):
        assert fr.is_supported("nope") is False
        assert fr.is_supported("") is False


class TestCapabilities:
    def test_xdit_capabilities(self):
        caps = fr.get_capabilities("xdit")
        assert caps.name == "xdit"
        assert caps.serving == "offline"
        assert caps.has_server is False
        assert caps.launch == "run_cmd"
        assert caps.modality == "diffusion"
        assert caps.kpi == "latency_per_image"
        assert caps.accuracy_gate == "image_diff"
        assert caps.search_knobs == ()  # no parameter sweeping
        assert caps.is_offline is True
        assert caps.is_online is False

    @pytest.mark.parametrize("fw", ["sglang", "vllm", "atom"])
    def test_llm_frameworks_are_online_servers(self, fw):
        caps = fr.get_capabilities(fw)
        assert caps.serving == "online"
        assert caps.has_server is True
        assert caps.launch == "server"
        assert caps.modality == "text"
        assert caps.kpi == "output_throughput"
        assert caps.accuracy_gate == "lm_eval"
        assert caps.is_online is True
        assert caps.is_offline is False

    @pytest.mark.parametrize("fw", ["xdit", "XDIT", "xDiT", " xdit "])
    def test_get_capabilities_case_and_whitespace_insensitive(self, fw):
        assert fr.get_capabilities(fw).name == "xdit"

    @pytest.mark.parametrize("fw", ["nope", "", "  "])
    def test_unset_or_unknown_falls_back_to_llm_default(self, fw):
        """Hyperloom is LLM-first: an unset/unknown framework must behave as the
        text-serving default so diffusion additions never perturb LLM paths."""
        caps = fr.get_capabilities(fw)
        assert caps.name == "sglang"
        assert caps.modality == "text"
        assert caps.accuracy_gate == "lm_eval"
        assert caps.is_diffusion is False
        assert caps.gates_on_image_diff is False
        assert caps.supports_param_search is True
        assert caps.supports_conc_sweep is True


class TestSemanticProperties:
    """The capability fields that call sites gate on (not has_server)."""

    def test_xdit_is_diffusion_llms_are_not(self):
        assert fr.get_capabilities("xdit").is_diffusion is True
        for fw in ("sglang", "vllm", "atom"):
            assert fr.get_capabilities(fw).is_diffusion is False

    def test_image_diff_gating(self):
        assert fr.get_capabilities("xdit").gates_on_image_diff is True
        for fw in ("sglang", "vllm", "atom"):
            assert fr.get_capabilities(fw).gates_on_image_diff is False

    def test_param_search_support(self):
        # xdit has empty search_knobs => nothing to sweep.
        assert fr.get_capabilities("xdit").supports_param_search is False
        for fw in ("sglang", "vllm", "atom"):
            caps = fr.get_capabilities(fw)
            assert caps.supports_param_search is True
            assert caps.supports_conc_sweep is True

    def test_xdit_has_no_conc_sweep(self):
        assert fr.get_capabilities("xdit").supports_conc_sweep is False
