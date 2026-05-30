"""Tests for capability detection."""

from hyperloom.capabilities import Capabilities, detect_capabilities


class TestCapabilities:
    def test_summary_format(self):
        caps = Capabilities(magpie=True, geak=False, torch_profiler=True)
        summary = caps.summary()
        assert "magpie: available" in summary
        assert "geak: not found" in summary
        assert "torch_profiler: available" in summary

    def test_detect_returns_capabilities(self):
        caps = detect_capabilities()
        assert isinstance(caps, Capabilities)
        assert isinstance(caps.magpie, bool)
        assert isinstance(caps.geak, bool)
