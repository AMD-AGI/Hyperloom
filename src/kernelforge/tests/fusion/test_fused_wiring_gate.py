# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The wiring gate: a fused module nothing calls is not a fusion."""

from pathlib import Path
from types import SimpleNamespace

from kernelforge.fusion import command as command_mod
from kernelforge.fusion.validate import SmokeVerdict
from kernelforge.fusion.validate import fused_symbol_invocation_evidence as evidence

_IMPORT_ONLY = """\
import os

logger = None

if os.environ.get("QWEN3_FUSED_QKNORM_ROPE_KVCACHE", "0") != "0":
    try:
        from vllm.model_executor.models.qwen3_fused_llm_qknorm_rope_kvcache import (
            fused_qknorm_rope_kvcache,  # noqa: F401
        )
    except Exception:
        pass


class Qwen3Attention:
    def forward(self, hidden_states, positions):
        qkv, _ = self.qkv_proj(hidden_states)
        return self.unfused(qkv, positions)
"""


def _write(tmp_path: Path, text: str, name: str = "qwen3.py") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_import_only_wiring_is_rejected(tmp_path):
    result = evidence(_write(tmp_path, _IMPORT_ONLY))
    assert result.verdict == "not_wired"
    assert "never references it" in result.reason
    assert "fused_qknorm_rope_kvcache" in result.reason


def test_a_call_site_in_the_forward_path_passes(tmp_path):
    text = _IMPORT_ONLY.replace(
        "        return self.unfused(qkv, positions)",
        "        return fused_qknorm_rope_kvcache(qkv, positions)",
    )
    result = evidence(_write(tmp_path, text))
    assert result.verdict == "wired"
    assert "fused_qknorm_rope_kvcache" in result.reason


def test_a_lazy_import_inside_the_call_site_passes(tmp_path):
    """Importing inside ``forward`` is a legitimate wiring style, not a miss."""
    text = """\
class Qwen3Attention:
    def forward(self, qkv, positions):
        from vllm.model_executor.models.qwen3_fused_x import fused_chain

        return fused_chain(qkv, positions)
"""
    assert evidence(_write(tmp_path, text)).verdict == "wired"


def test_module_alias_call_passes(tmp_path):
    text = """\
import vllm.model_executor.models.qwen3_fused_x as fx


def forward(qkv):
    return fx.fused_chain(qkv)
"""
    assert evidence(_write(tmp_path, text)).verdict == "wired"


def test_a_source_with_no_fused_import_is_not_judged(tmp_path):
    """No fused import is not evidence of a defect -- a fusion can be inline."""
    result = evidence(_write(tmp_path, "def forward(x):\n    return fused_norm(x)\n"))
    assert result.verdict == "unchecked"
    assert "imports no fused-kernel module" in result.reason


def test_the_gate_fails_open_when_it_cannot_inspect(tmp_path):
    """It demotes a provable defect, never a KEEP it could not read."""
    for source in (str(tmp_path / "missing.py"), _write(tmp_path, "def broken(:\n"), ""):
        assert evidence(source).verdict == "unchecked"


def test_an_unrelated_diffusion_module_is_not_mistaken_for_a_fusion(tmp_path):
    """``_is_fused_module_name`` excludes mid-word matches; rely on that here."""
    text = "from vllm.models.diffusion import unet  # noqa: F401\n"
    result = evidence(_write(tmp_path, text))
    assert result.verdict == "unchecked"


class TestTheArtifactRepeatsTheVerdict:
    """A KEEP note says whether the wiring was proved, not just that nothing was proved against it."""

    def _smoke(self, tmp_path, monkeypatch, source_file: str):
        monkeypatch.setattr(
            command_mod,
            "serving_smoke_verdict",
            lambda *a, **k: SmokeVerdict(True, "decoded 128 tokens"),
        )
        recipe = SimpleNamespace(
            pattern_id="residual_add_rmsnorm",
            env_flag="LFM2_FUSED_RESIDUAL",
            source_file=source_file,
        )
        return command_mod._run_serving_smoke(
            recipe,
            base_note="MICRO KEEP 1.31x",
            framework="sglang",
            out=tmp_path,
            gpu="gfx950",
            model_path="/models/lfm2",
            isl=128,
            osl=128,
        )

    def test_an_unchecked_wiring_is_named_in_the_note(self, tmp_path, monkeypatch):
        source = _write(tmp_path, "def forward(x):\n    return fused_norm(x)\n")

        disposition, note, blame = self._smoke(tmp_path, monkeypatch, source)

        assert (disposition, blame) == ("ok", "")
        # The gate could not read a call site here; recording that as confirmed wiring is how an INLINE fusion and a
        # wiring edit that was never made became the same artifact.
        assert "WIRING UNCHECKED" in note
        assert "imports no fused-kernel module" in note

    def test_a_proved_call_site_leaves_the_note_alone(self, tmp_path, monkeypatch):
        source = _write(
            tmp_path,
            _IMPORT_ONLY.replace(
                "        return self.unfused(qkv, positions)",
                "        return fused_qknorm_rope_kvcache(qkv, positions)",
            ),
        )

        _disposition, note, _blame = self._smoke(tmp_path, monkeypatch, source)

        assert "WIRING UNCHECKED" not in note
        assert note == "MICRO KEEP 1.31x | SERVING SMOKE OK"
