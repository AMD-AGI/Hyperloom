# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The wiring gate: a fused module nothing calls is not a fusion.

Taken from a real forge-fuse run on Qwen3-14B-FP8 that reported a 37.16x
microbench, SNR 52.1 dB and ``SERVING SMOKE OK`` for a patch whose entire
framework edit was a flag-gated ``# noqa: F401`` import. The fused kernel never
executed: the harness had timed the entry point directly, and the smoke had
booted stock code with the flag set.
"""

from pathlib import Path

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
    wired, reason = evidence(_write(tmp_path, _IMPORT_ONLY))
    assert wired is False
    assert "never references it" in reason
    assert "fused_qknorm_rope_kvcache" in reason


def test_a_call_site_in_the_forward_path_passes(tmp_path):
    text = _IMPORT_ONLY.replace(
        "        return self.unfused(qkv, positions)",
        "        return fused_qknorm_rope_kvcache(qkv, positions)",
    )
    wired, reason = evidence(_write(tmp_path, text))
    assert wired is True
    assert "fused_qknorm_rope_kvcache" in reason


def test_a_lazy_import_inside_the_call_site_passes(tmp_path):
    """Importing inside ``forward`` is a legitimate wiring style, not a miss."""
    text = """\
class Qwen3Attention:
    def forward(self, qkv, positions):
        from vllm.model_executor.models.qwen3_fused_x import fused_chain

        return fused_chain(qkv, positions)
"""
    assert evidence(_write(tmp_path, text))[0] is True


def test_module_alias_call_passes(tmp_path):
    text = """\
import vllm.model_executor.models.qwen3_fused_x as fx


def forward(qkv):
    return fx.fused_chain(qkv)
"""
    assert evidence(_write(tmp_path, text))[0] is True


def test_a_source_with_no_fused_import_is_not_judged(tmp_path):
    """No fused import is not evidence of a defect -- a fusion can be inline.

    ``test_smoke_salvage_contract`` builds exactly that shape: the fused call
    written straight into the framework file with nothing imported. Only a
    bound-and-unused import is provable, so every other shape fails open.
    """
    wired, reason = evidence(_write(tmp_path, "def forward(x):\n    return fused_norm(x)\n"))
    assert wired is True
    assert "unchecked" in reason


def test_the_gate_fails_open_when_it_cannot_inspect(tmp_path):
    """It demotes a provable defect, never a KEEP it could not read."""
    assert evidence(str(tmp_path / "missing.py"))[0] is True
    assert evidence(_write(tmp_path, "def broken(:\n"))[0] is True
    assert evidence("")[0] is True


def test_an_unrelated_diffusion_module_is_not_mistaken_for_a_fusion(tmp_path):
    """``_is_fused_module_name`` excludes mid-word matches; rely on that here."""
    text = "from vllm.models.diffusion import unet  # noqa: F401\n"
    wired, reason = evidence(_write(tmp_path, text))
    assert wired is True
    assert "unchecked" in reason
