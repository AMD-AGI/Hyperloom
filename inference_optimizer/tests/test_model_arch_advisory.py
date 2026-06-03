"""Advisory ``model_arch`` profile: loader, serialization, renderer, and
specialist warm-param injection.

Contracts pinned here (all advisory — ``model_arch`` drives NO
deterministic gating; it is prompt-context only and subordinate to live
TraceLens evidence):

* ``cli._load_model_arch`` reads ``<workspace_root>/model_arch.json``,
  soft-degrades to ``{}`` on missing / unreadable / invalid-JSON /
  non-dict / missing-or-mismatched ``model_name`` (stale-file guard).
* ``SharedState.model_arch`` round-trips through ``to_dict`` /
  ``from_dict`` so a ``--resume`` rehydrates the persisted profile.
* ``render_model_arch_compact`` drops empty fields, renders ``""`` for an
  empty / non-dict profile, and trails the free-text ``notes``.
* ``SharedState.to_prompt_summary`` injects the labeled advisory block
  when a profile is set and omits it entirely otherwise.
* ``Coordinator._warm_specialist_params`` populates ``arch_notes`` from
  ``SharedState.model_arch`` (and skips it when empty) so non-arch
  sessions render exactly as before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.cli import _load_model_arch
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import (
    SharedState,
    render_model_arch_compact,
)


_VALID_ARCH = {
    "model_name": "DeepSeek-R1-0528",
    "source": "gallery",
    "decoder_type": "Sparse MoE",
    "attention": "MLA",
    "layer_mix": "61 MLA",
    "kv_cache_per_token": "68.6 KiB",
    "active_params": "37B active / 671B total",
    "num_experts": 256,
    "experts_per_tok": 8,
    "mtp": True,
    "swa_window": None,
    "norm": "RMSNorm",
    "notes": "DeepSeek V3-style: dense prefix + shared expert + MTP-1 path",
}


def _write(workspace: Path, payload: Any) -> Path:
    p = workspace / "model_arch.json"
    p.write_text(
        payload if isinstance(payload, str) else json.dumps(payload),
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# 1. _load_model_arch — happy path + soft-degrade matrix
# ---------------------------------------------------------------------------
def test_load_model_arch_valid(tmp_path: Path):
    _write(tmp_path, _VALID_ARCH)
    out = _load_model_arch(tmp_path, "DeepSeek-R1-0528")
    assert out == _VALID_ARCH


def test_load_model_arch_matches_on_basename(tmp_path: Path):
    """The launched ``--model`` may be a full path; the guard compares
    basenames so a path / bare-name mismatch still matches."""
    _write(tmp_path, {**_VALID_ARCH, "model_name": "DeepSeek-R1-0528"})
    out = _load_model_arch(tmp_path, "/weights/nfs/DeepSeek-R1-0528")
    assert out["attention"] == "MLA"


def test_load_model_arch_missing_file_returns_empty(tmp_path: Path):
    assert _load_model_arch(tmp_path, "anything") == {}


def test_load_model_arch_invalid_json_returns_empty(tmp_path: Path):
    _write(tmp_path, "{not valid json")
    assert _load_model_arch(tmp_path, "anything") == {}


def test_load_model_arch_non_dict_returns_empty(tmp_path: Path):
    _write(tmp_path, ["a", "list"])
    assert _load_model_arch(tmp_path, "anything") == {}


def test_load_model_arch_missing_model_name_returns_empty(tmp_path: Path):
    payload = {k: v for k, v in _VALID_ARCH.items() if k != "model_name"}
    _write(tmp_path, payload)
    assert _load_model_arch(tmp_path, "DeepSeek-R1-0528") == {}


def test_load_model_arch_stale_mismatch_returns_empty(tmp_path: Path):
    """A leftover file from a previous run (different model) must be
    ignored — the convention path is shared across launches."""
    _write(tmp_path, {**_VALID_ARCH, "model_name": "Llama-3.1-8B"})
    assert _load_model_arch(tmp_path, "DeepSeek-R1-0528") == {}


# ---------------------------------------------------------------------------
# 2. SharedState serialization round-trip
# ---------------------------------------------------------------------------
def test_model_arch_round_trips_through_dict():
    state = SharedState(model_name="DeepSeek-R1-0528", model_arch=dict(_VALID_ARCH))
    revived = SharedState.from_dict(state.to_dict())
    assert revived.model_arch == _VALID_ARCH


def test_model_arch_defaults_to_empty_dict():
    state = SharedState(model_name="m")
    assert state.model_arch == {}
    revived = SharedState.from_dict(state.to_dict())
    assert revived.model_arch == {}


# ---------------------------------------------------------------------------
# 3. render_model_arch_compact
# ---------------------------------------------------------------------------
def test_render_empty_inputs_return_blank():
    assert render_model_arch_compact({}) == ""
    assert render_model_arch_compact(None) == ""
    assert render_model_arch_compact("not a dict") == ""  # type: ignore[arg-type]


def test_render_drops_empty_fields_and_trails_notes():
    line = render_model_arch_compact(_VALID_ARCH)
    assert "attention=MLA" in line
    assert "experts=256" in line
    # swa_window is None -> dropped.
    assert "swa_window" not in line
    # source/model_name are not structured render fields -> not shown.
    assert "model_name=" not in line
    # notes trails at the end.
    assert line.strip().endswith(_VALID_ARCH["notes"])
    assert line.index("decoder=") < line.index("notes=")


def test_render_skips_blank_notes():
    line = render_model_arch_compact({"attention": "MLA", "notes": "   "})
    assert line == "attention=MLA"


# ---------------------------------------------------------------------------
# 4. to_prompt_summary block
# ---------------------------------------------------------------------------
def test_prompt_summary_renders_block_when_set():
    state = SharedState(model_name="m", model_arch=dict(_VALID_ARCH))
    text = state.to_prompt_summary()
    assert "model_arch(advisory; subordinate to TraceLens analysis_md)=" in text
    assert "attention=MLA" in text


def test_prompt_summary_omits_block_when_empty():
    state = SharedState(model_name="m")
    text = state.to_prompt_summary()
    assert "model_arch" not in text


# ---------------------------------------------------------------------------
# 5. Coordinator._warm_specialist_params -> arch_notes
# ---------------------------------------------------------------------------
@dataclass
class _ArchState:
    """Minimal SharedState double for the warm-param path."""

    model_arch: dict = field(default_factory=dict)
    gpu_type: str = ""
    framework: str = ""
    tp: int = 0
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[Any] = field(default_factory=list)
    warm_start_lessons: list[Any] = field(default_factory=list)
    stack_fingerprint_meta: dict[str, Any] = field(default_factory=dict)


def _make_coord(tmp_path: Path, *, state: _ArchState) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = state
    c.knowledge_plane = None
    return c


@pytest.mark.asyncio
async def test_warm_populates_arch_notes_from_model_arch(tmp_path: Path):
    coord = _make_coord(tmp_path, state=_ArchState(model_arch=dict(_VALID_ARCH)))
    params: dict[str, Any] = {"domain": "serving_specialist"}
    await coord._warm_specialist_params(params)
    assert "arch_notes" in params
    assert "attention=MLA" in params["arch_notes"]


@pytest.mark.asyncio
async def test_warm_omits_arch_notes_when_model_arch_empty(tmp_path: Path):
    coord = _make_coord(tmp_path, state=_ArchState(model_arch={}))
    params: dict[str, Any] = {"domain": "serving_specialist"}
    await coord._warm_specialist_params(params)
    assert "arch_notes" not in params


@pytest.mark.asyncio
async def test_warm_respects_caller_supplied_arch_notes(tmp_path: Path):
    """``setdefault`` semantics: a caller-supplied value wins."""
    coord = _make_coord(tmp_path, state=_ArchState(model_arch=dict(_VALID_ARCH)))
    params: dict[str, Any] = {"domain": "serving_specialist", "arch_notes": "PRESET"}
    await coord._warm_specialist_params(params)
    assert params["arch_notes"] == "PRESET"
