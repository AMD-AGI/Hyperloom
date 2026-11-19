# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Focused unit coverage for small pure-logic helpers."""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- # orchestrator.knowledge.kb_writeback #
# --------------------------------------------------------------------------- #
async def test_kb_writeback_rejects_unknown_outcome() -> None:
    from hyperloom.orchestrator.knowledge import kb_writeback

    with pytest.raises(ValueError):
        await kb_writeback.write_framework_record(
            kb_writeback.FrameworkOutcome("u", "s", "p", "not_a_real_outcome", 1.0, "sess")
        )


# --------------------------------------------------------------------------- # recipe_kb.canonical_id #
# --------------------------------------------------------------------------- #
def test_canonical_id_roundtrip_and_errors() -> None:
    from hyperloom.orchestrator.knowledge.recipe_kb import canonical_id as cid

    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.cid_to_path_components("")
    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.cid_to_path_components("inference:only:three")
    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.cid_to_path_components("wrongprefix:a:b:c:d:e:f:g")
    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.cid_to_path_components("inference:a::c:d:e:f:g")

    full = "inference:model:hw:fw:mt:arch:fwv:prec"
    comps = cid.cid_to_path_components(full)
    assert comps == ("model", "hw", "fw", "mt", "arch", "fwv", "prec")


def test_canonical_id_for_path_errors(tmp_path) -> None:
    from hyperloom.orchestrator.knowledge.recipe_kb import canonical_id as cid

    # recipe_dir not under root.
    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.canonical_id_for_path(root=tmp_path / "root", recipe_dir=tmp_path / "elsewhere")

    # Wrong depth under root.
    root = tmp_path / "root"
    shallow = root / "only_one_level"
    with pytest.raises(cid.InvalidCanonicalIdError):
        cid.canonical_id_for_path(root=root, recipe_dir=shallow)


# --------------------------------------------------------------------------- # paths #
# --------------------------------------------------------------------------- #
def test_paths_asset_root_missing_override(monkeypatch, tmp_path) -> None:
    from hyperloom.inference_optimizer.session import paths

    monkeypatch.setenv(paths.ENV_OVERRIDE_ASSET_ROOT, str(tmp_path / "nope"))
    with pytest.raises(paths.AssetRootNotFound):
        paths.asset_root()
