# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Make the sheet's YAML the one this sandbox runs (design 4.8.3 step 4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.outside_facts import ROCR_VISIBLE_DEVICES_ENV
from hyperloom.measure.attachments import substitute
from hyperloom.measure.farm import Farm, place_farm
from hyperloom.orchestrator.actions.executors import strip_visible_devices_from_config

__all__ = ["Placement", "rewrite_config"]


_UTF8 = "utf-8"


def _without_device_masks(cfg: dict[str, Any], scratch: Path) -> dict[str, Any]:
    """The config minus its ``*_VISIBLE_DEVICES`` variables, by the function the old flow strips with."""
    scratch.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding=_UTF8)
    stripped = strip_visible_devices_from_config(scratch)
    result = yaml.safe_load(stripped.read_text(encoding=_UTF8)) or {}
    if stripped != scratch:
        stripped.unlink()
    scratch.unlink()
    return result


@dataclass(frozen=True)
class Placement:
    """What this sandbox puts into a sheet's YAML: the placed attachments, the pid directory, the InferenceX checkout,
    the device mask and the rebuilt framework, if any."""

    placed: dict[str, Path]
    pid_dir: Path
    inferencex_path: Path
    visible_devices: str | None
    farm: Farm | None


def rewrite_config(yaml_text: str, out_path: Path, placement: Placement) -> Path:
    """Write the rewritten YAML to ``out_path``.

    Attachment placeholders name the placed files, the device mask is this sandbox's, the server's pid files go to
    ``pid_dir``, the benchmark client comes from the runner's own InferenceX checkout, and the rebuilt framework, when
    there is one, is first on the server's import path.
    """
    config = yaml.safe_load(substitute(yaml_text, placement.placed)) or {}
    config = _without_device_masks(config, out_path.with_name(out_path.name + ".mask"))
    bench = config.setdefault("benchmark", {})
    envs = bench.setdefault("envs", {})
    if placement.visible_devices:
        envs[ROCR_VISIBLE_DEVICES_ENV] = placement.visible_devices
    if isinstance(bench.get("server_lifecycle"), dict):
        bench["server_lifecycle"]["pid_dir"] = str(placement.pid_dir)
    bench["inferencex_path"] = str(placement.inferencex_path)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
    if placement.farm is not None:
        place_farm(out_path, placement.farm)
    return out_path
