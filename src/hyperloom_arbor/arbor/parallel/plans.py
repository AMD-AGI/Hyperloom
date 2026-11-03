"""A round's plans, pinned: a copy in rounds/rNN/plans/ with its sha256, its ITEMS checked.

A direction's `plan` is a file of the dossier (relative to the 11.uK that names
it). The copy is what the lane runs: arbor/stages.py pins it again in the lane's
session, and refuses to judge anything once the copy and its pin disagree.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from arbor import stages


class PlanError(ValueError):
    pass


def resolve(plan: str, where: Path) -> Path:
    """A map's plan path: absolute, or relative to the directory of the 11.uK that names it."""
    return Path(plan) if os.path.isabs(plan) else Path(where) / plan


def check(path: Path) -> list[dict]:
    """The plan's items; PlanError when the file is missing or its ITEMS do not parse."""
    if not path.is_file():
        raise PlanError(f"plan {path} does not exist")
    items, errs = stages.parse_items(path.read_text(errors="replace"))
    if errs:
        raise PlanError(f"plan {path.name}: " + "; ".join(errs[:3]))
    return items


def has_items(path: Path) -> bool:
    return path.is_file() and bool(stages._blocks(path.read_text(errors="replace")))


def pin(src: Path, plans_dir: Path) -> dict:
    """Copy the plan into the round (idempotent for the same bytes); {path, sha256, source, items}."""
    data = src.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    dst = Path(plans_dir) / src.name
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != sha:
        dst = Path(plans_dir) / f"{src.stem}.{sha[:8]}{src.suffix}"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    items = stages.parse_items(data.decode("utf-8", errors="replace"))[0] if has_items(src) else []
    return {"path": str(dst), "sha256": sha, "source": str(src), "items": len(items)}


def verify(pinned: dict) -> bool:
    p = Path(pinned["path"])
    return p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == pinned["sha256"]
