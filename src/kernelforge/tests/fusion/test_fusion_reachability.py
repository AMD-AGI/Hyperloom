# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A fusion that nothing calls passes every other gate.

Compiling proves the module imports. Parity and the microbench call the entry
point from the harness directly. The serving smoke boots a server in which an
unreferenced fusion is inert, so it comes up clean. A DeepSeek-V4 run was
authored, validated, kept and exported with an entry point that had zero readers
in the framework tree.
"""

from __future__ import annotations

from pathlib import Path

from kernelforge.fusion.validate import unreached_fusion_symbols

# How the real run published its kernel: an attribute set on another module.
PUBLISHES = """
from mypkg import fused_mod
import mypkg.attention as _attn


def _install() -> None:
    _attn.fused_qk_norm = fused_mod.fused_qk_norm
"""

DEFINES = """
__all__ = ["fused_qk_norm"]


def fused_qk_norm(x):
    return x
"""

READS = """
import mypkg.attention as _attn


def forward(x):
    return _attn.fused_qk_norm(x)
"""

READS_DIRECTLY = """
from mypkg.fused_mod import fused_qk_norm


def forward(x):
    return fused_qk_norm(x)
"""

READS_DYNAMICALLY = """
def forward(mod, x):
    fn = getattr(mod, "fused_qk_norm", None)
    return fn(x) if fn else x
"""


def _tree(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    root = tmp_path / "fw"
    pkg = root / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "model.py").write_text(PUBLISHES, encoding="utf-8")
    (pkg / "fused_mod.py").write_text(DEFINES, encoding="utf-8")
    (pkg / "attention.py").write_text("def other(x):\n    return x\n", encoding="utf-8")
    for name, text in (extra or {}).items():
        (pkg / name).write_text(text, encoding="utf-8")
    return root


def test_a_published_symbol_nobody_reads_is_reported(tmp_path: Path) -> None:
    root = _tree(tmp_path)

    assert unreached_fusion_symbols(str(root), ["mypkg/model.py"]) == ["fused_qk_norm"]


def test_a_call_site_clears_it(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"decode.py": READS})

    assert unreached_fusion_symbols(str(root), ["mypkg/model.py"]) == []


def test_a_dynamic_lookup_counts_as_a_call_site(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"decode.py": READS_DYNAMICALLY})

    assert unreached_fusion_symbols(str(root), ["mypkg/model.py"]) == []


def test_the_module_that_defines_it_is_not_a_reader(tmp_path: Path) -> None:
    # fused_mod.py names the symbol in __all__ and in its def; neither is a call.
    root = _tree(tmp_path)

    assert unreached_fusion_symbols(str(root), ["mypkg/model.py"]) == ["fused_qk_norm"]


def test_an_authored_module_nothing_calls_is_reported(tmp_path: Path) -> None:
    # The second shape: no attribute is published anywhere, the kernel is just
    # defined and left. An audit of 27 landed fusions found two of these.
    root = _tree(tmp_path)

    assert unreached_fusion_symbols(str(root), ["mypkg/fused_mod.py"]) == ["fused_qk_norm"]


def test_an_authored_module_something_calls_is_not(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"decode.py": READS_DIRECTLY})

    assert unreached_fusion_symbols(str(root), ["mypkg/fused_mod.py"]) == []


def test_new_code_cited_only_by_new_code_is_still_unreached(tmp_path: Path) -> None:
    # An island of new definitions calling each other is not wiring: the chain
    # has to start somewhere the framework already goes.
    island = """
from mypkg.fused_mod import fused_qk_norm


def _island_entry(x):
    return fused_qk_norm(x)
"""
    root = _tree(tmp_path, {"fused_island.py": island})

    unreached = unreached_fusion_symbols(str(root), ["mypkg/fused_mod.py", "mypkg/fused_island.py"])

    assert unreached == ["_island_entry", "fused_qk_norm"]


def test_an_unknown_root_is_not_second_guessed(tmp_path: Path) -> None:
    assert unreached_fusion_symbols("", ["mypkg/model.py"]) == []
    assert unreached_fusion_symbols(str(tmp_path / "nope"), ["m.py"]) == []


def test_no_changed_files_reports_nothing(tmp_path: Path) -> None:
    root = _tree(tmp_path)

    assert unreached_fusion_symbols(str(root), []) == []


# The wiring an author actually writes, and the one the first version of this
# check could not see: resolve eligibility in __init__, branch in forward. Both
# live in the file being edited, and skipping that file while looking for
# readers rejected every fusion wired the normal way.
WIRED_IN_PLACE = """
from mypkg.fused_mod import fused_qk_norm


class Block:
    def __init__(self):
        self._use_fused = _enabled()

    def forward(self, x):
        if self._use_fused:
            return fused_qk_norm(x)
        return eager(x)
"""


def test_wiring_inside_the_edited_file_is_seen(tmp_path: Path) -> None:
    root = tmp_path / "fw"
    pkg = root / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "model.py").write_text(WIRED_IN_PLACE, encoding="utf-8")
    (pkg / "fused_mod.py").write_text(DEFINES, encoding="utf-8")

    assert unreached_fusion_symbols(str(root), ["mypkg/model.py"]) == []


def test_an_attribute_set_and_branched_on_in_place_is_seen(tmp_path: Path) -> None:
    # `self._use_fused` is set in __init__ and read in forward; an instance
    # attribute is not a publish, and reading it a few lines down is the wiring.
    root = tmp_path / "fw"
    pkg = root / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "model.py").write_text(WIRED_IN_PLACE, encoding="utf-8")
    (pkg / "fused_mod.py").write_text(DEFINES, encoding="utf-8")

    assert "_use_fused" not in unreached_fusion_symbols(str(root), ["mypkg/model.py"])


REFERENCE_IMPL = """
def fused_qk_norm_ref(x):
    # Eager reference the parity check compares against; the model never calls it.
    return x
"""


def test_a_reference_impl_does_not_fail_a_wired_fusion(tmp_path: Path) -> None:
    # Replaying 18 landed runs, the only false positive was a fusion reported
    # for shipping the eager reference its own parity check needs.
    root = tmp_path / "fw"
    pkg = root / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "model.py").write_text(WIRED_IN_PLACE + REFERENCE_IMPL, encoding="utf-8")
    (pkg / "fused_mod.py").write_text(DEFINES, encoding="utf-8")

    assert unreached_fusion_symbols(str(root), ["mypkg/model.py"]) == []


def test_an_authored_module_beside_the_source_is_checked_too(tmp_path: Path) -> None:
    # The caller passes the model source; the author also leaves a new module
    # next to it, and a kernel that lives there and is never called is dead in
    # exactly the way this looks for.
    root = tmp_path / "fw"
    pkg = root / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "model.py").write_text("def forward(x):\n    return x\n", encoding="utf-8")
    (pkg / "extra_fusion.py").write_text(DEFINES, encoding="utf-8")

    assert unreached_fusion_symbols(str(root), ["mypkg/model.py"]) == ["fused_qk_norm"]
