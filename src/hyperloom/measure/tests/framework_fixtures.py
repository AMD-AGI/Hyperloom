# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A stand-in for the image's aiter install, in a directory of the test's own: its tuned tables and its variables."""

from __future__ import annotations

from pathlib import Path

__all__ = ["CORE", "DAMAGED", "MODELS", "PRISTINE", "SHARED_GEMM", "make_framework"]

CORE = """
import os

AITER_ROOT_DIR = "/unused"
AITER_CONFIG_GEMM = os.getenv(
    "AITER_CONFIG_GEMM",
    f"{AITER_ROOT_DIR}/aiter/configs/gemm_tuned.csv",
)
AITER_CONFIG_MOE = (
    os.getenv("AITER_CONFIG_MOE", "").strip()
    or f"{AITER_ROOT_DIR}/aiter/configs/moe_tuned.csv"
)
AITER_CONFIG_SPLIT = os.getenv(
    "AITER_CONFIG_SPLIT",
    f"{AITER_ROOT_DIR}/aiter/configs/"
    f"split_name_tuned.csv",
)


class AITER_CONFIG:
    def get_config_file(self, env_name, default_file, tuned_file_name):
        return self.update_config_files((os.getenv(env_name) or "").strip() or default_file, tuned_file_name)

    def update_config_files(self, file_path, merge_name):
        paths = file_path.split(os.pathsep)
        if len(paths) <= 1:
            return file_path
        from pathlib import Path

        rows = [Path(path).read_text().splitlines() for path in paths if os.path.exists(path)]
        merged = [rows[0][0], *dict.fromkeys(row for table in rows for row in table[1:])]
        config_path = Path("/tmp/aiter_configs/")
        config_path.mkdir(parents=True, exist_ok=True)
        new_file_path = f"{config_path}/{os.environ.get('FAKE_AITER_MERGE_PREFIX', '')}{merge_name}.csv"
        Path(new_file_path + ".tmp").write_text("\\n".join(merged) + "\\n")
        os.replace(new_file_path + ".tmp", new_file_path)
        return new_file_path

    @property
    def AITER_CONFIG_GEMM_FILE(self):
        return self.get_config_file("AITER_CONFIG_GEMM", AITER_CONFIG_GEMM, "gemm_tuned")

    @property
    def AITER_CONFIG_MOE_FILE(self):
        return self.get_config_file("AITER_CONFIG_MOE", AITER_CONFIG_MOE, "moe_tuned")

    @property
    def AITER_CONFIG_SPLIT_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_SPLIT",
            AITER_CONFIG_SPLIT,
            "split_tuned",
        )


AITER_CONFIGS = AITER_CONFIG()
"""
PRISTINE = "M,N,K,us\n64,5120,5120,10\n"
DAMAGED = "M,N,K,us\n"
MODELS = ("a_gemm_tuned.csv", "b_gemm_tuned.csv", "c_gemm_tuned_untuned_v2.csv", "d_moe_tuned.csv")
SHARED_GEMM = "gemm_tuned.csv"


def make_framework(parent: Path) -> Path:
    """The tuned-table directory of a stand-in aiter install below ``parent``; the install is never the machine's."""
    root = parent / "sgl-workspace" / "aiter" / "aiter"
    (root / "jit").mkdir(parents=True)
    (root / "jit" / "core.py").write_text(CORE, encoding="utf-8")
    (root / "configs" / "model_configs").mkdir(parents=True)
    for name in (SHARED_GEMM, "moe_tuned.csv", "split_name_tuned.csv"):
        (root / "configs" / name).write_text(PRISTINE, encoding="utf-8")
    for name in MODELS:
        (root / "configs" / "model_configs" / name).write_text(PRISTINE + f"1,1,1,{len(name)}\n", encoding="utf-8")
    return root / "configs"
