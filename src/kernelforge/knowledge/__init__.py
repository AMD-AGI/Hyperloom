# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Public access to KernelForge local knowledge and durable KB records."""

from kernelforge.knowledge.local_wiki.index import build_forge_knowledge
from kernelforge.knowledge.kb_store.writer import write_run_experience
from kernelforge.knowledge.kb_store.reader import read_best_solution
from kernelforge.knowledge.kb_store.config import (
    KnowledgeConfig,
    KnowledgeStoreMode,
)

__all__ = [
    "build_forge_knowledge",
    "write_run_experience",
    "read_best_solution",
    "KnowledgeConfig",
    "KnowledgeStoreMode",
]
