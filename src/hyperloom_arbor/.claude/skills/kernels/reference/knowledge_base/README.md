# Knowledge base — citation conventions

The vendored playbooks (`ck/ck_playbook.md`, `aiter/asm_perf_playbook_v2.md`,
`aiter/asm_kernel_knowledge.md`, `hip/hipkittens_playbook.md`) are *analyses
of* external codebases. They cite source files in those repos using stable,
repo-rooted relative prefixes — never absolute machine paths.

The playbook prose, technique catalogs, tables, and ISA extracts are **fully
self-contained**. The citations are convenience pointers — an agent learns
the technique from the playbook text without ever opening a citation.

## Citation prefixes

| Prefix in the playbooks | What it refers to |
|---|---|
| `aiter-amd/...` | The aiter-amd repository (analysis is pinned to a specific commit — see `source_registry.json`) |
| `HipKittens/...` | The HipKittens repository |
| `sglang/...`, `sglang_v4_pr/...`, `vllm-amd/...` | The named upstream repository |
| `[pdf:pN]` (CDNA4 ISA spec) | The AMD CDNA4 ISA Architecture PDF; the chapter text extracts the playbooks rely on are vendored at `shared/cdna4-isa/` |
| `[asm-v2:§N]` | A section in `aiter/asm_perf_playbook_v2.md` |

How those prefixes resolve to filesystem locations is a per-machine
concern; this repo deliberately doesn't encode any one user's checkout
layout. The IntelliKit pack (`aiter/intellikit-asm-skills/`) is fully
vendored and has **no** external citations.

## Self-containment

On a machine without the upstream repos the citations are dangling pointers,
but the knowledge itself is intact: every technique, table, dispatch path,
and ISA chapter extract referenced by the playbooks is in-tree. An agent can
operate from these files alone.
