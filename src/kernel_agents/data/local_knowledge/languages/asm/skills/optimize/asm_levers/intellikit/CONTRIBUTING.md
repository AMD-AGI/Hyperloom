# Contributing

PRs welcome from humans and agents alike. If you discovered something useful about gfx950 assembly, share it here.

## What we're looking for

- **New instruction docs** — undocumented instructions, new MFMA variants, additional cycle count measurements
- **Corrections** — wrong NOP count, stale cycle measurement, inaccurate hazard rule
- **Agent-generated findings** — your agent found a hazard, pattern, or optimization on MI355X? Add it
- **New guides** — workflow improvements, kernel family deep-dives, profiling techniques
- **Other architectures** — measurements on MI300X, MI325X, MI350X are welcome
- **Tools and scripts** — workflow automation, MCP servers, profiling helpers

## Instruction doc template

Every instruction doc must have YAML frontmatter. Use this template:

```markdown
---
instruction: <opcode>
category: <MFMA | LDS | memory | scalar | VALU | conversion | sync>
architecture: gfx950
hazard_severity: <critical | moderate | low | none>
tags: [relevant, searchable, terms]
---

# <opcode>

<one-line description>

## Quick Reference

| Field | Value |
|-------|-------|
| Opcode | ... |
| Encoding | ... |
| Measured CPI | <value> (isa-bench on MI355X) |

## Operand Requirements

(register types, counts, alignment constraints)

## Counter

(vmcnt / lgkmcnt — which counter it increments, FIFO behavior)

## Hazards

(NOP requirements, WAW risks, clobber rules — with severity)

## Known Bugs

(empirical findings, workarounds — include the symptom and fix)

## Common Patterns

(asm code examples showing typical usage)

## Performance Notes

(measured throughput, comparison notes, roofline context)
```

## Standards

- **Empirical over theoretical.** "Measured on MI355X" is the standard. If you can't measure it, say so.
- **One file per instruction.** Named `<opcode>.md` (lowercase, underscores).
- **Frontmatter is required.** It's what agents use to find docs. Get the category and hazard_severity right.
- **Actionable content.** Every doc should answer: what does this instruction do, what will break if I use it wrong, and how do I use it correctly.
- **No internal references.** Don't reference agent sessions, internal tool paths, or specific cluster names. Keep docs standalone.

## Guide template

Guides are longer-form docs covering patterns, workflows, or kernel families. They should have frontmatter too:

```markdown
---
guide: <name>
category: <methodology | debugging | architecture | optimization>
architecture: gfx950
tags: [relevant, terms]
---
```

## Submitting

1. Fork the repo
2. Create a branch
3. Add or modify docs following the templates above
4. Open a PR with a brief description of what you found and how you validated it

AI-generated PRs are first-class contributions. If your agent produced the finding, say so in the PR description — it helps us understand the methodology.
