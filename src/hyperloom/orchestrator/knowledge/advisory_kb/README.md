# Advisory Knowledge Base (human-editable)

This directory holds Hyperloom's **advisory tuning knowledge** as plain markdown.
It is read at PRELUDE, rendered into the specialist "KB CONTEXT" prompt, and seeded
as advisory `gaps[]`. It is **advisory only** — the Critic still gates every kept
change; nothing here can reject a config.

## How routing works — the folder is the gate

A run reads `generic/` **plus** its own framework folder, and nothing else:

| Put a fact in… | It reaches… |
|---|---|
| `generic/` | **all** runs (vLLM, SGLang, ATOM) |
| `vllm/` | only vLLM runs |
| `sglang/` | only SGLang runs |
| `atom/` | only ATOM runs |

Rule of thumb: **the reasoning is generic, the exact knob is framework-specific.**
"If throughput-bound, raise the batch size" → `generic/`. The actual flag
(`--max-num-batched-tokens` vs `--max-total-tokens`) → the framework folder.

## How to add / edit knowledge (no code needed)

Open any `.md` and add an entry. Each entry is a `##` heading (the claim) followed
by an optional labeled block:

```markdown
## Short claim / knob name goes in the heading
- kind: hint            # "hint" (default) or "checklist"
- source: <required>    # URL / vllm#NNN / cph-perf-tuning:KNOWLEDGE.md#x / session:...
- impact: throughput    # optional: what it improves
- accuracy_risk: none   # optional
- domain_tags: framework # optional: routing hint to an EXPLORE specialist domain

Any prose here is kept and shown to the specialist as extra context.
```

Checklist entries (for the static-recon specialist) add gpu/precision gating:

```markdown
## rocm.fp4.some_gap
- kind: checklist
- source: <required>
- applies_when: gpu=rocm, precision=fp4   # only fires on matching runs
- domain_hint: kernel_switch_specialist
- source_dirs: vllm/model_executor/...
- consequence: <what regression the gap causes>
- bridge: <sketch of the fix>
- detect: <what to grep for / how to confirm>
```

Rules:
- **`source:` is required** — an entry without it is dropped (knowledge must be attributable).
- A missing/empty field is fine; the loader is forgiving.
- No code change is needed — the next run picks up your edit automatically.

## Layout

```
generic/   symptom_levers.md  hardware.md  quantization.md  parallelism.md  correctness_flags.md
vllm/      levers.md          correctness_flags.md          checklist.md
sglang/    levers.md          correctness_flags.md
atom/      levers.md          correctness_flags.md
```

Loader: `orchestrator/knowledge/advisory_kb.py`
(`hints_from_markdown(framework)`, `checklist_from_markdown(framework)`).
Override the KB root for tests/deploys with `HYPERLOOM_ADVISORY_KB_DIR`.
