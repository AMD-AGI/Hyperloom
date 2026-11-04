# Which pack to start from

Derive from the nearest pack that has worked, not from nothing. Sizes are of `pack.yaml` on `main` (2026-11-03).

| The campaign | Parent | Why |
| --- | --- | --- |
| Any LLM on vLLM or SGLang, a random or fixed ISL/OSL bench | `serving` (176 lines) | The template: `balanced`'s method with nothing model-specific in it. Its example config is `configs/serving-template.json`. The default. |
| The same, with kernel work in scope (the agent may write or port kernels) | `balanced` (217 lines) | The same method, plus the `hardware`, `serving-playbook` and `kernels` skills. Strip its Model M specifics. |
| An AgentX trace replay (InferenceX), from scratch | `agentx` (35 lines) | The smallest AgentX pack (GLM-5.2). Add `serving`'s directives and harness rules to it. |
| An AgentX trace replay with golden acceptance, a P90 or per-GPU objective | an AgentX pack with golden acceptance (not carried in this copy) | The rules that kept those campaigns honest (the `benchmark` plugin's `contract.md`). Leave its `results/` behind. |
| DeepSeek-V4.1-Flash on AgentX | `agentx-dsv41-6` (822 lines) | The newest of that line. Its size is the problem this skill exists for: take its method, move its model facts to the dossier. |
| Model M at another operating point | `model-m-c8-tp4` (189 lines) | The newest Model M pack. |
| A document campaign (a plan, a report, the dossier alone) | `model-m-plan` (50 lines) or `parallel-dossier` (47 lines) | The smallest real ones: a bench that scores a document, a correctness check that lints it. |

**A kernel campaign** (`workload: kernel`) has its own orchestrator (`evolve-asm`). It is not drafted from these.

When none fits, say so in the review note and start from `serving`.
