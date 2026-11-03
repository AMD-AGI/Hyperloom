# Configs

Campaign configs: each names a pack and says where the campaign runs. The keys:
[docs/reference.md](../docs/reference.md). Component mode's: [docs/hosts.md](../docs/hosts.md).

| Path | What | Status |
|---|---|---|
| `agentx-glm52-mi355x.json` | GLM-5.2 on MI355X, pack `agentx` | the dry run's config (`--dry-run`); its `<FILL>` values are placeholders |
| `agentx-glm52-mi355x.json.template` | the same as a template | |
| `serving-template.json` | the serving template's example: Qwen3-14B-FP8, vLLM, TP1, 1024/1024 at C64 (pack `serving`) | copy and set the `<...>` values; passes `--dry-run` |
| `mi350x/model-m-c32-{a,b,b2…b5}.json` | Model M campaigns A and B on Cluster A (MI350X), C32 | launched by `scripts/model-m-campaign/` |
| `mi350x/model-m-c32-balanced.json` | Model M with pack `balanced` (every harness rule) | |
| `mi350x/model-m-c8-{c,c2…c5}.json` | Model M campaign C, C8 (c2–c5 at TP4) | |
| `mi350x/model-m-c8-component-test.json` | the template of component mode's GPU test | used by `scripts/component/` |
| `mi350x/model-m-plan-c.json` | a document campaign: the round plan | used by `scripts/model-m-campaign/plan_campaign.sh` |
| `mi355x/dsv41flash-fp4-agentx-*.json` | DeepSeek-V4.1-Flash AgentX, one per pack variant (v2…v6, cg, ruppell) | legacy; launched by `scripts/dsv41-campaign/` |
| `mi355x/model-m-c32-b.json` | Model M campaign B on Cluster B (MI355X) | |
| `mi355x/{glm52,gptoss-120b,minimaxm3,qwen35-397b}-mxfp4.json` | configs of the retired Local Mode (no `pack`) | legacy; `run` cannot load them |
| `parallel/*.json` | parallel mode's run configs and gates | examples, never launched |

A config's `bench_script` and `correctness_script` are absolute paths to its pack's validators as
the container sees them, and many configs carry cluster paths: copy one and change the paths, the
model and the workspace for your node.
