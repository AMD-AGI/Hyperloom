# Inference A/B Test — User Guide

Compare two Hyperloom-side **plugin builds** end-to-end on the inference
optimization pipeline, across one or all models.

> Plugin model recap: a plugin is a server-side bundle of tools (MCP servers
> + a prompt skill that points at a runtime on WekaFS). The CI sends only
> `pluginId` to the SaFE / Claw API; the plugin transparently picks up its
> bundled tools list. Today plugin **4 (Hyperloom)** wraps the canonical
> wekafs `src/hyperloom/inference_optimizer/` Python package via tool 85
> (hyperloom-prompt). To A/B test a different runtime, register a new plugin
> and pass its ID here.

---

## Step 1: Prepare Your Plugins

Register the plugin you want to test on the
[Primus plugins console](https://oci-slc.primus-safe.amd.com/plugins) (or
ask the platform team to create one) and take note of its **plugin ID**.

Existing plugins:

| ID | Name        | Notes |
|----|-------------|-------|
| 4  | Hyperloom   | Current production plugin: tool 3 (`hyperloom-public-mcp`) + tool 85 (`hyperloom-prompt` → `@/wekafs/HyperloomV2/src/hyperloom/inference_optimizer/SKILL.md`) |

When a new plugin is registered, append it to this table.

## Step 2: Trigger the Workflow

Go to
[**Inference A/B Test**](https://github.com/AMD-AGI/Hyperloom/actions/workflows/inference-ab-test.yml)
and click **Run workflow**.

| Input | Example | Notes |
|-------|---------|-------|
| `plugin_id_a` | `4` | Plugin ID for group A (default `4` = current Hyperloom) |
| `label_a` | `Plugin 4 (current)` | Display label in the report |
| `plugin_id_b` | `5` | Plugin ID for group B (override to compare) |
| `label_b` | `Plugin 5 (experimental)` | Display label in the report |
| `mode` | `ab` or `single` | See below |
| `models` | `gptoss-fp4-mi355x-vllm` | Leave empty to run all models from `ci-config.yaml` |

**Modes:**
- `ab` — run both groups (2× cost, direct comparison)
- `single` — run only group B; the Compare step automatically uses the latest
  successful production CI run as group A (1× cost)

## Step 3: Read the Results

- **Teams channel** — one card per model per group, plus a final A/B
  comparison card at the end.
- **GitHub Job Summary** — markdown table on the run page.
- **Artifact** `ab-comparison` — `ab_comparison.md` for download.

The comparison shows:
1. **Plugin diff** — which plugin ID was used per group (highlighted when different)
2. **Per-model metrics** — baseline, optimized, gain, vs InferenceX, winner
3. **Overall** — aggregate win count across all models

If you need a tool-by-tool breakdown of what each plugin contains, query
`GET /claw-api/v1/plugins/<id>` directly — the workflow no longer expands
this automatically (plugins are treated as opaque labels).

---

## Common Recipes

**Reproducibility / control run** (same plugin both sides — verifies sandbox
variance)
```
plugin_id_a = 4    label_a = Plugin 4 run 1
plugin_id_b = 4    label_b = Plugin 4 run 2
mode        = ab
```

**Test a new plugin against current Hyperloom**
```
plugin_id_a = 4    label_a = Plugin 4 (control)
plugin_id_b = 5    label_b = Plugin 5 (experimental)
mode        = ab
```

**Test a new plugin cheaply (against yesterday's CI)**
```
plugin_id_b = 5    label_b = Plugin 5 (experimental)
mode        = single
```

> **Why no more `--update` flag?** The CI baseline must stay pinned to
> InferenceX's published image tag (otherwise tok/s comparisons across runs
> are apples-to-oranges). To use a different image, edit `ci-config.yaml`
> directly and commit the change so the diff is reviewable.

> **Why no more `tools_a` / `tools_b`?** The old design exposed individual
> Claw `tool_id`s. With the move to plugin-bundled skills (plugin 4 = wekafs
> `inference_optimizer`), tool selection is handled server-side; the client
> only needs to pick a plugin.
