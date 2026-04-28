# Inference A/B Test — User Guide

Compare two tool combinations (MCPs + Skill) end-to-end on the Hyperloom
inference optimization pipeline, across one or all models.

---

## Step 1: Prepare Your Tools

Register your MCP and Skill on
[Primus tools](https://oci-slc.example-internal-host.invalid/tools) and take note
of the **tool IDs**. Each group needs one MCP tool and one Skill tool.

Existing public tools:

| ID | Type  | Name |
|----|-------|------|
| 4  | mcp   | `geak-agent-mcp` |
| 9  | mcp   | `magpie` |
| 17 | mcp   | `OOB` |
| 22 | mcp   | `hyperloom-ci` (GEAK + OOB + TraceLens) |
| 24 | mcp   | `hyperloom-ci-b` (GEAK + OOB) |
| 20 | skill | `inference-optimization-magpie` |
| 23 | skill | `inference-optimization-CI` |
| 25 | skill | `inference-optimization-cli-b` |

## Step 2: Trigger the Workflow

Go to [**Inference A/B Test: TraceLens MCP vs CLI**](https://github.com/AMD-AGI/Hyperloom/actions/workflows/inference-ab-test.yml)
and click **Run workflow**.

| Input | Example | Notes |
|-------|---------|-------|
| `tools_a` | `22,23` | Tool IDs for group A |
| `label_a` | `Baseline` | Display label in the report |
| `tools_b` | `4,23` | Tool IDs for group B |
| `label_b` | `GEAK v2` | Display label in the report |
| `mode` | `ab` or `single` | See below |
| `models` | `gptoss-fp4-mi355x-vllm` | Leave empty to run all 6 models |

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
1. **Config diff** — which MCPs / Skill differ between A and B
2. **Per-model metrics** — baseline, optimized, gain, vs InferenceX, winner
3. **Overall** — aggregate win count across all models

---

## Common Recipes

**Test a new GEAK version**
```
tools_a = 22,23              label_a = GEAK current
tools_b = <new_geak>,23      label_b = GEAK experimental
mode    = ab
```

**Test a new Skill cheaply (against yesterday's CI)**
```
tools_b = 22,<new_skill>     label_b = My new skill
mode    = single
```

**Compare TraceLens MCP vs CLI**
```
tools_a = 22,23     label_a = MCP TraceLens
tools_b = 24,25     label_b = CLI TraceLens
mode    = ab
```
