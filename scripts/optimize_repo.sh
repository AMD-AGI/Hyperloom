#!/usr/bin/env bash
# =============================================================================
# optimize_repo.sh — bootstrap any repo with the generic-gpu-optimization skill
#
# Usage:
#   ./scripts/optimize_repo.sh <path-to-target-repo> [--force]
#
# What it does:
#   - Copies .cursor/skills/generic-gpu-optimization/ into <target>/.cursor/skills/
#   - Copies .cursor/mcp.json into <target>/.cursor/ (if not present)
#   - Copies .env.template into <target>/ (if not present)
#   - Runs the heuristic detector and prints what was found, so you can review
#     before opening Cursor
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPERLOOM_ROOT="$(dirname "$SCRIPT_DIR")"

TARGET="${1:?usage: optimize_repo.sh <target-repo> [--force]}"
FORCE="${2:-}"

[ -d "$TARGET" ] || { echo "ERROR: target $TARGET does not exist"; exit 1; }
TARGET=$(cd "$TARGET" && pwd)

SRC_SKILL="$HYPERLOOM_ROOT/.cursor/skills/generic-gpu-optimization"
DST_SKILL="$TARGET/.cursor/skills/generic-gpu-optimization"
SRC_MCP="$HYPERLOOM_ROOT/.cursor/mcp.json"
DST_MCP="$TARGET/.cursor/mcp.json"
SRC_ENV="$HYPERLOOM_ROOT/.env.template"
DST_ENV="$TARGET/.env.template"

echo "Hyperloom root: $HYPERLOOM_ROOT"
echo "Target repo:    $TARGET"
echo

# 1. Skill
mkdir -p "$TARGET/.cursor/skills"
if [ -d "$DST_SKILL" ] && [ "$FORCE" != "--force" ]; then
    echo "[skip] $DST_SKILL already exists (use --force to overwrite)"
else
    # Must rm -rf first: `cp -r src dst` with dst existing produces dst/src/,
    # which would silently leave stale files in the outer dir on re-bootstrap.
    rm -rf "$DST_SKILL"
    cp -r "$SRC_SKILL" "$DST_SKILL"
    echo "[ok]   copied skill to $DST_SKILL"
fi

# 2. MCP config
if [ -f "$DST_MCP" ] && [ "$FORCE" != "--force" ]; then
    echo "[skip] $DST_MCP already exists"
else
    cp "$SRC_MCP" "$DST_MCP"
    echo "[ok]   copied $DST_MCP"
fi

# 3. .env template
if [ -f "$DST_ENV" ] && [ "$FORCE" != "--force" ]; then
    echo "[skip] $DST_ENV already exists"
else
    cp "$SRC_ENV" "$DST_ENV"
    echo "[ok]   copied $DST_ENV — copy to .env and fill in AK_YOUR_API_KEY"
fi

# 4. Run the detector for a sanity preview
echo
echo "=== Project detection ==="
"$SRC_SKILL/scripts/detect_project.sh" "$TARGET" || {
    echo "WARN: detector failed; the agent will ask you for BUILD_COMMAND/BENCH_COMMAND on first run"
}

cat <<NEXT

=== Next steps ===
1. cd $TARGET
2. cp .env.template .env  &&  edit .env to set AK_YOUR_API_KEY
3. Open this folder in Cursor with the geak-agent + oci-oob-agent MCP toggles ENABLED
4. In Cursor chat:

   @.cursor/skills/generic-gpu-optimization/SKILL.md
   Optimize this repo for ${TARGET##*/} on $(rocm-smi --showproductname 2>/dev/null | grep -oE 'MI[0-9]+[A-Z]*' | head -1 || echo 'MI300X').

The agent will run setup -> detect -> build -> baseline -> profile -> DFS loop.
Override anything by passing it in the prompt:
   BENCH_COMMAND: ./build/bench/MY_BENCH --filter=foo
   TEST_COMMAND:  ctest -R MY_TEST
   TIME_BUDGET_MIN: 60
NEXT
