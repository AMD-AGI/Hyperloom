#!/usr/bin/env bash
# install.sh — install TraceLens and GEAK inside a Docker container for Hyperloom local-mode.
#
# Usage (run from the host, not inside the container):
#   bash local-mode/install.sh
#
# Reads local-mode/config.json for docker_container and workspace_root.
# All install paths are derived from workspace_root — no hardcoded /workspace.
#
# Idempotent: safe to re-run if a previous install was partial or interrupted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${LOCAL_MODE_CONFIG:-$SCRIPT_DIR/config.json}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: $CONFIG not found. Copy config.json.template to config.json and fill it in." >&2
    exit 1
fi

# Parse config via argv, not shell interpolation, to handle paths with spaces
read_config() {
    python3 - "$CONFIG" "$1" <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
key = sys.argv[2]
val = cfg.get(key) or cfg.get(key.replace("-", "_"))
if val is None:
    print(f"ERROR: '{key}' not found in config.json", file=sys.stderr)
    sys.exit(1)
print(val)
PYEOF
}

CONTAINER=$(read_config docker_container)
WORKSPACE_ROOT=$(read_config workspace_root)

echo "Container:      $CONTAINER"
echo "Workspace root: $WORKSPACE_ROOT"

# Verify container is running
if ! docker ps --filter "name=^${CONTAINER}$" --format "{{.Names}}" | grep -q "^${CONTAINER}$"; then
    echo "ERROR: container '$CONTAINER' is not running. Start it first." >&2
    exit 1
fi

# Verify workspace_root exists inside container (catches missing bind-mount early)
if ! docker exec "$CONTAINER" test -d "$WORKSPACE_ROOT" 2>/dev/null; then
    echo "ERROR: workspace_root '$WORKSPACE_ROOT' does not exist inside container '$CONTAINER'." >&2
    echo "       Make sure it is bind-mounted when you start the container:" >&2
    echo "         docker run ... -v /host/path:$WORKSPACE_ROOT ..." >&2
    exit 1
fi

# Derive all paths from workspace_root — no hardcoded /workspace anywhere
INSTALL_ROOT="$WORKSPACE_ROOT/hyperloom/runtime/source-mirrors"
KERNEL_AGENT_TOOLS="$WORKSPACE_ROOT/kernel-agent/tools"
SCRIPTS_DEST="$WORKSPACE_ROOT/local-mode/scripts"

TRACELENS_REPO="https://github.com/AMD-AGI/TraceLens.git"
TRACELENS_REF="c3405111a2f9270fd820a1baa8edaaf6f61e7646"
TRACELENS_ROOT="$INSTALL_ROOT/TraceLens"

GEAK_REPO="https://github.com/AMD-AGI/GEAK.git"
GEAK_REF="v3.2.2"
GEAK_ROOT="$INSTALL_ROOT/GEAK"

echo ""
echo "=== Installing TraceLens ==="
docker exec "$CONTAINER" bash -c "
    set -euo pipefail
    mkdir -p '$INSTALL_ROOT'

    # Full clone (not shallow) so the pinned commit SHA is reachable.
    # Shallow fetch cannot serve arbitrary interior commits on GitHub.
    if [ -f '$TRACELENS_ROOT/.git/HEAD' ]; then
        CURRENT=\$(git -C '$TRACELENS_ROOT' rev-parse HEAD 2>/dev/null || echo unknown)
        if [ \"\$CURRENT\" = '$TRACELENS_REF' ]; then
            echo 'TraceLens already at correct commit, skipping.'
        else
            echo \"TraceLens at \$CURRENT, updating to $TRACELENS_REF...\"
            git -C '$TRACELENS_ROOT' fetch origin 2>&1
            git -C '$TRACELENS_ROOT' checkout '$TRACELENS_REF' 2>&1
        fi
    else
        rm -rf '$TRACELENS_ROOT'
        TMP='$INSTALL_ROOT/.TraceLens.clone.\$\$'
        git clone '$TRACELENS_REPO' \"\$TMP\" 2>&1
        git -C \"\$TMP\" checkout '$TRACELENS_REF' 2>&1
        mv \"\$TMP\" '$TRACELENS_ROOT'
    fi

    python3 -m pip install -q --no-cache-dir --break-system-packages -e '$TRACELENS_ROOT' 2>&1
    echo 'TraceLens installed OK'
"

echo ""
echo "=== Installing GEAK ==="
docker exec "$CONTAINER" bash -c "
    set -euo pipefail
    mkdir -p '$INSTALL_ROOT'

    if [ -d '$GEAK_ROOT/.git' ]; then
        CURRENT_TAG=\$(git -C '$GEAK_ROOT' describe --tags --exact-match 2>/dev/null || echo unknown)
        if [ \"\$CURRENT_TAG\" = '$GEAK_REF' ]; then
            echo 'GEAK already at correct tag, skipping clone.'
        else
            echo \"GEAK at \$CURRENT_TAG, re-cloning to $GEAK_REF...\"
            rm -rf '$GEAK_ROOT'
            git clone --depth 1 --branch '$GEAK_REF' '$GEAK_REPO' '$GEAK_ROOT' 2>&1
        fi
    else
        rm -rf '$GEAK_ROOT'
        git clone --depth 1 --branch '$GEAK_REF' '$GEAK_REPO' '$GEAK_ROOT' 2>&1
    fi

    if [ -f '$GEAK_ROOT/setup.sh' ]; then
        bash '$GEAK_ROOT/setup.sh' 2>&1
    else
        python3 -m pip install -q --no-cache-dir --break-system-packages -e '$GEAK_ROOT' 2>&1
    fi
    echo 'GEAK installed OK'
"

echo ""
echo "=== Installing kernel-agent tools ==="
docker exec "$CONTAINER" bash -c "mkdir -p '$KERNEL_AGENT_TOOLS'"

# Copy tracelens_analysis.py from the Hyperloom repo into the container.
# Search sibling directories of this script — works for any repo layout.
TRACELENS_TOOL=$(find "$(dirname "$SCRIPT_DIR")" -name "tracelens_analysis.py" \
    -path "*/kernel*/tools/*" 2>/dev/null | head -1 || true)
if [[ -n "$TRACELENS_TOOL" && -f "$TRACELENS_TOOL" ]]; then
    docker cp "$TRACELENS_TOOL" "$CONTAINER:$KERNEL_AGENT_TOOLS/tracelens_analysis.py"
    echo "Copied tracelens_analysis.py from $TRACELENS_TOOL"
else
    echo "WARN: tracelens_analysis.py not found on host — skipping. If already present"
    echo "      in the container at $KERNEL_AGENT_TOOLS/tracelens_analysis.py, this is fine."
fi

echo ""
echo "=== Copying bench/correctness scripts into container ==="
docker exec "$CONTAINER" bash -c "mkdir -p '$SCRIPTS_DEST'"
docker cp "$SCRIPT_DIR/scripts/bench.sh"        "$CONTAINER:$SCRIPTS_DEST/bench.sh"
docker cp "$SCRIPT_DIR/scripts/correctness.sh"  "$CONTAINER:$SCRIPTS_DEST/correctness.sh"
docker exec "$CONTAINER" chmod +x "$SCRIPTS_DEST/bench.sh" "$SCRIPTS_DEST/correctness.sh"
echo "Scripts copied to $SCRIPTS_DEST"

echo ""
echo "=== Verifying ==="
docker exec "$CONTAINER" bash -c "
    echo -n 'TraceLens root: '; ls '$TRACELENS_ROOT/.git/HEAD' 2>/dev/null && echo OK || echo MISSING
    echo -n 'GEAK binary:    '; which geak 2>/dev/null || echo MISSING
    echo -n 'tracelens tool: '; ls '$KERNEL_AGENT_TOOLS/tracelens_analysis.py' 2>/dev/null && echo OK || echo MISSING
    echo -n 'bench.sh:       '; ls '$SCRIPTS_DEST/bench.sh' 2>/dev/null && echo OK || echo MISSING
    echo -n 'correctness.sh: '; ls '$SCRIPTS_DEST/correctness.sh' 2>/dev/null && echo OK || echo MISSING
"

echo ""
echo "All done. Add these to your config.json:"
echo "  \"workspace_root\":     \"$WORKSPACE_ROOT\""
echo "  \"tracelens_root\":     \"$TRACELENS_ROOT\""
echo "  \"session_root\":       \"$WORKSPACE_ROOT/sessions/local-mode\""
echo "  \"bench_script\":       \"$SCRIPTS_DEST/bench.sh\""
echo "  \"correctness_script\": \"$SCRIPTS_DEST/correctness.sh\""
