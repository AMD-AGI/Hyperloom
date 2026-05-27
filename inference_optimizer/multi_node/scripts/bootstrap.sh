#!/usr/bin/env bash
# Bootstrap the BYOI toolchain inside the RayJob head pod.
#
# Submitted by ``inference_optimizer.multi_node bootstrap`` via the Ray
# Dashboard REST ``/api/jobs/`` endpoint, which means this script runs
# INSIDE the RayJob head pod, not in the Claw sandbox.
#
# Idempotent: a marker file at ``$BOOTSTRAP_MARKER`` short-circuits the
# whole script on subsequent calls. To force a re-run pass ``--force`` or
# ``unset`` the marker first.
#
# Renders ``/etc/profile.d/hyperloom-env.sh`` from the env it can see,
# so every later REST job that does ``source /etc/profile.d/hyperloom-env.sh``
# picks up the same PATH / *_API_KEY / *_BASE_URL the bootstrap saw.

set -euo pipefail

BOOTSTRAP_MARKER="${BOOTSTRAP_MARKER:-/opt/hyperloom/.bootstrap_done}"
HYPERLOOM_VENV="${HYPERLOOM_VENV:-/opt/venv}"
LOG_DIR="${LOG_DIR:-/var/log/hyperloom}"
ENV_FILE="${ENV_FILE:-/etc/profile.d/hyperloom-env.sh}"

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) echo "bootstrap.sh: unknown arg: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$LOG_DIR" "$(dirname "$BOOTSTRAP_MARKER")"

if [ "$FORCE" -eq 0 ] && [ -f "$BOOTSTRAP_MARKER" ]; then
    echo "bootstrap.sh: marker $BOOTSTRAP_MARKER present; skipping. Pass --force to re-run."
    exit 0
fi

# --- 1. Verify the framework venv. The RayJob image (sglang/vllm) ships
#        /opt/venv with the framework already installed; we only fail loud
#        if it's missing.
if [ ! -x "$HYPERLOOM_VENV/bin/python3" ]; then
    echo "bootstrap.sh: ERROR — $HYPERLOOM_VENV/bin/python3 not found." >&2
    echo "             expected the framework image to ship a venv at /opt/venv." >&2
    exit 1
fi

# --- 2. (removed) OOB CLI install. The multi-node kernel-agent path uses
#        ``multi_node kernel-bench`` (NodeAffinity-hard-pinned to the head
#        pod via ``kernel_bench_multinode.py``); the head pod runs the
#        bench script directly, no OOB CLI is invoked inside the pod.
#        ``oob_submit.run_via_ray`` (the legacy single-node ray.remote
#        entrypoint) is not the path LLMs are prompted to take on
#        ``--nodes >= 2``. See ``multi_node/SKILL.md`` and
#        ``kernel-agent/tools/kernel_optimization.py:1244-1283``.

# --- 2c. Install claude + codex CLIs via npm (best-effort; require node>=18).
# Ubuntu 22.04's default apt nodejs is v12 (too old: @anthropic-ai/claude-code
# requires node>=18 and uses syntax v12 cannot parse). Mirror sandbox-side
# `kernel-agent/scripts/install.sh:ensure_node` and pull node 20 from
# NodeSource. `$NODE_TARBALL_PATH` is an env-driven escape hatch (agent can
# pass `--rayjob-extra-env NODE_TARBALL_PATH=<path-to-node-vXX-linux-x64.tar.xz>`
# when the pod has no public-network access to deb.nodesource.com).
NODE_OK=0
if command -v node >/dev/null 2>&1; then
    if node -e 'process.exit(parseInt(process.versions.node.split(".")[0],10) >= 18 ? 0 : 1)' 2>/dev/null; then
        NODE_OK=1
    fi
fi
if [ "$NODE_OK" -eq 0 ] && [ -n "${NODE_TARBALL_PATH:-}" ] && [ -f "$NODE_TARBALL_PATH" ]; then
    echo "bootstrap.sh: extracting node prebuilt from \$NODE_TARBALL_PATH ($NODE_TARBALL_PATH)"
    mkdir -p /opt/node && tar -xJf "$NODE_TARBALL_PATH" -C /opt/node --strip-components=1
    for b in node npm npx; do ln -sfn "/opt/node/bin/$b" "/usr/local/bin/$b"; done
    NODE_OK=1
fi
if [ "$NODE_OK" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "bootstrap.sh: installing curl/ca-certificates for NodeSource setup"
        apt-get update -qq 2>>"$LOG_DIR/bootstrap_apt.log" || true
        apt-get -y install ca-certificates curl gnupg 2>>"$LOG_DIR/bootstrap_apt.log" >/dev/null || true
    fi
    echo "bootstrap.sh: installing Node.js 20 from NodeSource (claude-code requires node>=18)"
    apt-get -y purge libnode-dev libnode72 nodejs nodejs-doc npm 2>>"$LOG_DIR/bootstrap_apt.log" >/dev/null || true
    NS_SCRIPT=/tmp/nodesource_setup_20.x
    if curl -fsSL https://deb.nodesource.com/setup_20.x -o "$NS_SCRIPT" 2>>"$LOG_DIR/bootstrap_apt.log" \
        && bash "$NS_SCRIPT" 2>>"$LOG_DIR/bootstrap_apt.log" >/dev/null \
        && apt-get -y install nodejs 2>>"$LOG_DIR/bootstrap_apt.log" >/dev/null; then
        NODE_OK=1
    else
        echo "bootstrap.sh: WARN — NodeSource setup failed; claude/codex CLIs will be missing (see $LOG_DIR/bootstrap_apt.log)" >&2
    fi
fi

if command -v npm >/dev/null 2>&1; then
    export NODE_TLS_REJECT_UNAUTHORIZED=0
    echo "bootstrap.sh: installing @anthropic-ai/claude-code via npm"
    npm install -g @anthropic-ai/claude-code 2>>"$LOG_DIR/bootstrap_npm.log" >/dev/null || \
        echo "bootstrap.sh: WARN — claude-code npm install failed (see $LOG_DIR/bootstrap_npm.log)" >&2
    echo "bootstrap.sh: installing @openai/codex via npm"
    npm install -g @openai/codex@0.100.0 2>>"$LOG_DIR/bootstrap_npm.log" >/dev/null || \
        echo "bootstrap.sh: WARN — codex npm install failed (see $LOG_DIR/bootstrap_npm.log)" >&2

    # The prebuilt node sets npm prefix to /opt/node so binaries live in
    # /opt/node/bin. Force-symlink them into /usr/local/bin so verify and
    # downstream callers find them on the standard PATH.
    for cli_bin in claude codex; do
        if [ -x "/opt/node/bin/$cli_bin" ]; then
            ln -sfn "/opt/node/bin/$cli_bin" "/usr/local/bin/$cli_bin"
        fi
    done
else
    echo "bootstrap.sh: WARN — npm not found; claude/codex CLIs will be missing" >&2
fi

# --- 3. Render /etc/profile.d/hyperloom-env.sh so every later REST job
#        can ``source`` it and inherit the same PATH + credentials. We
#        only emit keys that are actually set in this process env so we
#        don't leak ``KEY=`` empty placeholders (which would shadow real
#        env later). Keys we care about: ADDENDUM-13's credential set
#        plus the venv PATH.
echo "bootstrap.sh: writing $ENV_FILE"
{
    echo "# Generated by inference_optimizer.multi_node bootstrap.sh — DO NOT EDIT."
    echo "export PATH=\"$HYPERLOOM_VENV/bin:\${PATH}\""
    for k in OOB_API_KEY AMD_LLM_API_KEY LLM_API_KEY \
             ANTHROPIC_API_KEY OPENAI_API_KEY SAFE_API_KEY \
             OOB_BASE_URL ANTHROPIC_BASE_URL OPENAI_BASE_URL \
             ANTHROPIC_CUSTOM_HEADERS; do
        v="${!k:-}"
        if [ -n "$v" ]; then
            # Single-quote the value verbatim so embedded `$`/`"` don't
            # get re-evaluated when the env file is sourced later.
            esc=$(printf "%s" "$v" | sed "s/'/'\\\\''/g")
            echo "export $k='$esc'"
        fi
    done
} > "$ENV_FILE"
chmod 0644 "$ENV_FILE"

# --- 4. Mark done.
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$BOOTSTRAP_MARKER"
echo "bootstrap.sh: OK ($BOOTSTRAP_MARKER)"
