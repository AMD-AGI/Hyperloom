#!/usr/bin/env bash
#
# Apply TraceLens inference-analysis patches to the current Python environment.
#
# This script patches vLLM or SGLang (in-place, inside a container or venv)
# so that profiling traces include:
#   - CUDA graph capture traces (per batch-size, per capture mode)
#   - Roofline annotations (sq, sk, sqsq, sqsk per context/generation group)
#
# It also installs the TraceLens package (--no-deps) for post-collection
# analysis (report generation, trace splitting, etc.).
#
# Usage:
#   bash apply_tracelens_patches.sh [OPTIONS]
#
# Options:
#   --framework  vllm|sglang   (default: auto-detect from $FRAMEWORK or probe Python)
#   --version    v0.13|v0.14|v0.15|v0.16|v0.17  (vLLM only; default: auto-detect)
#   --tracelens  /path/to/TraceLens-internal     (overrides $TRACELENS_REPO)
#   --clone-to   /path/to/dir   clone TraceLens-internal here if not already present
#   --git-ref    branch/tag/sha to clone (default: main)
#   --install    also pip-install TraceLens       (default: true)
#   --no-install skip TraceLens pip install
#   --dry-run    show what would happen without applying
#
# Environment variables (all optional):
#   TRACELENS_REPO      path to existing local TraceLens-internal clone
#   TRACELENS_GIT_URL   git URL to clone from (default: AMD-AGI/TraceLens-internal on GitHub)
#   TRACELENS_GIT_REF   branch/tag/sha (default: main)
#   FRAMEWORK           "vllm" or "sglang" — skips auto-detection
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRACELENS_GIT_URL="${TRACELENS_GIT_URL:-https://github.com/AMD-AGI/TraceLens-internal.git}"
TRACELENS_GIT_REF="${TRACELENS_GIT_REF:-main}"

# Defaults
FRAMEWORK_ARG=""
VLLM_VERSION_ARG=""
TRACELENS_REPO="${TRACELENS_REPO:-}"
CLONE_TO=""
DO_INSTALL=true
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --framework)   FRAMEWORK_ARG="$2"; shift 2 ;;
        --version)     VLLM_VERSION_ARG="$2"; shift 2 ;;
        --tracelens)   TRACELENS_REPO="$2"; shift 2 ;;
        --clone-to)    CLONE_TO="$2"; shift 2 ;;
        --git-ref)     TRACELENS_GIT_REF="$2"; shift 2 ;;
        --install)     DO_INSTALL=true; shift ;;
        --no-install)  DO_INSTALL=false; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        -h|--help)
            sed -n '2,/^$/{ s/^# //; s/^#$//; p; }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Resolve or clone TraceLens repo ──────────────────────────────────────────

PATCH_SUBDIR="examples/custom_workflows/inference_analysis"

ensure_tracelens_repo() {
    # Already have a valid local path — nothing to do
    if [[ -n "$TRACELENS_REPO" && -d "$TRACELENS_REPO/$PATCH_SUBDIR" ]]; then
        echo "[TraceLens] Using existing repo: $TRACELENS_REPO"
        return 0
    fi

    # Determine clone destination
    local dest="${CLONE_TO:-}"
    if [[ -z "$dest" ]]; then
        if [[ -n "$TRACELENS_REPO" ]]; then
            dest="$TRACELENS_REPO"
        else
            dest="/tmp/TraceLens-internal"
        fi
    fi

    # If already cloned at dest, reuse it
    if [[ -d "$dest/$PATCH_SUBDIR" ]]; then
        echo "[TraceLens] Found existing clone at $dest"
        TRACELENS_REPO="$dest"
        return 0
    fi

    echo "[TraceLens] TraceLens-internal not found locally. Cloning..."
    echo "[TraceLens]   URL: $TRACELENS_GIT_URL"
    echo "[TraceLens]   Ref: $TRACELENS_GIT_REF"
    echo "[TraceLens]   Destination: $dest"

    if ! command -v git &>/dev/null; then
        echo "ERROR: git is not available — cannot clone TraceLens-internal." >&2
        echo "       Either install git or mount/copy TraceLens-internal manually." >&2
        exit 1
    fi

    git clone --depth 1 --branch "$TRACELENS_GIT_REF" "$TRACELENS_GIT_URL" "$dest"
    TRACELENS_REPO="$dest"

    if [[ ! -d "$TRACELENS_REPO/$PATCH_SUBDIR" ]]; then
        echo "ERROR: Cloned repo is missing $PATCH_SUBDIR — wrong repo or branch?" >&2
        exit 1
    fi

    echo "[TraceLens] Clone successful"
}

ensure_tracelens_repo

# ── Detect framework ─────────────────────────────────────────────────────────

detect_framework() {
    if [[ -n "$FRAMEWORK_ARG" ]]; then
        echo "$FRAMEWORK_ARG"
        return
    fi
    if [[ -n "${FRAMEWORK:-}" ]]; then
        case "$FRAMEWORK" in
            *sglang*) echo "sglang"; return ;;
            *vllm*)   echo "vllm"; return ;;
        esac
    fi
    if python3 -c "import vllm" 2>/dev/null; then
        echo "vllm"
    elif python3 -c "import sglang" 2>/dev/null; then
        echo "sglang"
    else
        echo "ERROR: Cannot detect framework. Neither vllm nor sglang is importable." >&2
        echo "       Pass --framework vllm|sglang explicitly." >&2
        exit 1
    fi
}

FW="$(detect_framework)"
echo "[TraceLens] Detected framework: $FW"

# ── Framework-specific logic ──────────────────────────────────────────────────

apply_vllm_patches() {
    local pkg_dir
    pkg_dir="$(python3 -c "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), '..'))")"
    pkg_dir="$(cd "$pkg_dir" && pwd)"
    echo "[TraceLens] vLLM package root: $pkg_dir"

    # Auto-detect vLLM version if not specified
    local ver="$VLLM_VERSION_ARG"
    if [[ -z "$ver" ]]; then
        local raw_ver
        raw_ver="$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")"
        echo "[TraceLens] Detected vLLM version: $raw_ver"
        case "$raw_ver" in
            0.13.*) ver="v0.13" ;;
            0.14.*) ver="v0.14" ;;
            0.15.*) ver="v0.15" ;;
            0.16.*) ver="v0.16" ;;
            0.17.*) ver="v0.17" ;;
            *)
                echo "ERROR: Cannot map vLLM version '$raw_ver' to a known patch." >&2
                echo "       Available patches: v0.13, v0.14, v0.15, v0.16, v0.17" >&2
                echo "       Pass --version v0.XX explicitly." >&2
                exit 1
                ;;
        esac
    fi

    # Map short version to patch filename
    local patch_file
    case "$ver" in
        v0.13|v13) patch_file="vllm_v0.13.0.patch" ;;
        v0.14|v14) patch_file="vllm_v0.14.0.patch" ;;
        v0.15|v15) patch_file="vllm_v0.15.0.patch" ;;
        v0.16|v16) patch_file="vllm_v0.16.0.patch" ;;
        v0.17|v17) patch_file="vllm_v0.17.0.patch" ;;
        *)
            echo "ERROR: Unknown vLLM version tag '$ver'." >&2
            exit 1
            ;;
    esac

    local patch_path="$TRACELENS_REPO/$PATCH_SUBDIR/$patch_file"
    if [[ ! -f "$patch_path" ]]; then
        echo "ERROR: Patch file not found: $patch_path" >&2
        exit 1
    fi

    echo "[TraceLens] Applying $patch_file to $pkg_dir"
    if $DRY_RUN; then
        echo "[DRY RUN] cd $pkg_dir && git apply --check $patch_path"
        cd "$pkg_dir" && git apply --check "$patch_path" 2>&1 || true
        return
    fi

    cd "$pkg_dir"
    if git apply "$patch_path" 2>/dev/null; then
        echo "[TraceLens] Patch applied successfully via git apply"
    elif patch -p1 --fuzz=10 < "$patch_path"; then
        echo "[TraceLens] Patch applied successfully via patch -p1 (fallback)"
    else
        echo "ERROR: Failed to apply $patch_file" >&2
        echo "       The vLLM version may not match the patch exactly." >&2
        exit 1
    fi
}

apply_sglang_patches() {
    local pkg_dir
    pkg_dir="$(python3 -c "import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))")"
    pkg_dir="$(cd "$pkg_dir" && pwd)"
    echo "[TraceLens] SGLang package root: $pkg_dir"

    local patch_dir="$TRACELENS_REPO/$PATCH_SUBDIR/sglang_roofline_patches"
    if [[ ! -d "$patch_dir" ]]; then
        echo "ERROR: SGLang patch directory not found: $patch_dir" >&2
        exit 1
    fi

    local patch_count=0
    for patch_path in "$patch_dir"/*.patch; do
        [[ -f "$patch_path" ]] || continue
        patch_count=$((patch_count + 1))
        local patch_name
        patch_name="$(basename "$patch_path")"

        if $DRY_RUN; then
            echo "[DRY RUN] cd $pkg_dir && git apply --check $patch_path"
            cd "$pkg_dir" && git apply --check "$patch_path" 2>&1 || true
            continue
        fi

        echo "[TraceLens] Applying $patch_name ..."
        cd "$pkg_dir"
        if ! git apply "$patch_path"; then
            echo "ERROR: Failed to apply $patch_name" >&2
            exit 1
        fi
    done

    echo "[TraceLens] Applied $patch_count SGLang patch(es)"
}

# ── Apply patches ─────────────────────────────────────────────────────────────

case "$FW" in
    vllm)   apply_vllm_patches ;;
    sglang) apply_sglang_patches ;;
    *)      echo "ERROR: Unsupported framework '$FW'" >&2; exit 1 ;;
esac

# ── Install TraceLens ─────────────────────────────────────────────────────────

if $DO_INSTALL && ! $DRY_RUN; then
    echo "[TraceLens] Installing TraceLens package (--no-deps) ..."
    pip install --no-deps --break-system-packages "$TRACELENS_REPO" 2>/dev/null \
        || pip install --no-deps "$TRACELENS_REPO"
    echo "[TraceLens] TraceLens installed successfully"
fi

echo "[TraceLens] Done."
