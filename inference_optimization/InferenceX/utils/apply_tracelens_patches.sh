#!/usr/bin/env bash
#
# Apply TraceLens inference-analysis patches to the current Python environment.
#
# Patches vLLM or SGLang in-place (inside a container) so profiling traces
# include CUDA graph capture traces and roofline annotations.
#
# The patch source is the public AMD-AGI/TraceLens repo by default.  The
# script will clone it automatically if no local copy is available.
#
# Usage:
#   bash apply_tracelens_patches.sh [OPTIONS]
#
# Options:
#   --framework  vllm|sglang          auto-detect from $FRAMEWORK or Python
#   --version    v0.18|v0.19|...      vLLM only; auto-detect if omitted
#   --tracelens  /path/to/TraceLens   use existing local clone
#   --clone-to   /path/to/dir         clone destination (default: /tmp/TraceLens)
#   --git-ref    branch/tag/sha       clone ref (default: main)
#   --install    pip-install TraceLens (default)
#   --no-install skip pip install
#   --dry-run    show what would happen without applying
#   --list       list available patches and exit
#
# Environment variables (all optional):
#   TRACELENS_REPO      path to existing local TraceLens clone
#   TRACELENS_GIT_URL   clone URL (default: https://github.com/AMD-AGI/TraceLens.git)
#   TRACELENS_GIT_REF   branch/tag (default: main)
#   FRAMEWORK           "vllm" or "sglang"
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRACELENS_GIT_URL="${TRACELENS_GIT_URL:-https://github.com/AMD-AGI/TraceLens.git}"
TRACELENS_GIT_REF="${TRACELENS_GIT_REF:-main}"

FRAMEWORK_ARG=""
VLLM_VERSION_ARG=""
TRACELENS_REPO="${TRACELENS_REPO:-}"
CLONE_TO=""
DO_INSTALL=true
DRY_RUN=false
LIST_PATCHES=false

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
        --list)        LIST_PATCHES=true; shift ;;
        -h|--help)
            sed -n '2,/^$/{ s/^# //; s/^#$//; p; }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Resolve or clone TraceLens repo ───────────────────────────────────────────

PATCH_SUBDIR="examples/custom_workflows/inference_analysis"

ensure_tracelens_repo() {
    if [[ -n "$TRACELENS_REPO" && -d "$TRACELENS_REPO/$PATCH_SUBDIR" ]]; then
        echo "[TraceLens] Using existing repo: $TRACELENS_REPO"
        return 0
    fi

    local dest="${CLONE_TO:-}"
    if [[ -z "$dest" ]]; then
        if [[ -n "$TRACELENS_REPO" ]]; then
            dest="$TRACELENS_REPO"
        else
            dest="/tmp/TraceLens"
        fi
    fi

    if [[ -d "$dest/$PATCH_SUBDIR" ]]; then
        echo "[TraceLens] Found existing clone at $dest"
        TRACELENS_REPO="$dest"
        return 0
    fi

    echo "[TraceLens] Cloning TraceLens..."
    echo "[TraceLens]   URL: $TRACELENS_GIT_URL"
    echo "[TraceLens]   Ref: $TRACELENS_GIT_REF"
    echo "[TraceLens]   Dest: $dest"

    if ! command -v git &>/dev/null; then
        echo "ERROR: git not available — cannot clone TraceLens." >&2
        echo "       Mount or copy TraceLens manually, then pass --tracelens /path" >&2
        exit 1
    fi

    git clone --depth 1 --branch "$TRACELENS_GIT_REF" "$TRACELENS_GIT_URL" "$dest"
    TRACELENS_REPO="$dest"

    if [[ ! -d "$TRACELENS_REPO/$PATCH_SUBDIR" ]]; then
        echo "ERROR: Cloned repo missing $PATCH_SUBDIR — wrong repo or branch?" >&2
        exit 1
    fi
    echo "[TraceLens] Clone successful"
}

ensure_tracelens_repo

# ── Discover available patches ────────────────────────────────────────────────

PATCH_DIR="$TRACELENS_REPO/$PATCH_SUBDIR"

discover_vllm_patches() {
    local -a patches=()
    for f in "$PATCH_DIR"/*vllm*.patch; do
        [[ -f "$f" ]] || continue
        patches+=("$(basename "$f")")
    done
    printf '%s\n' "${patches[@]}" | sort -V
}

if $LIST_PATCHES; then
    echo "Available vLLM patches:"
    discover_vllm_patches | while read -r p; do echo "  $p"; done
    echo ""
    if [[ -d "$PATCH_DIR/sglang_roofline_patches" ]]; then
        echo "Available SGLang patches:"
        for f in "$PATCH_DIR/sglang_roofline_patches/"*.patch; do
            [[ -f "$f" ]] && echo "  $(basename "$f")"
        done
    fi
    exit 0
fi

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
        echo "ERROR: Cannot detect framework. Neither vllm nor sglang importable." >&2
        echo "       Pass --framework vllm|sglang explicitly." >&2
        exit 1
    fi
}

FW="$(detect_framework)"
echo "[TraceLens] Detected framework: $FW"

# ── vLLM patching ────────────────────────────────────────────────────────────

apply_vllm_patches() {
    local pkg_dir
    pkg_dir="$(python3 -c "import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), '..'))")"
    pkg_dir="$(cd "$pkg_dir" && pwd)"
    echo "[TraceLens] vLLM package root: $pkg_dir"

    # ── Resolve version → patch file ──
    local ver="$VLLM_VERSION_ARG"
    local raw_ver=""

    if [[ -z "$ver" ]]; then
        raw_ver="$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")"
        echo "[TraceLens] Detected vLLM version: $raw_ver"

        # Extract minor version: 0.18.x → v0.18, 0.19.1 → v0.19, etc.
        case "$raw_ver" in
            0.*.*)
                local minor
                minor="$(echo "$raw_ver" | cut -d. -f1-2)"
                ver="v${minor}"
                ;;
            *)
                ver=""
                ;;
        esac
    fi

    # Normalize user-supplied short forms: v18 → v0.18, v19 → v0.19
    case "$ver" in
        v18) ver="v0.18" ;;
        v19) ver="v0.19" ;;
    esac

    # Search for a matching patch file.  Try two naming conventions:
    #   config_vllm_v0.XX.0.patch  (public TraceLens)
    #   vllm_v0.XX.0.patch         (TraceLens-internal)
    local patch_file=""
    if [[ -n "$ver" ]]; then
        local minor_num="${ver#v0.}"  # e.g. "18", "19"
        for candidate in \
            "config_vllm_v0.${minor_num}.0.patch" \
            "vllm_v0.${minor_num}.0.patch"; do
            if [[ -f "$PATCH_DIR/$candidate" ]]; then
                patch_file="$candidate"
                break
            fi
        done
    fi

    if [[ -z "$patch_file" ]]; then
        echo ""
        echo "ERROR: No matching patch found for vLLM version '${raw_ver:-$ver}'." >&2
        echo ""
        echo "Available patches in $PATCH_DIR:" >&2
        discover_vllm_patches | while read -r p; do echo "  $p" >&2; done
        echo ""
        echo "You can select one explicitly with --version, e.g.:" >&2
        echo "  --version v0.18" >&2
        echo "  --version v0.19" >&2
        exit 1
    fi

    local patch_path="$PATCH_DIR/$patch_file"
    echo "[TraceLens] Selected patch: $patch_file"
    echo "[TraceLens] Applying to: $pkg_dir"

    if $DRY_RUN; then
        echo "[DRY RUN] cd $pkg_dir && git apply --check $patch_path"
        cd "$pkg_dir" && git apply --check "$patch_path" 2>&1 || true
        return
    fi

    cd "$pkg_dir"
    if git apply "$patch_path" 2>/dev/null; then
        echo "[TraceLens] Patch applied via git apply"
    elif patch -p1 --fuzz=10 < "$patch_path"; then
        echo "[TraceLens] Patch applied via patch -p1 (fallback)"
    else
        echo "ERROR: Failed to apply $patch_file" >&2
        echo "       The installed vLLM may not match the patch." >&2
        echo "       Try --version to select a different patch." >&2
        exit 1
    fi
}

# ── SGLang patching ───────────────────────────────────────────────────────────

apply_sglang_patches() {
    local pkg_dir
    pkg_dir="$(python3 -c "import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))")"
    pkg_dir="$(cd "$pkg_dir" && pwd)"
    echo "[TraceLens] SGLang package root: $pkg_dir"

    local sglang_patch_dir="$PATCH_DIR/sglang_roofline_patches"
    if [[ ! -d "$sglang_patch_dir" ]]; then
        echo "ERROR: SGLang patch directory not found: $sglang_patch_dir" >&2
        exit 1
    fi

    local count=0
    for patch_path in "$sglang_patch_dir"/*.patch; do
        [[ -f "$patch_path" ]] || continue
        count=$((count + 1))
        local name
        name="$(basename "$patch_path")"

        if $DRY_RUN; then
            echo "[DRY RUN] cd $pkg_dir && git apply --check $patch_path"
            cd "$pkg_dir" && git apply --check "$patch_path" 2>&1 || true
            continue
        fi

        echo "[TraceLens] Applying $name ..."
        cd "$pkg_dir"
        if ! git apply "$patch_path"; then
            echo "ERROR: Failed to apply $name" >&2
            exit 1
        fi
    done
    echo "[TraceLens] Applied $count SGLang patch(es)"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

case "$FW" in
    vllm)   apply_vllm_patches ;;
    sglang) apply_sglang_patches ;;
    *)      echo "ERROR: Unsupported framework '$FW'" >&2; exit 1 ;;
esac

# ── Install TraceLens ─────────────────────────────────────────────────────────

if $DO_INSTALL && ! $DRY_RUN; then
    echo "[TraceLens] Installing TraceLens package (--no-deps) ..."
    install_src="$TRACELENS_REPO"
    if ! touch "$TRACELENS_REPO/.write_test" 2>/dev/null; then
        install_src="/tmp/TraceLens-install"
        echo "[TraceLens] Repo is read-only, copying to $install_src for pip install ..."
        rm -rf "$install_src"
        cp -r "$TRACELENS_REPO" "$install_src"
    else
        rm -f "$TRACELENS_REPO/.write_test"
    fi
    pip install --no-deps --break-system-packages "$install_src" 2>/dev/null \
        || pip install --no-deps "$install_src"
    echo "[TraceLens] TraceLens installed"
fi

echo "[TraceLens] Done."
