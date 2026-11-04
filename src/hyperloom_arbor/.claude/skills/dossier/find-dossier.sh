#!/usr/bin/env bash
# List the dossiers this checkout can see, each with the header rows that say
# whether it is the one you need (SKILL.md, "First: look it up").
#   bash find-dossier.sh [<root>...]
# Default roots: this repository's top level and, when it is a submodule (Arbor
# inside a project, as in the runner's external/arbor), the project around it.
# Then the dossiers campaigns shared (the knowledge collector's
# <share>/projects/<project>/<user>/<campaign>/knowledge/dossier/): read in
# place where the share is mounted (a Cluster B node), otherwise through
# `ssh arbor-share`. ARBOR_SHARE_ROOT and ARBOR_SHARE_HOST point elsewhere;
# ARBOR_DOSSIER_NO_SHARE=1 skips the share.
set -uo pipefail

# Each dossier index given: its header rows and how many parts are open.
# Exits 1 when none of the arguments is a file. Runs here, or on the share host.
report='
found=0
for index in "$@"; do
    [ -f "$index" ] || continue
    found=$((found + 1))
    echo "== $index"
    grep -E "^\| (Project|Pins|Platform|Workload|Written) \|" "$index" || echo "   (no header table)"
    # A part is an index row with a cell that starts with its status word;
    # emphasis and a trailing qualifier are allowed ("**partial**: no results").
    # Rows of other numbered tables (a list of gaps, say) have no status cell.
    parts=$(grep -c -i -E "^\| [0-9]+ \|.*\| [*_]*(written|partial|missing|not available)\b[^|]*\|" "$index")
    open=$(grep -c -i -E "^\| [0-9]+ \|.*\| [*_]*(missing|partial)\b[^|]*\|" "$index")
    echo "   parts listed: $parts; missing or partial: $open"
    # The updates table: one row per round, "| u1 | <date> | ...".
    round=$(grep -E "^\| u[0-9]+ \|" "$index" | tail -1 | cut -d"|" -f2,3 | tr -s " ")
    [ -n "$round" ] && echo "   latest update round:$round"
done
[ "$found" -gt 0 ]
'

roots=("$@")
if [ ${#roots[@]} -eq 0 ]; then
    top=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
    roots=("$top")
    super=$(git -C "$top" rev-parse --show-superproject-working-tree 2>/dev/null)
    [ -n "$super" ] && roots+=("$super")
fi

found=0
for root in "${roots[@]}"; do
    bash -c "$report" _ "$root"/docs/dossier/*/README.md && found=1
done
[ "$found" -gt 0 ] || echo "no dossier under: ${roots[*]} (looked for docs/dossier/*/README.md)"

[ "${ARBOR_DOSSIER_NO_SHARE:-0}" = 1 ] && exit 0
share_root=${ARBOR_SHARE_ROOT:-/shared/arbor}
share_glob="$share_root/projects/*/*/*/knowledge/dossier/README.md"
if [ -d "$share_root/projects" ]; then
    # shellcheck disable=SC2086  # the glob expands here
    bash -c "$report" _ $share_glob \
        || echo "no dossier on the share: $share_root/projects (looked for */*/*/knowledge/dossier/README.md)"
else
    host=${ARBOR_SHARE_HOST:-arbor-share}
    # The gateway takes commands, not piped stdin, so the listing travels inside
    # the command. It also exits 0 whatever happened, so a marker, not the exit
    # status, says the listing ran.
    b64=$(printf '%s' "$report" | base64 | tr -d '\n')
    out=$(timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" \
        "echo __ARBOR_SHARE__; echo $b64 | base64 -d | bash -s -- $share_glob" 2>&1 | tr -d '\r')
    if ! grep -q '^__ARBOR_SHARE__$' <<<"$out"; then
        echo "share not searched: ssh $host failed (${out##*$'\n'}); the realtime-collector skill sets it up"
    elif grep -q '^== ' <<<"$out"; then
        echo "-- on the share, $host:$share_root (read a file with: ssh $host cat <path>)"
        awk 'seen; /^__ARBOR_SHARE__$/ { seen = 1 }' <<<"$out"
    else
        echo "no dossier on the share: $host:$share_root/projects (looked for */*/*/knowledge/dossier/README.md)"
    fi
fi
