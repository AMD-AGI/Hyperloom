#!/usr/bin/env bash
# Re-vendor ruppell into this skill from a ruppell checkout (VENDOR.md).
#   bash sync-ruppell.sh <ruppell checkout> [<commit>]   default commit: origin/main
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
src=${1:?usage: sync-ruppell.sh <ruppell checkout> [<commit>]}

git -C "$src" fetch -q origin
commit=$(git -C "$src" rev-parse --verify --quiet "${2:-origin/main}^{commit}") \
    || { echo "unknown commit in $src: ${2:-origin/main}" >&2; exit 1; }
if [ -z "$(git -C "$src" branch -r --contains "$commit")" ]; then
    echo "refusing: $commit is on no remote branch of $src; push it first" >&2
    exit 1
fi

# The plan job folders stay in ruppell (VENDOR.md, "Left out").
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$src" -c core.autocrlf=false archive --format=tar "$commit" -- . \
    ':(exclude,glob)tests/p[0-9]*/**' ':(exclude)tests/d4' ':(exclude)tests/drafter' \
    | tar -x -C "$tmp"

rm -rf "$here/ruppell"
mkdir "$here/ruppell"
(cd "$tmp" && tar -cf - .) | tar -x -C "$here/ruppell"
# runtime/build.sh records the commit from git, or from .commit "where the
# tree was copied without .git": this copy, mounted in a container, is that
# case (Cluster A job 61205 failed its build without it).
echo "$commit" > "$here/ruppell/.commit"

date=$(git -C "$src" log -1 --format=%cd --date=short "$commit")
sed -i "s/^commit: .*/commit: $commit/; s/^date: .*/date: $date/" "$here/VENDOR.md"
echo "vendored ruppell $commit ($date); review the change before committing it"
