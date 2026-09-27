#!/usr/bin/env bash
# Step 1 of the review-pr skill: collect the PR evidence every later step reads.
#
# usage: fetch.sh <PR-NUMBER> [WORK_DIR]
# Writes the artifacts listed in SKILL.md into WORK_DIR and prints WORK_DIR last.
# Anything that cannot be collected exits non-zero with the reason: reviewing on
# partial evidence produces a confident review of a diff nobody read.

set -euo pipefail

die() {
  echo "fetch.sh: $1" >&2
  exit 1
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || die "usage: fetch.sh <PR-NUMBER> [WORK_DIR]"
case "$1" in '' | *[!0-9]*) die "usage: fetch.sh <PR-NUMBER> [WORK_DIR]" ;; esac

PR="$1"
WORK="${2:-/tmp/hl-review-$PR}"
command -v gh >/dev/null 2>&1 || die "gh (GitHub CLI) is required"

REPO="${HL_REPO:-$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)}"
[ -n "$REPO" ] || die "cannot resolve the repository: run inside a checkout or set HL_REPO=owner/name"

mkdir -p "$WORK"

# GitHub computes mergeability lazily: the first read of a cold PR returns UNKNOWN and only
# starts the background job. Queried once, meta.txt says UNKNOWN for every PR nobody looked at
# recently, and the conflict check in SKILL.md Step 8 then never fires on a conflicting PR.
for attempt in 1 2 3 4 5; do
  gh pr view "$PR" --repo "$REPO" \
    --json number,title,author,state,headRefOid,baseRefName,baseRefOid,url,mergeable \
    --template '{{printf "number: %v\ntitle: %v\nauthor: %v\nstate: %v\nhead: %v\nbase_ref: %v\nbase_tip: %v\nurl: %v\nmergeable: %v\n" .number .title .author.login .state .headRefOid .baseRefName .baseRefOid .url .mergeable}}' \
    > "$WORK/meta.txt" || die "gh pr view failed for #$PR"
  grep -q '^mergeable: UNKNOWN$' "$WORK/meta.txt" || break
  # A merged or closed PR has no mergeability to compute and stays UNKNOWN for good; only an
  # open one is expected to settle. Re-reviewing a merged PR is a supported case, so it must
  # not be the thing that aborts the fetch.
  grep -q '^state: OPEN$' "$WORK/meta.txt" || break
  [ "$attempt" = 5 ] && die "GitHub did not settle mergeability for open #$PR; rerun rather than review the conflict axis blind"
  sleep 3
done

sed -n 's/^title: //p' "$WORK/meta.txt" > "$WORK/title.txt"
HEAD_SHA=$(sed -n 's/^head: //p' "$WORK/meta.txt")
BASE_TIP=$(sed -n 's/^base_tip: //p' "$WORK/meta.txt")
[ -n "$HEAD_SHA" ] && [ -n "$BASE_TIP" ] || die "PR metadata carries no head sha or base sha"

gh pr view "$PR" --repo "$REPO" --json body --jq '.body // ""' > "$WORK/body.txt"

# The merge base, never the base-branch tip. A diff taken against the tip attributes
# every commit main gained since the branch point to this PR, which is how a
# pre-existing behaviour gets reported as a regression (rule V1).
#
# Resolved from the base sha the PR recorded, not from the base branch name: once the PR is
# merged, the branch contains its commits, so the merge base against the branch is the head
# itself and the diff comes back empty. Re-reviewing a merged PR is a supported case.
BASE_SHA=$(gh api "repos/$REPO/compare/$BASE_TIP...$HEAD_SHA" --jq '.merge_base_commit.sha')
case "$BASE_SHA" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
  *) die "no merge base for $BASE_TIP...$HEAD_SHA" ;;
esac
printf '%s\n' "$BASE_SHA" > "$WORK/base.txt"

# Both endpoints that list a PR's commits return the OLDEST ones and stop: gh pr view at 100,
# compare at 250. X2 cares about the newest commit, the one that pushed the description out of
# date, so a silent stop would hide exactly the commit the rule is about. Say what was dropped
# and name the head commit, which is the newest by definition.
gh api "repos/$REPO/compare/$BASE_SHA...$HEAD_SHA" \
  --jq '.total_commits, (.commits[].commit.message | split("\n")[0])' > "$WORK/.commits.raw"
TOTAL_COMMITS=$(head -1 "$WORK/.commits.raw")
tail -n +2 "$WORK/.commits.raw" > "$WORK/commits.txt"
rm -f "$WORK/.commits.raw"
LISTED_COMMITS=$(wc -l < "$WORK/commits.txt" | tr -d ' ')
if [ "$LISTED_COMMITS" -lt "$TOTAL_COMMITS" ]; then
  printf '# TRUNCATED: %s of %s commits listed, oldest first. Head commit: %s\n' \
    "$LISTED_COMMITS" "$TOTAL_COMMITS" \
    "$(gh api "repos/$REPO/commits/$HEAD_SHA" --jq '.commit.message | split("\n")[0]')" \
    >> "$WORK/commits.txt"
fi

gh api -H "Accept: application/vnd.github.v3.diff" \
  "repos/$REPO/compare/$BASE_SHA...$HEAD_SHA" > "$WORK/diff.txt"
[ -s "$WORK/diff.txt" ] || die "the diff against $BASE_SHA is empty"

# files.txt and numstat.txt come from diff.txt rather than a second API call, so the
# three can never disagree about which paths this review covers.
awk '
  function flush() {
    if (path != "") {
      if (binary) printf "-\t-\t%s\n", path
      else printf "%d\t%d\t%s\n", add, del, path
    }
    path = ""; add = 0; del = 0; binary = 0; inhunk = 0; apath = ""
  }
  /^diff --git / { flush(); i = index($0, " b/"); if (i > 0) path = substr($0, i + 3); next }
  /^@@/ { inhunk = 1; next }
  !inhunk && /^--- / { apath = substr($0, 5); next }
  !inhunk && /^\+\+\+ / {
    bpath = substr($0, 5)
    if (bpath != "/dev/null") path = substr(bpath, 3)
    else if (apath != "/dev/null") path = substr(apath, 3)
    next
  }
  !inhunk && /^Binary files / { binary = 1; next }
  inhunk && /^\+/ { add++; next }
  inhunk && /^-/ { del++; next }
  END { flush() }
' "$WORK/diff.txt" > "$WORK/numstat.txt"
cut -f3- "$WORK/numstat.txt" > "$WORK/files.txt"
[ -s "$WORK/files.txt" ] || die "the diff names no changed path"

grep -E '(^|/)tests/' "$WORK/files.txt" > "$WORK/testfiles.txt" || : > "$WORK/testfiles.txt"

# The X family's mechanical input, the counterpart of testfiles.txt for the T family. An empty
# docfiles.txt beside a changed src/ file is the shape X3 fires on, so the emptiness is the signal.
grep -E '(^|/)docs/|(^|/)prompts/|\.md$|\.rst$' "$WORK/files.txt" > "$WORK/docfiles.txt" \
  || : > "$WORK/docfiles.txt"

# Queried by head sha, not by PR: gh pr checks reports the PR's checks whatever commit
# they ran against, so a green from an earlier push reads as a pass for the current one.
# Commit statuses too -- an external reporter posts a status, not a check run, and
# missing it turns a failing required status into an invisible pass.
printf '# check runs at head %s\n' "$HEAD_SHA" > "$WORK/ci.txt"
gh api --paginate "repos/$REPO/commits/$HEAD_SHA/check-runs" \
  --jq '.check_runs[] | [.name, (.conclusion // .status), .html_url] | @tsv' >> "$WORK/ci.txt"
gh api --paginate "repos/$REPO/commits/$HEAD_SHA/status" \
  --jq '.statuses[] | [.context, .state, (.target_url // "")] | @tsv' >> "$WORK/ci.txt"

{
  gh api --paginate "repos/$REPO/pulls/$PR/reviews" \
    --jq '.[] | select((.body // "") != "") | "[REVIEW \(.user.login) \(.state)]\n\(.body)\n"'
  gh api --paginate "repos/$REPO/pulls/$PR/comments" \
    --jq '.[] | "[INLINE \(.user.login)] \(.path):\(.line // .original_line // 0)\n\(.body)\n"'
  gh api --paginate "repos/$REPO/issues/$PR/comments" \
    --jq '.[] | "[COMMENT \(.user.login)]\n\(.body)\n"'
} > "$WORK/comments.txt"

# Other open PRs whose changed paths intersect this one's (rule V4). One query: gh
# returns each open PR's file list, and the intersection is computed locally rather
# than with one REST call per PR.
# shellcheck disable=SC2016  # $p is a jq variable, not a shell one
gh pr list --repo "$REPO" --state open --limit 100 --json number,title,url,files,changedFiles \
  --jq '.[] | . as $p | $p.files[].path
        | [$p.number, $p.title, $p.url, ., ($p.files | length), $p.changedFiles] | @tsv' \
  > "$WORK/.openprs.tsv"
# gh lists only the first 100 files of each PR. A larger PR overlapping this one on file 101
# is then invisible here, and an empty openprs.txt reads as "no in-flight conflict" (rule V4),
# so the PRs whose lists were cut are named rather than dropped.
awk -v self="$PR" -F'\t' '
  NR == FNR { want[$0] = 1; next }
  $1 == self { next }
  { if ($5 < $6) cut[$1] = $6 }
  ($4 in want) {
    key = $1
    if (!(key in seen)) { seen[key] = 1; order[++n] = key; title[key] = $2; url[key] = $3 }
    count[key]++
    paths[key] = paths[key] "    " $4 "\n"
  }
  END {
    for (i = 1; i <= n; i++) {
      k = order[i]
      printf "#%s  %d overlapping file(s)  %s  %s\n%s", k, count[k], title[k], url[k], paths[k]
    }
    for (k in cut)
      printf "# PARTIAL: #%s changes %s files, only its first 100 were compared\n", k, cut[k]
  }
' "$WORK/files.txt" "$WORK/.openprs.tsv" > "$WORK/openprs.txt"
rm -f "$WORK/.openprs.tsv"

for artifact in meta.txt title.txt body.txt diff.txt files.txt numstat.txt commits.txt \
  base.txt ci.txt comments.txt testfiles.txt docfiles.txt openprs.txt; do
  printf '%-16s %s line(s)\n' "$artifact" "$(wc -l < "$WORK/$artifact" | tr -d ' ')"
done

printf '%s\n' "$WORK"
