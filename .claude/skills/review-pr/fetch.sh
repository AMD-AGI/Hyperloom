#!/usr/bin/env bash
# Step 1 of the review-pr skill: collect the PR evidence every later step reads.
#
# usage: fetch.sh <PR-NUMBER> [WORK_DIR]
# Writes the artifacts listed in SKILL.md into WORK_DIR and prints WORK_DIR as the
# last line of stdout. Exits non-zero when a mandatory artifact (meta, base, diff,
# files, numstat) could not be collected; a soft collector that cannot run writes an
# empty file plus a reason in collect_errors.txt and the run continues.

set -euo pipefail

usage() {
  echo "usage: fetch.sh <PR-NUMBER> [WORK_DIR]" >&2
  exit 2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
fi
case "$1" in
  '' | *[!0-9]*) usage ;;
esac

PR="$1"
WORK="${2:-/tmp/hl-review-$PR}"
MANDATORY_FAILED=0
HEAD_SHA=""
BASE_REF=""
BASE_SHA=""

if ! command -v gh >/dev/null 2>&1; then
  echo "gh (GitHub CLI) is required" >&2
  exit 2
fi

REPO="${HL_REPO:-}"
if [ -z "$REPO" ]; then
  if ! REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null); then
    echo "cannot resolve the repository: run inside a checkout or set HL_REPO=owner/name" >&2
    exit 2
  fi
fi

mkdir -p "$WORK"
: > "$WORK/collect_errors.txt"

note() {
  printf '%s\n' "$1" >> "$WORK/collect_errors.txt"
}

# An artifact that could not be collected is still created, empty: the rules read the
# emptiness of openprs.txt as a signal, so "absent" and "empty" must not be the same
# state, and the reason must be recorded rather than dropped.
soft_fail() {
  local artifact="$1" reason="$2"
  : > "$WORK/$artifact"
  note "$artifact: $reason"
}

hard_fail() {
  soft_fail "$1" "$2"
  MANDATORY_FAILED=1
}

first_err() {
  local file="$1"
  if [ -s "$file" ]; then
    tr -d '\r' < "$file" | head -n 1
  else
    echo "no error output"
  fi
}

# Append one paginated API query to an artifact, keeping whatever earlier queries
# already wrote. Returns non-zero on failure so the caller can record it once.
api_append() {
  local out="$1" endpoint="$2" filter="$3"
  local tmp="$WORK/.api.out" err="$WORK/.api.err" rc=0
  if gh api --paginate "$endpoint" --jq "$filter" > "$tmp" 2>"$err"; then
    cat "$tmp" >> "$out"
  else
    rc=1
    note "${out##*/}: $endpoint failed: $(first_err "$err")"
  fi
  rm -f "$tmp" "$err"
  return "$rc"
}

collect_meta() {
  local err="$WORK/.meta.err"
  if gh pr view "$PR" --repo "$REPO" \
      --json number,title,author,state,headRefOid,baseRefName,url,mergeable \
      --template '{{printf "number: %v\ntitle: %v\nauthor: %v\nstate: %v\nhead: %v\nbase_ref: %v\nurl: %v\nmergeable: %v\n" .number .title .author.login .state .headRefOid .baseRefName .url .mergeable}}' \
      > "$WORK/meta.txt" 2>"$err"; then
    HEAD_SHA=$(sed -n 's/^head: //p' "$WORK/meta.txt")
    BASE_REF=$(sed -n 's/^base_ref: //p' "$WORK/meta.txt")
    sed -n 's/^title: //p' "$WORK/meta.txt" > "$WORK/title.txt"
  else
    hard_fail meta.txt "gh pr view failed: $(first_err "$err")"
    soft_fail title.txt "PR metadata unavailable"
  fi
  rm -f "$err"
  if ! is_sha "$HEAD_SHA"; then
    HEAD_SHA=""
    hard_fail meta.txt "PR metadata carries no head sha"
  fi
  return 0
}

collect_body() {
  local err="$WORK/.body.err"
  if ! gh pr view "$PR" --repo "$REPO" --json body --jq '.body // ""' \
      > "$WORK/body.txt" 2>"$err"; then
    soft_fail body.txt "gh pr view --json body failed: $(first_err "$err")"
  fi
  rm -f "$err"
  return 0
}

collect_commits() {
  local err="$WORK/.commits.err"
  if ! gh pr view "$PR" --repo "$REPO" --json commits \
      --jq '.commits[].messageHeadline' > "$WORK/commits.txt" 2>"$err"; then
    soft_fail commits.txt "gh pr view --json commits failed: $(first_err "$err")"
  fi
  rm -f "$err"
  return 0
}

# The merge base, never the current base-branch tip. A diff taken against the tip
# attributes every commit main gained since the branch point to this PR, which is how a
# pre-existing behaviour gets reported as a regression (rule V1).
resolve_base_local() {
  local url="https://github.com/$REPO.git" base_tip=""
  git rev-parse --git-dir >/dev/null 2>&1 || return 1
  git fetch --quiet --no-tags "$url" "refs/pull/$PR/head" >/dev/null 2>&1 || return 1
  git fetch --quiet --no-tags "$url" "$BASE_REF" >/dev/null 2>&1 || return 1
  base_tip=$(git rev-parse FETCH_HEAD 2>/dev/null) || return 1
  git merge-base "$base_tip" "$HEAD_SHA" 2>/dev/null || return 1
}

is_sha() {
  case "$1" in
    *[!0-9a-f]*) return 1 ;;
    ???????*) return 0 ;;
    *) return 1 ;;
  esac
}

collect_base() {
  local err="$WORK/.base.err" sha=""
  if [ -z "$HEAD_SHA" ] || [ -z "$BASE_REF" ]; then
    hard_fail base.txt "PR metadata unavailable, so no merge base could be resolved"
    return 0
  fi
  sha=$(gh api "repos/$REPO/compare/$BASE_REF...$HEAD_SHA" \
    --jq '.merge_base_commit.sha' 2>"$err") || sha=""
  if ! is_sha "$sha"; then
    if sha=$(resolve_base_local); then
      note "base.txt: compare API gave no merge base ($(first_err "$err")); used git merge-base"
    else
      sha=""
    fi
  fi
  case "$sha" in
    [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*)
      printf '%s\n' "$sha" > "$WORK/base.txt"
      BASE_SHA="$sha"
      ;;
    *)
      hard_fail base.txt "merge base for $BASE_REF...$HEAD_SHA unresolved: $(first_err "$err")"
      ;;
  esac
  rm -f "$err"
  return 0
}

collect_diff() {
  local err="$WORK/.diff.err"
  if [ -z "$BASE_SHA" ]; then
    hard_fail diff.txt "no merge base, and a diff against anything else is not this PR"
    return 0
  fi
  if gh api -H "Accept: application/vnd.github.v3.diff" \
      "repos/$REPO/compare/$BASE_SHA...$HEAD_SHA" > "$WORK/diff.txt" 2>"$err" \
      && [ -s "$WORK/diff.txt" ]; then
    rm -f "$err"
    return 0
  fi
  # Fallback only: gh pr diff is also merge-base relative, but it is computed by the
  # server from the PR rather than from the sha recorded in base.txt.
  if gh pr diff "$PR" --repo "$REPO" > "$WORK/diff.txt" 2>>"$err" \
      && [ -s "$WORK/diff.txt" ]; then
    note "diff.txt: compare API failed ($(first_err "$err")); used gh pr diff"
    rm -f "$err"
    return 0
  fi
  hard_fail diff.txt "no diff could be fetched: $(first_err "$err")"
  rm -f "$err"
  return 0
}

# files.txt and numstat.txt are derived from diff.txt rather than from a second API
# call, so the three can never disagree about which paths this review covers.
derive_file_lists() {
  if [ ! -s "$WORK/diff.txt" ]; then
    hard_fail numstat.txt "no diff to derive changed paths from"
    hard_fail files.txt "no diff to derive changed paths from"
    return 0
  fi
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
  if [ ! -s "$WORK/files.txt" ]; then
    hard_fail files.txt "the diff names no changed path"
    hard_fail numstat.txt "the diff names no changed path"
  fi
  return 0
}

derive_testfiles() {
  : > "$WORK/testfiles.txt"
  [ -s "$WORK/files.txt" ] || return 0
  grep -E '(^|/)tests/' "$WORK/files.txt" > "$WORK/testfiles.txt" || true
  return 0
}

# Queried by head sha, not by PR: gh pr checks reports the PR's checks whatever commit
# they ran against, so a green from an earlier push reads as a pass for the current one.
collect_ci() {
  if [ -z "$HEAD_SHA" ]; then
    soft_fail ci.txt "no head sha, so no check runs could be queried"
    return 0
  fi
  printf '# check runs at head %s\n' "$HEAD_SHA" > "$WORK/ci.txt"
  api_append "$WORK/ci.txt" "repos/$REPO/commits/$HEAD_SHA/check-runs" \
    '.check_runs[] | [.name, (.conclusion // .status), .html_url] | @tsv' || true
  # Commit statuses too: an external reporter posts a status, not a check run, and
  # missing it turns a failing required status into an invisible pass.
  api_append "$WORK/ci.txt" "repos/$REPO/commits/$HEAD_SHA/status" \
    '.statuses[] | [.context, .state, (.target_url // "")] | @tsv' || true
  return 0
}

collect_comments() {
  : > "$WORK/comments.txt"
  api_append "$WORK/comments.txt" "repos/$REPO/pulls/$PR/reviews" \
    '.[] | select((.body // "") != "") | "[REVIEW \(.user.login) \(.state)]\n\(.body)\n"' || true
  api_append "$WORK/comments.txt" "repos/$REPO/pulls/$PR/comments" \
    '.[] | "[INLINE \(.user.login)] \(.path):\(.line // .original_line // 0)\n\(.body)\n"' || true
  api_append "$WORK/comments.txt" "repos/$REPO/issues/$PR/comments" \
    '.[] | "[COMMENT \(.user.login)]\n\(.body)\n"' || true
  return 0
}

# Other open PRs whose changed paths intersect this one's (rule V4). One GraphQL query:
# gh returns each open PR's file list, and the intersection is computed against
# files.txt locally rather than with one REST call per PR.
collect_openprs() {
  local err="$WORK/.openprs.err" raw="$WORK/.openprs.tsv"
  : > "$WORK/openprs.txt"
  if [ ! -s "$WORK/files.txt" ]; then
    soft_fail openprs.txt "no changed paths to intersect"
    return 0
  fi
  if ! gh pr list --repo "$REPO" --state open --limit 100 \
      --json number,title,url,files \
      --jq '.[] | {number, title, url, path: .files[].path} | [.number, .title, .url, .path] | @tsv' \
      > "$raw" 2>"$err"; then
    soft_fail openprs.txt "gh pr list failed: $(first_err "$err")"
    rm -f "$raw" "$err"
    return 0
  fi
  awk -v self="$PR" -F'\t' '
    NR == FNR { want[$0] = 1; next }
    $1 == self { next }
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
    }
  ' "$WORK/files.txt" "$raw" > "$WORK/openprs.txt"
  rm -f "$raw" "$err"
  return 0
}

collect_meta
collect_body
collect_commits
collect_base
collect_diff
derive_file_lists
derive_testfiles
collect_ci
collect_comments
collect_openprs

for artifact in meta.txt title.txt body.txt diff.txt files.txt numstat.txt commits.txt \
  base.txt ci.txt comments.txt testfiles.txt openprs.txt; do
  printf '%-16s %s line(s)\n' "$artifact" "$(wc -l < "$WORK/$artifact" | tr -d ' ')"
done

if [ -s "$WORK/collect_errors.txt" ]; then
  echo "collection errors (see $WORK/collect_errors.txt):"
  sed 's/^/  /' "$WORK/collect_errors.txt"
fi

if [ "$MANDATORY_FAILED" -ne 0 ]; then
  echo "a mandatory artifact is missing; reviewing on this evidence would be reviewing on" >&2
  echo "nothing. Fix the errors above and rerun." >&2
  printf '%s\n' "$WORK"
  exit 1
fi

printf '%s\n' "$WORK"
