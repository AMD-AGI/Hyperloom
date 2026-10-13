#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Step 0 of the analyze-upstream-pr skill: collect what review-pr's fetch.sh leaves out for a
# pull request in a repository this project does not own.
#
# usage: fetch_upstream.sh WORK OWNER/NAME NUMBER [TREE COMMIT]
# Writes into WORK, and prints WORK last:
#   comments_untrusted.txt  every issue comment, review and inline comment, oldest first, under a
#                           header that marks them as data. fetch.sh keeps only the author and
#                           users with write access, and GitHub shows who has write access only to
#                           a user with push permission, which upstream this project lacks.
#   linked.txt              the issues the PR closes, the items it supersedes, is superseded by,
#                           reverts or is reverted by, and every other item that mentions it
#   inclusion.txt           the PR's state, draft flag and merge commit, then the inclusion probe's
#                           answer for TREE at COMMIT, or a SKIPPED line when none was given
# Anything that cannot be collected exits non-zero, except a reference linked.txt cannot
# resolve, which its row names.

set -euo pipefail

# shellcheck disable=SC1091  # shellcheck checks lib.sh as a file of its own
. "$(dirname "${BASH_SOURCE[0]}")/../check-prior-art/lib.sh"

[ "$#" -eq 3 ] || [ "$#" -eq 5 ] || die "usage: fetch_upstream.sh WORK OWNER/NAME NUMBER [TREE COMMIT]"
WORK="$1"
REPO="$2"
PR="$3"
TREE="${4:-}"
COMMIT="${5:-}"
require_tools gh jq
check_repo "$REPO"
case "$PR" in
  '' | *[!0-9]*) die "not a pull request number: $PR" ;;
esac

mkdir -p "$WORK"
raw=$(mktemp -d)
trap 'rm -rf "$raw"' EXIT

gh api "repos/$REPO/pulls/$PR" < /dev/null > "$raw/pull.json" || die "cannot read $REPO#$PR"

fetch_comments "$REPO" "$PR" "$raw/comments.json"
render_comments "$raw/comments.json" > "$WORK/comments_untrusted.txt"

gh api --paginate "repos/$REPO/issues/$PR/timeline?per_page=100" < /dev/null \
  | jq -s 'add // []' > "$raw/timeline.json"
# shellcheck disable=SC2016  # $owner, $name and $number are GraphQL variables, not shell ones
gh api graphql -f owner="${REPO%%/*}" -f name="${REPO#*/}" -F number="$PR" -f query='
  query($owner: String!, $name: String!, $number: Int!) {
    repository(owner: $owner, name: $name) {
      pullRequest(number: $number) {
        closingIssuesReferences(first: 100) {
          nodes { number title state url repository { nameWithOwner } }
        }
      }
    }
  }' < /dev/null > "$raw/closing.json"

# A relation is a phrase, markdown emphasis allowed, followed by a run of references (#N,
# owner/name#N, or a pull or issue URL, each maybe a link's text) joined by commas, "and", "&",
# list bullets or whitespace. In this PR's own text the phrase names the relation. An item that
# mentions this PR stands in the inverse relation when such a phrase names this PR, is a revert
# when its title is one naming this PR, and otherwise only mentions it. Rows from this PR's text
# carry "?" until resolved below.
# shellcheck disable=SC2016  # $self, $pr and the rest are jq variables, not shell ones
jq -n -c --arg self "$REPO" --argjson pr "$PR" \
  --slurpfile pull "$raw/pull.json" --slurpfile comments "$raw/comments.json" \
  --slurpfile timeline "$raw/timeline.json" --slurpfile closing "$raw/closing.json" "$PR_JQ"'
  def relations: [
    ["supersedes|superseding|replaces|replacing|re-?lands?(?: of)?", "supersedes", "superseded-by"],
    ["superseded by|replaced by|in fav(?:o|ou)r of|closing for|continued in|duplicate of", "superseded-by", "supersedes"],
    ["revert(?:s|ing)?", "reverts", "reverted-by"],
    ["reverted (?:by|in)", "reverted-by", "reverts"]
  ];
  def ref_re: "(?:https?://github\\.com/([A-Za-z0-9-]+/[A-Za-z0-9._-]+)/(?:pull|issues)/|([A-Za-z0-9-]+/[A-Za-z0-9._-]+)?#)(\\d+)";
  def item: "(?:pr\\s+|pull request\\s+)?\\[?" + ref_re + "(?:\\]\\([^)\\s]*\\))?";
  def gap: "(?:[\\s,&:*_+-]|\\band\\b)+";
  def after($phrase): "(?i)(?<![a-z0-9])(?:" + $phrase + ")" + gap + item + "(?:" + gap + item + ")*";
  def refs($re; $home):
    [match($re; "g").string | scan(ref_re) | {repo: (.[0] // .[1] // $home), number: (.[2] | tonumber)}
     | select(.repo | test("/\\.\\.?$") | not)];
  def is_self: (.repo | ascii_downcase) == ($self | ascii_downcase) and .number == $pr;
  def back_relations($home):
    (.title + "\n" + (.body // "")) as $text
    | [relations[] as [$re, $rel, $inv] | select(any($text | refs(after($re); $home)[]; is_self)) | $inv]
      + (if (.title | test("(?i)^\\s*revert\\b")) and any(.title | refs(ref_re; $home)[]; is_self)
         then ["reverted-by"] else [] end)
    | unique | if length == 0 then ["mentioned-by"] else . end;

  ([{source: "title", text: $pull[0].title}, {source: "body", text: ($pull[0].body // "")}]
   + [$comments[0][] | {source: "\(.kind | ascii_downcase) \(.user) \(.at)", text: .body}]
   | .[] as $t
   | relations[] as [$re, $rel, $inv]
   | $t.text | refs(after($re); $self)[] | select(is_self | not)
   | {relation: $rel, repo, number, kind: "?", state: "?", title: "?", url: "?", source: $t.source}),
  ($timeline[0][] | select(.event == "cross-referenced") | .source.issue
   | .repository.full_name as $home
   | back_relations($home)[] as $rel
   | {relation: $rel, repo: $home, number, kind: (if .pull_request then "pr" else "issue" end),
      state: issue_state, title, url: .html_url, source: "mentions this PR"}),
  ($closing[0].data.repository.pullRequest.closingIssuesReferences.nodes[]
   | {relation: "closes", repo: .repository.nameWithOwner, number, kind: "issue",
      state: (.state | ascii_downcase), title, url, source: "closing reference"})
' > "$raw/links.jsonl"

# Resolves each reference named in this PR's text once. #N may be an issue or a PR: the issues
# endpoint answers for both and says which.
jq -r 'select(.kind == "?") | "\(.repo)\t\(.number)"' "$raw/links.jsonl" | LC_ALL=C sort -u \
  | while IFS=$'\t' read -r repo number; do
    if gh api "repos/$repo/issues/$number" < /dev/null > "$raw/ref.json" 2> "$raw/error"; then
      jq -c --arg repo "$repo" "$PR_JQ"'{repo: $repo, number, kind: (if .pull_request then "pr" else "issue" end),
        state: issue_state, title, url: .html_url}' "$raw/ref.json"
    else
      jq -n -c --arg repo "$repo" --argjson number "$number" --arg error "$(head -1 "$raw/error")" \
        '{repo: $repo, number: $number, kind: "-", state: "unresolved", title: $error, url: "-"}'
    fi
  done > "$raw/resolved.jsonl"

# One row per relation and item, their sources joined. An item that stands in a stronger
# relation is not listed again as one that only mentions this PR.
# shellcheck disable=SC2016  # $rel, $known and $related are jq variables, not shell ones
{
  printf 'relation\trepo\tnumber\tkind\tstate\ttitle\turl\tsource\n'
  jq -s -r --slurpfile resolved "$raw/resolved.jsonl" '
    def key: [(.repo | ascii_downcase), .number];
    def rank: .relation as $rel
      | ["closes", "supersedes", "superseded-by", "reverts", "reverted-by", "mentioned-by"] | index($rel);
    INDEX($resolved[]; key | tostring) as $known
    | map(if .kind == "?" then . + ($known[key | tostring] | {kind, state, title, url}) else . end)
    | [.[] | select(.relation != "mentioned-by") | key] as $related
    | map(select(.relation != "mentioned-by" or (key | IN($related[]) | not)))
    | group_by([rank, key])
    | .[] | .[0] + {source: (map(.source) | unique | join("; "))}
    | [.relation, .repo, .number, .kind, .state, .title, .url, .source] | @tsv' "$raw/links.jsonl"
} > "$WORK/linked.txt"

jq -r "$PR_JQ"'"state: \(pull_state)", "draft: \(.draft)", "merged_at: \(.merged_at // "-")",
  "merge_commit_sha: \(merge_sha)"' "$raw/pull.json" > "$WORK/inclusion.txt"
if [ -z "$TREE" ]; then
  printf 'SKIPPED: inclusion -- no tree and commit were given\n' >> "$WORK/inclusion.txt"
else
  probe=(--tree "$TREE" --commit "$COMMIT" --pr "$PR")
  merge=$(jq -r "$PR_JQ"'merge_sha' "$raw/pull.json")
  if [ "$merge" != - ]; then
    probe+=(--merge-commit-sha "$merge")
  fi
  # 0 included, 4 not included, 3 unknown; anything else is no answer, and its error is kept.
  status=0
  answer=$(python -m kernelforge.knowledge.pr_knowledge.inclusion "${probe[@]}" 2> "$raw/probe.err") || status=$?
  printf '%s#%s exit=%s %s\n' "$REPO" "$PR" "$status" "${answer:-$(tail -1 "$raw/probe.err")}" >> "$WORK/inclusion.txt"
fi

for artifact in comments_untrusted.txt linked.txt inclusion.txt; do
  printf '%-22s %s line(s)\n' "$artifact" "$(wc -l < "$WORK/$artifact" | tr -d ' ')"
done
printf '%s\n' "$WORK"
