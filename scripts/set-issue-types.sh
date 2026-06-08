#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

#
# Set Issue Types on existing issues based on their labels.
# Prerequisites: gh CLI authenticated with repo access
#
# Usage:
#   chmod +x scripts/set-issue-types.sh
#   ./scripts/set-issue-types.sh
#
set -euo pipefail

REPO="AMD-AGI/Hyperloom"
ORG="AMD-AGI"

echo "=== Step 1: Fetch available Issue Types ==="

TYPES_RESULT=$(gh api graphql -f query='
  query($org: String!) {
    organization(login: $org) {
      issueTypes(first: 20) {
        nodes {
          id
          name
        }
      }
    }
  }' -f org="$ORG")

echo "Available Issue Types:"
echo "$TYPES_RESULT" | jq -r '.data.organization.issueTypes.nodes[] | "  \(.name): \(.id)"'

BUG_TYPE_ID=$(echo "$TYPES_RESULT" | jq -r '.data.organization.issueTypes.nodes[] | select(.name == "Bug") | .id')
FEATURE_TYPE_ID=$(echo "$TYPES_RESULT" | jq -r '.data.organization.issueTypes.nodes[] | select(.name == "Feature") | .id')
TASK_TYPE_ID=$(echo "$TYPES_RESULT" | jq -r '.data.organization.issueTypes.nodes[] | select(.name == "Task") | .id')

echo ""
echo "  Bug     ID: $BUG_TYPE_ID"
echo "  Feature ID: $FEATURE_TYPE_ID"
echo "  Task    ID: $TASK_TYPE_ID"

echo ""
echo "=== Step 2: Fetch all open issues ==="

ISSUES=$(gh issue list --repo "$REPO" --state all --json number,title,labels,id --limit 200)
ISSUE_COUNT=$(echo "$ISSUES" | jq length)
echo "  Found $ISSUE_COUNT issues"

echo ""
echo "=== Step 3: Assign Issue Types based on labels ==="

set_issue_type() {
  local issue_node_id="$1"
  local type_id="$2"
  local issue_num="$3"
  local type_name="$4"

  if [ -z "$type_id" ] || [ "$type_id" = "null" ]; then
    echo "    Skipping #$issue_num: $type_name type not found in org"
    return
  fi

  gh api graphql -f query='
    mutation($issueId: ID!, $typeId: ID!) {
      updateIssue(input: {id: $issueId, issueTypeId: $typeId}) {
        issue { id number }
      }
    }' -f issueId="$issue_node_id" -f typeId="$type_id" 2>/dev/null && \
    echo "    #$issue_num -> $type_name" || \
    echo "    #$issue_num -> $type_name (failed)"
}

echo "$ISSUES" | jq -c '.[]' | while read -r issue; do
  num=$(echo "$issue" | jq -r '.number')
  title=$(echo "$issue" | jq -r '.title')
  node_id=$(echo "$issue" | jq -r '.id')
  labels=$(echo "$issue" | jq -r '.labels[].name' 2>/dev/null | tr '\n' ',' || true)

  # Determine type based on labels (check both old and new label names)
  if echo "$labels" | grep -qiE "bug|type:bug|Bug"; then
    set_issue_type "$node_id" "$BUG_TYPE_ID" "$num" "Bug"
  elif echo "$labels" | grep -qiE "feature|type:feature|Feature|Enhancement"; then
    set_issue_type "$node_id" "$FEATURE_TYPE_ID" "$num" "Feature"
  elif echo "$labels" | grep -qiE "task|type:task|Task"; then
    set_issue_type "$node_id" "$TASK_TYPE_ID" "$num" "Task"
  else
    echo "    #$num: no matching label, skipping (title: $title)"
  fi
done

echo ""
echo "=== Done! ==="
echo "Verify in the GitHub UI that Issue Types are set correctly."
