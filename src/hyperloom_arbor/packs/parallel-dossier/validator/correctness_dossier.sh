#!/usr/bin/env bash
# The parallel-dossier pack's correctness: the hard requirements on a draft, pass or fail.
#   - inputs/ is exactly as dossier_round.sh committed it (the commit in .git/PAR_INPUTS):
#     the evidence is read-only;
#   - the drafts are Markdown under reports/, each under 400 KB;
#   - nothing in the workspace's tracked files looks like a credential.
# Run by the harness on the host with WORKSPACE_ROOT and CORRECTNESS_OUTPUT set.
set -uo pipefail
ws="${WORKSPACE_ROOT:?WORKSPACE_ROOT is not set}"
out="${CORRECTNESS_OUTPUT:?CORRECTNESS_OUTPUT is not set}"
fail() { printf '{"passed": false, "score": 0.0, "error": "%s"}\n' "$1" > "$out"; exit 0; }

base=$(cat "$ws/.git/PAR_INPUTS" 2>/dev/null)
[ -n "$base" ] || fail "no .git/PAR_INPUTS: the workspace was not built by dossier_round.sh"
changed=$(git -C "$ws" diff --name-only "$base" -- inputs/ | head -5 | tr '\n' ' ')
[ -z "$changed" ] || fail "inputs/ changed since dossier_round.sh committed it: $changed"
for f in "$ws"/reports/*; do
    [ -e "$f" ] || continue
    case "$f" in *.md|*/.keep) ;; *) fail "reports/ holds $(basename "$f"): the drafts are Markdown" ;; esac
    [ "$(wc -c < "$f")" -lt 400000 ] || fail "$(basename "$f") is over 400 KB"
done

# Credentials: Hugging Face, Anthropic, GitHub, Azure gateway keys, docker auth.
hits=$(git -C "$ws" grep -I -l -E 'hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|Ocp-Apim-Subscription-Key|ANTHROPIC_(API_KEY|AUTH_TOKEN)[[:space:]]*=[[:space:]]*[^[:space:]$]|"auth"[[:space:]]*:[[:space:]]*"[A-Za-z0-9+/=]{20,}"' HEAD -- . ':(exclude)inputs/' 2>/dev/null | sed 's/^HEAD://' | head -3 | tr '\n' ' ')
[ -z "$hits" ] || fail "possible credential in: $hits"

printf '{"passed": true, "score": 1.0}\n' > "$out"
