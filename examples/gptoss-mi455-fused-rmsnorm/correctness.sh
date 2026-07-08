#!/usr/bin/env bash
# Spot-check that a running server produces correct answers.
#   PORT=8001 ./correctness.sh
set -u
PORT="${PORT:-8001}"
BASE="http://127.0.0.1:$PORT"
ask() {
  local q="$1" n="${2:-14}"
  printf '  %-44s -> ' "$q"
  curl -s -m 30 "$BASE/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"gpt-oss-120b\",\"prompt\":\"$q\",\"max_tokens\":$n,\"temperature\":0}" \
    | python3 -c "import sys,json; print(repr(json.load(sys.stdin)['choices'][0]['text']))" 2>&1
}
echo "Correctness spot-checks (port $PORT):"
ask "The capital of France is" 4
ask "17 * 23 =" 8
ask "List three prime numbers greater than 100:" 12
echo "Expect: Paris ; 391 ; 101, 103, 107"
