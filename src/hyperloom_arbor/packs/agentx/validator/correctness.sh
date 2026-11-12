#!/usr/bin/env bash
# Correctness check: three short questions with distinct numeric answers; the server passes when the final number of
# at least two replies is the right one.
# Called via: docker exec <container> bash <this_script>
# Env vars: PORT, MODEL_PATH, MODEL_NAME, CORRECTNESS_OUTPUT
#
# One question answered with any non-empty reply passed garbled output ("2 + 4 = 4" ends in the right number). Three
# questions with distinct answers, each read by its final number, catch a model that emits garbage. Two of three, not
# three: under simulated speculative acceptance a healthy model's greedy reply can carry an accepted draft token it
# would not have chosen, so one miss is not a broken model. This is a smoke check, not an eval: only a real eval says
# the served model is right.

set -euo pipefail

: "${PORT:?PORT env var not set}"
: "${MODEL_PATH:?MODEL_PATH env var not set}"
: "${CORRECTNESS_OUTPUT:?CORRECTNESS_OUTPUT env var not set}"

PORT="$PORT" MODEL_PATH="$MODEL_PATH" CORRECTNESS_OUTPUT="$CORRECTNESS_OUTPUT" python3 - <<'PYEOF'
import json, os, re, sys, urllib.error, urllib.request

QUESTIONS = [("What is 17 + 25? Answer with just the number.", 42),
             ("What is 9 times 7? Answer with just the number.", 63),
             ("What is 144 divided by 12? Answer with just the number.", 12)]
NEEDED = 2
port, model, output_path = os.environ["PORT"], os.environ["MODEL_PATH"], os.environ["CORRECTNESS_OUTPUT"]
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)


def write(result):
    with open(output_path, "w") as f:
        json.dump(result, f)


def ask(question):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": question}],
                       "max_tokens": 512, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode())
    msg = ((data.get("choices") or [{}])[0].get("message")) or {}
    # A thinking model's reasoning_content carries the reply when max_tokens ends it before the answer.
    return (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()[-200:]


answers = []
for question, want in QUESTIONS:
    try:
        reply = ask(question)
    except urllib.error.HTTPError as e:
        write({"passed": False, "score": 0.0, "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"})
        sys.exit(0)
    except (urllib.error.URLError, OSError, ValueError) as e:
        write({"passed": False, "score": 0.0, "error": f"server unreachable or no usable response: {e}"})
        sys.exit(0)
    numbers = re.findall(r"-?\d+", reply.replace(",", ""))
    got = int(numbers[-1]) if numbers else None
    answers.append({"question": question, "want": want, "got": got, "reply": reply[:120]})
    print(f"{question} -> {reply[:80]!r}", file=sys.stderr)

right = sum(a["got"] == a["want"] for a in answers)
result = {"passed": right >= NEEDED, "score": round(right / len(QUESTIONS), 3), "right": right,
          "answers": answers}
if right < NEEDED:
    result["error"] = f"{right} of {len(QUESTIONS)} replies end in the right number (at least {NEEDED} needed)"
write(result)
PYEOF
