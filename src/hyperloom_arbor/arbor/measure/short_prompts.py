"""The short-prompt check: one tree, two server processes, the same greedy answers (slice 3).

Hyperloom's MB found a KEEP whose tree gave wrong output on short prompts in about one server
process in three (AITER's vectorized_5d mha_batch_prefill), and GSM8K did not see it: a long,
batched, averaged evaluation. MB first compared each candidate with its parent and
dropped that: exact greedy matching fires on the largest gains (fp8 KV, quantized GEMMs move
greedy tokens early). A defect that is wrong in every process is one GSM8K already sees; the
gap is the defect that is wrong in some processes. So the check compares two processes of the
SAME tree (operator, 2026-10-28; docs/component-mode.md §8):

- **The probe**, MB's shape so the two teams' floors measure the same thing: 64 prompts of 1 to
  64 tokens, each starting with a different token (the prefix cache cannot serve one from
  another), 32 sent one at a time and 32 at once (short sequences batched in one prefill);
  greedy, 32 tokens, compared by token id. The prompts are built once per session through the
  server's /tokenize and reused, so every process gets the same ids. It runs after correctness
  on every measured iteration's live server: recorded, never a verdict.
- **At a KEEP** the harness boots the tree once more and probes again. The first KEEP's count of
  prompts sent one at a time that the two processes answer differently is the session's floor;
  a later KEEP above it is a defect, and the KEEP becomes REVERT. The batched half is recorded
  and refuses nothing: its answers move with the batch a boot happens to form (check_keep says
  how much). Two processes catch a one-in-three defect about 4 times in 9, and every later
  KEEP that carries it is another chance.
- **A check that cannot finish refuses nothing**, as MB's does: an incomplete probe, no time
  left before the deadline, or a second launch that never comes up is recorded and the KEEP
  stands. The host re-gates every KEEP anyway.

Component mode only; standalone never calls this.
"""

from __future__ import annotations

import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

GENERATOR = "arbor-short-prompts/1"
SEQUENTIAL, BATCHED = "sequential", "batched"
# MB's lengths (Hyperloom hl-short-prompts/1): single tokens, every odd length to 33, and the
# lengths around the block sizes a prefill kernel tiles by.
LENGTHS = {
    SEQUENTIAL: (1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 19, 21, 23, 25, 27, 29,
                 31, 32, 33, 35, 40, 47, 48, 55, 63, 64),
    BATCHED: (1, 2, 3, 5, 7, 9, 11, 12, 14, 17, 18, 20, 22, 24, 26, 28, 30, 31, 33, 34, 36, 37,
              39, 41, 43, 45, 49, 51, 57, 59, 61, 64),
}
MAX_TOKENS = 32
_START = """Autumn Bakers Candles Distant Every Fishermen Gardens Harbors Islands Journeys Kettles
Lanterns Markets Northern Orchards Pilgrims Quarries Rivers Sailors Tailors Umbrellas Valleys
Weavers Yesterday Zebras Across After Along Among Another Around Because Before Behind Below
Beneath Beside Between Beyond Bright Clouds Copper Desert Dozens Early Engines Evening Falling
Fields Forests Golden Granite Heavy Hidden Inside Kitchens Letters Lighthouses Meadows Midnight
Mountains Nothing Often Painters Perhaps Quiet Rain Several Silver Slowly Soldiers Songs Stone
Summer Through Today Toward Travelers Under Villages Water Whenever While Winter Within Wooden""".split()
_WORDS = """the a of to and in on at by with from over under near after before while because old
new small large quiet busy bright dark warm cold early late river town road bridge market
harbor garden kitchen window letter lantern boat engine bread apple stone wall square light
morning evening winter spring people children farmers teachers painters travelers walked talked
carried brought wrote opened closed lit sang crossed waited watched slowly quickly gently never
always often again together across along beyond inside behind each every some many few""".split()


class ProbeError(Exception):
    """The probe could not be built or run; the message says why."""


# ── the server's HTTP ───────────────────────────────────────────────────────────

def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def served_model(base: str, timeout: float = 30) -> str:
    """The first model the server lists: what a completion must name."""
    with urllib.request.urlopen(f"{base}/v1/models", timeout=timeout) as r:
        data = json.load(r).get("data") or []
    if not data:
        raise ProbeError(f"{base}/v1/models lists no model")
    return data[0]["id"]


def tokenizer(base: str, model: str, timeout: float = 30) -> Callable[[str], list[int]]:
    """The server's own tokenizer, through /tokenize, with no special tokens."""
    def tokenize(text: str) -> list[int]:
        out = _post(f"{base}/tokenize", {"model": model, "prompt": text, "add_special_tokens": False},
                    timeout)
        ids = out.get("tokens")
        if not isinstance(ids, list) or not ids:
            raise ProbeError(f"{base}/tokenize returned no tokens")
        return [int(t) for t in ids]
    return tokenize


# ── the prompt set ──────────────────────────────────────────────────────────────

def build_prompts(tokenize: Callable[[str], list[int]], seed: int = 0) -> list[dict]:
    """64 prompts of MB's lengths, each starting with a token no other prompt starts with.

    Each is the first ``length`` tokens of a passage of its own: a start word, then words drawn
    from a seeded generator. A start word whose first token is taken, or whose passage is too
    short, is passed over.
    """
    rng = random.Random(f"{GENERATOR}:{seed}")
    starts = list(_START)
    rng.shuffle(starts)
    used: set[int] = set()
    out = []
    for mode in (SEQUENTIAL, BATCHED):
        for i, length in enumerate(LENGTHS[mode]):
            while True:
                if not starts:
                    raise ProbeError("ran out of start words with a first token of their own")
                words = [starts.pop()] + [rng.choice(_WORDS) for _ in range(2 * length + 8)]
                ids = tokenize(" ".join(words))
                if len(ids) >= length and ids[0] not in used:
                    break
            used.add(ids[0])
            out.append({"id": f"{mode[0]}{i:02d}", "mode": mode, "length": length,
                        "tokens": ids[:length]})
    return out


def session_prompts(session_dir, base: str, model: str) -> list[dict]:
    """The session's prompt set: built once, then read, so every process gets the same ids."""
    path = Path(session_dir) / "short_prompts" / "prompts.json"
    if path.is_file():
        return json.loads(path.read_text())["prompts"]
    prompts = build_prompts(tokenizer(base, model))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".prompts.json.tmp")
    tmp.write_text(json.dumps({"generator": GENERATOR, "model": model, "prompts": prompts}) + "\n")
    tmp.replace(path)
    return prompts


# ── the probe ───────────────────────────────────────────────────────────────────

def _complete(base: str, model: str, p: dict, timeout: float) -> dict:
    body = {"model": model, "prompt": p["tokens"], "max_tokens": MAX_TOKENS, "temperature": 0,
            "logprobs": 1, "return_tokens_as_token_ids": True}
    try:
        choice = _post(f"{base}/v1/completions", body, timeout)["choices"][0]
    except Exception as e:  # one prompt's failure is that prompt's, not the probe's
        return {"id": p["id"], "error": f"{type(e).__name__}: {e}"[:300]}
    toks = ((choice.get("logprobs") or {}).get("tokens")) or []
    ids = None
    if toks and all(isinstance(t, str) and t.startswith("token_id:") for t in toks):
        ids = [int(t.split(":", 1)[1]) for t in toks]
    return {"id": p["id"], "text": choice.get("text", ""), "tokens": ids}


def probe(base: str, prompts: list[dict], seconds: float, model: str | None = None) -> dict:
    """Every prompt once: the sequential ones one at a time, the batched ones all at once."""
    end = time.time() + seconds
    model = model or served_model(base)
    left = lambda: max(1.0, min(120.0, end - time.time()))
    outputs = []
    for p in (p for p in prompts if p["mode"] == SEQUENTIAL):
        outputs.append(_complete(base, model, p, left()) if time.time() < end
                       else {"id": p["id"], "error": "no time left"})
    batched = [p for p in prompts if p["mode"] == BATCHED]
    if batched and time.time() < end:
        with ThreadPoolExecutor(max_workers=len(batched)) as pool:
            outputs += list(pool.map(lambda p: _complete(base, model, p, left()), batched))
    else:
        outputs += [{"id": p["id"], "error": "no time left"} for p in batched]
    done = sum("error" not in o for o in outputs)
    return {"generator": GENERATOR, "model": model, "prompts": len(prompts), "answered": done,
            "complete": done == len(prompts), "outputs": outputs}


def compare(a: dict, b: dict) -> dict:
    """The prompts two probes answered differently, and where each first differs.

    Token ids when both sides have them for every prompt, the text otherwise.
    """
    oa = {o["id"]: o for o in a.get("outputs", []) if "error" not in o}
    ob = {o["id"]: o for o in b.get("outputs", []) if "error" not in o}
    both = sorted(set(oa) & set(ob))
    by_ids = all(oa[i].get("tokens") is not None and ob[i].get("tokens") is not None for i in both)
    differ = {}
    for i in both:
        x, y = ((oa[i]["tokens"], ob[i]["tokens"]) if by_ids else (oa[i]["text"], ob[i]["text"]))
        if x != y:
            differ[i] = next((k for k, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
    return {"compared": len(both), "by": "token ids" if by_ids else "text", "differ": len(differ),
            "first_difference": differ}


# ── the harness's two uses ──────────────────────────────────────────────────────

def _write(path: Path, doc: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(path)


def probe_iteration(config: dict, session_dir, iter_dir, seconds: float) -> dict:
    """The probe on this iteration's live server, recorded in iter_N/short_prompts.json."""
    base = f"http://127.0.0.1:{config.get('port', 8888)}"
    try:
        model = served_model(base)
        doc = probe(base, session_prompts(session_dir, base, model), seconds, model)
    except Exception as e:
        doc = {"generator": GENERATOR, "complete": False, "error": f"{type(e).__name__}: {e}"[:400]}
    _write(Path(iter_dir) / "short_prompts.json", doc)
    return doc


def check_keep(config: dict, session_dir, iter_dir, *, relaunch: Callable[[], bool],
               probe_seconds: Callable[[], float | None]) -> dict:
    """At a KEEP: a second process of the tree, probed and compared with the first.

    ``relaunch`` stops the server Arbor runs and boots the tree again, True once healthy.
    ``probe_seconds`` is the probe's time (its phase cap, or the deadline), read after the
    boot. Returns the record written to iter_N/short_prompts_check.json, whose ``verdict``
    is ``defect`` (REVERT), ``ok``, ``floor`` (the session's first KEEP sets it) or
    ``not_run`` (the KEEP stands).
    """
    iter_dir, session_dir = Path(iter_dir), Path(session_dir)
    out = iter_dir / "short_prompts_check.json"
    first_path = iter_dir / "short_prompts.json"
    first = json.loads(first_path.read_text()) if first_path.is_file() else {}
    rec: dict = {"verdict": "not_run"}
    if not first.get("complete"):
        rec["reason"] = "the iteration's own probe did not finish: " + str(
            first.get("error") or f"{first.get('answered', 0)} of {first.get('prompts', 64)} answered")
        _write(out, rec)
        return rec
    if not relaunch():
        rec["reason"] = "the second launch of the tree did not come up healthy"
        _write(out, rec)
        return rec
    seconds = probe_seconds()
    if seconds is not None and seconds <= 0:
        rec["reason"] = "no time is left before the deadline for the second probe"
        _write(out, rec)
        return rec
    base = f"http://127.0.0.1:{config.get('port', 8888)}"
    prompts = json.loads((session_dir / "short_prompts" / "prompts.json").read_text())["prompts"]
    try:
        second = probe(base, prompts, seconds or 600, first.get("model"))
    except Exception as e:
        second = {"complete": False, "answered": 0, "prompts": len(prompts),
                  "error": f"{type(e).__name__}: {e}"[:400]}
    _write(iter_dir / "short_prompts_2.json", second)
    if not second.get("complete"):
        rec["reason"] = (f"the second probe did not finish ({second.get('answered', 0)} of "
                         f"{second.get('prompts', len(prompts))} answered)")
        _write(out, rec)
        return rec
    cmp = compare(first, second)
    # Only the prompts sent one at a time gate (operator, 2026-10-28). On Model M at TP4 an
    # unchanged tree's two processes differed on 0 of them every time, and on 19 to 23 of
    # the 32 batched ones in their first tokens (Cluster A job 102998): which prompts share a
    # prefill batch changes between boots, and that flips near-tied greedy picks. A pooled
    # count hid a small single-request defect under that noise and could refuse a good KEEP
    # on a noisy boot, so the batched half is recorded for the host and refuses nothing.
    modes = {p["id"]: p["mode"] for p in prompts}
    seq = [i for i in cmp["first_difference"] if modes.get(i) == SEQUENTIAL]
    n_seq = sum(1 for o in second.get("outputs", []) if modes.get(o["id"]) == SEQUENTIAL)
    rec.update(cmp, sequential=len(seq), batched=cmp["differ"] - len(seq))
    floor_path = session_dir / "short_prompts" / "floor.json"
    floor_doc = json.loads(floor_path.read_text()) if floor_path.is_file() else None
    batched = f"{rec['batched']} of {cmp['compared'] - n_seq} batched differ (recorded only)"
    if floor_doc is None:
        floor_path.parent.mkdir(parents=True, exist_ok=True)
        _write(floor_path, {"floor": len(seq), "batched": rec["batched"], "iteration": iter_dir.name,
                            "ts": time.time()})
        rec.update(verdict="floor", floor=len(seq),
                   reason=f"the session's first KEEP: its two processes disagree on {len(seq)} of "
                          f"{n_seq} prompts sent one at a time, the floor from here on; {batched}")
    elif len(seq) > floor_doc["floor"]:
        firsts = sorted(((i, cmp["first_difference"][i]) for i in seq), key=lambda kv: kv[1])[:8]
        rec.update(verdict="defect", floor=floor_doc["floor"],
                   reason=f"the tree's two processes disagree on {len(seq)} of {n_seq} short prompts "
                          f"sent one at a time (floor {floor_doc['floor']}); first differences at "
                          + ", ".join(f"{i}@{k}" for i, k in firsts) + f"; {batched}")
    else:
        rec.update(verdict="ok", floor=floor_doc["floor"],
                   reason=f"{len(seq)} of {n_seq} prompts sent one at a time differ, within the floor "
                          f"{floor_doc['floor']}; {batched}")
    _write(out, rec)
    return rec


def brief(rec: dict) -> dict:
    """What the gate row and keep.json carry."""
    return {k: rec[k] for k in ("verdict", "sequential", "batched", "differ", "compared", "floor",
                                "reason") if k in rec}
