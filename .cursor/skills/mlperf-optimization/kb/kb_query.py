#!/usr/bin/env python3
"""Query the MLPerf optimization knowledge base.

Usage:
    python3 kb_query.py "GPT-OSS-20B fusion flags"
    python3 kb_query.py --model "GPT-OSS-20B" --top-k 20
    python3 kb_query.py --category fusion_flags --compact
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
KB_FILE = SCRIPT_DIR / "entries.jsonl"


def load_entries() -> list[dict]:
    if not KB_FILE.exists():
        return []
    entries = []
    for line in KB_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _tfidf_vectors(corpus_tokens: list[list[str]], query_tokens: list[str]):
    all_docs = corpus_tokens + [query_tokens]
    df = Counter()
    for doc in all_docs:
        df.update(set(doc))
    n = len(all_docs)
    idf = {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}

    def tfidf_vec(tokens):
        tf = Counter(tokens)
        total = len(tokens) or 1
        return {t: (c / total) * idf.get(t, 1) for t, c in tf.items()}

    query_vec = tfidf_vec(query_tokens)
    scores = []
    for doc_tokens in corpus_tokens:
        doc_vec = tfidf_vec(doc_tokens)
        dot = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in set(query_vec) | set(doc_vec))
        mag_q = math.sqrt(sum(v * v for v in query_vec.values())) or 1
        mag_d = math.sqrt(sum(v * v for v in doc_vec.values())) or 1
        scores.append(dot / (mag_q * mag_d))
    return scores


def _entry_text(entry: dict) -> str:
    parts = [
        entry.get("model", ""), entry.get("gpu", ""), entry.get("framework", ""),
        entry.get("category", ""), entry.get("action", ""), entry.get("lesson", ""),
        entry.get("context", ""), " ".join(entry.get("tags", [])),
    ]
    result = entry.get("result", {})
    if result:
        parts.append(result.get("status", ""))
    return " ".join(p for p in parts if p)


def query(query_text="", model="", category="", top_k=10, **kwargs) -> list[dict]:
    entries = load_entries()
    if not entries:
        return []

    filtered = entries
    if model:
        filtered = [e for e in filtered if model.lower() in e.get("model", "").lower()]
    if category:
        filtered = [e for e in filtered if e.get("category") == category]

    if not filtered:
        return []
    if not query_text:
        filtered.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return filtered[:top_k]

    corpus_tokens = [_tokenize(_entry_text(e)) for e in filtered]
    query_tokens = _tokenize(query_text)
    scores = _tfidf_vectors(corpus_tokens, query_tokens)

    scored = sorted(zip(scores, filtered), key=lambda x: -x[0])
    return [dict(e, _relevance=round(s, 4)) for s, e in scored[:top_k]]


def main():
    parser = argparse.ArgumentParser(description="Query the MLPerf optimization KB")
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    results = query(query_text=args.query, model=args.model,
                    category=args.category, top_k=args.top_k)

    if not results:
        print("No entries found.")
        sys.exit(0)

    if args.compact:
        for e in results:
            rel = e.pop("_relevance", "")
            gain = e.get("result", {}).get("gain_pct", "")
            status = e.get("result", {}).get("status", "")
            gain_str = f" gain={gain}%" if gain != "" else ""
            status_str = f" [{status}]" if status else ""
            rel_str = f" (rel={rel})" if rel else ""
            print(f"[{e.get('model','?')}] {e['category']}: "
                  f"{e['action'][:80]}{status_str}{gain_str}{rel_str}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n--- {len(results)} result(s) ---")


if __name__ == "__main__":
    main()
