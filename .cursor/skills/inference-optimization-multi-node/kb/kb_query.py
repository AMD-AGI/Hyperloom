#!/usr/bin/env python3
"""Query the knowledge base using structured filters and text search.

Usage:
    # Free-text search
    python3 kb_query.py "MoE MLA torch.compile compatibility"

    # Filter by model
    python3 kb_query.py --model "GLM-5-FP8" "backend exploration"

    # Filter by category
    python3 kb_query.py --category backend_exploration --model "GLM-5-FP8"

    # List all entries for a model
    python3 kb_query.py --model "Qwen3-30B-A3B" --top-k 50

    # Get entry by ID
    python3 kb_query.py --id "abc123..."
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
    """Simple TF-IDF cosine similarity without external dependencies."""
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
    """Combine entry fields into a searchable text blob."""
    parts = [
        entry.get("model", ""),
        entry.get("gpu", ""),
        entry.get("framework", ""),
        entry.get("category", ""),
        entry.get("action", ""),
        entry.get("lesson", ""),
        entry.get("context", ""),
        " ".join(entry.get("tags", [])),
    ]
    result = entry.get("result", {})
    if result:
        parts.append(result.get("status", ""))
    return " ".join(p for p in parts if p)


def query(
    query_text: str = "",
    model: str = "",
    gpu: str = "",
    framework: str = "",
    category: str = "",
    entry_id: str = "",
    tags: list[str] | None = None,
    top_k: int = 10,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Query the KB with filters + text search. Returns entries sorted by relevance."""
    entries = load_entries()
    if not entries:
        return []

    if entry_id:
        return [e for e in entries if e["id"].startswith(entry_id)]

    filtered = entries
    if model:
        model_lower = model.lower()
        filtered = [e for e in filtered if model_lower in e.get("model", "").lower()]
    if gpu:
        gpu_lower = gpu.lower()
        filtered = [e for e in filtered if gpu_lower in e.get("gpu", "").lower()]
    if framework:
        fw_lower = framework.lower()
        filtered = [e for e in filtered if fw_lower in e.get("framework", "").lower()]
    if category:
        filtered = [e for e in filtered if e.get("category") == category]
    if tags:
        tag_set = set(t.lower() for t in tags)
        filtered = [e for e in filtered if tag_set & set(t.lower() for t in e.get("tags", []))]
    if min_confidence > 0:
        filtered = [e for e in filtered if e.get("confidence", 0.9) >= min_confidence]

    if not filtered:
        return []

    if not query_text:
        filtered.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return filtered[:top_k]

    corpus_tokens = [_tokenize(_entry_text(e)) for e in filtered]
    query_tokens = _tokenize(query_text)
    scores = _tfidf_vectors(corpus_tokens, query_tokens)

    scored = list(zip(scores, filtered))
    scored.sort(key=lambda x: (-x[0], -x[1].get("confidence", 0.9)))

    results = []
    for score, entry in scored[:top_k]:
        entry_copy = dict(entry)
        entry_copy["_relevance"] = round(score, 4)
        results.append(entry_copy)
    return results


def main():
    parser = argparse.ArgumentParser(description="Query the inference optimization KB")
    parser.add_argument("query", nargs="?", default="", help="Free-text search query")
    parser.add_argument("--model", default="", help="Filter by model name")
    parser.add_argument("--gpu", default="", help="Filter by GPU type")
    parser.add_argument("--framework", default="", help="Filter by framework")
    parser.add_argument("--category", default="", help="Filter by category")
    parser.add_argument("--id", default="", help="Get entry by ID prefix")
    parser.add_argument("--tags", default="", help="Comma-separated tags to filter")
    parser.add_argument("--top-k", type=int, default=10, help="Max results")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--compact", action="store_true", help="One-line-per-entry output")
    args = parser.parse_args()

    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None

    results = query(
        query_text=args.query,
        model=args.model,
        gpu=args.gpu,
        framework=args.framework,
        category=args.category,
        entry_id=args.id,
        tags=tag_list,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
    )

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
