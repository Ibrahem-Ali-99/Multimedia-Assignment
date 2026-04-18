"""Simple eval: run retrieval on benchmark questions (optional gold page)."""

import json

from colpali_rag.retrieve import retrieve
from colpali_rag.settings import ARTIFACTS

DEFAULT_QUERIES = [
    {"q": "What is this document about?", "gold_page_index": None},
    {"q": "What risks or problems are discussed?", "gold_page_index": None},
    {"q": "Are there numerical tables or financial figures?", "gold_page_index": None},
]


def run_eval(queries=None, top_k=5):
    idx = ARTIFACTS / "colqwen_pages.pt"
    if not idx.exists():
        return json.dumps(
            {
                "error": "No retrieval index found.",
                "hint": "Build it first: python main.py build [--pdf PATH] [--max-pages N]",
                "expected_file": str(idx.resolve()),
            },
            indent=2,
        )

    queries = queries or DEFAULT_QUERIES
    rows = []
    for item in queries:
        q = item["q"]
        gold = item.get("gold_page_index")
        hits = retrieve(q, top_k=top_k)
        top_pages = [h[1]["page_index"] for h in hits]
        hit = gold in top_pages if gold is not None else None
        rows.append({"query": q, "gold_page_index": gold, "hit": hit, "top_pages": top_pages})
    return json.dumps(rows, indent=2)
