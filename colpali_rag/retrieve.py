"""Late-interaction retrieval over ColQwen2 page embeddings."""

import torch

from colpali_rag.colqwen_backend import get_backend
from colpali_rag.rerank import rerank_hits
from colpali_rag.store import load_bundle

_CAND_MULT = 4
_CAND_CAP = 20


def retrieve(query, top_k=5):
    bundle = load_bundle()
    page_embs = bundle["embeddings"]
    meta = bundle["meta"]

    be = get_backend()
    q_embs = be.embed_queries([query])
    scores = be.score_pages(q_embs, page_embs)
    row = scores[0]
    n = int(row.shape[0])
    pool = min(n, max(top_k * _CAND_MULT, top_k), _CAND_CAP)
    order = torch.argsort(row, descending=True).tolist()[:pool]
    hits = [(float(row[i]), meta[i]) for i in order]
    hits = rerank_hits(query, hits)
    return hits[:top_k]
