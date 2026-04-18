"""Persist ColQwen2 page embeddings + page text for QA."""

import json
from pathlib import Path

import torch

from colpali_rag.colqwen_backend import get_backend, reset_backend
from colpali_rag.ingest import ingest_pdf
from colpali_rag.settings import ARTIFACTS, COLQWEN2_MODEL_ID, MAX_PDF_PAGES


def build_store(pdf_path, max_pages=None):
    max_pages = MAX_PDF_PAGES if max_pages is None else max_pages
    print(f"[build] Ingesting PDF: {pdf_path}", flush=True)
    pages = ingest_pdf(Path(pdf_path), max_pages=max_pages)
    images = [p.image for p in pages]
    print(f"[build] Ingested {len(pages)} page(s). Loading {COLQWEN2_MODEL_ID} …", flush=True)

    reset_backend()
    be = get_backend()
    print(f"[build] Embedding {len(images)} page image(s) …", flush=True)
    embs = be.embed_page_images(images)

    print("[build] Saving artifacts …", flush=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_id": COLQWEN2_MODEL_ID,
        "embeddings": embs,
        "meta": [
            {
                "page_index": p.page_index,
                "citation": p.citation,
                "text": p.text[:12000],
                "tables_markdown": (p.tables_markdown or "")[:8000],
                "text_chunks": [c[:800] for c in (p.text_chunks or [])[:24]],
            }
            for p in pages
        ],
    }
    out_pt = ARTIFACTS / "colqwen_pages.pt"
    torch.save(bundle, out_pt)
    with open(ARTIFACTS / "colqwen_meta.json", "w", encoding="utf-8") as f:
        json.dump({"n_pages": len(pages), "model_id": COLQWEN2_MODEL_ID}, f, indent=2)
    clear_bundle_cache()
    reset_backend()
    return out_pt


_bundle_cache = None


def load_bundle():
    global _bundle_cache
    p = ARTIFACTS / "colqwen_pages.pt"
    if not p.exists():
        raise FileNotFoundError("Missing index. Run: python main.py build")
    if _bundle_cache is None:
        try:
            _bundle_cache = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            _bundle_cache = torch.load(p, map_location="cpu")
    return _bundle_cache


def clear_bundle_cache():
    global _bundle_cache
    _bundle_cache = None
