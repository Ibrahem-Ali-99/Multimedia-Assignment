"""Multi-modal ingestion: page raster, plain text, detected tables (PyMuPDF)."""

import re
from pathlib import Path

import pymupdf
from PIL import Image

_CHUNK_MAX = 720
_MAX_CHUNKS_PER_PAGE = 24


def _chunk_page_text(text):
    if not (text or "").strip():
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks = []
    cur = ""
    for p in paras:
        if len(p) > _CHUNK_MAX:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(p), _CHUNK_MAX):
                chunks.append(p[i : i + _CHUNK_MAX])
            continue
        if len(cur) + len(p) + 2 <= _CHUNK_MAX:
            cur = (cur + "\n\n" + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks[:_MAX_CHUNKS_PER_PAGE]


class PageRecord:
    def __init__(self, page_index, citation, image, text, tables_markdown, text_chunks=None):
        self.page_index = page_index
        self.citation = citation
        self.image = image
        self.text = text
        self.tables_markdown = tables_markdown
        self.text_chunks = text_chunks if text_chunks is not None else []


def ingest_pdf(pdf_path, max_pages=None):
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(str(pdf_path))
    name = pdf_path.name
    n = len(doc)
    limit = n if max_pages is None else min(n, max_pages)
    pages = []

    for i in range(limit):
        page = doc[i]
        parts_table = []
        try:
            tf = page.find_tables()
            tabs = list(tf) if tf is not None else []
            for ti, tab in enumerate(tabs):
                md = None
                if hasattr(tab, "to_markdown"):
                    try:
                        md = tab.to_markdown()
                    except Exception:
                        md = None
                if not md:
                    raw = tab.extract()
                    if isinstance(raw, list):
                        lines = []
                        for row in raw:
                            if not row:
                                continue
                            lines.append("\t".join(str(c or "").strip() for c in row))
                        md = "\n".join(lines)
                    else:
                        md = str(raw)
                if md and md.strip():
                    parts_table.append(f"### Table {ti + 1}\n{md.strip()}")
        except Exception:
            pass
        tables_md = "\n\n".join(parts_table)

        text = (page.get_text("text") or "").strip()
        text_chunks = _chunk_page_text(text)

        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        citation = f"{name} page {i + 1}"
        pages.append(
            PageRecord(
                i,
                citation,
                img,
                text,
                tables_md,
                text_chunks,
            )
        )

    doc.close()
    return pages
