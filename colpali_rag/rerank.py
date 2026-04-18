import re

_BIO_EDU = re.compile(
    r"\b(where\s+did|stud(y|ied|ies)|university|college|education|degree|ph\.?d|"
    r"m\.?s\.?|b\.?s\.?|bachelor|master|affiliation|from\s+which|alma\s+mater)\b",
    re.I,
)

_REF_LINE = re.compile(r"^\s*\[\d+\]\s", re.M)
_ET_AL = re.compile(r"\bet\s+al\.?", re.I)
_PP = re.compile(r"\bpp\.?\s*\d", re.I)
_VOL_PP = re.compile(r"\bvol\.?\s*\d", re.I)
_REF_HEAD = re.compile(r"^\s*(references|bibliography)\s*$", re.I | re.M)


def _is_bio_edu_question(query):
    return bool(_BIO_EDU.search(query or ""))


def is_bio_edu_question(query):
    return _is_bio_edu_question(query)


def _page_blob(meta):
    parts = []
    t = (meta.get("text") or "").strip()
    if t:
        parts.append(t)
    tab = (meta.get("tables_markdown") or "").strip()
    if tab:
        parts.append(tab)
    chunks = meta.get("text_chunks")
    if isinstance(chunks, list):
        for c in chunks:
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
    return "\n".join(parts)


def _query_tokens(query):
    if not query:
        return []
    low = query.lower()
    toks = [t for t in re.split(r"[^\w]+", low) if len(t) >= 2]
    for p in query.split():
        p2 = re.sub(r"^\W+|\W+$", "", p)
        if len(p2) >= 3 and any(c.isupper() for c in p2):
            toks.append(p2.lower())
    out = []
    seen = set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _lexical_overlap(query, blob):
    if not blob:
        return 0
    blob_l = blob.lower()
    n = 0
    for t in _query_tokens(query):
        if len(t) >= 2 and t in blob_l:
            n += 1
    return n


def _bibliography_strength(blob):
    """Return 0..1 how much this page looks like references / bibliography."""
    if not blob:
        return 0.0
    head = blob[:1200]
    s = 0.0
    if _REF_HEAD.search(head):
        s += 0.35
    ref_lines = len(_REF_LINE.findall(blob))
    if ref_lines >= 3:
        s += min(0.45, 0.08 * ref_lines)
    et = len(_ET_AL.findall(blob))
    if et >= 2:
        s += min(0.35, 0.06 * et)
    pp = len(_PP.findall(blob)) + len(_VOL_PP.findall(blob))
    if pp >= 2:
        s += min(0.3, 0.05 * pp)
    # Dense years + commas often appear in reference blocks
    years = len(re.findall(r"\b(19|20)\d{2}\b", blob))
    if years >= 8 and len(blob) > 400:
        s += 0.15
    return min(1.0, s)


def rerank_hits(query, hits, bio_boost=True):
    """
    hits: list of (colqwen_score, meta) from ColQwen (typically descending).
    Returns the same tuples re-ordered by combined rerank score; tuple[0] stays the original MaxSim.
    """
    if not hits:
        return hits
    bio = bio_boost and _is_bio_edu_question(query)
    scores = [float(h[0]) for h in hits]
    lo, hi = min(scores), max(scores)
    span = hi - lo + 1e-9

    def sort_key(item):
        raw_s, meta = item
        norm = (float(raw_s) - lo) / span
        blob = _page_blob(meta)
        lex = _lexical_overlap(query, blob)
        lex_norm = min(lex / 6.0, 1.0)
        bib = _bibliography_strength(blob)
        if bio:
            bib = min(1.0, bib * 1.45)
        return norm + 0.14 * lex_norm - 0.38 * bib

    return sorted(hits, key=sort_key, reverse=True)
