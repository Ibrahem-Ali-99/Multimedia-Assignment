import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from colpali_rag.rerank import is_bio_edu_question
from colpali_rag.retrieve import retrieve
from colpali_rag.settings import GEN_MODEL_ID
from colpali_rag.store import load_bundle

_GEN_MAX_LEN = 512
_PER_HIT_TEXT = 520
_PER_HIT_TABLES = 320

_OVERVIEW_Q = re.compile(
    r"\b(what is this document|what is the document|what is this paper|what is the paper|"
    r"what is this about|summarize|summary|overview|describe this document|main topic|"
    r"purpose of (this|the) document)\b",
    re.I,
)

_WHERE_STUDY = re.compile(r"where\s+did\s+(.+?)\s+study\b", re.I | re.S)

_EDU_CUES = re.compile(
    r"\b(university|college|institute|polytechnic|school|b\.?s\.?|m\.?s\.?|ph\.?d\.?|bachelor'?s?|master'?s?|"
    r"doctorate|mba|studied\s+at|graduated|degree\s+from|received\s+.{0,48}(from|at)|"
    r"education\s+at|enrolled|alma\s+mater)\b",
    re.I,
)


def _is_low_quality_answer(text):
    t = text.strip()
    if len(t) < 28:
        return True
    if re.fullmatch(r"[\[\(]?\d+[\]\)]?", t):
        return True
    if re.fullmatch(r"(page|p\.?)\s*\d+", t, flags=re.I):
        return True
    if re.match(r"^sources?:", t, flags=re.I):
        return True
    return False


def _split_sentences(blob):
    if not blob:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", blob)
    out = []
    for p in pieces:
        p = p.strip()
        if len(p) > 12:
            out.append(p)
    return out if out else [blob.strip()]


def _name_match_in_text(name_raw, haystack_lower):
    parts = [p.lower() for p in re.split(r"\s+", (name_raw or "").strip()) if len(p) >= 2]
    if not parts:
        return False
    return all(p in haystack_lower for p in parts)


def _education_extractive_answer(question, hits):
    name_raw = None
    m = _WHERE_STUDY.search(question)
    if m:
        name_raw = m.group(1).strip()
        name_raw = re.sub(r"\s+according\s+to.*$", "", name_raw, flags=re.I).strip()
    elif is_bio_edu_question(question):
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][A-Za-z]+)+\b", question)
        if caps:
            name_raw = caps[0].strip()
    if not name_raw or len(name_raw) < 2:
        return None

    for _sc, meta in hits:
        cite = meta.get("citation", "document")
        blobs = []
        for c in meta.get("text_chunks") or []:
            if isinstance(c, str) and c.strip():
                blobs.append(c.strip())
        t = (meta.get("text") or "").strip()
        if t:
            blobs.append(t)
        tab = (meta.get("tables_markdown") or "").strip()
        if tab:
            blobs.append(tab)
        for blob in blobs:
            for sent in _split_sentences(blob):
                sl = sent.lower()
                if not _name_match_in_text(name_raw, sl):
                    continue
                if _EDU_CUES.search(sent):
                    return f"According to {cite}: {sent.strip()}"
    return None


def _context_blob_from_hits(hits):
    parts = []
    for _s, meta in hits:
        parts.append(meta.get("text") or "")
        parts.append(meta.get("tables_markdown") or "")
    return "\n".join(parts).lower()


def _answer_grounded_in_context(answer, hits):
    ctx = _context_blob_from_hits(hits)
    if not ctx.strip():
        return False
    alnum = re.findall(r"[a-z0-9]{7,}", answer.lower())
    if not alnum:
        return True
    hits_in_ctx = sum(1 for w in alnum if w in ctx)
    return hits_in_ctx >= max(1, min(2, len(alnum) // 3))


def _extractive_fallback(question, hits, bundle_meta):
    meta = None
    if _OVERVIEW_Q.search(question) and bundle_meta:
        meta = bundle_meta[0]
    if meta is None and hits:
        if _OVERVIEW_Q.search(question):
            for _s, m in sorted(hits, key=lambda x: x[1].get("page_index", 999)):
                if len((m.get("text") or "").strip()) > 120:
                    meta = m
                    break
        if meta is None:
            meta = hits[0][1]
    if meta is None:
        return "No retrieved pages to summarize."

    cite = meta.get("citation", "document")
    body = (meta.get("text") or "").strip()
    tab = (meta.get("tables_markdown") or "").strip()
    parts = []
    if body:
        parts.append(body[:1500])
    if tab:
        parts.append("Tables:\n" + tab[:900])
    blob = "\n\n".join(parts).strip()
    if not blob:
        return (
            "The retrieved page had almost no extractable text. "
            f"Try another question or inspect: {cite}."
        )
    return f"Grounded excerpt from {cite}:\n\n{blob}"


class ColpaliRAGChat:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
        self.gen = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_ID).to(self.device)
        self.gen.eval()
        self._enc_max = min(
            int(getattr(self.gen.config, "max_position_embeddings", _GEN_MAX_LEN) or _GEN_MAX_LEN),
            _GEN_MAX_LEN,
        )

    @torch.inference_mode()
    def answer(self, question, top_k=4, max_new_tokens=256):
        bundle = load_bundle()
        bundle_meta = bundle.get("meta") or []

        hits = retrieve(question, top_k=top_k)

        edu = _education_extractive_answer(question, hits)
        if edu:
            cites = []
            for _s, meta in hits:
                cites.append(meta["citation"])
            return {"answer": edu, "citations": list(dict.fromkeys(cites)), "hits": hits}

        blocks = []
        cites = []
        for score, meta in hits:
            cite = meta["citation"]
            cites.append(cite)
            body = meta.get("text") or ""
            tab = meta.get("tables_markdown") or ""
            blocks.append(
                f"({cite} | retrieval_score={score:.4f})\n"
                f"TEXT:\n{body[:_PER_HIT_TEXT]}\n"
                f"TABLES:\n{tab[:_PER_HIT_TABLES]}"
            )
        retrieved = "\n\n---\n\n".join(blocks)

        lead = ""
        if bundle_meta and not is_bio_edu_question(question):
            m0 = bundle_meta[0]
            t0 = (m0.get("text") or "").strip()
            tab0 = (m0.get("tables_markdown") or "").strip()
            bits = []
            if t0:
                bits.append(t0[:880])
            if tab0:
                bits.append("TABLES:\n" + tab0[:400])
            if bits:
                c0 = m0.get("citation", "page 1")
                lead = f"(Beginning of document — {c0})\n" + "\n\n".join(bits)

        if lead:
            context = lead + "\n\n---\n\nRetrieved pages (by relevance):\n\n" + retrieved
        else:
            context = retrieved

        if is_bio_edu_question(question):
            instruction = (
                "Read the Context (PDF excerpts only). Answer using facts from the Context in complete sentences. "
                "If the question asks where someone studied or was educated, use only sentences that clearly "
                "describe that person's education, university, or degree. Do NOT treat a journal or conference "
                "venue from a bibliography or reference list as where they studied. "
                "If the Context does not clearly state it, say the document does not state it clearly."
            )
        else:
            instruction = (
                "Read the Context (PDF excerpts only). Answer the Question in several complete sentences, "
                "using facts from the Context. Do not reply with only a page number, a bracketed number "
                "like [12], or a bare citation. If the Context does not support an answer, say the document "
                "does not state it clearly."
            )

        prompt = (
            f"{instruction}\n\nQuestion: {question.strip()}\n\nContext:\n{context}\n\nAnswer:"
        )
        enc = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._enc_max,
        ).to(self.device)
        out = self.gen.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
        )
        text = self.tok.decode(out[0], skip_special_tokens=True).strip()
        if _is_low_quality_answer(text):
            text = _extractive_fallback(question, hits, bundle_meta)
        elif not re.search(r"\b(does not|do not|not clearly|cannot find)\b", text, re.I):
            if not _answer_grounded_in_context(text, hits):
                text = (
                    "The answer could not be verified against the retrieved page text. "
                    "The document may not state this clearly in those pages, or try a different phrasing."
                )
        return {"answer": text, "citations": list(dict.fromkeys(cites)), "hits": hits}
