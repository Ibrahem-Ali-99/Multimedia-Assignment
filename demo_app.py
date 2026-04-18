import streamlit as st

from colpali_rag.hf_cache import ensure_hf_home

ensure_hf_home()

from colpali_rag.qa import ColpaliRAGChat
from colpali_rag.settings import ARTIFACTS, DEFAULT_PDF
from colpali_rag.store import build_store


@st.cache_resource
def get_chatbot():
    return ColpaliRAGChat()


def main():
    st.set_page_config(page_title="Multimodal Document RAG", layout="wide")
    st.title("Multi-modal document QA (ColQwen2 retrieval)")

    idx_path = ARTIFACTS / "colqwen_pages.pt"
    with st.sidebar:
        max_pages = st.number_input("Max pages (0 = all)", min_value=0, value=0, step=1)
        if st.button("Build / rebuild index"):
            mp = None if max_pages == 0 else int(max_pages)
            with st.spinner("Ingesting + embedding (GPU recommended)…"):
                build_store(DEFAULT_PDF, max_pages=mp)
            st.cache_resource.clear()
            st.success("Index saved under artifacts/")
        st.divider()
        st.markdown(
            "**CLI:** `python main.py build` · `python main.py chat` · `python main.py eval`"
        )

    if not idx_path.exists():
        st.warning("No index yet. Use the sidebar **Build / rebuild index** or run `python main.py build`.")
        return

    try:
        bot = get_chatbot()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return

    q = st.text_area("Your question", height=100)
    top_k = st.slider("Top-k pages for context", 1, 10, 4)
    if st.button("Ask", type="primary") and q.strip():
        with st.spinner("Retrieving + generating…"):
            r = bot.answer(q.strip(), top_k=int(top_k))
        st.subheader("Answer (grounded in retrieved context)")
        st.write(r["answer"])
        st.subheader("Supporting pages (citations)")
        for c in r["citations"]:
            st.markdown(f"- {c}")
        with st.expander("Retrieved context (scores)"):
            for score, meta in r["hits"]:
                st.markdown(f"**{score:.4f}** — `{meta.get('citation', '')}`")
                st.text((meta.get("text") or "")[:800])


if __name__ == "__main__":
    main()
