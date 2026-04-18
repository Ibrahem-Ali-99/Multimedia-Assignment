# Technical Report: Multi-Modal Document RAG (ColQwen2)

**Scope:** Architecture, design choices, benchmarks, and observations (≤2 pages).

## 1. Problem and goals

Financial and policy PDFs mix **text, tables, figures, and layout**. A text-only RAG pipeline loses structure and visuals. This prototype implements **multi-modal ingestion**, **unified visual retrieval** with a **ColVision-style** encoder (**ColQwen2**), **late-interaction scoring**, and a **small seq2seq generator** for **grounded answers with page-level citations**, plus a **Streamlit demo** and **CLI**.

## 2. Architecture

| Stage | Role |
|--------|------|
| **Ingestion** (`colpali_rag/ingest.py`) | PyMuPDF: per-page **raster image**, **plain text**, and **tables** (via `find_tables()`, exported as markdown/tab-separated text). |
| **Embedding** (`colpali_rag/colqwen_backend.py`) | **Hugging Face `ColQwen2ForRetrieval`** + **`ColQwen2Processor`** on checkpoint **`vidore/colqwen2-v1.0-hf`**. Each page image → **multi-vector** embedding (late-interaction compatible). |
| **Index** (`colpali_rag/store.py`) | Saves `artifacts/colqwen_pages.pt` (embeddings + per-page text/tables + citations). |
| **Retrieval** (`colpali_rag/retrieve.py`) | Embed the query with the same model; **MaxSim** via **`processor.score_retrieval`**. |
| **Generation** (`colpali_rag/qa.py`) | Top-k pages’ **text/tables** passed into **`google/flan-t5-small`** with an instruction to answer **only from context** and list **Sources** (citations). |
| **Evaluation** (`colpali_rag/eval_suite.py`) | JSON over placeholder queries + optional gold page index. |
| **Interfaces** | **CLI** (`main.py`: `build`, `chat`, `eval`) and **Streamlit** (`demo_app.py`). |

**HF cache:** `colpali_rag/hf_cache.py` sets **`HF_HOME`** to **`E:\hf_cache`** or **`D:\hf_cache`** when unset, so large weights avoid a full **C:** system drive.

## 3. Design choices

1. **`ColQwen2ForRetrieval` (Transformers) vs `colpali_engine.ColQwen2`:** The Hub config for `vidore/colqwen2-v1.0-hf` targets the **Transformers** class; using it avoids **missing/unexpected weight** mismatches while staying in the **ColPali / ColVision** line described in the course materials.
2. **Page as the retrieval unit:** Matches ColQwen2’s strength (whole-page “screenshot” semantics) and keeps the pipeline **simple**; table bodies are still **extracted** and fed to the **generator** as text for **grounding**.
3. **Flan-T5-small:** Lightweight **CPU/GPU** answer head; first run may download weights into the same HF cache. Trade-off: **not** a frontier LLM, but **clear attribution** and **low cost**.
4. **No separate vector DB:** Embeddings stored in **PyTorch** files; acceptable for a **course-scale** corpus and easier **reproducibility**.

## 4. Benchmarks and evaluation

`python main.py eval` runs a **small fixed query set** and reports **top retrieved page indices** (and optional **hit@k** if `gold_page_index` is filled in `eval_suite.py`). This is a **sanity / regression** harness, not a full ViDoRe benchmark: extending it with **labeled Q–page pairs** would be the next step for rigorous scores (**nDCG**, **Recall@k**).

**Observations:** Retrieval quality depends on **query phrasing** and **PDF raster quality**; scanned pages with poor OCR text still benefit from the **image branch** if the visual encoder aligns with the query in embedding space.

## 5. How to run (summary)

1. `conda activate assign1` (see **`env.yaml`** for pinned stack; add **`streamlit`** if not already installed).
2. `python main.py build [--pdf PATH] [--max-pages N]`
3. **CLI:** `python main.py chat` or **`streamlit run demo_app.py`** for the demo UI.
4. `python main.py eval` for the JSON evaluation dump.

## 6. Limitations and future work

- **Generator capacity:** Flan-T5-small may **hallucinate** on long contexts; mitigations include **smaller k**, **stronger models**, or **extractive** post-checks.
- **Chart “metadata”:** Currently implicit in the **page image**; explicit **chart detection** (e.g. region proposals) would improve **multi-modal coverage** scoring.
- **Scale:** For large corpora, move to **sharded indices**, **quantization**, or **FAISS/Plaid**-style backends compatible with late interaction.

---

*End of report (architecture, choices, benchmarks, observations).*
