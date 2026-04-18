"""
Multi-modal document RAG (ColQwen2 / ColVision — same family as ColPali).

  python main.py build [--pdf PATH] [--max-pages N]
  python main.py chat
  python main.py eval
  streamlit run demo_app.py

HF cache: set HF_HOME, or we use E:\\hf_cache / D:\\hf_cache (see colpali_rag.hf_cache).
Checkpoint: vidore/colqwen2-v1.0-hf (reuses your existing Hub cache).

Conda: conda env update -f env.yaml --prune
"""

import argparse
from pathlib import Path

from colpali_rag.hf_cache import ensure_hf_home

ensure_hf_home()


def cmd_build(args):
    from colpali_rag.settings import DEFAULT_PDF
    from colpali_rag.store import build_store

    pdf = Path(args.pdf) if args.pdf else DEFAULT_PDF
    out = build_store(pdf, max_pages=args.max_pages)
    print("Saved:", out)


def cmd_chat(args):
    from colpali_rag.qa import ColpaliRAGChat

    bot = ColpaliRAGChat()
    print("ColQwen2 RAG — type quit to exit.\n")
    while True:
        q = input("You> ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        r = bot.answer(q)
        print("\nAssistant>", r["answer"])
        print("\nCitations:", ", ".join(r["citations"]))
        print()


def cmd_eval(args):
    from colpali_rag.eval_suite import run_eval

    print(run_eval())


def main():
    ap = argparse.ArgumentParser(description="ColQwen2 multi-modal document RAG")
    sub = ap.add_subparsers(dest="command", required=True)

    p_b = sub.add_parser("build", help="Ingest PDF + ColQwen2 page embeddings")
    p_b.add_argument("--pdf", type=str, default=None)
    p_b.add_argument("--max-pages", type=int, default=None)
    p_b.set_defaults(func=cmd_build)

    sub.add_parser("chat", help="QA over indexed PDF").set_defaults(func=cmd_chat)
    sub.add_parser("eval", help="Retrieval benchmark JSON").set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
