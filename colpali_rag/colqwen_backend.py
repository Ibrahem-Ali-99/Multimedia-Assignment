"""ColQwen2 retrieval model (Hugging Face ColQwen2ForRetrieval — ColPali / ColVision line)."""

import torch
from PIL import Image
from transformers import ColQwen2ForRetrieval, ColQwen2Processor

from colpali_rag.settings import COLQWEN2_MODEL_ID, PAGE_ENCODE_BATCH

_backend = None


class ColQwenBackend:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print(
            f"[colqwen] Loading weights on {self.device} (dtype={dtype}) — can take several minutes …",
            flush=True,
        )
        self.model = ColQwen2ForRetrieval.from_pretrained(
            COLQWEN2_MODEL_ID,
            dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = ColQwen2Processor.from_pretrained(
            COLQWEN2_MODEL_ID,
            backend="pil",
        )
        print("[colqwen] Model and processor ready.", flush=True)

    @torch.inference_mode()
    def embed_page_images(self, images):
        """One multi-vector embedding tensor (seq, dim) per page, on CPU."""
        out_vecs = []
        n = len(images)
        for i in range(0, n, PAGE_ENCODE_BATCH):
            hi = min(i + PAGE_ENCODE_BATCH, n)
            print(f"[colqwen] Encoding page(s) {i + 1}-{hi}/{n} …", flush=True)
            batch_imgs = images[i:hi]
            feats = self.processor.process_images(batch_imgs, return_tensors="pt")
            feats = {k: v.to(self.device) for k, v in feats.items()}
            out = self.model(**feats)
            emb = out.embeddings
            for j in range(emb.shape[0]):
                out_vecs.append(emb[j].detach().float().cpu())
        return out_vecs

    @torch.inference_mode()
    def embed_queries(self, questions):
        feats = self.processor.process_queries(text=questions, return_tensors="pt")
        feats = {k: v.to(self.device) for k, v in feats.items()}
        out = self.model(**feats)
        emb = out.embeddings
        return [emb[i].detach().float().cpu() for i in range(emb.shape[0])]

    def score_pages(self, query_embs, page_embs):
        """MaxSim scores: shape (n_queries, n_pages)."""
        dev = self.device
        qs = [q.to(dev) for q in query_embs]
        ps = [p.to(dev) for p in page_embs]
        return self.processor.score_retrieval(qs, ps, output_device="cpu")


def get_backend():
    global _backend
    if _backend is None:
        _backend = ColQwenBackend()
    return _backend


def reset_backend():
    global _backend
    _backend = None
