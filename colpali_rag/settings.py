from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
DEFAULT_PDF = DATASET_DIR / "Financial_Banking_Dataset_for_Supervised_Machine_L.pdf"
ARTIFACTS = ROOT / "artifacts"

# Same repo id you already cached under HF_HOME (E:/hf_cache/...).
COLQWEN2_MODEL_ID = "vidore/colqwen2-v1.0-hf"

# Small answer head (downloads once into HF cache if missing).
GEN_MODEL_ID = "google/flan-t5-small"

MAX_PDF_PAGES = None
PAGE_ENCODE_BATCH = 1
