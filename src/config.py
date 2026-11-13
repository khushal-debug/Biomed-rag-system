from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = DATA_DIR / "pubmed_corpus.parquet"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
DB_PATH = DATA_DIR / "rag_logs.db"

# Stage 1 & Stage 2 Models
BI_ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval Top-K Settings
INITIAL_TOP_K = 20
FINAL_TOP_K = 5