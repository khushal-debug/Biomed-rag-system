# 🧬 Biomedical Question-Answering RAG System (Two-Stage Retrieval + MLOps)

An end-to-end, production-ready **Retrieval-Augmented Generation (RAG)** system for evidence-based biomedical question answering built on the **PubMedQA** corpus. Designed with a two-stage neural retrieval architecture, offline **MLflow** benchmarking, **FastAPI** + **SQLite** telemetry, **Docker** containerization, and an interactive **Streamlit** UI.

## 🏗️ System Architecture

User Query -> Stage 1: Dense Bi-Encoder (BAAI/bge-small-en-v1.5) + FAISS IndexFlatIP (Top-20) -> Stage 2: Cross-Encoder Re-Ranker (ms-marco-MiniLM-L-6-v2) -> Top-5 PubMed Abstracts -> Grounded LLM Generation (Groq) with Inline [PMID] Citations + SQLite Telemetry.

## 📊 Experimental Evaluation (MLflow Tracked)

Evaluated across **100 PubMedQA ground-truth question-abstract pairs**:

| Retrieval Pipeline | Embedding / Re-Ranker Model | Hit Rate@5 | MRR | Avg Retrieval Latency (CPU) |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline (FAISS Only)** | `BAAI/bge-small-en-v1.5` (`IndexFlatIP`) | **99.0%** | **0.9850** | **19.61 ms** |
| **Two-Stage (FAISS + Cross-Encoder)** | `+ cross-encoder/ms-marco-MiniLM-L-6-v2` | **99.0%** | **0.9783** | **2,520.39 ms** |

**Key Engineering Takeaway:** On structured biomedical abstracts (`PubMedQA`), normalized `bge-small-en-v1.5` embeddings achieve **99% Hit Rate@5** and **0.985 MRR** in under **20 ms**, whereas general-domain MS-MARCO Cross-Encoder re-ranking adds ~2.5s CPU overhead—justifying a toggleable re-ranker design in production.

## 🛠️ Tech Stack

- **Deep Learning & NLP:** PyTorch, Hugging Face `sentence-transformers`, `datasets`
- **Vector Search & LLM:** FAISS (`IndexFlatIP`), Groq API (`llama-3.1-8b-instant`)
- **MLOps & Tracking:** MLflow, Docker
- **Backend & SQL Analytics:** FastAPI, Uvicorn, Pydantic, SQLite (`analytics.sql` with `GROUP BY` & `CASE WHEN`)
- **Frontend & Deployment:** Streamlit, Streamlit Community Cloud

## 🚀 Quickstart

```bash
pip install -r requirements.txt
python -m src.ingest_and_index
streamlit run app.py
# Or run FastAPI: uvicorn api.main:app --reload