from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.database import get_metrics_summary, init_db, log_query
from src.rag_pipeline import BiomedRAGPipeline

rag_engine: Optional[BiomedRAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    init_db()
    rag_engine = BiomedRAGPipeline()
    yield


app = FastAPI(
    title="Biomedical Question-Answering RAG API",
    description="Two-Stage Retrieval (FAISS + Cross-Encoder) with Groq/Gemini LLM & SQLite Telemetry",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500, description="Biomedical question")
    use_reranker: bool = Field(default=True, description="Toggle Cross-Encoder re-ranking")


class SourceItem(BaseModel):
    doc_id: str
    context: str
    faiss_rank: int
    similarity_score: float
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    use_reranker: bool
    latency_ms: float
    sources: List[SourceItem]


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": rag_engine is not None}


@app.post("/query", response_model=QueryResponse)
def ask_biomed_question(payload: QueryRequest):
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    result = rag_engine.query(question=payload.question, use_reranker=payload.use_reranker)
    log_query(result)
    return result


@app.get("/metrics")
def get_telemetry_metrics():
    return get_metrics_summary()