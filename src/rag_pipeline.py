import os
import time
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from groq import Groq
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import (
    BI_ENCODER_MODEL,
    CORPUS_PATH,
    CROSS_ENCODER_MODEL,
    FAISS_INDEX_PATH,
    FINAL_TOP_K,
    INITIAL_TOP_K,
)

load_dotenv()


class BiomedRAGPipeline:
    def __init__(self):
        self.df = pd.read_parquet(CORPUS_PATH)
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        self.groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None
        self.gemini_client = genai.Client(api_key=self.gemini_key) if self.gemini_key else None

    def retrieve(
        self,
        query: str,
        use_reranker: bool = True,
        initial_k: int = INITIAL_TOP_K,
        final_k: int = FINAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        # Stage 1: Dense Retrieval via FAISS
        q_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        k_search = initial_k if use_reranker else final_k
        scores, indices = self.index.search(q_emb, k_search)

        candidates = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            row = self.df.iloc[idx]
            candidates.append(
                {
                    "doc_id": str(row["doc_id"]),
                    "context": row["context"],
                    "faiss_rank": rank,
                    "similarity_score": float(score),
                    "rerank_score": None,
                }
            )

        if not use_reranker:
            return candidates[:final_k]

        # Stage 2: Cross-Encoder Re-Ranking
        pairs = [(query, c["context"]) for c in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        for c, ce_score in zip(candidates, ce_scores):
            c["rerank_score"] = float(ce_score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:final_k]

    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        formatted_context = "\n\n".join(
            [f"[PMID: {c['doc_id']}] {c['context']}" for c in contexts]
        )

        system_prompt = (
            "You are an evidence-based biomedical research assistant. "
            "Answer the user's question strictly using ONLY the provided PubMed abstracts. "
            "Cite sources inline using [PMID: <id>]. If the context is insufficient, state that clearly."
        )
        user_prompt = f"Context:\n{formatted_context}\n\nQuestion: {query}\n\nAnswer:"

        if self.groq_client:
            for model_name in ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]:
                try:
                    resp = self.groq_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        max_tokens=512,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception:
                    continue
            raise RuntimeError("No active Groq model responded. Check your GROQ_API_KEY.")

        if self.gemini_client:
            resp = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system_prompt}\n\n{user_prompt}",
            )
            return resp.text.strip()

        return "LLM API key not configured. Retrieved contexts are listed below."

    def query(self, question: str, use_reranker: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()
        retrieved_docs = self.retrieve(question, use_reranker=use_reranker)
        answer = self.generate_answer(question, retrieved_docs)
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "question": question,
            "answer": answer,
            "use_reranker": use_reranker,
            "latency_ms": round(latency_ms, 2),
            "sources": retrieved_docs,
        }