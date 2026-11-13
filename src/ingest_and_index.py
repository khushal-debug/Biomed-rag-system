import time
import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import (
    BI_ENCODER_MODEL,
    CORPUS_PATH,
    CROSS_ENCODER_MODEL,
    FAISS_INDEX_PATH,
    FINAL_TOP_K,
    INITIAL_TOP_K,
)


def build_corpus_and_index() -> pd.DataFrame:
    print("1. Loading PubMedQA (pqa_labeled) dataset...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    records = []
    for row in dataset:
        abstract_sections = row["context"].get("contexts", [])
        full_context = " ".join(abstract_sections).strip()

        if full_context:
            records.append(
                {
                    "doc_id": str(row["pubid"]),
                    "question": row["question"].strip(),
                    "context": full_context,
                    "ground_truth_answer": row["long_answer"].strip(),
                }
            )

    df = pd.DataFrame(records).drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    df.to_parquet(CORPUS_PATH, index=False)
    print(f"   Saved {len(df)} documents to {CORPUS_PATH}")

    print(f"2. Encoding corpus with Bi-Encoder ({BI_ENCODER_MODEL})...")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    embeddings = bi_encoder.encode(
        df["context"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    print("3. Building and saving FAISS IndexFlatIP...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"   Indexed {index.ntotal} vectors -> {FAISS_INDEX_PATH}")

    return df


def verify_two_stage_retrieval(df: pd.DataFrame):
    print("\n4. Running Two-Stage Retrieval Sanity Check...")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    sample_row = df.iloc[0]
    query = sample_row["question"]
    target_doc_id = sample_row["doc_id"]

    t0 = time.perf_counter()
    q_emb = bi_encoder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, INITIAL_TOP_K)

    candidates = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        candidates.append(
            {
                "doc_id": df.iloc[idx]["doc_id"],
                "context": df.iloc[idx]["context"],
                "faiss_rank": rank,
                "faiss_score": float(score),
            }
        )

    pairs = [(query, c["context"]) for c in candidates]
    ce_scores = cross_encoder.predict(pairs)
    for c, ce_score in zip(candidates, ce_scores):
        c["rerank_score"] = float(ce_score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:FINAL_TOP_K]
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\nQuery: {query}")
    print(f"Target Ground-Truth Doc ID: {target_doc_id} | Latency: {latency_ms:.1f} ms")
    print("-" * 75)
    for new_rank, item in enumerate(reranked, start=1):
        marker = " <-- GROUND TRUTH" if item["doc_id"] == target_doc_id else ""
        print(
            f"Final Rank #{new_rank} (was FAISS #{item['faiss_rank']}) | "
            f"Doc ID: {item['doc_id']} | CE Score: {item['rerank_score']:.3f} | "
            f"Cosine: {item['faiss_score']:.3f}{marker}"
        )


if __name__ == "__main__":
    corpus_df = build_corpus_and_index()
    verify_two_stage_retrieval(corpus_df)