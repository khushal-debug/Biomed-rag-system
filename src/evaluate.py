import time
import mlflow
import numpy as np
from src.config import (
    BI_ENCODER_MODEL,
    CROSS_ENCODER_MODEL,
    FINAL_TOP_K,
    INITIAL_TOP_K,
)
from src.rag_pipeline import BiomedRAGPipeline


def evaluate_retrieval(pipeline: BiomedRAGPipeline, num_samples: int = 100, use_reranker: bool = True):
    eval_df = pipeline.df.sample(n=min(num_samples, len(pipeline.df)), random_state=42)

    hits = 0
    reciprocal_ranks = []
    latencies = []

    for _, row in eval_df.iterrows():
        query = row["question"]
        target_id = str(row["doc_id"])

        t0 = time.perf_counter()
        results = pipeline.retrieve(query, use_reranker=use_reranker)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_ids = [r["doc_id"] for r in results]
        if target_id in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(target_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    hit_rate_at_5 = hits / len(eval_df)
    mrr = float(np.mean(reciprocal_ranks))
    avg_latency_ms = float(np.mean(latencies))

    return {
        "hit_rate_at_5": round(hit_rate_at_5, 4),
        "mrr": round(mrr, 4),
        "avg_retrieval_latency_ms": round(avg_latency_ms, 2),
    }


def main():
    mlflow.set_experiment("Biomed_RAG_Retrieval_Benchmark")
    pipeline = BiomedRAGPipeline()

    configs = [
        {"run_name": "Baseline_FAISS_Only", "use_reranker": False},
        {"run_name": "TwoStage_FAISS_CrossEncoder", "use_reranker": True},
    ]

    for cfg in configs:
        with mlflow.start_run(run_name=cfg["run_name"]):
            mlflow.log_param("bi_encoder", BI_ENCODER_MODEL)
            mlflow.log_param("cross_encoder", CROSS_ENCODER_MODEL if cfg["use_reranker"] else "None")
            mlflow.log_param("use_reranker", cfg["use_reranker"])
            mlflow.log_param("initial_top_k", INITIAL_TOP_K)
            mlflow.log_param("final_top_k", FINAL_TOP_K)
            mlflow.log_param("eval_queries", 100)

            metrics = evaluate_retrieval(pipeline, num_samples=100, use_reranker=cfg["use_reranker"])
            mlflow.log_metrics(metrics)

            print(f"\n=== {cfg['run_name']} ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()