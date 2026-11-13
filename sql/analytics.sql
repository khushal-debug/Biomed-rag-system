-- 1. Compare Baseline FAISS vs. Two-Stage Cross-Encoder Performance
SELECT
    CASE
        WHEN use_reranker = 1 THEN 'Two-Stage (FAISS + Cross-Encoder)'
        ELSE 'Baseline (FAISS Only)'
    END AS retrieval_mode,
    COUNT(*) AS total_queries,
    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
    ROUND(MIN(latency_ms), 2) AS min_latency_ms,
    ROUND(MAX(latency_ms), 2) AS max_latency_ms,
    ROUND(AVG(top_similarity_score), 4) AS avg_cosine_similarity,
    ROUND(AVG(top_rerank_score), 4) AS avg_cross_encoder_score
FROM query_logs
GROUP BY use_reranker;

-- 2. Latency SLA Bucket Analysis using CASE WHEN
SELECT
    CASE
        WHEN latency_ms < 500 THEN 'Fast (<500ms)'
        WHEN latency_ms BETWEEN 500 AND 1500 THEN 'Moderate (500ms-1.5s)'
        ELSE 'Slow (>1.5s)'
    END AS latency_tier,
    COUNT(*) AS query_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM query_logs), 2) AS percentage_share
FROM query_logs
GROUP BY latency_tier
ORDER BY query_count DESC;