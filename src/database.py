import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.config import DB_PATH


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                use_reranker INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                top_similarity_score REAL,
                top_rerank_score REAL,
                num_sources INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def log_query(result: Dict[str, Any]) -> None:
    sources: List[Dict[str, Any]] = result.get("sources", [])
    top_sim = sources[0]["similarity_score"] if sources else None
    top_rerank = sources[0]["rerank_score"] if sources else None

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO query_logs (
                timestamp, question, use_reranker, latency_ms,
                top_similarity_score, top_rerank_score, num_sources
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                result["question"],
                int(result["use_reranker"]),
                result["latency_ms"],
                top_sim,
                top_rerank,
                len(sources),
            ),
        )
        conn.commit()


def get_metrics_summary() -> Dict[str, Any]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                CASE WHEN use_reranker = 1 THEN 'TwoStage_CrossEncoder' ELSE 'Baseline_FAISS' END AS mode,
                COUNT(*) AS total_queries,
                ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
                ROUND(AVG(top_similarity_score), 4) AS avg_top_similarity,
                ROUND(AVG(top_rerank_score), 4) AS avg_top_rerank_score
            FROM query_logs
            GROUP BY use_reranker
            """
        ).fetchall()
    return {"breakdown": [dict(r) for r in rows]}