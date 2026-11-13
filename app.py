import streamlit as st
from src.database import get_metrics_summary, init_db, log_query
from src.ingest_and_index import build_corpus_and_index
from src.config import CORPUS_PATH, FAISS_INDEX_PATH
from src.rag_pipeline import BiomedRAGPipeline

st.set_page_config(
    page_title="Biomedical QA RAG System",
    page_icon="🧬",
    layout="wide",
)

@st.cache_resource
def load_pipeline():
    init_db()
    if not CORPUS_PATH.exists() or not FAISS_INDEX_PATH.exists():
        build_corpus_and_index()
    return BiomedRAGPipeline()

pipeline = load_pipeline()

st.title("🧬 Biomedical Question-Answering RAG System")
st.caption("Two-Stage Retrieval (BAAI/bge-small FAISS + Cross-Encoder Re-Ranking) • PubMedQA • Groq Llama-3.3-70B")

# Sidebar Controls & Telemetry
with st.sidebar:
    st.header("⚙️ Retrieval Settings")
    use_reranker = st.toggle(
        "Enable Cross-Encoder Re-Ranker",
        value=True,
        help="Stage 1 retrieves top-20 via FAISS; Stage 2 re-ranks top-5 via Cross-Encoder.",
    )
    st.markdown("---")
    st.subheader("📊 Offline MLflow Benchmark (100 Queries)")
    st.markdown(
        """
        - **FAISS Baseline:** `99.0%` Hit@5 | `0.985` MRR | `19.6 ms`
        - **+ Cross-Encoder:** `99.0%` Hit@5 | `0.978` MRR | `2520 ms (CPU)`
        """
    )
    st.markdown("---")
    st.subheader("🗄️ Live SQLite Telemetry")
    summary = get_metrics_summary()["breakdown"]
    if summary:
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Ask a question to populate SQLite query logs.")

# Sample Questions
sample_questions = [
    "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
    "Is preoperative statin therapy associated with reduced postoperative atrial fibrillation?",
    "Does vitamin D supplementation reduce the risk of acute respiratory tract infections?",
]

selected_sample = st.selectbox("💡 Try a sample biomedical question (or type your own below):", [""] + sample_questions)
user_query = st.text_input(
    "Enter your biomedical research question:",
    value=selected_sample,
    placeholder="e.g., What is the effect of metformin on cardiovascular outcomes in type 2 diabetes?",
)

if st.button("🔍 Generate Evidence-Based Answer", type="primary") and user_query.strip():
    with st.spinner("Retrieving PubMed abstracts and synthesizing cited response..."):
        result = pipeline.query(question=user_query.strip(), use_reranker=use_reranker)
        log_query(result)

    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval Mode", "Two-Stage (FAISS + CE)" if use_reranker else "Baseline (FAISS Only)")
    col2.metric("End-to-End Latency", f"{result['latency_ms']} ms")
    col3.metric("Cited PubMed Abstracts", len(result["sources"]))

    st.subheader("🩺 Grounded Answer")
    st.success(result["answer"])

    st.subheader("📚 Retrieved PubMed Evidence & Ranking Scores")
    for idx, src in enumerate(result["sources"], start=1):
        ce_badge = f" | Cross-Encoder Score: `{src['rerank_score']:.3f}`" if src["rerank_score"] is not None else ""
        with st.expander(
            f"Rank #{idx} — PMID: {src['doc_id']} (FAISS Rank #{src['faiss_rank']} | Cosine: {src['similarity_score']:.3f}{ce_badge})",
            expanded=(idx == 1),
        ):
            st.write(src["context"])