import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict
import json
import pandas as pd
import numpy as np
import streamlit as st

from src.recommenders.hybrid import RecommendationEngine
from src.utils.logger import get_logger

logger = get_logger("dashboard")

# Page Configuration
st.set_page_config(
    page_title="Retail Customer Intelligence & Recommendation Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .source-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .explanation-box {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_artifacts():
    """Load precomputed engine, features, and metrics."""
    try:
        engine = RecommendationEngine.load(str(_PROJECT_ROOT / "artifacts"))
    except Exception as e:
        logger.warning(f"Could not load RecommendationEngine: {e}")
        engine = None

    cust_path = _PROJECT_ROOT / "artifacts" / "features" / "customer_features.parquet"
    prod_path = _PROJECT_ROOT / "artifacts" / "features" / "product_features.parquet"
    eval_path = _PROJECT_ROOT / "outputs" / "metrics" / "evaluation_metrics.csv"
    ablation_path = _PROJECT_ROOT / "outputs" / "metrics" / "ablation_study.csv"
    seg_eval_path = _PROJECT_ROOT / "outputs" / "metrics" / "segment_evaluation.csv"
    seg_prof_path = _PROJECT_ROOT / "outputs" / "segment_profiles.csv"
    dq_path = _PROJECT_ROOT / "outputs" / "data_quality_report.json"

    cust_features = pd.read_parquet(cust_path) if cust_path.exists() else pd.DataFrame()
    prod_features = pd.read_parquet(prod_path) if prod_path.exists() else pd.DataFrame()
    eval_metrics = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    ablation_df = pd.read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()
    segment_eval_df = pd.read_csv(seg_eval_path) if seg_eval_path.exists() else pd.DataFrame()
    segment_profiles = pd.read_csv(seg_prof_path) if seg_prof_path.exists() else pd.DataFrame()
    
    dq_report = {}
    if dq_path.exists():
        with open(dq_path, "r", encoding="utf-8") as f:
            dq_report = json.load(f)

    return engine, cust_features, prod_features, eval_metrics, ablation_df, segment_eval_df, segment_profiles, dq_report


engine, cust_df, prod_df, eval_df, ablation_df, seg_eval_df, seg_profiles_df, dq_report = load_all_artifacts()

# Sidebar Navigation
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Select Module:",
    [
        "🛍️ Customer Recommendations",
        "📦 Product & Content Explorer",
        "👥 Customer Segmentation",
        "📊 Model Evaluation & Ablation",
        "🛡️ Data Quality & Architecture",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
if engine is not None and engine.id_builder is not None:
    st.sidebar.success("✅ Models Loaded: Production Ready")
    st.sidebar.caption(f"👥 Catalog Customers: {len(cust_df):,}")
    st.sidebar.caption(f"📦 Catalog Products: {len(prod_df):,}")
else:
    st.sidebar.error("⚠️ Artifacts not found. Run pipeline first.")


# -----------------------------------------------------------------------------
# PAGE 1: CUSTOMER RECOMMENDATIONS & PROFILE EXPLORER
# -----------------------------------------------------------------------------
if page == "🛍️ Customer Recommendations":
    st.markdown('<div class="main-header">Personalized Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-strategy candidate generation, hybrid ranking, and explainable product recommendations.</div>', unsafe_allow_html=True)

    if engine is None or cust_df.empty:
        st.warning("Please run the pipeline script to train models and generate artifacts first (`python scripts/run_pipeline.py`).")
        st.stop()

    col_input, col_filters = st.columns([1.2, 1.8])

    with col_input:
        st.markdown("#### 1. Select Customer")
        sample_customers = [20000, 20001, 20005, 20010, 20050, 999999]  # 999999 is cold-start
        cust_choice = st.selectbox(
            "Customer ID (or type custom ID):",
            options=sample_customers + [c for c in cust_df["customer_id"].head(50).tolist() if c not in sample_customers],
            index=0,
        )

    with col_filters:
        st.markdown("#### 2. Recommendation Controls")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            top_n = st.slider("Top-N Items:", min_value=1, max_value=20, value=8)
        with f_col2:
            all_cats = ["All Categories"] + sorted(prod_df["category"].dropna().unique().tolist()) if not prod_df.empty else ["All Categories"]
            cat_filter = st.selectbox("Category Constraint:", all_cats)
        with f_col3:
            max_p = st.number_input("Max Budget ($):", min_value=0.0, value=0.0, step=50.0, help="Set 0 for no price ceiling")

    st.markdown("---")

    # Customer Profile Section
    cust_row = cust_df[cust_df["customer_id"] == cust_choice]
    is_cold = cust_row.empty

    st.markdown(f"### 👤 Customer Profile: `ID #{cust_choice}`")
    
    if not is_cold:
        r = cust_row.iloc[0]
        seg_id = engine.customer_segments.get(cust_choice, 0)
        seg_label = engine.segment_labels.get(seg_id, f"Segment {seg_id}")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Segment", f"Cluster {seg_id}", seg_label)
        m2.metric("Total Spend", f"${r['monetary_value']:,.2f}")
        m3.metric("Purchases / Events", f"{int(r['frequency'])}")
        m4.metric("Avg Order Value", f"${r['average_order_value']:,.2f}")
        m5.metric("Recency", f"{r['recency']:.1f} days ago")
        m6.metric("Discount Usage", f"{r['discount_usage_rate']*100:.1f}%")
    else:
        st.info("🆕 **Brand New Customer Detected** (0 interaction history). Routing dynamically to **Cold-Start Strategy** (Popularity + Trending momentum + Category Context).")

    # Generate Recommendations
    context: Dict[str, Any] = {}
    if cat_filter != "All Categories":
        context["allowed_categories"] = [cat_filter]
        context["category_context"] = cat_filter
    if max_p > 0:
        context["max_price"] = max_p

    rec_result = engine.recommend(customer_id=cust_choice, n=top_n, context=context)

    st.markdown(f"### 🎯 Recommendations (`Strategy: {rec_result['recommendation_type'].upper()}`)")
    st.caption(f"Engine routed to: **{rec_result['recommendation_type']}** | Customer Segment: **{rec_result['segment_label']}**")

    recs = rec_result["recommendations"]
    if not recs:
        st.warning("No recommendations met the active business filters.")
    else:
        # Display as cards in a responsive grid
        cols_per_row = 4
        for i in range(0, len(recs), cols_per_row):
            row_recs = recs[i:i+cols_per_row]
            cols = st.columns(len(row_recs))
            for c_idx, rec in enumerate(row_recs):
                with cols[c_idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.85rem; color: #6B7280; font-weight: bold;">RANK #{rec['rank']} • SCORE: {rec['score']:.3f}</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #111827; margin: 4px 0;">Product #{rec['product_id']}</div>
                        <div style="font-size: 0.85rem; color: #374151;"><b>Category:</b> {rec.get('category', 'General')}</div>
                        <div style="font-size: 0.85rem; color: #374151;"><b>Brand:</b> {rec.get('brand', 'Generic')}</div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #059669; margin: 4px 0;">${rec.get('unit_price', 0.0):,.2f}</div>
                        <div style="margin: 6px 0;">
                            {''.join([f'<span class="source-badge">{s}</span>' for s in rec['sources']])}
                        </div>
                        <div class="explanation-box">
                            💡 {rec['explanation']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 2: PRODUCT & CONTENT EXPLORER
# -----------------------------------------------------------------------------
elif page == "📦 Product & Content Explorer":
    st.markdown('<div class="main-header">Product Catalog & Similarity Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect item collaborative filtering neighbors, content embeddings, and test new product cold-start.</div>', unsafe_allow_html=True)

    if prod_df.empty:
        st.warning("Product feature artifacts not found.")
        st.stop()

    tab_explore, tab_cold_prod = st.tabs(["🔍 Explore Catalog Products", "✨ New Product Cold-Start Simulator"])

    with tab_explore:
        p_ids = prod_df["product_id"].tolist()
        selected_prod = st.selectbox("Select Target Product ID:", options=p_ids, index=0)
        
        prod_row = prod_df[prod_df["product_id"] == selected_prod].iloc[0]
        
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Category", str(prod_row["category"]))
        p2.metric("Brand", str(prod_row["brand"]))
        p3.metric("Unit Price", f"${prod_row['average_price']:,.2f}")
        p4.metric("Total Purchases", f"{int(prod_row['purchase_count'])}")
        p5.metric("Unique Customers", f"{int(prod_row['unique_customer_count'])}")

        st.caption(f"**Description:** {prod_row['description']}")
        st.markdown("---")

        st.markdown("#### 🔗 Similar Products")
        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:
            st.markdown("##### 👥 Collaborative Filtering Neighbors (Item-Item Cosine)")
            if engine and engine.item_item_model:
                cf_sims = engine.item_item_model.get_similar_items(selected_prod, n=5)
                if cf_sims:
                    for s_id, score in cf_sims:
                        s_meta = engine.product_metadata.get(s_id, {})
                        st.markdown(f"- **Product #{s_id}** (`{s_meta.get('category')}`, {s_meta.get('brand')} - ${s_meta.get('unit_price'):.2f}) — **Similarity: {score:.3f}**")
                else:
                    st.info("No collaborative neighbors found (sparse co-interaction).")

        with sim_col2:
            st.markdown("##### 📝 Content-Based Semantic Neighbors (TF-IDF + Metadata)")
            if engine and engine.content_model:
                content_sims = engine.content_model.get_similar_products(selected_prod, n=5)
                if content_sims:
                    for s_id, score in content_sims:
                        s_meta = engine.product_metadata.get(s_id, {})
                        st.markdown(f"- **Product #{s_id}** (`{s_meta.get('category')}`, {s_meta.get('brand')} - ${s_meta.get('unit_price'):.2f}) — **Similarity: {score:.3f}**")
                else:
                    st.info("No content neighbors found.")

    with tab_cold_prod:
        st.markdown("#### 🚀 Simulate Brand New Unseen Product")
        st.markdown("When a merchant uploads a new product with zero transactions, content vectorization matches catalog embeddings to recommend it immediately.")

        np_c1, np_c2 = st.columns(2)
        with np_c1:
            new_cat = st.selectbox("New Item Category:", sorted(prod_df["category"].dropna().unique().tolist()))
            new_brand = st.selectbox("New Item Brand:", sorted(prod_df["brand"].dropna().unique().tolist()))
        with np_c2:
            new_price = st.number_input("New Item Price ($):", min_value=1.0, value=79.99, step=10.0)
            new_desc = st.text_area("New Item Description:", value="High performance athletic gear designed with breathable moisture-wicking material.")

        if st.button("Generate Cold-Start Matches", type="primary"):
            new_meta = {
                "category": new_cat,
                "brand": new_brand,
                "unit_price": new_price,
                "description": new_desc,
            }
            if engine and engine.content_model:
                matches = engine.content_model.recommend_for_new_product(new_meta, n=5)
                st.success(f"Matched top {len(matches)} catalog items for new product cold-start routing:")
                for m_id, sim in matches:
                    m_info = engine.product_metadata.get(m_id, {})
                    st.markdown(f"- **Product #{m_id}** (`{m_info.get('category')}`, {m_info.get('brand')}, ${m_info.get('unit_price'):.2f}) — **Content Match Score: {sim:.3f}**")


# -----------------------------------------------------------------------------
# PAGE 3: CUSTOMER SEGMENTATION
# -----------------------------------------------------------------------------
elif page == "👥 Customer Segmentation":
    st.markdown('<div class="main-header">Customer Segmentation & Behavioral Personas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unsupervised K-Means clustering with deterministic, rule-derived behavioral profiles.</div>', unsafe_allow_html=True)

    if seg_profiles_df.empty:
        st.warning("Segment profiles table not found.")
        st.stop()

    st.markdown("### 📊 Discovered Customer Segments Summary")
    st.dataframe(
        seg_profiles_df[[
            "cluster_id", "label", "customer_count", "percentage_of_customers",
            "avg_monetary_value", "avg_frequency", "avg_recency_days", "avg_order_value",
            "avg_discount_usage_rate", "avg_category_diversity"
        ]],
        use_container_width=True,
    )

    st.markdown("---")
    v_col1, v_col2 = st.columns([1.2, 1])

    with v_col1:
        st.markdown("#### 🗺️ 2D PCA Cluster Projection")
        pca_img_path = _PROJECT_ROOT / "outputs" / "figures" / "customer_segments_pca.png"
        if pca_img_path.exists():
            st.image(str(pca_img_path), caption="Customer Segments in 2D PCA Dimensionality Reduction Space", use_container_width=True)
        else:
            st.info("PCA visualization plot not found.")

    with v_col2:
        st.markdown("#### 📈 Segment Spend & Frequency Distribution")
        st.bar_chart(
            data=seg_profiles_df.set_index("label")[["avg_monetary_value", "avg_order_value"]],
        )

    st.markdown("### 🏷️ Rule-Derived Persona Interpretations")
    for _, s_row in seg_profiles_df.iterrows():
        with st.expander(f"**Segment {s_row['cluster_id']}**: {s_row['label']} ({s_row['customer_count']:,} customers, {s_row['percentage_of_customers']}%)", expanded=True):
            st.markdown(f"""
            - **Average Total Spend:** ${s_row['avg_monetary_value']:,.2f} | **AOV:** ${s_row['avg_order_value']:,.2f}
            - **Engagement Volume:** {s_row['avg_frequency']} interactions across {s_row['avg_active_days']} active days
            - **Recency:** {s_row['avg_recency_days']} days since last interaction
            - **Catalog Diversity:** Interacted with {s_row['avg_unique_products']} unique products in {s_row['avg_unique_categories']} categories (Diversity: {s_row['avg_category_diversity']})
            - **Price Sensitivity:** {s_row['avg_discount_usage_rate']*100:.1f}% discount usage rate
            """)


# -----------------------------------------------------------------------------
# PAGE 4: MODEL EVALUATION & ABLATION
# -----------------------------------------------------------------------------
elif page == "📊 Model Evaluation & Ablation":
    st.markdown('<div class="main-header">Offline Recommender Evaluation & Ablation Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Strict temporal test-set evaluation across ranking accuracy, catalog coverage, diversity, and novelty.</div>', unsafe_allow_html=True)

    if eval_df.empty:
        st.warning("Evaluation metrics table not found.")
        st.stop()

    tab_benchmark, tab_ablation, tab_seg_eval = st.tabs([
        "🏆 Comprehensive Model Benchmark",
        "🔬 Component Ablation Study",
        "👥 Segment-Stratified Performance",
    ])

    with tab_benchmark:
        st.markdown("#### 📋 Offline Evaluation Metrics (Temporal Hold-Out Test Set)")
        st.dataframe(eval_df, use_container_width=True)

        st.markdown("#### 📊 Metric Comparisons")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Ranking Quality (Recall@10 vs NDCG@10)")
            st.bar_chart(eval_df.set_index("model")[["recall@10", "ndcg@10", "precision@10"]])
        with c2:
            st.markdown("##### Catalog Exploration & Coverage")
            st.bar_chart(eval_df.set_index("model")[["catalog_coverage", "diversity@10"]])

    with tab_ablation:
        st.markdown("#### 🔬 Systematic Component Ablation Study")
        st.markdown("Evaluating component combinations demonstrates the exact marginal lift provided by each algorithm.")
        if not ablation_df.empty:
            st.dataframe(ablation_df, use_container_width=True)
            st.markdown("##### Lift in Recall@20 and Catalog Coverage across Ablation Stages")
            st.line_chart(ablation_df.set_index("model")[["recall@20", "catalog_coverage", "ndcg@20"]])

    with tab_seg_eval:
        st.markdown("#### 👥 Segment-Stratified Evaluation")
        st.markdown("Measuring recommendation performance separately for each customer segment.")
        if not seg_eval_df.empty:
            st.dataframe(seg_eval_df, use_container_width=True)
            st.bar_chart(seg_eval_df.set_index("segment_label")[["recall@10", "ndcg@10", "catalog_coverage"]])


# -----------------------------------------------------------------------------
# PAGE 5: DATA QUALITY & ARCHITECTURE
# -----------------------------------------------------------------------------
elif page == "🛡️ Data Quality & Architecture":
    st.markdown('<div class="main-header">Data Quality & Architecture Specifications</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data validation logs, sparsity metrics, and temporal leakage controls.</div>', unsafe_allow_html=True)

    if dq_report:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Validated Rows", f"{dq_report.get('row_count', 0):,}")
        q2.metric("Total Customers", f"{dq_report.get('customer_count', 0):,}")
        q3.metric("Total Products", f"{dq_report.get('product_count', 0):,}")
        q4.metric("Interaction Sparsity", f"{dq_report.get('sparsity', 0.0)*100:.3f}%")

        st.markdown("### 📋 Event Distribution")
        st.json(dq_report.get("event_distribution", {}))

        st.markdown("### 🔍 Data Cleaning & Filtering Decisions Log")
        filtering_log = dq_report.get("filtering_log", [])
        if filtering_log:
            for log_item in filtering_log:
                st.markdown(f"- {log_item}")
        else:
            st.success("Clean dataset with zero discarded rows.")
