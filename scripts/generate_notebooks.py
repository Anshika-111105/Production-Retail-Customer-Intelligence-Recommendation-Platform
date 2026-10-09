"""Generate clean, self-contained Jupyter Notebooks for the Retail Intelligence Platform."""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code],
    }


PATH_BOOTSTRAP = """import sys, os
from pathlib import Path
import pandas as pd
import numpy as np

# Robust root resolver whether run from notebooks/ or workspace root
_curr = Path('.').resolve()
PROJECT_ROOT = _curr if (_curr / 'src').exists() else (_curr.parent if (_curr.parent / 'src').exists() else (_curr.parents[1] if len(_curr.parents) > 1 and (_curr.parents[1] / 'src').exists() else _curr))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Switch working directory to project root so all relative paths ('data/...', 'artifacts/...') resolve seamlessly
os.chdir(PROJECT_ROOT)
"""



def create_all_notebooks():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 01_data_quality.ipynb
    # -------------------------------------------------------------
    nb1 = make_notebook([
        md_cell("# 01: Data Quality, Schema Validation & Temporal Splitting\nThis notebook demonstrates loading raw transactional interactions, performing comprehensive data validation, checking sparsity, and applying a strict temporal split to prevent data leakage."),
        code_cell(PATH_BOOTSTRAP + """
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocess import DataPreprocessor
"""),
        md_cell("## 1. Load Raw Transactions and Product Catalog"),
        code_cell("""loader = DataLoader('data/raw/retail_events.csv', 'data/raw/product_catalog.csv')
events_df, catalog_df = loader.load_and_merge()
print(f"Loaded {len(events_df):,} events across {events_df['customer_id'].nunique():,} customers and {events_df['product_id'].nunique():,} products.")
events_df.head()
"""),
        md_cell("## 2. Execute Data Validation & Quality Checks"),
        code_cell("""validator = DataValidator(output_report_path='outputs/data_quality_report.json')
clean_df, dq_report = validator.validate_and_clean(events_df)
print("Data Quality Report Summary:")
print(f"- Validated Rows: {dq_report['row_count']:,}")
print(f"- Interaction Sparsity: {dq_report['sparsity']*100:.3f}%")
print(f"- Event Distribution: {dq_report['event_distribution']}")
"""),
        md_cell("## 3. Strict Temporal Train / Validation / Test Split"),
        code_cell("""preprocessor = DataPreprocessor(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
train_df, val_df, test_df, cutoffs = preprocessor.temporal_split(clean_df)
print(f"Train events: {len(train_df):,} ({train_df['timestamp'].min()} to {cutoffs['train_val_cutoff']})")
print(f"Val events  : {len(val_df):,} ({cutoffs['train_val_cutoff']} to {cutoffs['val_test_cutoff']})")
print(f"Test events : {len(test_df):,} ({cutoffs['val_test_cutoff']} to {test_df['timestamp'].max()})")
"""),
    ])
    with open(nb_dir / "01_data_quality.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)

    # -------------------------------------------------------------
    # 02_customer_segmentation.ipynb
    # -------------------------------------------------------------
    nb2 = make_notebook([
        md_cell("# 02: Customer Feature Engineering, RFM Analysis & K-Means Segmentation\nThis notebook walks through calculating RFM and multi-dimensional engagement metrics, transforming skewed distributions with log1p and robust scaling, evaluating optimal K, and generating deterministic descriptive cluster profiles."),
        code_cell(PATH_BOOTSTRAP + """
import matplotlib.pyplot as plt
from src.features.customer_features import CustomerFeatureEngineer
from src.segmentation.clustering import CustomerClustering
from src.segmentation.profiling import SegmentProfiler
from src.segmentation.visualization import SegmentVisualizer
"""),
        md_cell("## 1. Extract Customer Features & RFM Metrics on Train Split"),
        code_cell("""train_events_df = pd.read_parquet('data/processed/train_events.parquet')
cust_fe = CustomerFeatureEngineer(scaler_type='robust')
features_df = cust_fe.extract_features(train_events_df)
transformed_df, scaled_matrix = cust_fe.transform_and_scale(features_df, fit=True)
features_df.head()
"""),
        md_cell("## 2. Evaluate Optimal K via Inertia & Silhouette Diagnostics"),
        code_cell("""clustering = CustomerClustering(n_clusters=4, random_state=42)
diag = clustering.find_optimal_k(scaled_matrix, k_range=[2, 3, 4, 5, 6, 7, 8])
cluster_labels = clustering.fit_predict(scaled_matrix)
features_df['cluster'] = cluster_labels
"""),
        md_cell("## 3. Automated Deterministic Segment Profiling (No Hardcoding)"),
        code_cell("""profiler = SegmentProfiler()
profiles_df, labels_map = profiler.profile_segments(features_df)
print("Discovered Customer Segments:")
print(profiles_df[['cluster_id', 'label', 'customer_count', 'percentage_of_customers', 'avg_monetary_value', 'avg_frequency', 'avg_recency_days']])
"""),
        md_cell("## 4. 2D PCA Dimensionality Reduction Projection"),
        code_cell("""viz = SegmentVisualizer(n_components=2)
viz.plot_pca_clusters(scaled_matrix, cluster_labels, cluster_names=labels_map, output_image_path='outputs/figures/customer_segments_pca.png')
from IPython.display import Image
Image('outputs/figures/customer_segments_pca.png')
"""),
    ])
    with open(nb_dir / "02_customer_segmentation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)

    # -------------------------------------------------------------
    # 03_item_similarity.ipynb
    # -------------------------------------------------------------
    nb3 = make_notebook([
        md_cell("# 03: CSR Interaction Matrix & Item-Item Collaborative Filtering\nThis notebook demonstrates constructing an implicit interaction matrix with event weights and damped logarithmic scaling, computing sparse cosine item similarity, and generating explainable recommendations."),
        code_cell(PATH_BOOTSTRAP + """
from src.features.interaction_features import InteractionMatrixBuilder
from src.recommenders.item_item import ItemItemRecommender
"""),
        md_cell("## 1. Build Damped Interaction Matrix"),
        code_cell("""train_df = pd.read_parquet('data/processed/train_events.parquet')
id_builder = InteractionMatrixBuilder().fit_mappings(train_df)
csr_mat, agg_df = id_builder.build_matrix(train_df)
print(f"Matrix shape: {csr_mat.shape}, Non-zeros: {csr_mat.nnz:,}")
"""),
        md_cell("## 2. Train Item-Item Collaborative Filtering"),
        code_cell("""item_model = ItemItemRecommender(k_nearest=50).fit(csr_mat, id_builder)
test_prod_id = list(id_builder.item_to_idx.keys())[0]
similar_items = item_model.get_similar_items(test_prod_id, n=5)
print(f"Top-5 items similar to Product #{test_prod_id}:")
for pid, sim in similar_items:
    print(f"  - Product #{pid}: Similarity = {sim:.4f}")
"""),
        md_cell("## 3. Recommend with Explainability Traces"),
        code_cell("""test_cust_id = list(id_builder.user_to_idx.keys())[0]
recs = item_model.recommend(test_cust_id, n=5)
print(f"Recommendations for Customer #{test_cust_id}:")
for pid, score in recs:
    explanation = item_model.explain(test_cust_id, pid)
    print(f"  - Product #{pid} (Score: {score:.3f}): {explanation['explanation']}")
"""),
    ])
    with open(nb_dir / "03_item_similarity.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)

    # -------------------------------------------------------------
    # 04_matrix_factorization.ipynb
    # -------------------------------------------------------------
    nb4 = make_notebook([
        md_cell("# 04: Implicit ALS Matrix Factorization vs TruncatedSVD Baseline\nThis notebook compares Implicit Feedback ALS (with confidence scaling $C_{ui} = 1 + \\alpha r_{ui}$) against the classical TruncatedSVD baseline."),
        code_cell(PATH_BOOTSTRAP + """
from src.features.interaction_features import InteractionMatrixBuilder
from src.recommenders.als import ALSRecommender
from src.recommenders.svd import SVDRecommender
"""),
        md_cell("## 1. Load Matrix & Train ALS Model"),
        code_cell("""train_df = pd.read_parquet('data/processed/train_events.parquet')
id_builder = InteractionMatrixBuilder().fit_mappings(train_df)
csr_mat, _ = id_builder.build_matrix(train_df)

als = ALSRecommender(factors=64, regularization=0.05, iterations=20, alpha_confidence=40.0)
als.fit(csr_mat, id_builder)
print("ALS Model Fitted.")
"""),
        md_cell("## 2. Train TruncatedSVD Baseline"),
        code_cell("""svd = SVDRecommender(n_components=32)
svd.fit(csr_mat, id_builder)
print(f"TruncatedSVD Fitted. Explained Variance Sum: {svd.svd.explained_variance_ratio_.sum():.4f}")
"""),
        md_cell("## 3. Compare Personalization Recommendations"),
        code_cell("""sample_user = list(id_builder.user_to_idx.keys())[0]
als_recs = als.recommend(sample_user, n=5)
svd_recs = svd.recommend(sample_user, n=5)

print(f"ALS Top-5 for Customer #{sample_user}: {als_recs}")
print(f"SVD Top-5 for Customer #{sample_user}: {svd_recs}")
"""),
    ])
    with open(nb_dir / "04_matrix_factorization.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=2)

    # -------------------------------------------------------------
    # 05_recommender_evaluation.ipynb
    # -------------------------------------------------------------
    nb5 = make_notebook([
        md_cell("# 05: Offline Recommender Evaluation, Ablation Study & Hybrid Demo\nThis notebook demonstrates full temporal offline evaluation, ablation study, and the unified RecommendationEngine with cold-start, sparse, and mature customer routing."),
        code_cell(PATH_BOOTSTRAP + """
from src.recommenders.hybrid import RecommendationEngine
"""),
        md_cell("## 1. Load Production Recommendation Engine & Evaluation Results"),
        code_cell("""engine = RecommendationEngine.load('artifacts')
eval_df = pd.read_csv('outputs/metrics/evaluation_metrics.csv')
ablation_df = pd.read_csv('outputs/metrics/ablation_study.csv')
print("Offline Evaluation Summary:")
eval_df
"""),
        md_cell("## 2. Component Ablation Study Analysis"),
        code_cell("""print("Ablation Study Results:")
ablation_df[['model', 'recall@10', 'recall@20', 'hit_rate@10', 'hit_rate@20', 'ndcg@10', 'catalog_coverage']]
"""),
        md_cell("## 3. Live Recommendation Engine Demo across Customer States"),
        code_cell("""# Case A: Brand New Customer (Cold-Start)
cold_res = engine.recommend(customer_id=9999999, n=5)
print(f"Cold-Start User Strategy: {cold_res['recommendation_type']}")
for r in cold_res['recommendations']:
    print(f"  Rank #{r['rank']} Product #{r['product_id']} (Score: {r['score']:.3f}, Sources: {r['sources']}): {r['explanation']}")

# Case B: Mature Customer (Full Hybrid)
mature_user = list(engine.user_interaction_counts.keys())[0]
mature_res = engine.recommend(customer_id=mature_user, n=5)
print(f"\\nMature User Strategy ({mature_res['segment_label']}): {mature_res['recommendation_type']}")
for r in mature_res['recommendations']:
    print(f"  Rank #{r['rank']} Product #{r['product_id']} (Score: {r['score']:.3f}, Sources: {r['sources']}): {r['explanation']}")
"""),
    ])
    with open(nb_dir / "05_recommender_evaluation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb5, f, indent=2)

    print("All 5 Jupyter notebooks generated successfully in notebooks/.")


if __name__ == "__main__":
    create_all_notebooks()
