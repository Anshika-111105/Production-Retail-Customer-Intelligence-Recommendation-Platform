"""Master Pipeline Execution Script for Retail Customer Intelligence & Recommendation Platform."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import mlflow

from src.utils.logger import get_logger
from src.utils.config import AppConfig
from src.utils.reproducibility import set_seed
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocess import DataPreprocessor
from src.features.customer_features import CustomerFeatureEngineer
from src.segmentation.clustering import CustomerClustering
from src.segmentation.profiling import SegmentProfiler
from src.segmentation.visualization import SegmentVisualizer
from src.features.interaction_features import InteractionMatrixBuilder
from src.features.product_features import ProductFeatureEngineer
from src.features.segment_features import SegmentPreferenceMiner
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.trending import TrendingRecommender
from src.recommenders.item_item import ItemItemRecommender
from src.recommenders.als import ALSRecommender
from src.recommenders.svd import SVDRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import RecommendationEngine
from src.evaluation.evaluator import RecommenderEvaluator
from src.evaluation.ablation import AblationStudy
from src.evaluation.segment_evaluation import SegmentEvaluator

logger = get_logger("master_pipeline")


def run_pipeline() -> None:
    """Execute end-to-end data processing, modeling, evaluation, and artifact generation."""
    logger.info("================================================================================")
    logger.info("STARTING END-TO-END RETAIL CUSTOMER INTELLIGENCE & RECOMMENDATION PIPELINE")
    logger.info("================================================================================")

    # 1. Load Configurations
    config = AppConfig()
    seed = config.base.get("project", {}).get("random_seed", 42)
    set_seed(seed)

    # 2. Check and Load Data
    raw_events_path = Path(config.base["data"]["raw_events_path"])
    raw_catalog_path = Path(config.base["data"]["raw_catalog_path"])

    if not raw_events_path.exists():
        logger.warning(f"Raw events file not found at {raw_events_path}. Auto-generating synthetic benchmark dataset...")
        from scripts.generate_sample_data import generate_retail_dataset
        generate_retail_dataset(num_customers=3000, num_products=800, num_interactions=60000, random_seed=seed)

    loader = DataLoader(str(raw_events_path), str(raw_catalog_path) if raw_catalog_path.exists() else None)
    raw_events_df, raw_catalog_df = loader.load_and_merge()

    # 3. Data Validation & Quality Checks
    validator = DataValidator(
        min_price=config.base["validation"]["min_price_threshold"],
        max_price=config.base["validation"]["max_price_threshold"],
        max_quantity=config.base["validation"]["max_quantity_threshold"],
        output_report_path="outputs/data_quality_report.json",
    )
    clean_events_df, dq_report = validator.validate_and_clean(raw_events_df)

    # 4. Strict Temporal Data Splitting (Prevent Temporal Leakage)
    preprocessor = DataPreprocessor(
        train_ratio=config.base["split"]["train_ratio"],
        val_ratio=config.base["split"]["val_ratio"],
        test_ratio=config.base["split"]["test_ratio"],
        processed_dir=config.base["data"]["processed_dir"],
    )
    train_events_df, val_events_df, test_events_df, cutoffs = preprocessor.temporal_split(clean_events_df)
    preprocessor.save_splits(train_events_df, val_events_df, test_events_df)

    # 5. Customer Feature Engineering & RFM Extraction (Trained strictly on Train Events)
    logger.info("Step 5: Customer Feature Engineering & RFM Extraction...")
    cust_fe = CustomerFeatureEngineer(
        reference_date=cutoffs["train_val_cutoff"],
        scaler_type=config.segmentation["clustering"]["scaler"],
    )
    train_cust_features_df = cust_fe.extract_features(train_events_df)
    train_cust_transformed_df, train_scaled_matrix = cust_fe.transform_and_scale(train_cust_features_df, fit=True)
    cust_fe.save("artifacts/models/customer_scaler.joblib")
    train_cust_features_df.to_parquet("artifacts/features/customer_features.parquet", index=False)

    # 6. Customer Clustering (K-Means) & Diagnostics
    logger.info("Step 6: Customer Segmentation & Clustering...")
    clustering = CustomerClustering(
        n_clusters=config.segmentation["clustering"]["default_k"],
        random_state=seed,
        n_init=config.segmentation["clustering"]["n_init"],
        max_iter=config.segmentation["clustering"]["max_iter"],
    )
    # Run optimal K diagnostics
    diagnostics = clustering.find_optimal_k(
        train_scaled_matrix,
        k_range=config.segmentation["clustering"]["k_search_range"],
    )
    cluster_labels = clustering.fit_predict(train_scaled_matrix)
    clustering.save("artifacts/models/customer_kmeans.joblib")

    train_cust_features_df["cluster"] = cluster_labels
    train_cust_transformed_df["cluster"] = cluster_labels

    # 7. Segment Profiling (Deterministic Rule-Based Interpretations)
    logger.info("Step 7: Automated Segment Profiling & Label Generation...")
    profiler = SegmentProfiler(metrics=config.segmentation["profiling"]["metrics"])
    profiles_df, segment_labels_map = profiler.profile_segments(train_cust_features_df)
    profiler.generate_markdown_report(
        profiles_df,
        output_csv_path="outputs/segment_profiles.csv",
        output_md_path="reports/segment_profiles.md",
    )

    # 8. PCA Visualization
    logger.info("Step 8: PCA 2D Cluster Visualization...")
    viz = SegmentVisualizer(n_components=2, random_state=seed)
    viz.plot_pca_clusters(
        train_scaled_matrix,
        cluster_labels,
        cluster_names=segment_labels_map,
        output_image_path="outputs/figures/customer_segments_pca.png",
    )

    # 9. Product Feature Engineering
    logger.info("Step 9: Product Feature Engineering...")
    prod_fe = ProductFeatureEngineer(recent_window_days=config.recommendation["popularity"]["recent_window_days"])
    prod_features_df = prod_fe.extract_features(
        events_df=train_events_df,
        catalog_df=raw_catalog_df,
        reference_date=cutoffs["train_val_cutoff"],
    )
    prod_fe.save(prod_features_df, "artifacts/features/product_features.parquet")

    # 10. Interaction Matrix & ID Mappings
    logger.info("Step 10: Interaction Matrix & Sparse CSR Construction...")
    id_builder = InteractionMatrixBuilder(event_weights=config.recommendation["interactions"]["event_weights"])
    id_builder.fit_mappings(train_events_df)
    train_user_item_csr, _ = id_builder.build_matrix(train_events_df)
    id_builder.save("artifacts/mappings/id_mappings.joblib")

    # 11. Train Baseline Models (Popularity & Trending)
    logger.info("Step 11: Training Baseline Recommenders (Popularity & Trending)...")
    pop_model = PopularityRecommender(
        alpha_unique_customers=config.recommendation["popularity"]["alpha_unique_customers"],
        beta_recent_purchases=config.recommendation["popularity"]["beta_recent_purchases"],
        recent_window_days=config.recommendation["popularity"]["recent_window_days"],
    ).fit(train_events_df, catalog_df=raw_catalog_df)
    pop_model.save("artifacts/models/popularity_model.joblib")

    trending_model = TrendingRecommender(
        decay_lambda=config.recommendation["trending"]["decay_lambda"]
    ).fit(train_events_df, reference_date=cutoffs["train_val_cutoff"])
    trending_model.save("artifacts/models/trending_model.joblib")

    # 12. Train Item-Item Collaborative Filtering
    logger.info("Step 12: Training Item-Item Collaborative Filtering...")
    item_item_model = ItemItemRecommender(
        k_nearest=config.recommendation["item_item"]["k_nearest_items"],
        min_similarity=config.recommendation["item_item"]["min_similarity_threshold"],
    ).fit(train_user_item_csr, id_builder)
    item_item_model.save("artifacts/models/item_item_model.joblib")

    # 13. Train Matrix Factorization (ALS & SVD)
    logger.info("Step 13: Training Matrix Factorization Models (Implicit ALS & SVD Baseline)...")
    als_model = ALSRecommender(
        factors=config.recommendation["als"]["factors"],
        regularization=config.recommendation["als"]["regularization"],
        iterations=config.recommendation["als"]["iterations"],
        alpha_confidence=config.recommendation["als"]["alpha_confidence"],
        random_state=seed,
    ).fit(train_user_item_csr, id_builder)
    als_model.save("artifacts/models/als_model.joblib")

    svd_model = SVDRecommender(
        n_components=config.recommendation["svd"]["n_components"],
        random_state=seed,
    ).fit(train_user_item_csr, id_builder)
    svd_model.save("artifacts/models/svd_model.joblib")

    # 14. Train Content-Based Recommender
    logger.info("Step 14: Training Content-Based Product Recommender...")
    content_catalog = raw_catalog_df if raw_catalog_df is not None else prod_features_df
    content_model = ContentBasedRecommender(
        tfidf_max_features=config.recommendation["content_based"]["tfidf_max_features"],
    ).fit(content_catalog)
    content_model.save("artifacts/models/content_based_model.joblib")

    # 15. Mine Customer Segment Preferences
    logger.info("Step 15: Mining Customer Segment Product Affinities...")
    segment_miner = SegmentPreferenceMiner().fit(train_events_df, train_cust_features_df)
    segment_miner.save("artifacts/features/segment_preferences.joblib")

    # 16. Assemble Complete Production Recommendation Engine
    logger.info("Step 16: Assembling Unified Recommendation Engine...")
    cust_seg_map = dict(zip(train_cust_features_df["customer_id"], train_cust_features_df["cluster"]))
    user_counts_map = train_events_df.groupby("customer_id").size().to_dict()
    user_purchased_map = train_events_df[train_events_df["event_type"] == "purchase"].groupby("customer_id")["product_id"].apply(set).to_dict()

    engine = RecommendationEngine(
        id_builder=id_builder,
        popularity_model=pop_model,
        trending_model=trending_model,
        item_item_model=item_item_model,
        als_model=als_model,
        svd_model=svd_model,
        content_model=content_model,
        segment_miner=segment_miner,
        customer_segments=cust_seg_map,
        segment_labels=segment_labels_map,
        user_interaction_counts=user_counts_map,
        user_purchased_items=user_purchased_map,
        product_metadata=content_model.product_metadata_,
        sparse_user_threshold=config.recommendation["cold_start"]["sparse_user_threshold"],
        weights=config.recommendation["hybrid"]["weights"],
    )
    engine.save("artifacts")

    # 17. Offline Temporal Evaluation on Unseen Test Split
    logger.info("Step 17: Running Comprehensive Offline Temporal Evaluation...")
    catalog_items_set = set(prod_features_df["product_id"])
    item_cat_map = dict(zip(prod_features_df["product_id"], prod_features_df["category"]))
    item_pop_prob = (prod_features_df["total_interactions"] / max(1, prod_features_df["total_interactions"].sum())).to_dict()

    evaluator = RecommenderEvaluator(
        k_list=config.recommendation["evaluation"]["k_list"],
        catalog_items=catalog_items_set,
        item_category_map=item_cat_map,
        item_popularity_prob=item_pop_prob,
    )
    ground_truth = evaluator.extract_ground_truth(test_events_df)
    train_purchased_dict = train_events_df.groupby("customer_id")["product_id"].apply(set).to_dict()

    models_to_evaluate = {
        "Popularity Baseline": pop_model,
        "Trending Baseline": trending_model,
        "Item-Item Collaborative Filtering": item_item_model,
        "ALS Matrix Factorization": als_model,
        "TruncatedSVD Baseline": svd_model,
        "Hybrid Engine": engine,
    }

    results_df = evaluator.evaluate_all(
        models_dict=models_to_evaluate,
        ground_truth=ground_truth,
        filter_items_dict=train_purchased_dict,
        user_interaction_counts=user_counts_map,
        sparse_threshold=config.recommendation["cold_start"]["sparse_user_threshold"],
    )
    evaluator.generate_report(results_df, output_csv_path="outputs/metrics/evaluation_metrics.csv", output_md_path="reports/recommender_evaluation.md")

    # 18. Ablation Study
    logger.info("Step 18: Executing Ablation Study across Component Combinations...")
    ablation = AblationStudy(evaluator=evaluator)
    ablation_df = ablation.run_ablation(
        als_model=als_model,
        item_item_model=item_item_model,
        segment_miner=segment_miner,
        popularity_model=pop_model,
        trending_model=trending_model,
        customer_segments=cust_seg_map,
        ground_truth=ground_truth,
        filter_items_dict=train_purchased_dict,
        output_csv_path="outputs/metrics/ablation_study.csv",
    )

    # 19. Segment-Stratified Performance Evaluation
    logger.info("Step 19: Executing Segment-Stratified Recommender Evaluation...")
    seg_evaluator = SegmentEvaluator(evaluator=evaluator)
    seg_eval_df = seg_evaluator.evaluate_by_segment(
        recommend_fn=lambda uid, n=10, filter_items=None: engine.recommend(uid, n=n, filter_items=filter_items)["recommendations"],
        customer_segments=cust_seg_map,
        segment_labels=segment_labels_map,
        ground_truth=ground_truth,
        filter_items_dict=train_purchased_dict,
        output_csv_path="outputs/metrics/segment_evaluation.csv",
    )

    # 20. MLflow Experiment Tracking
    logger.info("Step 20: Logging Experiment Metrics to MLflow...")
    try:
        mlflow.set_experiment("retail_recommendation_engine")
        with mlflow.start_run(run_name="full_production_pipeline"):
            # Log params
            mlflow.log_params({
                "kmeans_k": config.segmentation["clustering"]["default_k"],
                "als_factors": config.recommendation["als"]["factors"],
                "als_regularization": config.recommendation["als"]["regularization"],
                "als_iterations": config.recommendation["als"]["iterations"],
                "sparse_threshold": config.recommendation["cold_start"]["sparse_user_threshold"],
            })

            # Log hybrid metrics
            hybrid_row = results_df[results_df["model"] == "Hybrid Engine"]
            if not hybrid_row.empty:
                for col in ["precision@10", "recall@10", "hit_rate@10", "map@10", "ndcg@10", "catalog_coverage", "diversity@10"]:
                    if col in hybrid_row.columns:
                        safe_metric_name = col.replace("@", "_at_")
                        mlflow.log_metric(safe_metric_name, float(hybrid_row[col].values[0]))

            # Log artifacts
            if Path("outputs/metrics/evaluation_metrics.csv").exists():
                mlflow.log_artifact("outputs/metrics/evaluation_metrics.csv")
            if Path("outputs/figures/customer_segments_pca.png").exists():
                mlflow.log_artifact("outputs/figures/customer_segments_pca.png")
            if Path("outputs/data_quality_report.json").exists():
                mlflow.log_artifact("outputs/data_quality_report.json")
    except Exception as e:
        logger.warning(f"MLflow logging encountered notice: {e}")

    logger.info("================================================================================")
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("================================================================================")


if __name__ == "__main__":
    run_pipeline()
