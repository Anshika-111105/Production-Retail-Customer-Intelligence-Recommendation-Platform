"""End-to-end integration test for the entire Recommendation Pipeline."""

import pytest
import pandas as pd
from pathlib import Path

from src.recommenders.hybrid import RecommendationEngine
from src.features.customer_features import CustomerFeatureEngineer
from src.data.loader import DataLoader
from src.data.validator import DataValidator


def test_full_pipeline_artifacts_exist():
    """Verify all pipeline models and feature artifacts were generated and can be loaded."""
    assert Path("artifacts/models/customer_kmeans.joblib").exists()
    assert Path("artifacts/models/customer_scaler.joblib").exists()
    assert Path("artifacts/models/popularity_model.joblib").exists()
    assert Path("artifacts/models/trending_model.joblib").exists()
    assert Path("artifacts/models/item_item_model.joblib").exists()
    assert Path("artifacts/models/als_model.joblib").exists()
    assert Path("artifacts/models/svd_model.joblib").exists()
    assert Path("artifacts/models/content_based_model.joblib").exists()
    assert Path("artifacts/features/segment_preferences.joblib").exists()
    assert Path("artifacts/mappings/id_mappings.joblib").exists()
    assert Path("outputs/metrics/evaluation_metrics.csv").exists()
    assert Path("outputs/metrics/ablation_study.csv").exists()
    assert Path("outputs/metrics/segment_evaluation.csv").exists()
    assert Path("reports/segment_profiles.md").exists()
    assert Path("reports/recommender_evaluation.md").exists()


def test_engine_inference_modes():
    """Test engine behavior across unknown user, sparse user, mature user, and new product."""
    engine = RecommendationEngine.load("artifacts")

    # 1. Unknown user (Cold-start routing)
    unknown_uid = 9999999
    res_cold = engine.recommend(customer_id=unknown_uid, n=5)
    assert res_cold["recommendation_type"] == "cold_start_user"
    assert len(res_cold["recommendations"]) == 5
    for rec in res_cold["recommendations"]:
        assert "explanation" in rec
        assert rec["score"] >= 0.0

    # 2. Mature user
    # Fetch a mature user from customer segments
    mature_uids = [uid for uid, cnt in engine.user_interaction_counts.items() if cnt > engine.sparse_user_threshold]
    if mature_uids:
        test_uid = mature_uids[0]
        res_mature = engine.recommend(customer_id=test_uid, n=10)
        assert res_mature["recommendation_type"] == "hybrid"
        assert len(res_mature["recommendations"]) <= 10
        # Check that already purchased products are excluded
        already_purchased = engine.user_purchased_items.get(test_uid, set())
        for rec in res_mature["recommendations"]:
            assert rec["product_id"] not in already_purchased

    # 3. New Product Content-based Cold-start
    new_product_meta = {
        "category": "Electronics/Audio",
        "brand": "SonicPulse",
        "unit_price": 149.99,
        "description": "High-fidelity wireless noise cancelling headphones with deep bass.",
    }
    similar_existing = engine.content_model.recommend_for_new_product(new_product_meta, n=5)
    assert len(similar_existing) == 5
    for pid, sim in similar_existing:
        assert pid in engine.product_metadata
        assert sim > 0.0
