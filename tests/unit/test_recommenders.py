"""Unit tests for all recommender algorithms, hybrid engine, and offline evaluation metrics."""

import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features.interaction_features import InteractionMatrixBuilder
from src.features.segment_features import SegmentPreferenceMiner
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.trending import TrendingRecommender
from src.recommenders.item_item import ItemItemRecommender
from src.recommenders.als import ALSRecommender
from src.recommenders.svd import SVDRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import RecommendationEngine
from src.evaluation.metrics import precision_at_k, recall_at_k, hit_rate_at_k, ndcg_at_k, map_at_k


@pytest.fixture
def mock_dataset():
    """Create mock dataset for testing."""
    users = [101, 102, 103, 104, 105, 106]
    items = [201, 202, 203, 204, 205, 206, 207, 208]
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    
    events_df = pd.DataFrame({
        "customer_id": np.random.choice(users, size=60),
        "product_id": np.random.choice(items, size=60),
        "timestamp": dates,
        "quantity": 1,
        "unit_price": 25.0,
        "category": np.random.choice(["Tech", "Apparel"], size=60),
        "brand": np.random.choice(["BrandA", "BrandB"], size=60),
        "description": "Premium test item with great quality.",
        "discount": 0.0,
        "event_type": np.random.choice(["view", "add_to_cart", "purchase"], size=60),
    })

    catalog_df = pd.DataFrame({
        "product_id": items,
        "category": ["Tech", "Tech", "Apparel", "Apparel", "Tech", "Apparel", "Tech", "Apparel"],
        "brand": ["BrandA", "BrandB", "BrandA", "BrandB", "BrandA", "BrandB", "BrandA", "BrandB"],
        "unit_price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        "description": ["Tech product A", "Tech product B", "Shirt A", "Shirt B", "Phone", "Shoes", "Laptop", "Jacket"],
    })

    return events_df, catalog_df


def test_ranking_metrics():
    """Test metric calculations."""
    rec = [1, 2, 3, 4, 5]
    actual = {2, 5, 8}

    p5 = precision_at_k(rec, actual, k=5)
    r5 = recall_at_k(rec, actual, k=5)
    hr5 = hit_rate_at_k(rec, actual, k=5)
    ndcg5 = ndcg_at_k(rec, actual, k=5)
    map5 = map_at_k(rec, actual, k=5)

    assert p5 == 2 / 5
    assert r5 == 2 / 3
    assert hr5 == 1.0
    assert 0.0 < ndcg5 <= 1.0
    assert 0.0 < map5 <= 1.0


def test_popularity_and_trending(mock_dataset):
    """Test popularity and trending baseline models."""
    events_df, catalog_df = mock_dataset
    pop = PopularityRecommender()
    pop.fit(events_df)
    recs = pop.recommend(101, n=5)
    assert len(recs) == 5
    assert pop.is_fitted

    trend = TrendingRecommender()
    trend.fit(events_df)
    t_recs = trend.recommend(101, n=5)
    assert len(t_recs) == 5
    assert trend.is_fitted


def test_collaborative_and_matrix_factorization(mock_dataset):
    """Test Item-Item, ALS, and SVD recommenders."""
    events_df, catalog_df = mock_dataset
    id_builder = InteractionMatrixBuilder().fit_mappings(events_df)
    csr_matrix, _ = id_builder.build_matrix(events_df)

    # Item-Item
    ii_model = ItemItemRecommender(k_nearest=10)
    ii_model.fit(csr_matrix, id_builder)
    ii_recs = ii_model.recommend(101, n=5)
    assert isinstance(ii_recs, list)

    # ALS
    als_model = ALSRecommender(factors=8, iterations=5)
    als_model.fit(csr_matrix, id_builder)
    als_recs = als_model.recommend(101, n=5)
    assert isinstance(als_recs, list)

    # SVD
    svd_model = SVDRecommender(n_components=4)
    svd_model.fit(csr_matrix, id_builder)
    svd_recs = svd_model.recommend(101, n=5)
    assert isinstance(svd_recs, list)


def test_content_based_and_hybrid_engine(mock_dataset, tmp_path):
    """Test Content-Based recommender and unified Hybrid RecommendationEngine."""
    events_df, catalog_df = mock_dataset
    id_builder = InteractionMatrixBuilder().fit_mappings(events_df)
    csr_matrix, _ = id_builder.build_matrix(events_df)

    # Content-based
    content_model = ContentBasedRecommender(k_similar=5)
    content_model.fit(catalog_df)
    sim_items = content_model.get_similar_products(201, n=3)
    assert len(sim_items) > 0

    # Pop, Trend, ItemItem, ALS
    pop = PopularityRecommender().fit(events_df)
    trend = TrendingRecommender().fit(events_df)
    ii = ItemItemRecommender().fit(csr_matrix, id_builder)
    als = ALSRecommender(factors=8, iterations=5).fit(csr_matrix, id_builder)
    svd = SVDRecommender(n_components=4).fit(csr_matrix, id_builder)

    clusters_df = pd.DataFrame({
        "customer_id": [101, 102, 103, 104, 105, 106],
        "cluster": [0, 0, 1, 1, 0, 1]
    })
    seg_miner = SegmentPreferenceMiner().fit(events_df, clusters_df)

    engine = RecommendationEngine(
        id_builder=id_builder,
        popularity_model=pop,
        trending_model=trend,
        item_item_model=ii,
        als_model=als,
        svd_model=svd,
        content_model=content_model,
        segment_miner=seg_miner,
        customer_segments={101: 0, 102: 0, 103: 1},
        segment_labels={0: "Loyal High-Value", 1: "Casual Bargain"},
        user_interaction_counts={101: 15, 102: 2, 103: 0},
        user_purchased_items={101: {201}},
        product_metadata=content_model.product_metadata_,
        sparse_user_threshold=3,
    )

    # Case A: Cold start (0 interactions)
    res_cold = engine.recommend(103, n=5)
    assert res_cold["recommendation_type"] == "cold_start_user"
    assert len(res_cold["recommendations"]) > 0

    # Case B: Sparse user (2 interactions)
    res_sparse = engine.recommend(102, n=5)
    assert res_sparse["recommendation_type"] == "sparse_user"
    assert len(res_sparse["recommendations"]) > 0

    # Case C: Mature user (15 interactions)
    res_mature = engine.recommend(101, n=5)
    assert res_mature["recommendation_type"] == "hybrid"
    assert len(res_mature["recommendations"]) > 0
    # Check already purchased item is filtered
    rec_ids = [r["product_id"] for r in res_mature["recommendations"]]
    assert 201 not in rec_ids

    # Test persistence and loading
    art_dir = tmp_path / "artifacts"
    engine.save(str(art_dir))
    loaded_engine = RecommendationEngine.load(str(art_dir))
    res_loaded = loaded_engine.recommend(101, n=5)
    assert len(res_loaded["recommendations"]) > 0
