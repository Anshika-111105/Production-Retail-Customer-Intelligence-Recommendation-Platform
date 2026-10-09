"""Unit tests for Customer Feature Engineering and Segmentation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.customer_features import CustomerFeatureEngineer
from src.segmentation.clustering import CustomerClustering
from src.segmentation.profiling import SegmentProfiler
from src.segmentation.visualization import SegmentVisualizer


@pytest.fixture
def sample_interactions_df():
    """Create fixture dataframe of customer interactions."""
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "customer_id": np.random.choice([101, 102, 103, 104, 105], size=100),
        "product_id": np.random.choice([201, 202, 203, 204], size=100),
        "timestamp": dates,
        "quantity": np.random.choice([1, 2, 3], size=100),
        "unit_price": np.random.uniform(10.0, 100.0, size=100),
        "category": np.random.choice(["Electronics", "Apparel", "Home"], size=100),
        "discount": np.random.choice([0.0, 0.1, 0.2], size=100),
        "event_type": np.random.choice(["view", "add_to_cart", "purchase"], size=100, p=[0.5, 0.3, 0.2]),
    })
    return df


def test_customer_feature_extraction(sample_interactions_df):
    """Test RFM and engagement feature generation."""
    fe = CustomerFeatureEngineer()
    features_df = fe.extract_features(sample_interactions_df)

    assert len(features_df) == sample_interactions_df["customer_id"].nunique()
    assert "recency" in features_df.columns
    assert "frequency" in features_df.columns
    assert "monetary_value" in features_df.columns
    assert (features_df["recency"] >= 0).all()
    assert (features_df["frequency"] > 0).all()

    transformed_df, scaled_matrix = fe.transform_and_scale(features_df, fit=True)
    assert scaled_matrix.shape[0] == len(features_df)
    assert not np.isnan(scaled_matrix).any()


def test_clustering_and_profiling(sample_interactions_df, tmp_path):
    """Test K-Means clustering and deterministic profiling."""
    fe = CustomerFeatureEngineer()
    features_df = fe.extract_features(sample_interactions_df)
    _, scaled_matrix = fe.transform_and_scale(features_df, fit=True)

    clustering = CustomerClustering(n_clusters=2, random_state=42)
    labels = clustering.fit_predict(scaled_matrix)
    assert len(labels) == len(features_df)

    features_df["cluster"] = labels
    profiler = SegmentProfiler()
    profiles_df, labels_map = profiler.profile_segments(features_df)

    assert len(profiles_df) == 2
    assert len(labels_map) == 2
    for cid, label in labels_map.items():
        assert isinstance(label, str)
        assert len(label) > 0

    csv_path = tmp_path / "profiles.csv"
    md_path = tmp_path / "profiles.md"
    profiler.generate_markdown_report(profiles_df, str(csv_path), str(md_path))
    assert csv_path.exists()
    assert md_path.exists()


def test_pca_visualization(sample_interactions_df, tmp_path):
    """Test PCA 2D plot creation."""
    fe = CustomerFeatureEngineer()
    features_df = fe.extract_features(sample_interactions_df)
    _, scaled_matrix = fe.transform_and_scale(features_df, fit=True)

    clustering = CustomerClustering(n_clusters=2, random_state=42)
    labels = clustering.fit_predict(scaled_matrix)

    viz = SegmentVisualizer()
    out_img = tmp_path / "pca.png"
    viz.plot_pca_clusters(scaled_matrix, labels, output_image_path=str(out_img))
    assert out_img.exists()
