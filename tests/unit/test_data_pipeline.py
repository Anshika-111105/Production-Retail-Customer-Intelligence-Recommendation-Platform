"""Unit tests for the Data pipeline components."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocess import DataPreprocessor
from src.data.schema import RawInteractionRecord, ProductCatalogRecord


def test_raw_interaction_schema():
    """Test validation of raw interaction records."""
    rec = RawInteractionRecord(
        customer_id=101,
        product_id=202,
        timestamp="2026-01-01 10:00:00",
        quantity=2,
        unit_price=49.99,
        event_type="purchase"
    )
    assert rec.customer_id == 101
    assert rec.event_type == "purchase"
    assert rec.quantity == 2


def test_data_loader_and_validator(tmp_path):
    """Test loading and validating synthetic data."""
    # Create sample csv
    df_raw = pd.DataFrame({
        "visitorid": [1, 2, 2, 3, None],
        "itemid": [10, 20, 20, 30, 40],
        "timestamp": ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03", "2026-01-04"],
        "event": ["view", "add_to_cart", "add_to_cart", "purchase", "view"],
        "price": [10.0, -5.0, 25.0, 999999.0, 15.0]
    })
    csv_file = tmp_path / "raw_events.csv"
    df_raw.to_csv(csv_file, index=False)

    loader = DataLoader(str(csv_file))
    loaded_df = loader.load_events()
    assert "customer_id" in loaded_df.columns
    assert "product_id" in loaded_df.columns

    report_path = tmp_path / "dq_report.json"
    validator = DataValidator(output_report_path=str(report_path))
    clean_df, report = validator.validate_and_clean(loaded_df)

    assert report["row_count"] < len(df_raw)  # Null customer_id and duplicate removed
    assert clean_df["customer_id"].isnull().sum() == 0
    assert (clean_df["unit_price"] > 0).all()
    assert Path(report_path).exists()


def test_temporal_split():
    """Test temporal split order and no leakage."""
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "customer_id": np.random.choice([1, 2, 3], size=100),
        "product_id": np.random.choice([10, 20, 30], size=100),
        "timestamp": dates,
        "quantity": 1,
        "unit_price": 20.0,
        "event_type": "purchase"
    })

    preprocessor = DataPreprocessor(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df, cutoffs = preprocessor.temporal_split(df)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()
