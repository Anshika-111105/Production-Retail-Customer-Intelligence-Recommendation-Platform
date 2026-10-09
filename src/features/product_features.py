"""Product-level feature engineering and metadata aggregation."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("product_features")


class ProductFeatureEngineer:
    """Extracts product sales statistics, engagement metrics, and catalog metadata."""

    def __init__(self, recent_window_days: int = 30):
        self.recent_window_days = recent_window_days

    def extract_features(
        self,
        events_df: pd.DataFrame,
        catalog_df: Optional[pd.DataFrame] = None,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Extract product-level interaction metrics and merge with catalog metadata."""
        logger.info("Computing product-level features and catalog aggregates...")
        df = events_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ref_date = reference_date if reference_date is not None else df["timestamp"].max()
        cutoff_date = ref_date - pd.Timedelta(days=self.recent_window_days)

        # Revenue per event: quantity * unit_price * (1 - discount)
        spend = df["quantity"] * df["unit_price"] * (1.0 - df["discount"].fillna(0.0))
        df["calculated_spend"] = spend.clip(lower=0.0)

        # Base aggregates across all events
        all_groups = df.groupby("product_id")
        total_interactions = all_groups.size()
        unique_customers = all_groups["customer_id"].nunique()
        avg_price = all_groups["unit_price"].mean()
        avg_discount = all_groups["discount"].mean()

        # Purchase specific aggregates
        purchase_df = df[df["event_type"] == "purchase"]
        p_groups = purchase_df.groupby("product_id")
        purchase_count = p_groups.size().reindex(total_interactions.index, fill_value=0)
        revenue = p_groups["calculated_spend"].sum().reindex(total_interactions.index, fill_value=0.0)

        # Recent purchases
        recent_purchases = purchase_df[purchase_df["timestamp"] >= cutoff_date]
        recent_p_count = recent_purchases.groupby("product_id").size().reindex(total_interactions.index, fill_value=0)

        # Most frequent category & brand from events
        mode_cat = all_groups["category"].agg(lambda x: x.mode()[0] if not x.empty else "General")
        mode_brand = all_groups["brand"].agg(lambda x: x.mode()[0] if not x.empty else "Generic")
        mode_desc = all_groups["description"].agg(lambda x: x.mode()[0] if not x.empty else "")

        prod_features = pd.DataFrame({
            "product_id": total_interactions.index,
            "total_interactions": total_interactions.values,
            "purchase_count": purchase_count.values,
            "unique_customer_count": unique_customers.values,
            "revenue": revenue.values,
            "average_price": avg_price.values,
            "average_discount": avg_discount.values,
            "recent_purchase_count": recent_p_count.values,
            "category": mode_cat.values,
            "brand": mode_brand.values,
            "description": mode_desc.values,
        }).reset_index(drop=True)

        # If external catalog exists, merge to include products with zero interaction events
        if catalog_df is not None:
            catalog_clean = catalog_df.copy()
            merged = catalog_clean.merge(prod_features, on="product_id", how="left", suffixes=("_cat", ""))
            
            merged["total_interactions"] = merged["total_interactions"].fillna(0).astype(int)
            merged["purchase_count"] = merged["purchase_count"].fillna(0).astype(int)
            merged["unique_customer_count"] = merged["unique_customer_count"].fillna(0).astype(int)
            merged["revenue"] = merged["revenue"].fillna(0.0)
            merged["average_price"] = merged["average_price"].fillna(merged.get("unit_price", 10.0))
            merged["average_discount"] = merged["average_discount"].fillna(0.0)
            merged["recent_purchase_count"] = merged["recent_purchase_count"].fillna(0).astype(int)
            merged["category"] = merged["category"].fillna(merged.get("category_cat", "General"))
            merged["brand"] = merged["brand"].fillna(merged.get("brand_cat", "Generic"))
            merged["description"] = merged["description"].fillna(merged.get("description_cat", ""))

            cols = [
                "product_id", "total_interactions", "purchase_count", "unique_customer_count",
                "revenue", "average_price", "average_discount", "recent_purchase_count",
                "category", "brand", "description"
            ]
            prod_features = merged[cols].drop_duplicates(subset=["product_id"]).reset_index(drop=True)

        logger.info(f"Extracted product features for {len(prod_features)} products.")
        return prod_features

    def save(self, df: pd.DataFrame, filepath: str = "artifacts/features/product_features.parquet") -> None:
        """Persist product features."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved product features to {path}")
