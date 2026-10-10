import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib

from src.utils.logger import get_logger

logger = get_logger("customer_features")


class CustomerFeatureEngineer:
    """Calculates customer RFM metrics, engagement statistics, and transformations."""

    def __init__(
        self,
        reference_date: Optional[pd.Timestamp] = None,
        scaler_type: str = "robust",
    ):
        self.reference_date = reference_date
        self.scaler_type = scaler_type
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.feature_columns = [
            "recency",
            "frequency",
            "monetary_value",
            "average_order_value",
            "average_basket_size",
            "unique_products",
            "unique_categories",
            "category_diversity",
            "discount_usage_rate",
            "active_days",
            "repeat_purchase_rate",
        ]
        self.transformed_feature_columns = [f"{c}_norm" for c in self.feature_columns]

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute raw customer features from interaction events with fast vectorized operations."""
        logger.info("Computing customer features and RFM metrics...")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ref_date = self.reference_date if self.reference_date is not None else df["timestamp"].max()

        # Calculate spend per row (quantity * unit_price * (1 - discount))
        spend = df["quantity"] * df["unit_price"] * (1.0 - df["discount"].fillna(0.0))
        df["calculated_spend"] = spend.clip(lower=0.0)
        df["event_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

        customer_groups = df.groupby("customer_id")

        # 1. Frequency
        frequency = customer_groups.size()

        # 2. Recency
        last_interaction = customer_groups["timestamp"].max()
        recency_days = (ref_date - last_interaction).values / np.timedelta64(1, "D")
        recency = pd.Series(recency_days, index=frequency.index).clip(lower=0.0)


        # 3. Monetary value
        monetary = customer_groups["calculated_spend"].sum()

        # 4. Purchases count & Average Order Value
        purchase_mask = df["event_type"] == "purchase"
        purchase_df = df[purchase_mask]
        purchase_groups = purchase_df.groupby("customer_id")
        purchase_counts = purchase_groups.size().reindex(frequency.index, fill_value=0)
        purchase_spend = purchase_groups["calculated_spend"].sum().reindex(frequency.index, fill_value=0.0)
        
        p_counts = purchase_counts.values
        p_spend = purchase_spend.values
        f_counts = frequency.values
        m_spend = monetary.values

        aov = np.where(
            p_counts > 0,
            p_spend / np.maximum(1, p_counts),
            m_spend / np.maximum(1, f_counts)
        )


        # 5. Average basket size
        basket_size = customer_groups["quantity"].mean().clip(lower=1.0)

        # 6. Unique products & categories
        unique_prods = customer_groups["product_id"].nunique()
        unique_cats = customer_groups["category"].nunique() if "category" in df.columns else pd.Series(1, index=frequency.index)

        # 7. Category diversity
        cat_diversity = unique_cats / np.log1p(frequency).clip(lower=1.0)

        # 8. Discount usage rate
        discount_rate = (df["discount"] > 0).groupby(df["customer_id"]).mean()

        # 9. Active days count
        active_days = df.groupby("customer_id")["event_date"].nunique()

        # 10. Vectorized Repeat purchase rate
        # Pair counts: (customer_id, product_id)
        cust_prod_counts = df.groupby(["customer_id", "product_id"]).size().reset_index(name="cnt")
        repeat_counts = cust_prod_counts[cust_prod_counts["cnt"] > 1].groupby("customer_id").size().reindex(frequency.index, fill_value=0)
        total_unique_items = unique_prods.replace(0, 1)
        repeat_rate = (repeat_counts / total_unique_items).clip(lower=0.0, upper=1.0)

        features_df = pd.DataFrame({
            "customer_id": frequency.index,
            "recency": recency.values,
            "frequency": frequency.values,
            "monetary_value": monetary.values,
            "average_order_value": aov,
            "average_basket_size": basket_size.values,
            "unique_products": unique_prods.values,
            "unique_categories": unique_cats.values,
            "category_diversity": cat_diversity.values,
            "discount_usage_rate": discount_rate.values,
            "active_days": active_days.values,
            "repeat_purchase_rate": repeat_rate.values,
        }).reset_index(drop=True)

        logger.info(f"Extracted customer features for {len(features_df)} customers.")
        return features_df

    def transform_and_scale(self, features_df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply log1p transformations for skewness reduction and scale features."""
        logger.info(f"Applying log1p transformations and {self.scaler_type} scaling...")
        transformed_df = features_df.copy()

        log_cols = ["recency", "frequency", "monetary_value", "average_order_value", "unique_products", "active_days"]
        for col in log_cols:
            if col in transformed_df.columns:
                clipped_vals = np.clip(transformed_df[col].values, a_min=0, a_max=None)
                transformed_df[f"{col}_log"] = pd.Series(np.log1p(clipped_vals), index=transformed_df.index)

        cols_to_scale = [
            "recency_log",
            "frequency_log",
            "monetary_value_log",
            "average_order_value_log",
            "average_basket_size",
            "unique_products_log",
            "unique_categories",
            "category_diversity",
            "discount_usage_rate",
            "active_days_log",
            "repeat_purchase_rate",
        ]

        raw_matrix = transformed_df[cols_to_scale].values

        if fit:
            scaled_matrix = self.scaler.fit_transform(raw_matrix)
        else:
            scaled_matrix = self.scaler.transform(raw_matrix)

        for idx, col in enumerate(self.transformed_feature_columns):
            transformed_df[col] = pd.Series(scaled_matrix[:, idx], index=transformed_df.index)

        return transformed_df, np.asarray(scaled_matrix)


    def save(self, filepath: str = "artifacts/models/customer_scaler.joblib") -> None:
        """Persist fitted feature scaler."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"Saved customer scaler to {path}")

    @classmethod
    def load_scaler(cls, filepath: str = "artifacts/models/customer_scaler.joblib", scaler_type: str = "robust") -> "CustomerFeatureEngineer":
        """Load fitted scaler."""
        engineer = cls(scaler_type=scaler_type)
        engineer.scaler = joblib.load(filepath)
        return engineer
