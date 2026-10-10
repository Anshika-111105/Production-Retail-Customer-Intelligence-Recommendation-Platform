"""Interaction feature engineering, event weighting, and sparse matrix construction."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib

from src.utils.logger import get_logger

logger = get_logger("interaction_features")

DEFAULT_EVENT_WEIGHTS = {
    "purchase": 5.0,
    "add_to_cart": 3.0,
    "wishlist": 2.0,
    "view": 1.0,
}


class InteractionMatrixBuilder:
    """Constructs CSR implicit interaction matrices with event weighting and ID mappings."""

    def __init__(self, event_weights: Optional[Dict[str, float]] = None):
        self.event_weights = event_weights or DEFAULT_EVENT_WEIGHTS
        self.user_to_idx: Dict[Union[int, str], int] = {}
        self.idx_to_user: Dict[int, Union[int, str]] = {}
        self.item_to_idx: Dict[Union[int, str], int] = {}
        self.idx_to_item: Dict[int, Union[int, str]] = {}

    def fit_mappings(self, df: pd.DataFrame) -> "InteractionMatrixBuilder":
        """Fit bidirectional mappings between raw IDs and contiguous 0-indexed integers."""
        unique_users = sorted(df["customer_id"].unique())
        unique_items = sorted(df["product_id"].unique())

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for idx, uid in enumerate(unique_users)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item = {idx: iid for idx, iid in enumerate(unique_items)}

        logger.info(f"Built ID mappings: {len(self.user_to_idx)} unique users, {len(self.item_to_idx)} unique items.")
        return self

    def build_matrix(self, df: pd.DataFrame) -> Tuple[sp.csr_matrix, pd.DataFrame]:
        """Compute damped implicit interaction strengths and assemble CSR user-item matrix."""
        logger.info("Computing weighted implicit interactions and building CSR matrix...")
        df = df.copy()
        
        # Apply event weights
        df["base_weight"] = df["event_type"].map(self.event_weights).fillna(1.0)
        
        # Group by (customer_id, product_id)
        # Interaction strength: sum(base_weight) or base_weight * log1p(count)
        agg_df = df.groupby(["customer_id", "product_id"]).agg(
            total_events=("event_type", "count"),
            weighted_sum=("base_weight", "sum")
        ).reset_index()

        # Damped strength formula: (weighted_sum / total_events) * log1p(total_events)
        agg_df["interaction_strength"] = (agg_df["weighted_sum"] / agg_df["total_events"]) * np.log1p(agg_df["total_events"])

        # Filter to known mapped entities
        agg_df = agg_df[agg_df["customer_id"].isin(self.user_to_idx) & agg_df["product_id"].isin(self.item_to_idx)].copy()
        
        row_indices = [self.user_to_idx[uid] for uid in agg_df["customer_id"]]
        col_indices = [self.item_to_idx[iid] for iid in agg_df["product_id"]]
        data_values = agg_df["interaction_strength"].values.astype(np.float32)

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        user_item_csr = sp.csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(n_users, n_items),
            dtype=np.float32
        )

        logger.info(f"Interaction matrix assembled: {user_item_csr.shape} with {user_item_csr.nnz:,} non-zero entries.")
        return user_item_csr, agg_df

    def save(self, filepath: str = "artifacts/mappings/id_mappings.joblib") -> None:
        """Persist ID mappings."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "event_weights": self.event_weights,
        }
        joblib.dump(payload, path)
        logger.info(f"Saved ID mappings to {path}")

    @classmethod
    def load(cls, filepath: str = "artifacts/mappings/id_mappings.joblib") -> "InteractionMatrixBuilder":
        """Load persisted mappings."""
        payload = joblib.load(filepath)
        builder = cls(event_weights=payload.get("event_weights"))
        builder.user_to_idx = payload["user_to_idx"]
        builder.idx_to_user = payload["idx_to_user"]
        builder.item_to_idx = payload["item_to_idx"]
        builder.idx_to_item = payload["idx_to_item"]
        return builder
