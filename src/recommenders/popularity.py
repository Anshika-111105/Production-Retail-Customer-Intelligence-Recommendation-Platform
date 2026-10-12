"""Popularity Recommender baseline with configurable recency and customer breadth weighting."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import joblib

from src.recommenders.base import BaseRecommender
from src.utils.logger import get_logger

logger = get_logger("popularity_recommender")


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommendation engine combining volume, unique reach, and recent momentum."""

    def __init__(
        self,
        alpha_unique_customers: float = 0.5,
        beta_recent_purchases: float = 1.0,
        recent_window_days: int = 30,
    ):
        super().__init__(name="popularity")
        self.alpha = alpha_unique_customers
        self.beta = beta_recent_purchases
        self.recent_window_days = recent_window_days
        self.product_scores_: Dict[Union[int, str], float] = {}
        self.ranked_products_: List[Union[int, str]] = []
        self.category_ranked_products_: Dict[str, List[Union[int, str]]] = {}
        self.product_metadata_: Dict[Union[int, str], Dict[str, Any]] = {}

    def fit(self, events_df: pd.DataFrame, catalog_df: Optional[pd.DataFrame] = None) -> "PopularityRecommender":
        """Fit popularity scores from event transactions."""
        logger.info("Training Popularity Recommender...")
        df = events_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        max_ts = df["timestamp"].max()
        cutoff_recent = max_ts - pd.Timedelta(days=self.recent_window_days)

        # 1. Total purchases / interactions
        purchase_df = df[df["event_type"] == "purchase"] if "purchase" in df["event_type"].values else df
        purchase_counts = purchase_df.groupby("product_id").size()

        # 2. Unique customers
        unique_cust = df.groupby("product_id")["customer_id"].nunique()

        # 3. Recent purchases
        recent_df = purchase_df[purchase_df["timestamp"] >= cutoff_recent]
        recent_counts = recent_df.groupby("product_id").size().reindex(purchase_counts.index, fill_value=0)

        # Unique product index union
        all_prods = list(set(purchase_counts.index).union(set(unique_cust.index)))
        p_counts = purchase_counts.reindex(all_prods, fill_value=0).values.astype(float)
        u_counts = unique_cust.reindex(all_prods, fill_value=0).values.astype(float)
        r_counts = recent_counts.reindex(all_prods, fill_value=0).values.astype(float)

        # Normalize individual components into [0, 1] range to avoid magnitude distortion
        def min_max_norm(arr):
            mx, mn = np.max(arr), np.min(arr)
            return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

        norm_p = min_max_norm(p_counts)
        norm_u = min_max_norm(u_counts)
        norm_r = min_max_norm(r_counts)

        # Combined popularity formula
        raw_scores = norm_p + self.alpha * norm_u + self.beta * norm_r
        final_scores = min_max_norm(raw_scores)

        self.product_scores_ = {all_prods[i]: float(final_scores[i]) for i in range(len(all_prods))}
        
        # Sort ranked products descending by score
        sorted_items = sorted(self.product_scores_.items(), key=lambda x: x[1], reverse=True)
        self.ranked_products_ = [item[0] for item in sorted_items]

        # Category-stratified popularity
        if "category" in df.columns:
            cat_map = df.groupby("product_id")["category"].agg(lambda x: x.mode()[0] if not x.empty else "General").to_dict()
            for prod, score in sorted_items:
                cat = cat_map.get(prod, "General")
                if cat not in self.category_ranked_products_:
                    self.category_ranked_products_[cat] = []
                self.category_ranked_products_[cat].append(prod)

        self.is_fitted = True
        logger.info(f"Popularity Recommender trained across {len(self.ranked_products_)} products.")
        return self

    def recommend(
        self,
        user_id: Optional[Union[int, str]] = None,
        n: int = 10,
        filter_items: Optional[Set[Union[int, str]]] = None,
        category_context: Optional[str] = None,
    ) -> List[Tuple[Union[int, str], float]]:
        """Generate top-N popular products, optionally filtered by user history or category."""
        filter_set = filter_items or set()
        
        if category_context and category_context in self.category_ranked_products_:
            candidates = self.category_ranked_products_[category_context]
        else:
            candidates = self.ranked_products_

        results = []
        for prod in candidates:
            if prod not in filter_set:
                score = self.product_scores_.get(prod, 0.0)
                results.append((prod, score))
            if len(results) >= n:
                break

        return results

    def explain(self, user_id: Union[int, str], item_id: Union[int, str]) -> Dict[str, Any]:
        """Explain popularity recommendation."""
        score = self.product_scores_.get(item_id, 0.0)
        return {
            "recommended_product": item_id,
            "method": "popularity",
            "score": round(score, 4),
            "explanation": "Top-selling and trending product across the store.",
        }

    def save(self, filepath: str = "artifacts/models/popularity_model.joblib") -> None:
        """Persist popularity model."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "beta": self.beta,
            "recent_window_days": self.recent_window_days,
            "product_scores_": self.product_scores_,
            "ranked_products_": self.ranked_products_,
            "category_ranked_products_": self.category_ranked_products_,
        }
        joblib.dump(payload, path)
        logger.info(f"Saved popularity model to {path}")

    @classmethod
    def load(cls, filepath: str = "artifacts/models/popularity_model.joblib") -> "PopularityRecommender":
        """Load persisted popularity model."""
        payload = joblib.load(filepath)
        model = cls(
            alpha_unique_customers=payload["alpha"],
            beta_recent_purchases=payload["beta"],
            recent_window_days=payload["recent_window_days"],
        )
        model.product_scores_ = payload["product_scores_"]
        model.ranked_products_ = payload["ranked_products_"]
        model.category_ranked_products_ = payload["category_ranked_products_"]
        model.is_fitted = True
        return model
