"""Time-decay Trending Recommender model."""

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

logger = get_logger("trending_recommender")


class TrendingRecommender(BaseRecommender):
    """Trending recommender using exponential recency decay over interaction timestamps."""

    def __init__(self, decay_lambda: float = 0.05):
        super().__init__(name="trending")
        self.decay_lambda = decay_lambda
        self.trend_scores_: Dict[Union[int, str], float] = {}
        self.ranked_products_: List[Union[int, str]] = []

    def fit(self, events_df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> "TrendingRecommender":
        """Compute exponential time-decay trending scores for each product."""
        logger.info(f"Training Trending Recommender with decay_lambda={self.decay_lambda}...")
        df = events_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ref_date = reference_date if reference_date is not None else df["timestamp"].max()

        # Age in days
        age_days = (ref_date - df["timestamp"]).dt.total_seconds() / 86400.0
        age_days = np.clip(age_days, a_min=0.0, a_max=None)

        # Decay weights: exp(-lambda * age_days)
        decay_weights = np.exp(-self.decay_lambda * age_days)
        
        # Multiply by event type multiplier (purchases matter more than views)
        event_multipliers = {"purchase": 5.0, "add_to_cart": 3.0, "wishlist": 2.0, "view": 1.0}
        mults = df["event_type"].map(event_multipliers).fillna(1.0).values
        decay_weights = decay_weights * mults

        df["decay_weight"] = decay_weights
        raw_trend = df.groupby("product_id")["decay_weight"].sum()

        # Normalize scores to [0, 1]
        mx = raw_trend.max()
        mn = raw_trend.min()
        if mx > mn:
            norm_trend = (raw_trend - mn) / (mx - mn)
        else:
            norm_trend = raw_trend / max(1e-4, mx)

        self.trend_scores_ = {prod: float(score) for prod, score in norm_trend.items()}
        sorted_trend = sorted(self.trend_scores_.items(), key=lambda x: x[1], reverse=True)
        self.ranked_products_ = [item[0] for item in sorted_trend]

        self.is_fitted = True
        logger.info(f"Trending Recommender trained across {len(self.ranked_products_)} products.")
        return self

    def recommend(
        self,
        user_id: Optional[Union[int, str]] = None,
        n: int = 10,
        filter_items: Optional[Set[Union[int, str]]] = None,
    ) -> List[Tuple[Union[int, str], float]]:
        """Recommend top trending items."""
        filter_set = filter_items or set()
        results = []
        for prod in self.ranked_products_:
            if prod not in filter_set:
                score = self.trend_scores_.get(prod, 0.0)
                results.append((prod, score))
            if len(results) >= n:
                break
        return results

    def explain(self, user_id: Union[int, str], item_id: Union[int, str]) -> Dict[str, Any]:
        """Explain trending recommendation."""
        score = self.trend_scores_.get(item_id, 0.0)
        return {
            "recommended_product": item_id,
            "method": "trending",
            "score": round(score, 4),
            "explanation": "Trending with rapid surge in recent interactions and purchases.",
        }

    def save(self, filepath: str = "artifacts/models/trending_model.joblib") -> None:
        """Persist trending model."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "decay_lambda": self.decay_lambda,
            "trend_scores_": self.trend_scores_,
            "ranked_products_": self.ranked_products_,
        }
        joblib.dump(payload, path)
        logger.info(f"Saved trending model to {path}")

    @classmethod
    def load(cls, filepath: str = "artifacts/models/trending_model.joblib") -> "TrendingRecommender":
        """Load persisted trending model."""
        payload = joblib.load(filepath)
        model = cls(decay_lambda=payload["decay_lambda"])
        model.trend_scores_ = payload["trend_scores_"]
        model.ranked_products_ = payload["ranked_products_"]
        model.is_fitted = True
        return model
