"""Implicit Alternating Least Squares (ALS) Matrix Factorization recommender."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse as sp
import joblib

try:
    from implicit.als import AlternatingLeastSquares
except ImportError:
    AlternatingLeastSquares = None

from src.recommenders.base import BaseRecommender
from src.features.interaction_features import InteractionMatrixBuilder
from src.utils.logger import get_logger

logger = get_logger("als_recommender")


class ALSRecommender(BaseRecommender):
    """Implicit feedback matrix factorization using Alternating Least Squares."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 20,
        alpha_confidence: float = 40.0,
        random_state: int = 42,
    ):
        super().__init__(name="als")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha_confidence = alpha_confidence
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.user_item_csr_: Optional[sp.csr_matrix] = None
        self.id_builder_: Optional[InteractionMatrixBuilder] = None

    def fit(self, user_item_csr: sp.csr_matrix, id_builder: InteractionMatrixBuilder) -> "ALSRecommender":
        """Train ALS model on confidence-scaled user-item sparse interaction matrix."""
        logger.info(
            f"Training Implicit ALS (factors={self.factors}, reg={self.regularization}, "
            f"iter={self.iterations}, alpha={self.alpha_confidence})..."
        )
        self.id_builder_ = id_builder
        self.user_item_csr_ = user_item_csr

        if AlternatingLeastSquares is None:
            raise ImportError("implicit package is not installed. Please install with 'pip install implicit'.")

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )

        # Scale interaction matrix with confidence: C_ui = 1 + alpha * r_ui
        confidence_matrix = (user_item_csr * self.alpha_confidence).astype(np.float32)

        # Fit model
        self.model.fit(confidence_matrix, show_progress=False)
        self.is_fitted = True
        logger.info("ALS model training complete.")
        return self

    def recommend(
        self,
        user_id: Union[int, str],
        n: int = 10,
        filter_items: Optional[Set[Union[int, str]]] = None,
    ) -> List[Tuple[Union[int, str], float]]:
        """Generate top-N ALS personalized recommendations."""
        if not self.is_fitted or self.model is None or not self.id_builder_:
            raise RuntimeError("ALSRecommender is not fitted.")

        u_idx = self.id_builder_.user_to_idx.get(user_id)
        if u_idx is None or u_idx >= self.user_item_csr_.shape[0]:
            return []

        filter_set = filter_items or set()

        # Call implicit recommend asking for n + len(filter_set)
        fetch_n = min(self.user_item_csr_.shape[1], n + len(filter_set) + 20)
        item_indices, scores = self.model.recommend(
            userid=u_idx,
            user_items=self.user_item_csr_[u_idx],
            N=fetch_n,
            filter_already_liked_items=False,
        )

        # Normalize scores to [0, 1]
        max_score = float(np.max(scores)) if len(scores) > 0 else 1.0
        min_score = float(np.min(scores)) if len(scores) > 0 else 0.0
        
        results = []
        for idx, score in zip(item_indices, scores):
            item_raw = self.id_builder_.idx_to_item[int(idx)]
            if item_raw in filter_set:
                continue

            if max_score > min_score:
                norm_score = float((score - min_score) / (max_score - min_score))
            else:
                norm_score = float(score / max(1e-4, max_score))
            results.append((item_raw, norm_score))

            if len(results) >= n:
                break

        return results

    def explain(self, user_id: Union[int, str], item_id: Union[int, str]) -> Dict[str, Any]:
        """Explain ALS recommendation based on latent similarity and user factors."""
        return {
            "recommended_product": item_id,
            "method": "als",
            "explanation": "Recommended based on collaborative latent preferences with similar shoppers.",
        }

    def save(self, filepath: str = "artifacts/models/als_model.joblib") -> None:
        """Persist ALS model parameters and factors."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha_confidence": self.alpha_confidence,
            "random_state": self.random_state,
            "user_factors": self.model.user_factors if self.model else None,
            "item_factors": self.model.item_factors if self.model else None,
            "user_item_csr_": self.user_item_csr_,
        }
        joblib.dump(payload, path)
        logger.info(f"Saved ALS model to {path}")

    @classmethod
    def load(cls, filepath: str = "artifacts/models/als_model.joblib", id_builder: Optional[InteractionMatrixBuilder] = None) -> "ALSRecommender":
        """Load persisted ALS model."""
        payload = joblib.load(filepath)
        model = cls(
            factors=payload["factors"],
            regularization=payload["regularization"],
            iterations=payload["iterations"],
            alpha_confidence=payload["alpha_confidence"],
            random_state=payload["random_state"],
        )
        if AlternatingLeastSquares is not None:
            model.model = AlternatingLeastSquares(
                factors=model.factors,
                regularization=model.regularization,
                iterations=model.iterations,
                random_state=model.random_state,
            )
            model.model.user_factors = payload["user_factors"]
            model.model.item_factors = payload["item_factors"]
        model.user_item_csr_ = payload["user_item_csr_"]
        model.id_builder_ = id_builder
        model.is_fitted = True
        return model
