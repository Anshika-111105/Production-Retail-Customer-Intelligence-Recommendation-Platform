"""TruncatedSVD Matrix Factorization Baseline Recommender."""

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
from sklearn.decomposition import TruncatedSVD
import joblib

from src.recommenders.base import BaseRecommender
from src.features.interaction_features import InteractionMatrixBuilder
from src.utils.logger import get_logger

logger = get_logger("svd_recommender")


class SVDRecommender(BaseRecommender):
    """TruncatedSVD baseline matrix factorization recommender.
    
    Architectural Note:
    TruncatedSVD assumes unobserved interactions represent true zero ratings, whereas
    implicit ALS models missing entries as unobserved interactions with lower confidence weights.
    SVD serves here as a classical linear dimensionality-reduction baseline.
    """

    def __init__(self, n_components: int = 32, random_state: int = 42):
        super().__init__(name="svd")
        self.n_components = n_components
        self.random_state = random_state
        self.svd: Optional[TruncatedSVD] = None
        self.user_embeddings_: Optional[np.ndarray] = None
        self.item_components_: Optional[np.ndarray] = None
        self.user_interacted_: Dict[int, Set[int]] = {}
        self.id_builder_: Optional[InteractionMatrixBuilder] = None

    def fit(self, user_item_csr: sp.csr_matrix, id_builder: InteractionMatrixBuilder) -> "SVDRecommender":
        """Fit TruncatedSVD on interaction matrix and compute latent embeddings."""
        logger.info(f"Training TruncatedSVD baseline with n_components={self.n_components}...")
        self.id_builder_ = id_builder
        n_users, n_items = user_item_csr.shape
        actual_components = min(self.n_components, n_items - 1, n_users - 1)

        self.svd = TruncatedSVD(n_components=actual_components, random_state=self.random_state)
        self.user_embeddings_ = self.svd.fit_transform(user_item_csr)
        self.item_components_ = self.svd.components_  # shape: (n_components, n_items)

        # Record user interactions for filtering
        coo = user_item_csr.tocoo()
        self.user_interacted_ = {}
        for u_idx, i_idx in zip(coo.row, coo.col):
            if u_idx not in self.user_interacted_:
                self.user_interacted_[u_idx] = set()
            self.user_interacted_[u_idx].add(i_idx)

        self.is_fitted = True
        logger.info(
            f"TruncatedSVD baseline trained (explained variance ratio sum: {self.svd.explained_variance_ratio_.sum():.4f})."
        )
        return self

    def recommend(
        self,
        user_id: Union[int, str],
        n: int = 10,
        filter_items: Optional[Set[Union[int, str]]] = None,
    ) -> List[Tuple[Union[int, str], float]]:
        """Generate top-N SVD reconstructed affinity recommendations."""
        if not self.is_fitted or self.user_embeddings_ is None or not self.id_builder_:
            raise RuntimeError("SVDRecommender is not fitted.")

        u_idx = self.id_builder_.user_to_idx.get(user_id)
        if u_idx is None or u_idx >= self.user_embeddings_.shape[0]:
            return []

        u_vec = self.user_embeddings_[u_idx]  # shape: (n_components,)
        # Predicted scores: u_vec @ item_components_ (1, n_components) x (n_components, n_items) -> (n_items,)
        pred_scores = np.dot(u_vec, self.item_components_)

        already_interacted = self.user_interacted_.get(u_idx, set())
        filter_indices = set(already_interacted)
        if filter_items:
            for it in filter_items:
                if it in self.id_builder_.item_to_idx:
                    filter_indices.add(self.id_builder_.item_to_idx[it])

        # Mask out filtered items
        pred_scores[list(filter_indices)] = -np.inf

        top_indices = np.argsort(pred_scores)[::-1][:n]
        valid_indices = [idx for idx in top_indices if pred_scores[idx] > -np.inf]

        if not valid_indices:
            return []

        raw_scores = pred_scores[valid_indices]
        mx = np.max(raw_scores)
        mn = np.min(raw_scores)
        
        results = []
        for idx, score in zip(valid_indices, raw_scores):
            item_raw = self.id_builder_.idx_to_item[int(idx)]
            norm_score = float((score - mn) / (mx - mn)) if mx > mn else 1.0
            results.append((item_raw, norm_score))

        return results

    def explain(self, user_id: Union[int, str], item_id: Union[int, str]) -> Dict[str, Any]:
        """Explain SVD recommendation."""
        return {
            "recommended_product": item_id,
            "method": "svd_baseline",
            "explanation": "Recommended by classical TruncatedSVD linear matrix factorization baseline.",
        }

    def save(self, filepath: str = "artifacts/models/svd_model.joblib") -> None:
        """Persist SVD model."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_components": self.n_components,
            "random_state": self.random_state,
            "svd": self.svd,
            "user_embeddings_": self.user_embeddings_,
            "item_components_": self.item_components_,
            "user_interacted_": self.user_interacted_,
        }
        joblib.dump(payload, path)
        logger.info(f"Saved SVD model to {path}")

    @classmethod
    def load(cls, filepath: str = "artifacts/models/svd_model.joblib", id_builder: Optional[InteractionMatrixBuilder] = None) -> "SVDRecommender":
        """Load persisted SVD model."""
        payload = joblib.load(filepath)
        model = cls(
            n_components=payload["n_components"],
            random_state=payload["random_state"],
        )
        model.svd = payload["svd"]
        model.user_embeddings_ = payload["user_embeddings_"]
        model.item_components_ = payload["item_components_"]
        model.user_interacted_ = payload["user_interacted_"]
        model.id_builder_ = id_builder
        model.is_fitted = True
        return model
