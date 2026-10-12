"""Item-Item Collaborative Filtering recommender with sparse cosine similarity and explainability."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from src.recommenders.base import BaseRecommender
from src.features.interaction_features import InteractionMatrixBuilder
from src.utils.logger import get_logger

logger = get_logger("item_item_recommender")


class ItemItemRecommender(BaseRecommender):
    """Item-Item collaborative filtering using sparse item cosine similarity."""

    def __init__(self, k_nearest: int = 50, min_similarity: float = 0.01):
        super().__init__(name="item_item")
        self.k_nearest = k_nearest
        self.min_similarity = min_similarity
        self.similarity_dict_: Dict[int, List[Tuple[int, float]]] = {}  # item_idx -> list of (similar_item_idx, sim_score)
        self.item_norms_: Optional[np.ndarray] = None
        self.user_history_: Dict[int, Dict[int, float]] = {}  # user_idx -> {item_idx: weight}
        self.id_builder_: Optional[InteractionMatrixBuilder] = None

    def fit(self, user_item_csr: sp.csr_matrix, id_builder: InteractionMatrixBuilder) -> "ItemItemRecommender":
        """Compute top-K item-item cosine similarities from sparse interaction matrix."""
        logger.info(f"Training Item-Item CF with top-{self.k_nearest} neighbors...")
        self.id_builder_ = id_builder
        n_users, n_items = user_item_csr.shape

        # Transpose to Item-User matrix for item-item dot product
        item_user_csc = user_item_csr.tocsc().T

        # Store user history for fast inference lookup
        user_item_csr_coo = user_item_csr.tocoo()
        self.user_history_ = {}
        for u_idx, i_idx, val in zip(user_item_csr_coo.row, user_item_csr_coo.col, user_item_csr_coo.data):
            u = int(u_idx)
            i = int(i_idx)
            if u not in self.user_history_:
                self.user_history_[u] = {}
            self.user_history_[u][i] = float(val)

        # L2 normalize rows of item_user_csc (items)
        norms = sp.linalg.norm(item_user_csc, axis=1)
        norms[norms == 0] = 1.0
        norm_diag = sp.diags(1.0 / norms.ravel())
        normalized_item_user: sp.csr_matrix = (norm_diag @ item_user_csc).tocsr()

        # Compute cosine similarity per batch of items to ensure sparse memory efficiency
        batch_size = 200
        self.similarity_dict_ = {}

        for start_idx in range(0, n_items, batch_size):
            end_idx = min(n_items, start_idx + batch_size)
            batch = normalized_item_user[start_idx:end_idx]
            # Batch similarity: batch @ normalized_item_user.T
            batch_sim = batch.dot(normalized_item_user.T).toarray()

            for offset, item_idx in enumerate(range(start_idx, end_idx)):
                row = batch_sim[offset]
                row[item_idx] = 0.0  # Zero self-similarity
                
                # Filter indices above threshold
                valid_mask = row >= self.min_similarity
                valid_indices = np.where(valid_mask)[0]
                valid_scores = row[valid_indices]

                if len(valid_scores) > 0:
                    top_k_order = np.argsort(valid_scores)[::-1][:self.k_nearest]
                    top_indices = valid_indices[top_k_order]
                    top_scores = valid_scores[top_k_order]
                    self.similarity_dict_[item_idx] = [
                        (int(top_indices[j]), float(top_scores[j])) for j in range(len(top_indices))
                    ]
                else:
                    self.similarity_dict_[item_idx] = []

        self.is_fitted = True
        logger.info(f"Item-Item similarity graph computed across {len(self.similarity_dict_)} items.")
        return self

    def recommend(
        self,
        user_id: Union[int, str],
        n: int = 10,
        filter_items: Optional[Set[Union[int, str]]] = None,
    ) -> List[Tuple[Union[int, str], float]]:
        """Recommend items similar to the customer's interacted history."""
        if not self.is_fitted or not self.id_builder_:
            raise RuntimeError("ItemItemRecommender must be fitted before generating recommendations.")

        u_idx = self.id_builder_.user_to_idx.get(user_id)
        if u_idx is None or u_idx not in self.user_history_:
            # Cold-start user
            return []

        user_history = self.user_history_[u_idx]
        user_interacted_items = set(user_history.keys())
        
        filter_item_indices = set()
        if filter_items:
            for item in filter_items:
                if item in self.id_builder_.item_to_idx:
                    filter_item_indices.add(self.id_builder_.item_to_idx[item])
        
        # Accumulate similarity scores
        candidate_scores: Dict[int, float] = {}
        for history_item_idx, interaction_val in user_history.items():
            similar_items = self.similarity_dict_.get(history_item_idx, [])
            for sim_item_idx, sim_score in similar_items:
                if sim_item_idx not in user_interacted_items and sim_item_idx not in filter_item_indices:
                    candidate_scores[sim_item_idx] = candidate_scores.get(sim_item_idx, 0.0) + (interaction_val * sim_score)

        if not candidate_scores:
            return []

        # Sort descending
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Normalize top scores to [0, 1]
        max_score = sorted_candidates[0][1] if sorted_candidates else 1.0
        results = [
            (self.id_builder_.idx_to_item[idx], float(score / max_score) if max_score > 0 else 0.0)
            for idx, score in sorted_candidates
        ]
        return results

    def get_similar_items(
        self,
        product_id: Union[int, str],
        n: int = 10,
    ) -> List[Tuple[Union[int, str], float]]:
        """Fetch top-N similar products for a given target product."""
        if not self.id_builder_ or product_id not in self.id_builder_.item_to_idx:
            return []
        p_idx = self.id_builder_.item_to_idx[product_id]
        sim_list = self.similarity_dict_.get(p_idx, [])[:n]
        return [(self.id_builder_.idx_to_item[idx], sim) for idx, sim in sim_list]

    def explain(self, user_id: Union[int, str], item_id: Union[int, str]) -> Dict[str, Any]:
        """Provide detailed item-item trace explanation."""
        if not self.id_builder_ or user_id not in self.id_builder_.user_to_idx or item_id not in self.id_builder_.item_to_idx:
            return {
                "recommended_product": item_id,
                "method": "item_item",
                "similarity": 0.0,
                "source_product": None,
                "explanation": "Recommended via collaborative filtering similarity.",
            }

        u_idx = self.id_builder_.user_to_idx[user_id]
        rec_item_idx = self.id_builder_.item_to_idx[item_id]
        user_history = self.user_history_.get(u_idx, {})

        best_source_item = None
        best_sim = 0.0

        for hist_idx, hist_val in user_history.items():
            sim_list = dict(self.similarity_dict_.get(hist_idx, []))
            if rec_item_idx in sim_list:
                sim = sim_list[rec_item_idx]
                if sim > best_sim:
                    best_sim = sim
                    best_source_item = self.id_builder_.idx_to_item[hist_idx]

        return {
            "recommended_product": item_id,
            "source_product": best_source_item,
            "similarity": round(float(best_sim), 4),
            "method": "item_item",
            "explanation": f"Recommended because it is highly similar ({best_sim:.2f}) to product {best_source_item} previously purchased/viewed.",
        }

    def save(self, filepath: str = "artifacts/models/item_item_model.joblib") -> None:
        """Persist Item-Item model and similarities."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k_nearest": self.k_nearest,
            "min_similarity": self.min_similarity,
            "similarity_dict_": self.similarity_dict_,
            "user_history_": self.user_history_,
        }
        joblib.dump(payload, path)
        logger.info(f"Saved Item-Item model to {path}")

    @classmethod
    def load(cls, filepath: str = "artifacts/models/item_item_model.joblib", id_builder: Optional[InteractionMatrixBuilder] = None) -> "ItemItemRecommender":
        """Load persisted Item-Item model."""
        payload = joblib.load(filepath)
        model = cls(
            k_nearest=payload["k_nearest"],
            min_similarity=payload["min_similarity"],
        )
        model.similarity_dict_ = payload["similarity_dict_"]
        model.user_history_ = payload["user_history_"]
        model.id_builder_ = id_builder
        model.is_fitted = True
        return model
