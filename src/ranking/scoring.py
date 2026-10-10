"""Multi-source score normalization and hybrid ranking model."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from src.ranking.candidate_generation import CandidatePool
from src.utils.logger import get_logger

logger = get_logger("scoring")

DEFAULT_WEIGHTS = {
    "als": 0.35,
    "item_item": 0.25,
    "segment": 0.20,
    "popularity": 0.10,
    "trending": 0.10,
}


class HybridScorer:
    """Normalizes and linearly blends heterogeneous multi-model candidate scores."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        normalization_method: str = "min_max",
    ):
        self.weights = weights or DEFAULT_WEIGHTS
        self.normalization_method = normalization_method

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array of scores to [0, 1]."""
        if len(arr) == 0:
            return arr
        mx = np.max(arr)
        mn = np.min(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        elif mx > 0:
            return arr / mx
        return np.zeros_like(arr)

    def rank_candidates(
        self,
        candidate_pool: CandidatePool,
        allowed_items: Optional[List[Union[int, str]]] = None,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Compute normalized weighted hybrid scores for filtered candidates and rank top-N."""
        target_items = allowed_items if allowed_items is not None else candidate_pool.get_all_item_ids()
        if not target_items:
            return []

        sources = list(self.weights.keys())
        raw_matrix: Dict[str, List[float]] = {s: [] for s in sources}

        for item in target_items:
            for s in sources:
                raw_matrix[s].append(candidate_pool.get_source_score(item, s))

        # Normalize score vector per source
        norm_matrix: Dict[str, np.ndarray] = {}
        for s in sources:
            norm_matrix[s] = self._normalize_array(np.array(raw_matrix[s], dtype=np.float32))

        # Blend scores
        final_scores = np.zeros(len(target_items), dtype=np.float32)
        total_weight = sum(self.weights.values())

        for s in sources:
            w = self.weights.get(s, 0.0)
            final_scores += w * norm_matrix[s]

        if total_weight > 0:
            final_scores /= total_weight

        # Sort descending by final score
        order = np.argsort(-final_scores, kind="stable")
        top_indices = order[:top_n]

        # Build output objects with contribution breakdowns for top-N items
        ranked_items: List[Dict[str, Any]] = []
        for r_idx, idx in enumerate(top_indices):
            idx_int = int(idx)
            item = target_items[idx_int]
            item_sources = candidate_pool.get_sources_for_item(item)
            source_scores = {
                s: round(float(norm_matrix[s][idx_int]), 4)
                for s in sources
                if norm_matrix[s][idx_int] > 0
            }
            ranked_items.append({
                "product_id": item,
                "score": round(float(final_scores[idx_int]), 4),
                "rank": r_idx + 1,
                "sources": item_sources,
                "source_scores": source_scores,
            })

        return ranked_items

