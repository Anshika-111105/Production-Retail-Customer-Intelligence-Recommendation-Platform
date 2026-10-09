"""Base Recommender abstract class interface."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class BaseRecommender(ABC):
    """Abstract base class for all recommendation models."""

    def __init__(self, name: str = "base_recommender"):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseRecommender":
        """Train recommendation model on input data."""
        pass

    @abstractmethod
    def recommend(self, user_id: Union[int, str], n: int = 10, filter_items: Optional[Set[Union[int, str]]] = None,) -> List[Tuple[Union[int, str], float]]:
        """Generate top-N recommendations for a single customer."""
        pass

    def recommend_batch(
        self,
        user_ids: List[Union[int, str]],
        n: int = 10,
        filter_items_dict: Optional[Dict[Union[int, str], Set[Union[int, str]]]] = None,
    ) -> Dict[Union[int, str], List[Tuple[Union[int, str], float]]]:
        """Generate recommendations across a batch of customers."""
        results = {}
        for uid in user_ids:
            filter_items = filter_items_dict.get(uid) if filter_items_dict else None
            results[uid] = self.recommend(uid, n=n, filter_items=filter_items)
        return results

    def explain(
        self,
        user_id: Union[int, str],
        item_id: Union[int, str],
    ) -> Dict[str, Any]:
        """Provide explainability metadata for a specific recommended item."""
        return {
            "recommended_product": item_id,
            "method": self.name,
            "explanation": f"Recommended by {self.name} algorithm.",
        }

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist trained model artifacts."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseRecommender":
        """Load trained model artifacts."""
        pass
