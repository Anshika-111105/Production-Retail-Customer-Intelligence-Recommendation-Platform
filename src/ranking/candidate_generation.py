"""Multi-source candidate generation and candidate pool deduplication."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from src.utils.logger import get_logger

logger = get_logger("candidate_generation")


class CandidatePool:
    """Pool of recommendation candidates aggregated across multiple algorithmic sources."""

    def __init__(self):
        # item_id -> dict with {source: raw_score}
        self.candidates: Dict[Union[int, str], Dict[str, float]] = {}

    def add_candidates(self, source_name: str, candidate_list: List[Tuple[Union[int, str], float]]) -> None:
        """Add candidate list from a specific recommender model."""
        for item_id, score in candidate_list:
            if item_id not in self.candidates:
                self.candidates[item_id] = {}
            self.candidates[item_id][source_name] = float(score)

    def get_all_item_ids(self) -> List[Union[int, str]]:
        """Return list of distinct candidate item IDs."""
        return list(self.candidates.keys())

    def get_sources_for_item(self, item_id: Union[int, str]) -> List[str]:
        """Return list of model sources that proposed this item."""
        return list(self.candidates.get(item_id, {}).keys())

    def get_source_score(self, item_id: Union[int, str], source_name: str) -> float:
        """Get raw score of an item for a given source."""
        return self.candidates.get(item_id, {}).get(source_name, 0.0)


class MultiSourceCandidateGenerator:
    """Generates candidate pools from multiple algorithmic sources."""

    def __init__(
        self,
        als_top_k: int = 50,
        item_item_top_k: int = 50,
        segment_top_k: int = 30,
        popularity_top_k: int = 30,
        trending_top_k: int = 20,
    ):
        self.k_als = als_top_k
        self.k_item = item_item_top_k
        self.k_segment = segment_top_k
        self.k_pop = popularity_top_k
        self.k_trend = trending_top_k

    def generate(
        self,
        user_id: Union[int, str],
        user_segment_id: Optional[int] = None,
        als_model: Optional[Any] = None,
        item_item_model: Optional[Any] = None,
        segment_miner: Optional[Any] = None,
        popularity_model: Optional[Any] = None,
        trending_model: Optional[Any] = None,
        filter_items: Optional[Set[Union[int, str]]] = None,
    ) -> CandidatePool:
        """Generate deduplicated multi-source candidate pool."""
        pool = CandidatePool()

        # 1. ALS Candidates
        if als_model and hasattr(als_model, "recommend"):
            als_recs = als_model.recommend(user_id, n=self.k_als, filter_items=filter_items)
            pool.add_candidates("als", als_recs)

        # 2. Item-Item CF Candidates
        if item_item_model and hasattr(item_item_model, "recommend"):
            item_recs = item_item_model.recommend(user_id, n=self.k_item, filter_items=filter_items)
            pool.add_candidates("item_item", item_recs)

        # 3. Segment Preference Candidates
        if segment_miner and user_segment_id is not None:
            seg_recs = segment_miner.get_top_candidates(user_segment_id, top_n=self.k_segment)
            if filter_items:
                seg_recs = [r for r in seg_recs if r[0] not in filter_items]
            pool.add_candidates("segment", seg_recs)

        # 4. Popularity Candidates
        if popularity_model and hasattr(popularity_model, "recommend"):
            pop_recs = popularity_model.recommend(user_id, n=self.k_pop, filter_items=filter_items)
            pool.add_candidates("popularity", pop_recs)

        # 5. Trending Candidates
        if trending_model and hasattr(trending_model, "recommend"):
            trend_recs = trending_model.recommend(user_id, n=self.k_trend, filter_items=filter_items)
            pool.add_candidates("trending", trend_recs)

        return pool


# Alias for convenience
CandidateGenerator = MultiSourceCandidateGenerator

