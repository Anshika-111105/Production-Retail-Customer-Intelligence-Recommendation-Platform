import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger("segment_features")


class SegmentPreferenceMiner:
    """Computes segment-level product affinity scores and preference distributions."""

    def __init__(self):
        self.segment_product_scores: Dict[int, Dict[Union[int, str], float]] = {}
        self.segment_top_products: Dict[int, List[Union[int, str]]] = {}
        self.segment_category_affinities: Dict[int, Dict[str, float]] = {}
        self.segment_brand_affinities: Dict[int, Dict[str, float]] = {}

    def fit(self, events_df: pd.DataFrame, customer_clusters_df: pd.DataFrame) -> "SegmentPreferenceMiner":
        """Compute normalized product affinity and top item lists for each cluster."""
        logger.info("Mining segment preferences and product affinities...")
        merged = events_df.merge(customer_clusters_df[["customer_id", "cluster"]], on="customer_id", how="inner")

        for cluster_id, group in merged.groupby("cluster"):
            cid = int(float(str(cluster_id)))
            
            # Weighted interaction count per product in this segment
            prod_counts = group["product_id"].value_counts()
            
            # Normalize to [0, 1] range using min-max scaling per segment
            if len(prod_counts) > 0:
                max_cnt = prod_counts.max()
                min_cnt = prod_counts.min()
                if max_cnt > min_cnt:
                    norm_scores = (prod_counts - min_cnt) / (max_cnt - min_cnt)
                else:
                    norm_scores = prod_counts / max_cnt
                self.segment_product_scores[cid] = norm_scores.to_dict()
                self.segment_top_products[cid] = prod_counts.head(100).index.tolist()
            else:
                self.segment_product_scores[cid] = {}
                self.segment_top_products[cid] = []

            # Category & Brand distributions
            if "category" in group.columns:
                cat_dist = (group["category"].value_counts(normalize=True)).to_dict()
                self.segment_category_affinities[cid] = cat_dist
            if "brand" in group.columns:
                brand_dist = (group["brand"].value_counts(normalize=True)).to_dict()
                self.segment_brand_affinities[cid] = brand_dist

            logger.info(f"Segment {cid}: Mined {len(self.segment_product_scores[cid])} product affinities.")

        return self

    def get_preference_score(self, segment_id: int, product_id: Union[int, str]) -> float:
        """Fetch normalized preference score of a product for a given segment."""
        if segment_id not in self.segment_product_scores:
            return 0.0
        return float(self.segment_product_scores[segment_id].get(product_id, 0.0))

    def get_top_candidates(self, segment_id: int, top_n: int = 30) -> List[Tuple[Union[int, str], float]]:
        """Retrieve top candidate products for a segment."""
        if segment_id not in self.segment_product_scores:
            return []
        top_items = self.segment_top_products.get(segment_id, [])[:top_n]
        return [(item, self.get_preference_score(segment_id, item)) for item in top_items]

    def save(self, filepath: str = "artifacts/features/segment_preferences.joblib") -> None:
        """Persist segment preferences."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "segment_product_scores": self.segment_product_scores,
            "segment_top_products": self.segment_top_products,
            "segment_category_affinities": self.segment_category_affinities,
            "segment_brand_affinities": self.segment_brand_affinities,
        }
        joblib.dump(payload, path)
        logger.info(f"Saved segment preferences to {path}")

    @classmethod
    def load(cls, filepath: str = "artifacts/features/segment_preferences.joblib") -> "SegmentPreferenceMiner":
        """Load persisted segment preferences."""
        payload = joblib.load(filepath)
        miner = cls()
        miner.segment_product_scores = payload["segment_product_scores"]
        miner.segment_top_products = payload["segment_top_products"]
        miner.segment_category_affinities = payload.get("segment_category_affinities", {})
        miner.segment_brand_affinities = payload.get("segment_brand_affinities", {})
        return miner
