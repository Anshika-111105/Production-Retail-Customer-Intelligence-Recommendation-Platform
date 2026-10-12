"""Production Hybrid Recommendation Engine orchestrating routing, candidate generation, ranking, and explainability."""

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
from src.recommenders.popularity import PopularityRecommender
from src.recommenders.trending import TrendingRecommender
from src.recommenders.item_item import ItemItemRecommender
from src.recommenders.als import ALSRecommender
from src.recommenders.svd import SVDRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.features.interaction_features import InteractionMatrixBuilder
from src.features.segment_features import SegmentPreferenceMiner
from src.ranking.candidate_generation import CandidatePool, MultiSourceCandidateGenerator
from src.ranking.filters import (
    BusinessRuleFilterChain,
    AlreadyPurchasedFilter,
    CategoryFilter,
    PriceFilter,
    ProductAvailabilityFilter,
)
from src.ranking.scoring import HybridScorer
from src.utils.logger import get_logger

logger = get_logger("hybrid_engine")


class RecommendationEngine:
    """Production-grade unified retail recommendation engine."""

    def __init__(
        self,
        id_builder: Optional[InteractionMatrixBuilder] = None,
        popularity_model: Optional[PopularityRecommender] = None,
        trending_model: Optional[TrendingRecommender] = None,
        item_item_model: Optional[ItemItemRecommender] = None,
        als_model: Optional[ALSRecommender] = None,
        svd_model: Optional[SVDRecommender] = None,
        content_model: Optional[ContentBasedRecommender] = None,
        segment_miner: Optional[SegmentPreferenceMiner] = None,
        customer_segments: Optional[Dict[Union[int, str], int]] = None,
        segment_labels: Optional[Dict[int, str]] = None,
        user_interaction_counts: Optional[Dict[Union[int, str], int]] = None,
        user_purchased_items: Optional[Dict[Union[int, str], Set[Union[int, str]]]] = None,
        product_metadata: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
        sparse_user_threshold: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.id_builder = id_builder
        self.popularity_model = popularity_model
        self.trending_model = trending_model
        self.item_item_model = item_item_model
        self.als_model = als_model
        self.svd_model = svd_model
        self.content_model = content_model
        self.segment_miner = segment_miner
        
        self.customer_segments = customer_segments or {}
        self.segment_labels = segment_labels or {}
        self.user_interaction_counts = user_interaction_counts or {}
        self.user_purchased_items = user_purchased_items or {}
        self.product_metadata = product_metadata or {}
        
        self.sparse_user_threshold = sparse_user_threshold
        self.candidate_generator = MultiSourceCandidateGenerator()
        self.scorer = HybridScorer(weights=weights)

    def recommend(
        self,
        customer_id: Union[int, str],
        n: int = 10,
        context: Optional[Dict[str, Any]] = None,
        filter_items: Optional[Set[Union[int, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate personalized recommendations with automated strategy routing and explanations."""
        context = context or {}
        n_interactions = self.user_interaction_counts.get(customer_id, 0)
        segment_id = self.customer_segments.get(customer_id)
        segment_label = self.segment_labels.get(segment_id, "New / Unsegmented Customer") if segment_id is not None else "New Customer"
        already_purchased = set(self.user_purchased_items.get(customer_id, set()))
        if filter_items:
            already_purchased.update(filter_items)

        # Determine Recommendation Routing Strategy
        if n_interactions == 0:
            rec_type = "cold_start_user"
            recommendations = self._recommend_cold_start(customer_id, n=n, context=context, already_purchased=already_purchased)
        elif n_interactions <= self.sparse_user_threshold:
            rec_type = "sparse_user"
            recommendations = self._recommend_sparse(customer_id, segment_id=segment_id, n=n, context=context, already_purchased=already_purchased)
        else:
            rec_type = "hybrid"
            recommendations = self._recommend_hybrid(customer_id, segment_id=segment_id, n=n, context=context, already_purchased=already_purchased)

        # Build explainability traces for each recommended product
        for rec in recommendations:
            prod_id = rec["product_id"]
            sources = rec.get("sources", [])
            rec["explanation"] = self._generate_explanation(customer_id, prod_id, sources, segment_label)
            
            # Attach product catalog metadata if present
            if prod_id in self.product_metadata:
                rec["category"] = self.product_metadata[prod_id].get("category", "General")
                rec["brand"] = self.product_metadata[prod_id].get("brand", "Generic")
                rec["unit_price"] = self.product_metadata[prod_id].get("unit_price", 0.0)

        return {
            "customer_id": customer_id,
            "segment_id": segment_id,
            "segment_label": segment_label,
            "interaction_count": n_interactions,
            "recommendation_type": rec_type,
            "recommendations": recommendations,
        }

    def _recommend_cold_start(
        self,
        customer_id: Union[int, str],
        n: int,
        context: Dict[str, Any],
        already_purchased: Set[Union[int, str]],
    ) -> List[Dict[str, Any]]:
        """Cold-start routing: Popularity + Trending + Category Context."""
        cat_ctx = context.get("category_context")
        pop_recs = self.popularity_model.recommend(customer_id, n=n*2, filter_items=already_purchased, category_context=cat_ctx) if self.popularity_model else []
        trend_recs = self.trending_model.recommend(customer_id, n=n, filter_items=already_purchased) if self.trending_model else []

        pool = CandidatePool()
        pool.add_candidates("popularity", pop_recs)
        pool.add_candidates("trending", trend_recs)

        allowed_items = self._apply_business_filters(pool.get_all_item_ids(), context, customer_id, already_purchased)
        scorer = HybridScorer(weights={"popularity": 0.6, "trending": 0.4})
        return scorer.rank_candidates(pool, allowed_items=allowed_items, top_n=n)

    def _recommend_sparse(
        self,
        customer_id: Union[int, str],
        segment_id: Optional[int],
        n: int,
        context: Dict[str, Any],
        already_purchased: Set[Union[int, str]],
    ) -> List[Dict[str, Any]]:
        """Sparse-user routing: Item-Item + Popularity + Segment preferences."""
        item_recs = self.item_item_model.recommend(customer_id, n=n*2, filter_items=already_purchased) if self.item_item_model else []
        pop_recs = self.popularity_model.recommend(customer_id, n=n, filter_items=already_purchased) if self.popularity_model else []
        seg_recs = self.segment_miner.get_top_candidates(segment_id, top_n=n) if (self.segment_miner and segment_id is not None) else []
        seg_recs = [r for r in seg_recs if r[0] not in already_purchased]

        pool = CandidatePool()
        pool.add_candidates("item_item", item_recs)
        pool.add_candidates("popularity", pop_recs)
        pool.add_candidates("segment", seg_recs)

        allowed_items = self._apply_business_filters(pool.get_all_item_ids(), context, customer_id, already_purchased)
        scorer = HybridScorer(weights={"item_item": 0.50, "segment": 0.30, "popularity": 0.20})
        return scorer.rank_candidates(pool, allowed_items=allowed_items, top_n=n)

    def _recommend_hybrid(
        self,
        customer_id: Union[int, str],
        segment_id: Optional[int],
        n: int,
        context: Dict[str, Any],
        already_purchased: Set[Union[int, str]],
    ) -> List[Dict[str, Any]]:
        """Mature-user routing: Full multi-candidate generation + hybrid score blending."""
        pool = self.candidate_generator.generate(
            user_id=customer_id,
            user_segment_id=segment_id,
            als_model=self.als_model,
            item_item_model=self.item_item_model,
            segment_miner=self.segment_miner,
            popularity_model=self.popularity_model,
            trending_model=self.trending_model,
            filter_items=already_purchased,
        )

        allowed_items = self._apply_business_filters(pool.get_all_item_ids(), context, customer_id, already_purchased)
        return self.scorer.rank_candidates(pool, allowed_items=allowed_items, top_n=n)

    def _apply_business_filters(
        self,
        candidate_items: List[Union[int, str]],
        context: Dict[str, Any],
        customer_id: Union[int, str],
        already_purchased: Set[Union[int, str]],
    ) -> List[Union[int, str]]:
        """Run modular filter chain."""
        filter_chain = BusinessRuleFilterChain([
            AlreadyPurchasedFilter(user_history_dict={customer_id: already_purchased}),
            CategoryFilter(product_metadata=self.product_metadata),
            PriceFilter(product_metadata=self.product_metadata),
            ProductAvailabilityFilter(),
        ])
        filter_context = {
            "user_id": customer_id,
            "already_purchased": already_purchased,
            **context,
        }
        return filter_chain.apply(candidate_items, context=filter_context)

    def _generate_explanation(
        self,
        customer_id: Union[int, str],
        product_id: Union[int, str],
        sources: List[str],
        segment_label: str,
    ) -> str:
        """Derive natural explainability text strictly anchored in active model signals."""
        if "item_item" in sources and self.item_item_model:
            exp_meta = self.item_item_model.explain(customer_id, product_id)
            if exp_meta.get("source_product"):
                return f"Similar to Product #{exp_meta['source_product']} that you previously interacted with."
        
        if "segment" in sources:
            return f"Popular among customers in your segment ({segment_label})."

        if "als" in sources:
            return "Recommended based on collaborative latent preferences with similar shoppers."

        if "trending" in sources:
            return "Trending with high momentum across the store right now."

        if "popularity" in sources:
            return "One of the most frequently purchased products in our catalog."

        return "Recommended for you based on catalog activity."

    def save(self, artifacts_dir: str = "artifacts") -> None:
        """Persist engine metadata and sub-models."""
        out_dir = Path(artifacts_dir)
        (out_dir / "models").mkdir(parents=True, exist_ok=True)
        (out_dir / "mappings").mkdir(parents=True, exist_ok=True)
        (out_dir / "features").mkdir(parents=True, exist_ok=True)

        if self.id_builder:
            self.id_builder.save(str(out_dir / "mappings" / "id_mappings.joblib"))
        if self.popularity_model:
            self.popularity_model.save(str(out_dir / "models" / "popularity_model.joblib"))
        if self.trending_model:
            self.trending_model.save(str(out_dir / "models" / "trending_model.joblib"))
        if self.item_item_model:
            self.item_item_model.save(str(out_dir / "models" / "item_item_model.joblib"))
        if self.als_model:
            self.als_model.save(str(out_dir / "models" / "als_model.joblib"))
        if self.svd_model:
            self.svd_model.save(str(out_dir / "models" / "svd_model.joblib"))
        if self.content_model:
            self.content_model.save(str(out_dir / "models" / "content_based_model.joblib"))
        if self.segment_miner:
            self.segment_miner.save(str(out_dir / "features" / "segment_preferences.joblib"))

        # Engine metadata
        engine_meta = {
            "customer_segments": self.customer_segments,
            "segment_labels": self.segment_labels,
            "user_interaction_counts": self.user_interaction_counts,
            "user_purchased_items": self.user_purchased_items,
            "product_metadata": self.product_metadata,
            "sparse_user_threshold": self.sparse_user_threshold,
            "weights": self.scorer.weights,
        }
        joblib.dump(engine_meta, out_dir / "models" / "engine_metadata.joblib")
        logger.info(f"Persisted complete Recommendation Engine state to {out_dir}")

    @classmethod
    def load(cls, artifacts_dir: str = "artifacts") -> "RecommendationEngine":
        """Load fully assembled production RecommendationEngine."""
        p = Path(artifacts_dir)
        if not p.is_absolute() and not p.exists() and (_PROJECT_ROOT / p).exists():
            p = _PROJECT_ROOT / p
        id_builder = InteractionMatrixBuilder.load(str(p / "mappings" / "id_mappings.joblib")) if (p / "mappings" / "id_mappings.joblib").exists() else None

        popularity = PopularityRecommender.load(str(p / "models" / "popularity_model.joblib")) if (p / "models" / "popularity_model.joblib").exists() else None
        trending = TrendingRecommender.load(str(p / "models" / "trending_model.joblib")) if (p / "models" / "trending_model.joblib").exists() else None
        item_item = ItemItemRecommender.load(str(p / "models" / "item_item_model.joblib"), id_builder=id_builder) if (p / "models" / "item_item_model.joblib").exists() else None
        als = ALSRecommender.load(str(p / "models" / "als_model.joblib"), id_builder=id_builder) if (p / "models" / "als_model.joblib").exists() else None
        svd = SVDRecommender.load(str(p / "models" / "svd_model.joblib"), id_builder=id_builder) if (p / "models" / "svd_model.joblib").exists() else None
        content = ContentBasedRecommender.load(str(p / "models" / "content_based_model.joblib")) if (p / "models" / "content_based_model.joblib").exists() else None
        segment_miner = SegmentPreferenceMiner.load(str(p / "features" / "segment_preferences.joblib")) if (p / "features" / "segment_preferences.joblib").exists() else None

        meta_path = p / "models" / "engine_metadata.joblib"
        if meta_path.exists():
            meta = joblib.load(meta_path)
        else:
            meta = {}

        return cls(
            id_builder=id_builder,
            popularity_model=popularity,
            trending_model=trending,
            item_item_model=item_item,
            als_model=als,
            svd_model=svd,
            content_model=content,
            segment_miner=segment_miner,
            customer_segments=meta.get("customer_segments", {}),
            segment_labels=meta.get("segment_labels", {}),
            user_interaction_counts=meta.get("user_interaction_counts", {}),
            user_purchased_items=meta.get("user_purchased_items", {}),
            product_metadata=meta.get("product_metadata", {}),
            sparse_user_threshold=meta.get("sparse_user_threshold", 3),
            weights=meta.get("weights"),
        )
