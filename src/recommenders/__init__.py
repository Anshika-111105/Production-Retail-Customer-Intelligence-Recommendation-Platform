"""Recommendation algorithm models and the hybrid routing engine."""

from .base import BaseRecommender
from .popularity import PopularityRecommender
from .trending import TrendingRecommender
from .item_item import ItemItemRecommender
from .als import ALSRecommender
from .svd import SVDRecommender
from .content_based import ContentBasedRecommender
from .hybrid import RecommendationEngine

__all__ = [
    "BaseRecommender",
    "PopularityRecommender",
    "TrendingRecommender",
    "ItemItemRecommender",
    "ALSRecommender",
    "SVDRecommender",
    "ContentBasedRecommender",
    "RecommendationEngine",
]
