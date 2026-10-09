"""Feature engineering for customers, products, interactions, and segments."""

from .customer_features import CustomerFeatureEngineer
from .product_features import ProductFeatureEngineer
from .interaction_features import InteractionMatrixBuilder
from .segment_features import SegmentPreferenceMiner

__all__ = [
    "CustomerFeatureEngineer",
    "ProductFeatureEngineer",
    "InteractionMatrixBuilder",
    "SegmentPreferenceMiner",
]
