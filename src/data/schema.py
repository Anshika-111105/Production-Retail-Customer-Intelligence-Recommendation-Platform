"""Data schemas and contract definitions for retail intelligence."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import List, Optional, Union
from pydantic import BaseModel, Field


class RawInteractionRecord(BaseModel):
    """Schema for an individual interaction record."""
    customer_id: Union[int, str]
    product_id: Union[int, str]
    timestamp: str
    transaction_id: Optional[Union[int, str]] = None
    quantity: Optional[int] = 1
    unit_price: Optional[float] = None
    category: Optional[str] = "Unknown"
    brand: Optional[str] = "Unknown"
    description: Optional[str] = None
    discount: Optional[float] = 0.0
    event_type: Optional[str] = "purchase"


class ProductCatalogRecord(BaseModel):
    """Schema for product catalog metadata."""
    product_id: Union[int, str]
    category: Optional[str] = "Unknown"
    brand: Optional[str] = "Unknown"
    unit_price: Optional[float] = None
    description: Optional[str] = None


class DataQualitySummary(BaseModel):
    """Data quality validation summary report schema."""
    row_count: int
    customer_count: int
    product_count: int
    missing_values: dict
    duplicate_rows: int
    invalid_transactions: int
    date_range: dict
    sparsity: float
    event_distribution: dict
    filtering_actions: List[str] = Field(default_factory=list)


# Schema aliases
RawInteractionSchema = RawInteractionRecord
CleanedInteractionSchema = RawInteractionRecord

