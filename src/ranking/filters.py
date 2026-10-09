"""Modular business rule filtering components for recommendation post-processing."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union


class BaseFilter(ABC):
    """Abstract base class for business rule filters."""

    @abstractmethod
    def filter(
        self,
        candidate_items: List[Union[int, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Union[int, str]]:
        """Filter candidate item IDs based on specific business rules."""
        pass


class AlreadyPurchasedFilter(BaseFilter):
    """Filters out products the customer has already purchased or interacted with."""

    def __init__(self, user_history_dict: Optional[Dict[Union[int, str], Set[Union[int, str]]]] = None):
        self.user_history_dict = user_history_dict or {}

    def filter(
        self,
        candidate_items: List[Union[int, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Union[int, str]]:
        if not context:
            return candidate_items
        
        user_id = context.get("user_id")
        raw_purchased = context.get("already_purchased")
        already_purchased = set(raw_purchased) if raw_purchased is not None else set()
        if user_id and user_id in self.user_history_dict:
            already_purchased.update(self.user_history_dict[user_id])

        if not already_purchased:
            return candidate_items

        return [item for item in candidate_items if item not in already_purchased]


class CategoryFilter(BaseFilter):
    """Constrains recommendations to specified categories if requested."""

    def __init__(self, product_metadata: Optional[Dict[Union[int, str], Dict[str, Any]]] = None):
        self.product_metadata = product_metadata or {}

    def filter(
        self,
        candidate_items: List[Union[int, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Union[int, str]]:
        if not context or "allowed_categories" not in context:
            return candidate_items

        allowed = set(context["allowed_categories"])
        if not allowed:
            return candidate_items

        valid_items = []
        for item in candidate_items:
            cat = self.product_metadata.get(item, {}).get("category")
            if cat in allowed:
                valid_items.append(item)
        return valid_items


class PriceFilter(BaseFilter):
    """Filters products outside of a minimum or maximum price budget."""

    def __init__(self, product_metadata: Optional[Dict[Union[int, str], Dict[str, Any]]] = None):
        self.product_metadata = product_metadata or {}

    def filter(
        self,
        candidate_items: List[Union[int, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Union[int, str]]:
        if not context:
            return candidate_items

        min_price = context.get("min_price", 0.0)
        max_price = context.get("max_price", float("inf"))

        valid_items = []
        for item in candidate_items:
            price = float(self.product_metadata.get(item, {}).get("unit_price", 0.0))
            if min_price <= price <= max_price:
                valid_items.append(item)
        return valid_items


class ProductAvailabilityFilter(BaseFilter):
    """Filters out inactive, discontinued, or out-of-stock products."""

    def __init__(self, unavailable_items: Optional[Set[Union[int, str]]] = None):
        self.unavailable_items = unavailable_items or set()

    def filter(
        self,
        candidate_items: List[Union[int, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Union[int, str]]:
        unavail = self.unavailable_items
        if context and "unavailable_items" in context:
            unavail = unavail.union(set(context["unavailable_items"]))

        if not unavail:
            return candidate_items

        return [item for item in candidate_items if item not in unavail]


class BusinessRuleFilterChain:
    """Chains multiple business filters and applies them sequentially."""

    def __init__(self, filters: Optional[List[BaseFilter]] = None):
        self.filters: List[BaseFilter] = filters or []

    def add_filter(self, rule_filter: BaseFilter) -> "BusinessRuleFilterChain":
        """Add a filter to the chain."""
        self.filters.append(rule_filter)
        return self

    def apply(
        self,
        candidate_items: List[Union[int, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Union[int, str]]:
        """Sequentially execute all active filters."""
        current_items = candidate_items
        for f in self.filters:
            current_items = f.filter(current_items, context=context)
        return current_items


# Convenient aliases matching architectural specification
InventoryFilter = ProductAvailabilityFilter
FilterChain = BusinessRuleFilterChain

