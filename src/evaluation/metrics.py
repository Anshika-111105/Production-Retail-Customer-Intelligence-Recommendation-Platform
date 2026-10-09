"""Offline Recommender System Evaluation Metrics."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Dict, List, Optional, Set, Union
import numpy as np


def precision_at_k(recommended: List[Union[int, str]], actual: Set[Union[int, str]], k: int = 10) -> float:
    """Calculate Precision@K."""
    if not actual or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in actual)
    return float(hits / k)


def recall_at_k(recommended: List[Union[int, str]], actual: Set[Union[int, str]], k: int = 10) -> float:
    """Calculate Recall@K."""
    if not actual or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in actual)
    return float(hits / len(actual))


def hit_rate_at_k(recommended: List[Union[int, str]], actual: Set[Union[int, str]], k: int = 10) -> float:
    """Calculate HitRate@K (1 if at least one hit in top-K, 0 otherwise)."""
    if not actual or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    return 1.0 if any(item in actual for item in rec_k) else 0.0


def map_at_k(recommended: List[Union[int, str]], actual: Set[Union[int, str]], k: int = 10) -> float:
    """Calculate Mean Average Precision (MAP@K) for a single user."""
    if not actual or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    score = 0.0
    num_hits = 0.0
    for i, item in enumerate(rec_k):
        if item in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return float(score / min(len(actual), k))


def ndcg_at_k(recommended: List[Union[int, str]], actual: Set[Union[int, str]], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
    if not actual or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)  # 0-indexed: i=0 -> rank 1 -> log2(2)=1

    # Ideal DCG
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(actual), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def catalog_coverage(all_recommendations: List[List[Union[int, str]]], catalog_items: Set[Union[int, str]]) -> float:
    """Calculate catalog coverage (proportion of catalog items recommended at least once)."""
    if not catalog_items:
        return 0.0
    unique_recommended = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)
    return float(len(unique_recommended.intersection(catalog_items)) / len(catalog_items))


def intra_list_diversity(
    recommended: List[Union[int, str]],
    item_category_map: Dict[Union[int, str], str],
) -> float:
    """Calculate intra-list category diversity (1 - mean pairwise category similarity)."""
    if len(recommended) <= 1:
        return 1.0
    cats = [item_category_map.get(it, "Unknown") for it in recommended]
    pairs_total = 0
    pairs_dissimilar = 0
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            pairs_total += 1
            if cats[i] != cats[j]:
                pairs_dissimilar += 1
    return float(pairs_dissimilar / pairs_total) if pairs_total > 0 else 0.0


def novelty_at_k(
    recommended: List[Union[int, str]],
    item_popularity_prob: Dict[Union[int, str], float],
    k: int = 10,
) -> float:
    """Calculate recommendation novelty (-log2(P(item))). Higher = more serendipitous/less head-heavy."""
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    self_info = []
    for item in rec_k:
        p = item_popularity_prob.get(item, 1e-5)
        self_info.append(-np.log2(max(1e-6, p)))
    return float(np.mean(self_info))


class RankingMetrics:
    """Namespace wrapper for ranking evaluation metrics."""
    precision_at_k = staticmethod(precision_at_k)
    recall_at_k = staticmethod(recall_at_k)
    hit_rate_at_k = staticmethod(hit_rate_at_k)
    map_at_k = staticmethod(map_at_k)
    ndcg_at_k = staticmethod(ndcg_at_k)
    catalog_coverage = staticmethod(catalog_coverage)
    intra_list_diversity = staticmethod(intra_list_diversity)
    novelty_at_k = staticmethod(novelty_at_k)

