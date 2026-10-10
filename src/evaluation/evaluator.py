import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    map_at_k,
    ndcg_at_k,
    catalog_coverage,
    intra_list_diversity,
    novelty_at_k,
)
from src.utils.logger import get_logger

logger = get_logger("recommender_evaluator")


class RecommenderEvaluator:
    """Evaluates recommender strategies on temporal hold-out test sets."""

    def __init__(
        self,
        k_list: List[int] = [5, 10, 20],
        catalog_items: Optional[Set[Union[int, str]]] = None,
        item_category_map: Optional[Dict[Union[int, str], str]] = None,
        item_popularity_prob: Optional[Dict[Union[int, str], float]] = None,
    ):
        self.k_list = k_list
        self.catalog_items = catalog_items or set()
        self.item_category_map = item_category_map or {}
        self.item_popularity_prob = item_popularity_prob or {}

    def extract_ground_truth(self, test_df: pd.DataFrame) -> Dict[Union[int, str], Set[Union[int, str]]]:
        """Extract set of actual interacted items per customer in temporal test window."""
        gt_dict: Dict[Union[int, str], Set[Union[int, str]]] = {}
        for uid, grp in test_df.groupby("customer_id"):
            user_id: Union[int, str] = int(uid) if isinstance(uid, (int, np.integer)) else str(uid)
            gt_dict[user_id] = set(grp["product_id"].tolist())
        return gt_dict

    def evaluate_model(
        self,
        model_name: str,
        recommend_fn: Any,
        ground_truth: Dict[Union[int, str], Set[Union[int, str]]],
        user_ids: Optional[List[Union[int, str]]] = None,
        filter_items_dict: Optional[Dict[Union[int, str], Set[Union[int, str]]]] = None,
    ) -> Dict[str, Any]:
        """Compute ranking and diversity metrics for a given recommendation strategy."""
        eval_users = user_ids if user_ids is not None else list(ground_truth.keys())
        max_k = max(self.k_list)

        all_recs: List[List[Union[int, str]]] = []
        metrics_acc: Dict[str, List[float]] = {
            f"{m}@{k}": [] for k in self.k_list for m in ["precision", "recall", "hit_rate", "map", "ndcg"]
        }
        metrics_acc["diversity@10"] = []
        metrics_acc["novelty@10"] = []

        for uid in eval_users:
            actual = ground_truth.get(uid, set())
            if not actual:
                continue

            filter_items = filter_items_dict.get(uid) if filter_items_dict else None
            
            # Generate recommendations
            raw_recs = recommend_fn(uid, n=max_k, filter_items=filter_items)
            
            # Extract list of recommended item IDs
            if isinstance(raw_recs, list) and len(raw_recs) > 0:
                if isinstance(raw_recs[0], tuple):
                    rec_items = [r[0] for r in raw_recs]
                elif isinstance(raw_recs[0], dict):
                    rec_items = [r["product_id"] for r in raw_recs]
                else:
                    rec_items = raw_recs
            else:
                rec_items = []

            all_recs.append(rec_items)

            # Compute metric values per K
            for k in self.k_list:
                metrics_acc[f"precision@{k}"].append(precision_at_k(rec_items, actual, k=k))
                metrics_acc[f"recall@{k}"].append(recall_at_k(rec_items, actual, k=k))
                metrics_acc[f"hit_rate@{k}"].append(hit_rate_at_k(rec_items, actual, k=k))
                metrics_acc[f"map@{k}"].append(map_at_k(rec_items, actual, k=k))
                metrics_acc[f"ndcg@{k}"].append(ndcg_at_k(rec_items, actual, k=k))

            # Diversity and novelty at default K=10
            metrics_acc["diversity@10"].append(intra_list_diversity(rec_items[:10], self.item_category_map))
            metrics_acc["novelty@10"].append(novelty_at_k(rec_items[:10], self.item_popularity_prob, k=10))

        # Aggregate means
        summary: Dict[str, Any] = {"model": model_name, "eval_users": len(eval_users)}
        for m_name, vals in metrics_acc.items():
            summary[m_name] = round(float(np.mean(vals)), 4) if vals else 0.0

        # Coverage across all evaluation recommendations
        summary["catalog_coverage"] = round(float(catalog_coverage(all_recs, self.catalog_items)), 4)
        return summary

    def evaluate_all(
        self,
        models_dict: Dict[str, Any],
        ground_truth: Dict[Union[int, str], Set[Union[int, str]]],
        filter_items_dict: Optional[Dict[Union[int, str], Set[Union[int, str]]]] = None,
        user_interaction_counts: Optional[Dict[Union[int, str], int]] = None,
        sparse_threshold: int = 3,
    ) -> pd.DataFrame:
        """Run complete benchmark comparing all recommender algorithms and user cohorts."""
        logger.info(f"Initiating evaluation across {len(models_dict)} model strategies on {len(ground_truth)} test customers...")
        results = []

        for model_name, model_obj in models_dict.items():
            logger.info(f"Evaluating model: {model_name}...")
            
            # Recommendation wrapper
            if hasattr(model_obj, "recommend"):
                def rec_fn(uid, n=10, filter_items=None):
                    res = model_obj.recommend(uid, n=n, filter_items=filter_items)
                    # If dict returned by RecommendationEngine
                    if isinstance(res, dict) and "recommendations" in res:
                        return res["recommendations"]
                    return res
            elif callable(model_obj):
                rec_fn = model_obj
            else:
                continue

            summary = self.evaluate_model(
                model_name=model_name,
                recommend_fn=rec_fn,
                ground_truth=ground_truth,
                filter_items_dict=filter_items_dict,
            )
            results.append(summary)

        # Stratified evaluation by customer maturity cohort on Hybrid Engine
        if "Hybrid Engine" in models_dict and user_interaction_counts is not None:
            hybrid_obj = models_dict["Hybrid Engine"]
            all_users = list(ground_truth.keys())
            
            sparse_users = [u for u in all_users if user_interaction_counts.get(u, 0) <= sparse_threshold]
            mature_users = [u for u in all_users if user_interaction_counts.get(u, 0) > sparse_threshold]

            if sparse_users:
                logger.info(f"Evaluating Hybrid on Sparse User cohort (n={len(sparse_users)})...")
                sparse_summary = self.evaluate_model(
                    model_name="Hybrid (Sparse Cohort)",
                    recommend_fn=lambda uid, n=10, filter_items=None: hybrid_obj.recommend(uid, n=n, filter_items=filter_items)["recommendations"],
                    ground_truth=ground_truth,
                    user_ids=sparse_users,
                    filter_items_dict=filter_items_dict,
                )
                results.append(sparse_summary)

            if mature_users:
                logger.info(f"Evaluating Hybrid on Mature User cohort (n={len(mature_users)})...")
                mature_summary = self.evaluate_model(
                    model_name="Hybrid (Mature Cohort)",
                    recommend_fn=lambda uid, n=10, filter_items=None: hybrid_obj.recommend(uid, n=n, filter_items=filter_items)["recommendations"],
                    ground_truth=ground_truth,
                    user_ids=mature_users,
                    filter_items_dict=filter_items_dict,
                )
                results.append(mature_summary)

        results_df = pd.DataFrame(results)
        return results_df

    def generate_report(
        self,
        results_df: pd.DataFrame,
        output_csv_path: str = "outputs/metrics/evaluation_metrics.csv",
        output_md_path: str = "reports/recommender_evaluation.md",
    ) -> str:
        """Persist evaluation metrics to CSV and generate Markdown evaluation report."""
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved evaluation metrics CSV to {csv_path}")

        md_path = Path(output_md_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)

        md_content = [
            "# Offline Recommender Evaluation Report\n",
            "## 1. Methodology & Data Leakage Controls\n",
            "- **Temporal Hold-Out Split**: Interactions were strictly split along the global timeline into Train (70%), Validation (15%), and Test (15%).",
            "- **Evaluation Protocol**: Models were trained exclusively on training interactions. Ground truth evaluation measures actual unseen interactions occurring in the future test window.",
            "- **Cold-Start & History Isolation**: Test-period interactions were never visible to item similarity graphs, ALS matrix factorization, or segment preferences.",
            "- **Score Normalization**: Multi-model candidate scores were min-max normalized prior to weighted hybrid combination.\n",
            "## 2. Comprehensive Model Benchmark\n\n",
            results_df.to_markdown(index=False),
            "\n\n## 3. Metric Interpretation & Key Insights\n",
            "- **Precision@K & Recall@K**: Measure accuracy and coverage of ground-truth future items in top-K recommendations.",
            "- **NDCG@K**: Evaluates ranking quality, heavily penalizing relevant items placed lower in the recommended list.",
            "- **Catalog Coverage**: Proportion of all distinct items in the catalog recommended at least once, highlighting long-tail discovery.",
            "- **Intra-List Diversity**: Measures cross-category diversity within user recommendation sets, preventing single-category concentration.",
            "- **Novelty@10**: Information-theoretic measure of recommendation serendipity vs popular bias.\n",
        ]

        full_md = "\n".join(md_content)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_md)

        logger.info(f"Generated Recommender Evaluation Markdown report at {md_path}")
        return full_md
