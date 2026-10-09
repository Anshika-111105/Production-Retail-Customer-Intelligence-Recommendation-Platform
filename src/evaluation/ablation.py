"""Ablation study module analyzing component contributions to the hybrid recommender."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.evaluation.evaluator import RecommenderEvaluator
from src.ranking.candidate_generation import CandidatePool, MultiSourceCandidateGenerator
from src.ranking.scoring import HybridScorer
from src.utils.logger import get_logger

logger = get_logger("ablation_study")


class AblationStudy:
    """Runs systematic ablation experiments by enabling/disabling individual algorithmic components."""

    def __init__(self, evaluator: RecommenderEvaluator):
        self.evaluator = evaluator

    def run_ablation(
        self,
        als_model: Any,
        item_item_model: Any,
        segment_miner: Any,
        popularity_model: Any,
        trending_model: Any,
        customer_segments: Dict[Union[int, str], int],
        ground_truth: Dict[Union[int, str], Set[Union[int, str]]],
        filter_items_dict: Optional[Dict[Union[int, str], Set[Union[int, str]]]] = None,
        output_csv_path: str = "outputs/metrics/ablation_study.csv",
    ) -> pd.DataFrame:
        """Execute ablation configurations across all candidate sources."""
        logger.info("Executing Recommender Component Ablation Study...")

        configurations = [
            ("Popularity Only", {"popularity": 1.0}),
            ("Popularity + Segment", {"popularity": 0.5, "segment": 0.5}),
            ("Item-Item CF Only", {"item_item": 1.0}),
            ("ALS Matrix Factorization Only", {"als": 1.0}),
            ("ALS + Item-Item", {"als": 0.6, "item_item": 0.4}),
            ("ALS + Segment", {"als": 0.7, "segment": 0.3}),
            ("ALS + Item-Item + Segment", {"als": 0.45, "item_item": 0.35, "segment": 0.20}),
            ("Full Hybrid Engine", {"als": 0.35, "item_item": 0.25, "segment": 0.20, "popularity": 0.10, "trending": 0.10}),
        ]

        candidate_gen = MultiSourceCandidateGenerator()
        ablation_results = []

        for config_name, weights in configurations:
            logger.info(f"Running ablation config: {config_name} (weights: {weights})...")
            scorer = HybridScorer(weights=weights)

            def make_rec_fn(w_dict):
                def rec_fn(uid, n=10, filter_items=None):
                    seg_id = customer_segments.get(uid)
                    pool = candidate_gen.generate(
                        user_id=uid,
                        user_segment_id=seg_id,
                        als_model=als_model if "als" in w_dict else None,
                        item_item_model=item_item_model if "item_item" in w_dict else None,
                        segment_miner=segment_miner if "segment" in w_dict else None,
                        popularity_model=popularity_model if "popularity" in w_dict else None,
                        trending_model=trending_model if "trending" in w_dict else None,
                        filter_items=filter_items,
                    )
                    allowed = [it for it in pool.get_all_item_ids() if not filter_items or it not in filter_items]
                    recs = scorer.rank_candidates(pool, allowed_items=allowed, top_n=n)
                    return [r["product_id"] for r in recs]
                return rec_fn

            eval_summary = self.evaluator.evaluate_model(
                model_name=config_name,
                recommend_fn=make_rec_fn(weights),
                ground_truth=ground_truth,
                filter_items_dict=filter_items_dict,
            )
            ablation_results.append(eval_summary)

        ablation_df = pd.DataFrame(ablation_results)
        
        # Save results
        out_csv = Path(output_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        ablation_df.to_csv(out_csv, index=False)
        logger.info(f"Ablation study saved to {out_csv}")
        return ablation_df
