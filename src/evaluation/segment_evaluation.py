"""Segment-stratified recommender performance evaluation."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd

from src.evaluation.evaluator import RecommenderEvaluator
from src.utils.logger import get_logger

logger = get_logger("segment_evaluation")


class SegmentEvaluator:
    """Evaluates recommendation quality across stratified customer clusters."""

    def __init__(self, evaluator: RecommenderEvaluator):
        self.evaluator = evaluator

    def evaluate_by_segment(
        self,
        recommend_fn: Any,
        customer_segments: Dict[Union[int, str], int],
        segment_labels: Dict[int, str],
        ground_truth: Dict[Union[int, str], Set[Union[int, str]]],
        filter_items_dict: Optional[Dict[Union[int, str], Set[Union[int, str]]]] = None,
        output_csv_path: str = "outputs/metrics/segment_evaluation.csv",
    ) -> pd.DataFrame:
        """Run evaluation metrics broken down by customer segment."""
        logger.info("Computing segment-stratified recommendation performance...")

        # Group test users by cluster
        segment_users: Dict[int, List[Union[int, str]]] = {}
        for uid in ground_truth.keys():
            seg_id = customer_segments.get(uid)
            if seg_id is not None:
                if seg_id not in segment_users:
                    segment_users[seg_id] = []
                segment_users[seg_id].append(uid)

        results = []
        for seg_id in sorted(segment_users.keys()):
            u_list = segment_users[seg_id]
            label = segment_labels.get(seg_id, f"Segment {seg_id}")
            logger.info(f"Evaluating Segment {seg_id} ('{label}') with {len(u_list)} test customers...")

            summary = self.evaluator.evaluate_model(
                model_name=f"Segment {seg_id}: {label}",
                recommend_fn=recommend_fn,
                ground_truth=ground_truth,
                user_ids=u_list,
                filter_items_dict=filter_items_dict,
            )
            summary["segment_id"] = seg_id
            summary["segment_label"] = label
            results.append(summary)

        results_df = pd.DataFrame(results)
        
        # Save output
        out_csv = Path(output_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_csv, index=False)
        logger.info(f"Saved segment-stratified evaluation to {out_csv}")
        return results_df
