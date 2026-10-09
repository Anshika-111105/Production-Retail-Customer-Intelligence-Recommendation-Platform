"""Data quality checks and validation reporter."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data_validator")


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        return super().default(obj)


class DataValidator:
    """Validates raw interaction logs and produces comprehensive DQ reports."""

    def __init__(
        self,
        min_price: float = 0.01,
        max_price: float = 50000.0,
        max_quantity: int = 1000,
        output_report_path: str = "outputs/data_quality_report.json",
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.max_quantity = max_quantity
        self.output_report_path = Path(output_report_path)

    def validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run all validation checks, log decisions, filter anomalies, and generate DQ report."""
        logger.info("Initiating Data Quality and Validation checks...")
        df = df.copy()
        initial_len = len(df)
        filtering_log: List[str] = []

        # 1. Null ID checks
        null_cust = int(df["customer_id"].isnull().sum())
        null_prod = int(df["product_id"].isnull().sum())
        null_ts = int(df["timestamp"].isnull().sum())

        if null_cust > 0:
            filtering_log.append(f"Removed {null_cust} rows with missing customer_id")
            df = df[df["customer_id"].notnull()].copy()
        if null_prod > 0:
            filtering_log.append(f"Removed {null_prod} rows with missing product_id")
            df = df[df["product_id"].notnull()].copy()
        if null_ts > 0:
            filtering_log.append(f"Removed {null_ts} rows with missing timestamp")
            df = df[df["timestamp"].notnull()].copy()

        # 2. Timestamp parsing
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        invalid_ts_count = int(df["timestamp"].isnull().sum())
        if invalid_ts_count > 0:
            filtering_log.append(f"Removed {invalid_ts_count} rows with unparseable timestamps")
            df = df[df["timestamp"].notnull()].copy()

        # 3. Duplicate checks
        dup_mask = df.duplicated(subset=["customer_id", "product_id", "timestamp", "event_type"], keep="first")
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            filtering_log.append(f"Removed {dup_count} exact duplicate interaction rows")
            df = df[~dup_mask].copy()

        # 4. Numerical validation (quantities and prices)
        invalid_qty = (df["quantity"] < 0) | (df["quantity"] > self.max_quantity)
        invalid_qty_count = int(invalid_qty.sum())
        if invalid_qty_count > 0:
            filtering_log.append(f"Sanitized/clamped {invalid_qty_count} rows with negative or extreme quantities")
            df.loc[df["quantity"] < 0, "quantity"] = 0
            df.loc[df["quantity"] > self.max_quantity, "quantity"] = self.max_quantity

        if "unit_price" in df.columns:
            invalid_price = (df["unit_price"] < self.min_price) | (df["unit_price"] > self.max_price)
            invalid_price_count = int(invalid_price.sum())
            if invalid_price_count > 0:
                median_price = float(df["unit_price"].median()) if not df["unit_price"].empty else 10.0
                filtering_log.append(f"Imputed {invalid_price_count} anomalous prices with median ({median_price:.2f})")
                df.loc[invalid_price, "unit_price"] = median_price

        # 5. Missing values breakdown
        missing_dict = {str(col): int(df[col].isnull().sum()) for col in df.columns}

        # 6. Interaction Matrix Sparsity
        n_cust = int(df["customer_id"].nunique())
        n_prod = int(df["product_id"].nunique())
        total_possible = n_cust * n_prod if (n_cust > 0 and n_prod > 0) else 1
        sparsity = 1.0 - (len(df) / float(total_possible))

        # 7. Date range & event distribution
        min_date = str(df["timestamp"].min())
        max_date = str(df["timestamp"].max())
        event_dist = {str(k): int(v) for k, v in df["event_type"].value_counts().items()} if "event_type" in df.columns else {}

        report = {
            "row_count": int(len(df)),
            "initial_row_count": int(initial_len),
            "rows_filtered": int(initial_len - len(df)),
            "customer_count": int(n_cust),
            "product_count": int(n_prod),
            "missing_values": missing_dict,
            "duplicate_rows": int(dup_count),
            "invalid_transactions": int(invalid_ts_count + null_cust + null_prod),
            "date_range": {"min": min_date, "max": max_date},
            "sparsity": round(float(sparsity), 6),
            "event_distribution": event_dist,
            "filtering_log": filtering_log,
        }

        # Write data quality report
        self.output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, cls=NumpyJSONEncoder)

        logger.info(f"Data Quality report written to {self.output_report_path}")
        logger.info(f"Valid dataset: {len(df)} rows, {n_cust} customers, {n_prod} products, Sparsity: {sparsity:.4%}")

        return df.reset_index(drop=True), report
