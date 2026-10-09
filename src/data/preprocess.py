"""Data Preprocessing and Temporal Splitting pipeline preventing data leakage."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("preprocess")


class DataPreprocessor:
    """Preprocesses retail events and executes strict temporal train/val/test splits."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        processed_dir: str = "data/processed",
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.processed_dir = Path(processed_dir)

    def temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.Timestamp]]:
        """Split interactions temporally by global timestamp index ordering to prevent temporal leakage."""
        logger.info("Executing strict temporal train/val/test data split...")
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)

        n = len(df_sorted)
        train_end_idx = int(n * self.train_ratio)
        val_end_idx = int(n * (self.train_ratio + self.val_ratio))

        train_df = df_sorted.iloc[:train_end_idx].copy().reset_index(drop=True)
        val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy().reset_index(drop=True)
        test_df = df_sorted.iloc[val_end_idx:].copy().reset_index(drop=True)

        cutoffs = {
            "min_timestamp": df_sorted["timestamp"].min(),
            "train_val_cutoff": train_df["timestamp"].max() if not train_df.empty else None,
            "val_test_cutoff": val_df["timestamp"].max() if not val_df.empty else None,
            "max_timestamp": df_sorted["timestamp"].max(),
        }

        logger.info(
            f"Temporal Split Complete:\n"
            f"  Train : {len(train_df):,} events ({train_df['timestamp'].min()} to {cutoffs['train_val_cutoff']})\n"
            f"  Val   : {len(val_df):,} events ({val_df['timestamp'].min()} to {cutoffs['val_test_cutoff']})\n"
            f"  Test  : {len(test_df):,} events ({test_df['timestamp'].min()} to {cutoffs['max_timestamp']})"
        )

        return train_df, val_df, test_df, cutoffs

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Persist processed splits to parquet and csv."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_parquet(self.processed_dir / "train_events.parquet", index=False)
        val_df.to_parquet(self.processed_dir / "val_events.parquet", index=False)
        test_df.to_parquet(self.processed_dir / "test_events.parquet", index=False)

        train_df.to_csv(self.processed_dir / "train_events.csv", index=False)
        val_df.to_csv(self.processed_dir / "val_events.csv", index=False)
        test_df.to_csv(self.processed_dir / "test_events.csv", index=False)

        logger.info(f"Saved processed dataset splits to {self.processed_dir}")
