"""Data loading adapter supporting standard retail transactional logs and RetailRocket formats."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data_loader")


class DataLoader:
    """Configurable loader for raw retail datasets."""

    def __init__(self, raw_events_path: Union[str, Path], raw_catalog_path: Optional[Union[str, Path]] = None):
        self.raw_events_path = self._resolve_path(raw_events_path)
        self.raw_catalog_path = self._resolve_path(raw_catalog_path)

    @staticmethod
    def _resolve_path(path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolve path relative to CWD or project root."""
        if path is None:
            return None
        p = Path(path)
        if p.exists():
            return p
        if not p.is_absolute() and (_PROJECT_ROOT / p).exists():
            return _PROJECT_ROOT / p
        return p

    def load_events(self) -> pd.DataFrame:
        """Load and normalize event log into standard schema."""
        if not self.raw_events_path or not self.raw_events_path.exists():
            raise FileNotFoundError(f"Events file not found: {self.raw_events_path}")


        logger.info(f"Loading raw interactions from {self.raw_events_path}...")
        if self.raw_events_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.raw_events_path)
        else:
            df = pd.read_csv(self.raw_events_path)

        df = self._standardize_events_schema(df)
        logger.info(f"Loaded {len(df)} raw interaction records.")
        return df

    def load_catalog(self) -> Optional[pd.DataFrame]:
        """Load product catalog if available."""
        if not self.raw_catalog_path or not self.raw_catalog_path.exists():
            logger.info("No external product catalog provided or file does not exist.")
            return None

        logger.info(f"Loading product catalog from {self.raw_catalog_path}...")
        if self.raw_catalog_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.raw_catalog_path)
        else:
            df = pd.read_csv(self.raw_catalog_path)

        df = self._standardize_catalog_schema(df)
        logger.info(f"Loaded catalog with {len(df)} products.")
        return df

    def load_and_merge(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load events and catalog, merging metadata when catalog exists."""
        events_df = self.load_events()
        catalog_df = self.load_catalog()

        if catalog_df is not None:
            # Merge catalog columns if missing in events
            merge_cols = [c for c in catalog_df.columns if c != "product_id" and (c not in events_df.columns or events_df[c].isnull().all())]
            if merge_cols:
                logger.info(f"Merging catalog metadata columns {merge_cols} into events...")
                events_df = events_df.merge(
                    catalog_df[["product_id"] + merge_cols],
                    on="product_id",
                    how="left",
                    suffixes=("", "_catalog")
                )
                for col in merge_cols:
                    if f"{col}_catalog" in events_df.columns:
                        events_df[col] = events_df[col].fillna(events_df[f"{col}_catalog"])
                        events_df.drop(columns=[f"{col}_catalog"], inplace=True)

        return events_df, catalog_df

    def _standardize_events_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map heterogeneous raw column names to canonical schema."""
        col_map = {
            # RetailRocket mappings
            "visitorid": "customer_id",
            "visitor_id": "customer_id",
            "user_id": "customer_id",
            "userid": "customer_id",
            "CustomerID": "customer_id",
            "itemid": "product_id",
            "item_id": "product_id",
            "productid": "product_id",
            "ProductID": "product_id",
            "StockCode": "product_id",
            "event": "event_type",
            "transactionid": "transaction_id",
            "transaction_id": "transaction_id",
            "InvoiceNo": "transaction_id",
            "InvoiceDate": "timestamp",
            "Quantity": "quantity",
            "UnitPrice": "unit_price",
            "price": "unit_price",
            "category_code": "category",
            "category_id": "category",
            "brand_name": "brand",
        }

        rename_dict = {col: col_map[col] for col in df.columns if col in col_map}
        if rename_dict:
            df = df.rename(columns=rename_dict)

        # Validate required columns
        required = ["customer_id", "product_id", "timestamp"]
        missing_req = [c for c in required if c not in df.columns]
        if missing_req:
            raise ValueError(f"Raw data missing required canonical columns: {missing_req}. Available: {list(df.columns)}")

        # Fill recommended defaults if missing
        if "quantity" not in df.columns:
            df["quantity"] = 1
        if "unit_price" not in df.columns:
            df["unit_price"] = 1.0
        if "event_type" not in df.columns:
            df["event_type"] = "purchase"
        if "category" not in df.columns:
            df["category"] = "General"
        if "brand" not in df.columns:
            df["brand"] = "Generic"
        if "discount" not in df.columns:
            df["discount"] = 0.0
        if "description" not in df.columns:
            df["description"] = "No description provided"
        if "transaction_id" not in df.columns:
            df["transaction_id"] = None

        return df

    def _standardize_catalog_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize catalog columns."""
        col_map = {
            "itemid": "product_id",
            "item_id": "product_id",
            "productid": "product_id",
            "ProductID": "product_id",
            "StockCode": "product_id",
            "category_code": "category",
            "price": "unit_price",
            "UnitPrice": "unit_price",
        }
        rename_dict = {col: col_map[col] for col in df.columns if col in col_map}
        if rename_dict:
            df = df.rename(columns=rename_dict)

        if "product_id" not in df.columns:
            raise ValueError("Product catalog must contain 'product_id' column.")

        if "category" not in df.columns:
            df["category"] = "General"
        if "brand" not in df.columns:
            df["brand"] = "Generic"
        if "unit_price" not in df.columns:
            df["unit_price"] = 10.0
        if "description" not in df.columns:
            df["description"] = "No description provided"

        return df
