"""Content-Based Product Recommender with text, categorical, and numerical multi-modal embeddings."""

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
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from src.recommenders.base import BaseRecommender
from src.utils.logger import get_logger

logger = get_logger("content_based_recommender")


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender utilizing product metadata for new-product cold-start and similarity matching."""

    def __init__(
        self,
        tfidf_max_features: int = 500,
        k_similar: int = 30,
    ):
        super().__init__(name="content_based")
        self.tfidf_max_features = tfidf_max_features
        self.k_similar = k_similar
        
        self.tfidf_ = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english")
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.scaler_ = StandardScaler()

        self.product_ids_: List[Union[int, str]] = []
        self.product_to_idx_: Dict[Union[int, str], int] = {}
        self.idx_to_product_: Dict[int, Union[int, str]] = {}
        self.content_matrix_: Optional[sp.csr_matrix] = None
        self.similarity_dict_: Dict[int, List[Tuple[int, float]]] = {}
        self.product_metadata_: Dict[Union[int, str], Dict[str, Any]] = {}

    def fit(self, catalog_df: pd.DataFrame) -> "ContentBasedRecommender":
        """Transform product catalog metadata into unified sparse content feature representation."""
        logger.info(f"Training Content-Based Recommender on {len(catalog_df)} catalog items...")
        df = catalog_df.drop_duplicates(subset=["product_id"]).reset_index(drop=True).copy()

        self.product_ids_ = df["product_id"].tolist()
        self.product_to_idx_ = {pid: idx for idx, pid in enumerate(self.product_ids_)}
        self.idx_to_product_ = {idx: pid for idx, pid in enumerate(self.product_ids_)}

        # Store metadata for quick querying
        for _, row in df.iterrows():
            self.product_metadata_[row["product_id"]] = {
                "category": row.get("category", "General"),
                "brand": row.get("brand", "Generic"),
                "unit_price": float(row.get("unit_price", 10.0)),
                "description": str(row.get("description", "")),
            }

        # 1. Text TF-IDF on description / title
        descriptions = df["description"].fillna("").astype(str).tolist()
        if any(len(d.strip()) > 0 for d in descriptions):
            text_features = self.tfidf_.fit_transform(descriptions)
        else:
            text_features = sp.csr_matrix((len(df), 1))

        # 2. Categorical One-Hot on category and brand
        cat_cols = []
        if "category" in df.columns:
            cat_cols.append("category")
        if "brand" in df.columns:
            cat_cols.append("brand")
        
        if cat_cols:
            cat_features = self.ohe_.fit_transform(df[cat_cols].fillna("Unknown"))
        else:
            cat_features = sp.csr_matrix((len(df), 1))

        # 3. Numerical Scaler on unit_price
        if "unit_price" in df.columns:
            price_vals = df[["unit_price"]].fillna(df["unit_price"].median()).values
            scaled_price = self.scaler_.fit_transform(price_vals)
            price_sparse = sp.csr_matrix(scaled_price)
        else:
            price_sparse = sp.csr_matrix((len(df), 1))

        # Combine sparse blocks
        self.content_matrix_ = sp.hstack([text_features, cat_features, price_sparse]).tocsr()

        # Compute cosine similarities
        logger.info("Computing content cosine similarities...")
        sim_mat = cosine_similarity(self.content_matrix_, dense_output=False)
        
        self.similarity_dict_ = {}
        for idx in range(len(self.product_ids_)):
            row = sim_mat[idx].toarray().ravel()
            row[idx] = 0.0  # zero self-similarity
            top_indices = np.argsort(row)[::-1][:self.k_similar]
            self.similarity_dict_[idx] = [
                (int(t_idx), float(row[t_idx])) for t_idx in top_indices if row[t_idx] > 0.0
            ]

        self.is_fitted = True
        logger.info(f"Content-Based model fitted with feature dimension {self.content_matrix_.shape[1]}.")
        return self

    def get_similar_products(
        self,
        product_id: Union[int, str],
        n: int = 10,
    ) -> List[Tuple[Union[int, str], float]]:
        """Find products with most similar content attributes to the target product."""
        if not self.is_fitted:
            raise RuntimeError("ContentBasedRecommender is not fitted.")

        p_idx = self.product_to_idx_.get(product_id)
        if p_idx is None:
            return []

        sim_list = self.similarity_dict_.get(p_idx, [])[:n]
        return [(self.idx_to_product_[idx], score) for idx, score in sim_list]

    def recommend_for_new_product(
        self,
        new_product_metadata: Dict[str, Any],
        n: int = 10,
    ) -> List[Tuple[Union[int, str], float]]:
        """Cold-start strategy for a brand-new unseen product: embed attributes and find similar existing catalog items."""
        if not self.is_fitted:
            raise RuntimeError("ContentBasedRecommender is not fitted.")

        desc = [str(new_product_metadata.get("description", ""))]
        text_vec = self.tfidf_.transform(desc)

        cat = str(new_product_metadata.get("category", "Unknown"))
        brand = str(new_product_metadata.get("brand", "Unknown"))
        cat_df = pd.DataFrame([{"category": cat, "brand": brand}])
        cat_vec = self.ohe_.transform(cat_df)

        price = float(new_product_metadata.get("unit_price", 10.0))
        price_vec = sp.csr_matrix(self.scaler_.transform([[price]]))

        query_vec = sp.hstack([text_vec, cat_vec, price_vec]).tocsr()
        sim_scores = cosine_similarity(query_vec, self.content_matrix_).ravel()

        top_indices = np.argsort(sim_scores)[::-1][:n]
        return [(self.idx_to_product_[idx], float(sim_scores[idx])) for idx in top_indices]

    def recommend(
        self,
        user_id: Union[int, str] = None,
        n: int = 10,
        filter_items: Optional[Set[Union[int, str]]] = None,
    ) -> List[Tuple[Union[int, str], float]]:
        """Base recommend interface (content similarity fallback)."""
        return []

    def explain(self, user_id: Union[int, str], item_id: Union[int, str]) -> Dict[str, Any]:
        """Explain content-based recommendation."""
        meta = self.product_metadata_.get(item_id, {})
        cat = meta.get("category", "General")
        brand = meta.get("brand", "Generic")
        return {
            "recommended_product": item_id,
            "method": "content_based",
            "category": cat,
            "brand": brand,
            "explanation": f"Recommended because its attributes ({cat}, {brand}) closely match your profile preferences.",
        }

    def save(self, filepath: str = "artifacts/models/content_based_model.joblib") -> None:
        """Persist content-based model."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tfidf_max_features": self.tfidf_max_features,
            "k_similar": self.k_similar,
            "tfidf_": self.tfidf_,
            "ohe_": self.ohe_,
            "scaler_": self.scaler_,
            "product_ids_": self.product_ids_,
            "product_to_idx_": self.product_to_idx_,
            "idx_to_product_": self.idx_to_product_,
            "content_matrix_": self.content_matrix_,
            "similarity_dict_": self.similarity_dict_,
            "product_metadata_": self.product_metadata_,
        }
        joblib.dump(payload, path)
        logger.info(f"Saved Content-Based model to {path}")

    @classmethod
    def load(cls, filepath: str = "artifacts/models/content_based_model.joblib") -> "ContentBasedRecommender":
        """Load persisted Content-Based model."""
        payload = joblib.load(filepath)
        model = cls(
            tfidf_max_features=payload["tfidf_max_features"],
            k_similar=payload["k_similar"],
        )
        model.tfidf_ = payload["tfidf_"]
        model.ohe_ = payload["ohe_"]
        model.scaler_ = payload["scaler_"]
        model.product_ids_ = payload["product_ids_"]
        model.product_to_idx_ = payload["product_to_idx_"]
        model.idx_to_product_ = payload["idx_to_product_"]
        model.content_matrix_ = payload["content_matrix_"]
        model.similarity_dict_ = payload["similarity_dict_"]
        model.product_metadata_ = payload["product_metadata_"]
        model.is_fitted = True
        return model
