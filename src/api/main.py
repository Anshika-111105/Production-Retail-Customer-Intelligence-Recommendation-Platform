"""FastAPI Production Recommendation & Customer Intelligence Service."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from src.recommenders.hybrid import RecommendationEngine
from src.features.customer_features import CustomerFeatureEngineer
from src.utils.logger import get_logger
from src.utils.config import AppConfig

logger = get_logger("api_service")

app = FastAPI(
    title="Retail Customer Intelligence & Recommendation Platform",
    description="High-performance, modular e-commerce customer segmentation and hybrid recommendation engine.",
    version="1.0.0",
)

# Enable CORS for Streamlit dashboard and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine & Artifact State
engine: Optional[RecommendationEngine] = None
customer_features_df: Optional[pd.DataFrame] = None
segment_profiles_df: Optional[pd.DataFrame] = None
evaluation_metrics_df: Optional[pd.DataFrame] = None
config: Optional[AppConfig] = None


@app.on_event("startup")
def load_artifacts():
    """Load engine and feature tables on service startup."""
    global engine, customer_features_df, segment_profiles_df, evaluation_metrics_df, config
    logger.info("Initializing API and loading model artifacts...")
    config = AppConfig()
    
    art_path = Path("artifacts")
    if (art_path / "models" / "engine_metadata.joblib").exists():
        engine = RecommendationEngine.load("artifacts")
        logger.info("RecommendationEngine loaded successfully.")
    else:
        logger.warning("Artifacts not found in artifacts/. Running fallback loading or awaiting pipeline execution.")

    if Path("artifacts/features/customer_features.parquet").exists():
        customer_features_df = pd.read_parquet("artifacts/features/customer_features.parquet")
    
    if Path("outputs/segment_profiles.csv").exists():
        segment_profiles_df = pd.read_csv("outputs/segment_profiles.csv")
    
    if Path("outputs/metrics/evaluation_metrics.csv").exists():
        evaluation_metrics_df = pd.read_csv("outputs/metrics/evaluation_metrics.csv")


# --- Pydantic Response Schemas ---

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    num_customers: int
    num_products: int


class RecommendationItem(BaseModel):
    rank: int
    product_id: Union[int, str]
    score: float
    sources: List[str]
    source_scores: Dict[str, float] = Field(default_factory=dict)
    category: Optional[str] = "General"
    brand: Optional[str] = "Generic"
    unit_price: Optional[float] = 0.0
    explanation: str


class RecommendationResponse(BaseModel):
    customer_id: Union[int, str]
    segment_id: Optional[int]
    segment_label: str
    interaction_count: int
    recommendation_type: str
    recommendations: List[RecommendationItem]


class CustomerProfileResponse(BaseModel):
    customer_id: Union[int, str]
    segment_id: Optional[int]
    segment_label: str
    recency_days: float
    frequency: int
    monetary_value: float
    average_order_value: float
    average_basket_size: float
    unique_products: int
    unique_categories: int
    discount_usage_rate: float
    active_days: int
    repeat_purchase_rate: float


class SimilarProductItem(BaseModel):
    product_id: Union[int, str]
    similarity_score: float
    category: Optional[str] = "General"
    brand: Optional[str] = "Generic"
    unit_price: Optional[float] = 0.0


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint exposing system readiness and model loading state."""
    is_ready = engine is not None and engine.id_builder is not None
    n_cust = len(customer_features_df) if customer_features_df is not None else 0
    n_prod = len(engine.product_metadata) if engine else 0
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        version="1.0.0",
        models_loaded=is_ready,
        num_customers=n_cust,
        num_products=n_prod,
    )


@app.get("/customers/{customer_id}", response_model=CustomerProfileResponse)
def get_customer_profile(customer_id: int):
    """Retrieve customer RFM profile, segment membership, and behavioral metrics."""
    if customer_features_df is None or customer_features_df.empty:
        raise HTTPException(status_code=503, detail="Customer feature table not loaded.")

    row = customer_features_df[customer_features_df["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID {customer_id} not found in database.")

    r = row.iloc[0]
    seg_id = engine.customer_segments.get(customer_id) if engine else None
    seg_label = engine.segment_labels.get(seg_id, "Unassigned") if (engine and seg_id is not None) else "Unassigned"

    return CustomerProfileResponse(
        customer_id=customer_id,
        segment_id=seg_id,
        segment_label=seg_label,
        recency_days=round(float(r["recency"]), 2),
        frequency=int(r["frequency"]),
        monetary_value=round(float(r["monetary_value"]), 2),
        average_order_value=round(float(r["average_order_value"]), 2),
        average_basket_size=round(float(r["average_basket_size"]), 2),
        unique_products=int(r["unique_products"]),
        unique_categories=int(r["unique_categories"]),
        discount_usage_rate=round(float(r["discount_usage_rate"]), 3),
        active_days=int(r["active_days"]),
        repeat_purchase_rate=round(float(r["repeat_purchase_rate"]), 3),
    )


@app.get("/recommendations/{customer_id}", response_model=RecommendationResponse)
def get_recommendations(
    customer_id: int,
    n: int = Query(default=10, ge=1, le=50, description="Number of recommendations to return"),
    category_context: Optional[str] = Query(default=None, description="Optional category filter/context"),
    min_price: Optional[float] = Query(default=None, ge=0.0, description="Minimum price filter"),
    max_price: Optional[float] = Query(default=None, ge=0.0, description="Maximum price filter"),
):
    """Generate explainable, personalized top-N recommendations for a customer with automated routing."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine is not initialized.")

    context = {}
    if category_context:
        context["allowed_categories"] = [category_context]
    if min_price is not None:
        context["min_price"] = min_price
    if max_price is not None:
        context["max_price"] = max_price

    try:
        rec_output = engine.recommend(customer_id=customer_id, n=n, context=context)
        return rec_output
    except Exception as e:
        logger.error(f"Error generating recommendations for customer {customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal recommendation error: {str(e)}")


@app.get("/products/{product_id}/similar", response_model=List[SimilarProductItem])
def get_similar_products(
    product_id: int,
    n: int = Query(default=10, ge=1, le=50),
    method: str = Query(default="collaborative", pattern="^(collaborative|content)$"),
):
    """Find products similar to a given product using item-item CF or content attributes."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine is not initialized.")

    if method == "collaborative" and engine.item_item_model:
        sim_tuples = engine.item_item_model.get_similar_items(product_id, n=n)
    elif engine.content_model:
        sim_tuples = engine.content_model.get_similar_products(product_id, n=n)
    else:
        sim_tuples = []

    results = []
    for pid, score in sim_tuples:
        meta = engine.product_metadata.get(pid, {})
        results.append(SimilarProductItem(
            product_id=pid,
            similarity_score=round(float(score), 4),
            category=meta.get("category", "General"),
            brand=meta.get("brand", "Generic"),
            unit_price=meta.get("unit_price", 0.0),
        ))
    return results


@app.get("/segments")
def get_segment_profiles():
    """Retrieve customer segment summaries and cluster definitions."""
    if segment_profiles_df is None or segment_profiles_df.empty:
        raise HTTPException(status_code=503, detail="Segment profiles table not loaded.")
    return segment_profiles_df.to_dict(orient="records")


@app.get("/model-info")
def get_model_info():
    """Retrieve model training metadata, configurations, and offline evaluation benchmark metrics."""
    eval_records = evaluation_metrics_df.to_dict(orient="records") if evaluation_metrics_df is not None else []
    return {
        "platform": "Retail Customer Intelligence & Recommendation Platform",
        "version": "1.0.0",
        "algorithms": {
            "segmentation": "K-Means + Robust Scaler + Log1p Transformation",
            "collaborative_filtering": "Item-Item Cosine Similarity",
            "implicit_mf": "Alternating Least Squares (ALS)",
            "baseline_mf": "Truncated SVD",
            "cold_start": "Popularity + Time-Decay Trending + Content-Based",
            "hybrid": "Candidate Generation + Min-Max Score Normalization + Weighted Ranker",
        },
        "evaluation_metrics": eval_records,
    }
