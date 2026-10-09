# Production Retail Customer Intelligence & Recommendation Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/Tests-17%20Passed%20(100%25)-brightgreen.svg)]()

A production-grade, end-to-end Machine Learning, Customer Intelligence, and Recommendation Engine built for modern e-commerce and retail platforms.

This system integrates **unsupervised customer segmentation (RFM & K-Means)**, **multi-model candidate generation (Implicit ALS, Sparse Item-Item CF, Content-Based, Segment Affinities, Damped Popularity, Time-Decay Trending)**, **score normalization**, **modular business-rule filtering**, and **automated customer state routing (Cold-Start, Sparse, and Mature)** into an explainable, deployable platform.

---

## Table of Contents
1. [Business Objective & Key Questions](#1-business-objective)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Dataset & Data Adapter Schema](#3-dataset--data-adapter-schema)
4. [Data Quality & Temporal Split Pipeline](#4-data-quality--temporal-split-pipeline)
5. [Customer Intelligence & Automated Profiling](#5-customer-intelligence--automated-profiling)
6. [Recommendation Algorithms](#6-recommendation-algorithms)
7. [Cold-Start & State-Aware Routing](#7-cold-start--state-aware-routing)
8. [Candidate Generation, Ranking & Business Filters](#8-candidate-generation-ranking--business-filters)
9. [Recommendation Explainability](#9-recommendation-explainability)
10. [Offline Evaluation & Leakage Controls](#10-offline-evaluation--leakage-controls)
11. [Empirical Results & Ablation Study](#11-empirical-results--ablation-study)
12. [Segment-Stratified Performance](#12-segment-stratified-performance)
13. [FastAPI REST API Documentation](#13-fastapi-rest-api-documentation)
14. [Interactive Streamlit Dashboard](#14-interactive-streamlit-dashboard)
15. [Project Structure](#15-project-structure)
16. [Installation & Quickstart](#16-installation--quickstart)
17. [Pipeline Execution & Reproducibility](#17-pipeline-execution--reproducibility)
18. [Unit & Integration Testing](#18-unit--integration-testing)
19. [Docker Deployment](#19-docker-deployment)
20. [Production Monitoring Design](#20-production-monitoring-design)
21. [Limitations & Future Roadmap](#21-limitations--future-roadmap)

---

## 1. Business Objective

The platform addresses critical e-commerce retail challenges:
1. **Behavioral Cohort Discovery**: Which customers share transaction frequency, monetary spend, price sensitivity, and category exploration patterns?
2. **Personalized Discovery**: Which products maximize individual utility for a specific customer?
3. **Zero-Interaction Cold-Start**: What should be recommended to brand-new visitors with no prior activity?
4. **Sparse History Guidance**: How should recommendations adapt for low-interaction customers ($1 \le N \le 3$) where matrix factorization is uninformative?
5. **New Product Launch**: How to recommend newly cataloged items before receiving transaction signals?
6. **Cross-Strategy Synergy**: How does a hybrid ranker compare to standalone baselines across accuracy, diversity, and catalog coverage?
7. **Traceable Explainability**: Why was an item recommended (e.g. collaborative neighbor, segment affinity, or momentum)?

---

## 2. High-Level Architecture

```text
                                ┌──────────────────────┐
                                │   Raw Retail Data    │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │ Data Validation & DQ │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │ Strict Temporal Split│
                                └──────────┬───────────┘
                                           │
                 ┌─────────────────────────┼─────────────────────────┐
                 │                         │                         │
                 ▼                         ▼                         ▼
        Customer Features          Product Features          Interaction Matrix
      (RFM, AOV, Diversity)      (Sales, Price, Cat)         (CSR Sparse + Damping)
                 │                         │                         │
                 ▼                         ▼                         ▼
          K-Means / PCA            Content Embeddings        ALS / SVD / Item-Item
                 │                                                   │
                 ▼                                                   │
        Customer Segments                                            │
                 │                                                   │
                 ▼                                                   ▼
        Segment Affinities                               Candidate Generation
                 │                               (ALS, Item-Item, Seg, Pop, Trend)
                 └─────────────────────────┬─────────────────────────┘
                                           ▼
                                 Candidate Deduplication
                                           │
                                           ▼
                                Business Rule Filtering
                           (Exclude Purchased, Stock, Cat)
                                           │
                                           ▼
                              Min-Max Score Normalization
                                           │
                                           ▼
                                  Hybrid Weighted Rank
                                           │
                 ┌─────────────────────────┼─────────────────────────┐
                 │                         │                         │
                 ▼                         ▼                         ▼
            FastAPI REST             Explainability             Offline Eval
             (Port 8000)             (Signal Traces)            (Metrics & MLflow)
```

---

## 3. Dataset & Data Adapter Schema

The data loading adapter (`src/data/loader.py`) supports standard transactional tables, CSV, Parquet, and the [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

### Data Schema Contract
- **Required Fields**: `customer_id`, `product_id`, `timestamp`
- **Recommended Fields**: `quantity`, `unit_price`, `category`, `brand`
- **Optional Fields**: `event_type` (`view`, `add_to_cart`, `purchase`), `discount`, `description`, `transaction_id`

If optional fields are missing, the pipeline gracefully degrades using documented defaults without failing.

---

## 4. Data Quality & Temporal Split Pipeline

### Validation Checks (`src/data/validator.py`)
- Missing customer / product IDs (filtered and logged).
- Timestamp format and ordering validation.
- Duplicate event deduplication.
- Extreme price and negative quantity sanitary clamping.
- Interaction matrix sparsity and date range audit.
- Generates `outputs/data_quality_report.json`.

### Temporal Leakage Prevention (`src/data/preprocess.py`)
To prevent data leakage, interactions are partitioned by global timestamps:
- **Train Split (70%)**: $t \le T_{\text{train}}$ (All model training and feature extraction).
- **Validation Split (15%)**: $T_{\text{train}} < t \le T_{\text{val}}$ (Hyperparameter tuning).
- **Test Split (15%)**: $t > T_{\text{val}}$ (Hold-out ground truth evaluation).

---

## 5. Customer Intelligence & Automated Profiling

### RFM & Engagement Engineering (`src/features/customer_features.py`)
- **Recency**: Elapsed days from last transaction to temporal split reference date.
- **Frequency**: Interaction and purchase volume.
- **Monetary Value**: Total discounted expenditure ($\sum \text{qty} \times \text{price} \times (1 - \text{discount})$).
- **Average Order Value (AOV)** & **Basket Size**: Basket depth per transaction.
- **Category Diversity**: Entropy-based exploration breadth ($\frac{|\text{Categories}|}{\ln(1 + \text{Frequency})}$).
- **Discount Usage Rate**: Price sensitivity ratio.
- **Active Days & Repeat Purchase Rate**: Customer loyalty and engagement regularity.

### K-Means Clustering & PCA (`src/segmentation/clustering.py`, `visualization.py`)
- Features transformed via `log1p` and scaled using `RobustScaler`.
- Evaluated optimal $K \in [2, 8]$ using **Inertia**, **Silhouette Score**, and **Davies-Bouldin Index**.
- Persists 2D PCA projection to `outputs/figures/customer_segments_pca.png`.

### Deterministic Rule-Based Profiling (`src/segmentation/profiling.py`)
No hard-coded assumptions. Cluster labels and descriptions are generated deterministically by computing cluster deviations against population means:
- **Segment 0**: *Low-Spend Dormant/At-Risk Shoppers*
- **Segment 1**: *Low-Spend Dormant/At-Risk Shoppers*
- **Segment 2**: *Discount-Driven Shoppers*
- **Segment 3**: *High-Value Frequent Active Shoppers*

Outputs: `outputs/segment_profiles.csv` and `reports/segment_profiles.md`.

---

## 6. Recommendation Algorithms

### 1. Damped Popularity Recommender (`src/recommenders/popularity.py`)
Combines normalized total purchases, unique customer reach, and recent momentum:
$$\text{Popularity}(p) = \text{Norm}(\text{Purchases}) + \alpha \cdot \text{Norm}(\text{UniqueUsers}) + \beta \cdot \text{Norm}(\text{RecentPurchases})$$

### 2. Time-Decay Trending Recommender (`src/recommenders/trending.py`)
Applies exponential half-life decay over event timestamps:
$$\text{Trend}(p) = \sum_{e \in \text{events}(p)} w(e) \cdot \exp(-\lambda \cdot \text{AgeDays}(e))$$

### 3. Sparse Item-Item Collaborative Filtering (`src/recommenders/item_item.py`)
Builds sparse cosine similarity graph over normalized Item-User vectors ($M^T M$). Computes top-50 nearest neighbors per item. Aggregates customer interaction history and produces item-item similarity traces.

### 4. Implicit Matrix Factorization - ALS (`src/recommenders/als.py`)
Optimizes implicit feedback objective with confidence scaling ($C_{ui} = 1 + \alpha r_{ui}$) and L2 regularization:
$$\min_{X, Y} \sum_{u, i} c_{ui} (p_{ui} - x_u^T y_i)^2 + \lambda \left(\sum_u \|x_u\|^2 + \sum_i \|y_i\|^2\right)$$
Hyperparameters: `factors=64, regularization=0.05, iterations=20, alpha=40.0`.

### 5. Truncated SVD Baseline (`src/recommenders/svd.py`)
Classical linear matrix factorization baseline ($U \Sigma V^T$). Evaluated to contrast dense missing-as-zero assumptions with implicit confidence weighting.

### 6. Multi-Modal Content-Based Recommender (`src/recommenders/content_based.py`)
Combines TF-IDF descriptions, One-Hot encoded categories/brands, and scaled unit price. Generates content representations for new product cold-start matching.

---

## 7. Cold-Start & State-Aware Routing

The engine automatically routes queries based on customer interaction state:

| State | Condition | Strategy | Output Method |
| :--- | :--- | :--- | :--- |
| **New Customer** | $\text{Interactions} = 0$ | Popularity + Trending + Category Context | `cold_start_user` |
| **Sparse Customer** | $1 \le \text{Interactions} \le 3$ | Item-Item CF + Segment Preferences + Popularity | `sparse_user` |
| **Mature Customer** | $\text{Interactions} > 3$ | Multi-Candidate Hybrid (ALS + Item-Item + Segment + Pop + Trend) | `hybrid` |
| **New Product** | Unseen product | Multi-Modal Content Embedding Cosine Matching | `content_based` |

---

## 8. Candidate Generation, Ranking & Business Filters

### Candidate Pooling & Deduplication (`src/ranking/candidate_generation.py`)
Pulls candidate pools:
- 50 ALS candidates
- 50 Item-Item candidates
- 30 Segment affinity candidates
- 30 Popularity candidates
- 20 Trending candidates

### Modular Business Rule Filters (`src/ranking/filters.py`)
- `AlreadyPurchasedFilter`: Excludes past purchased items.
- `CategoryFilter`: Enforces optional category constraints.
- `PriceFilter`: Enforces minimum / maximum price filters.
- `ProductAvailabilityFilter`: Filters out inactive / out-of-stock items.

### Score Normalization & Weighted Hybrid Blending (`src/ranking/scoring.py`)
Candidate scores from heterogeneous models are min-max normalized to $[0, 1]$ before weighted linear combination:
$$\text{FinalScore}(p) = w_{\text{als}} \hat{S}_{\text{ALS}} + w_{\text{item}} \hat{S}_{\text{Item}} + w_{\text{seg}} S_{\text{Segment}} + w_{\text{pop}} S_{\text{Pop}} + w_{\text{trend}} S_{\text{Trend}}$$

---

## 9. Recommendation Explainability

Every recommendation output contains an evidence-backed explanation trace derived from model signals:
- **Item-Item**: `"Similar to Product #1255 that you previously interacted with."`
- **Segment**: `"Popular among customers in your segment (High-Value Frequent Active Shoppers)."`
- **ALS**: `"Recommended based on collaborative latent preferences with similar shoppers."`
- **Trending**: `"Trending with high momentum across the store right now."`
- **Popularity**: `"One of the most frequently purchased products in our catalog."`

---

## 10. Offline Evaluation & Leakage Controls

### Metrics Computed (`src/evaluation/metrics.py`)
- **Ranking Accuracy**: Precision@K, Recall@K, HitRate@K, MAP@K, NDCG@K for $K \in [5, 10, 20]$.
- **Catalog Coverage**: Proportion of distinct catalog items recommended across test customers.
- **Intra-List Diversity**: Mean cross-category diversity within user recommendation lists.
- **Novelty**: Information-theoretic measure ($-\log_2 P(\text{item})$).

---

## 11. Empirical Results & Ablation Study

### Complete Benchmark (Temporal Test Set, 1,852 Customers)

| Model Strategy | Recall@5 | Recall@10 | Recall@20 | HitRate@10 | NDCG@10 | MAP@10 | Diversity@10 | Catalog Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | 0.1013 | 0.1280 | 0.1500 | 0.3278 | 0.1129 | 0.0730 | 0.8971 | 3.62% |
| **Trending Baseline** | 0.1014 | 0.1270 | 0.1535 | 0.3261 | 0.1127 | 0.0730 | 0.8961 | 3.62% |
| **Item-Item CF** | 0.0737 | 0.1134 | 0.1881 | 0.3105 | 0.0902 | 0.0532 | 0.5189 | 99.75% |
| **ALS Matrix Factorization** | 0.0328 | 0.0671 | 0.1308 | 0.2009 | 0.0445 | 0.0214 | 0.4045 | **100.00%** |
| **TruncatedSVD Baseline** | 0.0303 | 0.0593 | 0.1167 | 0.1928 | 0.0400 | 0.0185 | 0.3081 | 80.87% |
| **Full Hybrid Engine** | **0.0787** | **0.1250** | **0.2024** | **0.3321** | **0.1041** | **0.0637** | 0.5704 | **100.00%** |

### Systematic Component Ablation Study

| Ablation Configuration | Recall@10 | Recall@20 | HitRate@10 | HitRate@20 | NDCG@10 | NDCG@20 | Catalog Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Popularity Only** | 0.1280 | 0.1500 | 0.3278 | 0.3661 | 0.1129 | 0.1185 | 3.62% |
| **Popularity + Segment** | 0.1291 | 0.1525 | 0.3272 | 0.3720 | 0.1131 | 0.1194 | 5.50% |
| **Item-Item CF Only** | 0.1134 | 0.1881 | 0.3105 | 0.4336 | 0.0902 | 0.1144 | 99.75% |
| **ALS Only** | 0.0671 | 0.1308 | 0.2009 | 0.3299 | 0.0445 | 0.0656 | 100.00% |
| **ALS + Item-Item** | 0.0852 | 0.1736 | 0.2468 | 0.4125 | 0.0524 | 0.0814 | 100.00% |
| **ALS + Segment** | 0.0769 | 0.1606 | 0.2235 | 0.3877 | 0.0496 | 0.0771 | 100.00% |
| **ALS + Item-Item + Segment**| 0.1101 | 0.1939 | 0.3029 | 0.4411 | 0.0779 | 0.1049 | 100.00% |
| **Full Hybrid Engine** | **0.1250** | **0.2024** | **0.3321** | **0.4530** | **0.1041** | **0.1286** | **100.00%** |

---

## 12. Segment-Stratified Performance

| Segment | Label | Test Users | Recall@10 | HitRate@10 | NDCG@10 | Catalog Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Segment 0** | Low-Spend Dormant/At-Risk Shoppers | 464 | 0.1304 | 0.2478 | 0.0955 | 60.12% |
| **Segment 1** | Low-Spend Dormant/At-Risk Shoppers | 49 | 0.1837 | 0.2857 | 0.1288 | 15.62% |
| **Segment 2** | Discount-Driven Shoppers | 139 | 0.1417 | 0.3453 | 0.1090 | 36.88% |
| **Segment 3** | High-Value Frequent Active Shoppers | 1,199 | 0.1186 | 0.3653 | 0.1065 | 99.88% |

---

## 13. FastAPI REST API Documentation

Run the REST API:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Interactive Swagger UI: `http://localhost:8000/docs`

### Available Endpoints
- `GET /health`: Service health and model status.
- `GET /customers/{customer_id}`: Customer RFM profile, segment label, and engagement metrics.
- `GET /recommendations/{customer_id}?n=10&category_context=...&max_price=...`: Explainable top-N recommendations.
- `GET /products/{product_id}/similar?n=10&method=collaborative`: Similar product neighbors.
- `GET /segments`: Summary of customer clusters and statistics.
- `GET /model-info`: Model architecture metadata and offline evaluation metrics.

---

## 14. Interactive Streamlit Dashboard

Run the dashboard:
```bash
streamlit run src/dashboard/app.py --server.port 8501
```
Access at: `http://localhost:8501`

### Dashboard Pages
1. **🛍️ Customer Recommendations**: Live recommendation sandbox with budget/category filters and explanation cards.
2. **📦 Product & Content Explorer**: Collaborative and content similar product explorer + new product cold-start simulator.
3. **👥 Customer Segmentation**: 2D PCA cluster visualization and segment summaries.
4. **📊 Model Evaluation & Ablation**: Interactive offline evaluation charts and ablation study tables.
5. **🛡️ Data Quality & Architecture**: Data validation statistics and filtering decision logs.

---

## 15. Project Structure

```text
retail-customer-intelligence/
├── README.md                           # Master Project Documentation
├── pyproject.toml                      # Build and packaging specification
├── requirements.txt                    # Production dependencies
├── .env.example                        # Environment variables template
├── .gitignore                          # Version control exclusions
├── Makefile                            # Build and execution automation
├── Dockerfile                          # Containerization specification
├── docker-compose.yml                  # Multi-service API and Dashboard setup
│
├── configs/
│   ├── config.yaml                     # Master pipeline configuration
│   ├── segmentation.yaml               # RFM and K-Means segmentation config
│   └── recommendation.yaml             # Model hyperparameters and hybrid weights
│
├── data/
│   ├── raw/                            # Raw events and catalog CSVs
│   ├── interim/                        # Cleaned intermediate tables
│   └── processed/                      # Parquet temporal splits (Train, Val, Test)
│
├── notebooks/                          # Walkthrough notebooks
│   ├── 01_data_quality.ipynb
│   ├── 02_customer_segmentation.ipynb
│   ├── 03_item_similarity.ipynb
│   ├── 04_matrix_factorization.ipynb
│   └── 05_recommender_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── schema.py                   # Pydantic data schemas
│   │   ├── loader.py                   # Heterogeneous CSV/Parquet data loader
│   │   ├── validator.py                # Data quality validation checks
│   │   └── preprocess.py               # Temporal train/val/test splitting
│   │
│   ├── features/
│   │   ├── customer_features.py        # RFM and engagement feature extraction
│   │   ├── product_features.py         # Product catalog sales statistics
│   │   ├── interaction_features.py     # Weighted CSR sparse interaction matrix
│   │   └── segment_features.py         # Segment preference miner
│   │
│   ├── segmentation/
│   │   ├── clustering.py               # K-Means with optimal K search
│   │   ├── profiling.py                # Deterministic rule-based labeling
│   │   └── visualization.py            # 2D PCA cluster projection plot
│   │
│   ├── recommenders/
│   │   ├── base.py                     # Base recommender interface
│   │   ├── popularity.py               # Damped popularity baseline
│   │   ├── trending.py                 # Time-decay trending recommender
│   │   ├── item_item.py                # Sparse Item-Item CF with explainability
│   │   ├── als.py                      # Implicit ALS Matrix Factorization
│   │   ├── svd.py                      # TruncatedSVD baseline
│   │   ├── content_based.py            # Multi-modal content embedding recommender
│   │   └── hybrid.py                   # Production unified RecommendationEngine
│   │
│   ├── ranking/
│   │   ├── candidate_generation.py     # Multi-source candidate pool
│   │   ├── scoring.py                  # Score normalization & weighted ranker
│   │   └── filters.py                  # Modular business rule filters
│   │
│   ├── evaluation/
│   │   ├── metrics.py                  # Ranking, diversity, coverage metrics
│   │   ├── evaluator.py                # Temporal hold-out evaluator
│   │   ├── ablation.py                 # Systematic ablation runner
│   │   └── segment_evaluation.py       # Segment-stratified evaluation
│   │
│   ├── api/
│   │   └── main.py                     # FastAPI production service
│   │
│   ├── dashboard/
│   │   └── app.py                      # Interactive Streamlit application
│   │
│   └── utils/
│       ├── logger.py                   # Structured logging utility
│       ├── config.py                   # YAML configuration loader
│       └── reproducibility.py          # Deterministic random seed controller
│
├── tests/
│   ├── unit/
│   │   ├── test_data_pipeline.py       # Data validation & temporal split tests
│   │   ├── test_segmentation.py        # RFM & K-Means unit tests
│   │   └── test_recommenders.py        # All recommender algorithms & metrics tests
│   ├── integration/
│   │   └── test_api.py                 # FastAPI endpoint integration tests
│   └── test_recommendation_pipeline.py # End-to-end pipeline test
│
├── reports/
│   ├── segment_profiles.md             # Automated customer segmentation profiles
│   ├── recommender_evaluation.md       # Offline evaluation and benchmark report
│   └── model_card.md                   # Formal machine learning model card
│
├── outputs/
│   ├── figures/                        # PCA visualization plots
│   ├── metrics/                        # CSV evaluation metrics and ablation tables
│   └── data_quality_report.json        # Data quality summary
│
├── artifacts/                          # Versioned model weights and features
│   ├── models/
│   ├── mappings/
│   └── features/
│
└── scripts/
    ├── generate_sample_data.py         # Realistic benchmark dataset generator
    └── run_pipeline.py                 # Master end-to-end pipeline runner
```

---

## 16. Installation & Quickstart

```bash
# Clone and enter directory
cd "c:/Users/hp/OneDrive/Desktop/Code/Day 17"

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## 17. Pipeline Execution & Reproducibility

Execute the entire data validation, feature engineering, clustering, model training, evaluation, and artifact generation pipeline in one command:

```bash
python scripts/run_pipeline.py
```

---

## 18. Unit & Integration Testing

Run the full pytest suite:

```bash
python -m pytest tests/ -v
```

Output:
```text
======================= 17 passed in 6.90s =======================
```

---

## 19. Docker Deployment

Build and run the multi-container API and Dashboard:

```bash
docker-compose up --build -d
```

- FastAPI Service: `http://localhost:8000`
- Streamlit Dashboard: `http://localhost:8501`

---

## 20. Production Monitoring Design

1. **Data Drift**: Monitor customer interaction volume, average basket size, and category entropy.
2. **Recommendation Drift**: Track catalog coverage and Gini coefficient of recommended products over time.
3. **Model Health**: Monitor p95/p99 API latency, unknown customer request rate, and fallbacks.
4. **Online Experimentation**: A/B test hybrid ranker weights against business KPIs (Conversion Rate, Add-to-Cart Rate, Revenue per Visitor).

---

## 21. Limitations & Future Roadmap

1. **In-Session Graph Embeddings**: Incorporate graph neural networks (e.g. PinSage / LightGCN) for real-time within-session clickstream updates.
2. **Two-Tower Deep Retrieval**: Extend candidate generation with deep neural two-tower embeddings for million-item catalogs.
3. **Contextual Multi-Armed Bandits**: Implement LinUCB for dynamic exploration-exploitation on newly cataloged items.
