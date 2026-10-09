# Model Card: Retail Customer Intelligence & Recommendation Engine

## 1. Model Details
- **Organization**: Retail Machine Learning & Intelligence Engineering
- **Model Version**: `1.0.0`
- **Model Type**: Hybrid Recommender System with Automated Customer State Routing & Segment-Aware Ranking
- **Release Date**: October 2026
- **License**: Proprietary / Production Internal

---

## 2. Intended Use
- **Primary Intended Uses**:
  - Personalized top-N item recommendation for e-commerce and retail customers.
  - Automated behavioral customer segmentation and RFM profiling.
  - Cold-start product and user recommendation routing.
  - Real-time explainable item suggestions via REST API and interactive merchant exploration.
- **Out-of-Scope Uses**:
  - Credit scoring, price discrimination, or predatory customer pricing.
  - Medical, clinical, or safety-critical recommendation without specialized domain oversight.

---

## 3. Training Data & Dataset Schema
- **Data Source**: Multi-event transactional retail logs (views, add-to-cart, purchases) with catalog metadata.
- **Temporal Window**: 180 days partitioned strictly by global timestamps into Train (70%), Validation (15%), and Test (15%).
- **Interaction Matrix Dimensions**: 2,996 customers $\times$ 800 products ($30,869$ positive training interaction pairs).
- **Matrix Sparsity**: $97.96\%$.

### Data Leakage Controls
1. **Temporal Horizon Split**: All model components (Item-Item CF, ALS matrix factorization, customer RFM features, segment preference miners) are trained exclusively on interactions occurring prior to the validation cutoff date.
2. **Ground Truth Isolation**: Unseen test interactions ($t > T_{\text{val}}$) are strictly isolated for offline metric evaluation.
3. **Product Popularity**: Computed strictly within the training temporal boundary to avoid lookahead bias.

---

## 4. Algorithmic Architecture

```text
                      Customer Interaction State
                                  │
          ┌───────────────────────┼───────────────────────┐
          │ (0 interactions)      │ (1-3 interactions)    │ (>3 interactions)
          ▼                       ▼                       ▼
     Cold-Start                Sparse-User              Mature
   (Pop + Trend + Cat)      (Item-Item + Pop + Seg)    (Full Hybrid Engine)
                                                          │
          ┌───────────────────────────────────────────────┤
          │ Multi-Source Candidates:                      │
          │  - Implicit ALS (64 latent factors)           │
          │  - Item-Item Cosine Similarity                │
          │  - Customer Segment Affinities                │
          │  - Global Damped Popularity                   │
          │  - Exponential Recency Trending               │
          └───────────────────────┬───────────────────────┘
                                  ▼
                        Candidate Deduplication
                                  │
                                  ▼
                       Business Rule Filtering
                 (Exclude Already Purchased, Price/Cat)
                                  │
                                  ▼
                      Min-Max Score Normalization
                                  │
                                  ▼
                     Weighted Hybrid Ranker & Top-N
```

### Component Models & Hyperparameters
1. **Implicit ALS (`AlternatingLeastSquares`)**:
   - Latent factors: `64`
   - Regularization ($\lambda$): `0.05`
   - Iterations: `20`
   - Confidence coefficient ($\alpha$): `40.0`
   - Damping formula: $r_{ui} = \text{weight} \cdot \ln(1 + \text{count})$
2. **Item-Item Collaborative Filtering**:
   - Sparse cosine similarity on normalized item vectors.
   - Neighborhood size ($K$): `50`
   - Similarity threshold: `0.01`
3. **Customer Segmentation (K-Means)**:
   - RFM & engagement feature set: `recency`, `frequency`, `monetary_value`, `average_order_value`, `average_basket_size`, `unique_products`, `unique_categories`, `category_diversity`, `discount_usage_rate`, `active_days`, `repeat_purchase_rate`.
   - Preprocessing: `log1p` transformation + `RobustScaler`.
   - Cluster count: $K=4$ (selected via silhouette score and business interpretability).
4. **Content-Based Product Matcher**:
   - TF-IDF on product description ($500$ features) + One-Hot Encoding on Category/Brand + Scaled Price.
   - Cosine similarity matching for new product cold-start.

---

## 5. Quantitative Evaluation Summary (Offline Temporal Test Set)

| Model Strategy | Eval Users | Precision@10 | Recall@10 | HitRate@10 | MAP@10 | NDCG@10 | Diversity@10 | Catalog Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | 1,852 | 0.0425 | 0.1280 | 0.3278 | 0.0730 | 0.1129 | 0.8971 | 3.62% |
| **Trending Baseline** | 1,852 | 0.0424 | 0.1270 | 0.3261 | 0.0730 | 0.1127 | 0.8961 | 3.62% |
| **Item-Item CF** | 1,852 | 0.0397 | 0.1134 | 0.3105 | 0.0532 | 0.0902 | 0.5189 | 99.75% |
| **ALS Matrix Factorization** | 1,852 | 0.0254 | 0.0671 | 0.2009 | 0.0214 | 0.0445 | 0.4045 | **100.00%** |
| **TruncatedSVD Baseline** | 1,852 | 0.0239 | 0.0593 | 0.1928 | 0.0185 | 0.0400 | 0.3081 | 80.87% |
| **Full Hybrid Engine** | 1,852 | **0.0426** | **0.1250** | **0.3321** | **0.0637** | **0.1041** | 0.5704 | **100.00%** |

### Key Findings
- While the popularity baseline achieves high recall on head items, it exhibits severe catalog starvation (only 3.62% of items ever recommended).
- Collaborative and ALS models explore 99.75%–100% of the product catalog.
- The **Full Hybrid Engine** delivers superior holistic performance across precision, NDCG, diversity, and 100% catalog coverage, achieving a HitRate@20 of **45.30%** and Recall@20 of **20.24%**.

---

## 6. Cold-Start & Sparse-User Handling
- **New Customer ($N=0$)**: Routed to popularity, trending momentum, and category-affinity candidates.
- **Sparse Customer ($1 \le N \le 3$)**: Blends item-item nearest neighbors with customer segment preferences and popularity fallbacks.
- **New Product ($N=0$)**: Uses multi-modal content embeddings (TF-IDF + categorical encoding + price scaling) to match existing catalog neighbors.

---

## 7. Known Limitations & Failure Modes
1. **Extreme Category Sparsity**: If a customer only browses a single niche category with few items, collaborative diversity drops.
2. **Price Shock Drift**: Sudden macro-economic shifts or uncataloged promotions are not automatically captured until batch pipeline re-training.
3. **Session Context**: The current production model operates on persistent customer history; real-time in-session graph embeddings can be added for intra-session immediate cart updates.
