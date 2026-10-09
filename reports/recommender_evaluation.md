# Offline Recommender Evaluation Report

## 1. Methodology & Data Leakage Controls

- **Temporal Hold-Out Split**: Interactions were strictly split along the global timeline into Train (70%), Validation (15%), and Test (15%).
- **Evaluation Protocol**: Models were trained exclusively on training interactions. Ground truth evaluation measures actual unseen interactions occurring in the future test window.
- **Cold-Start & History Isolation**: Test-period interactions were never visible to item similarity graphs, ALS matrix factorization, or segment preferences.
- **Score Normalization**: Multi-model candidate scores were min-max normalized prior to weighted hybrid combination.

## 2. Comprehensive Model Benchmark


| model                             |   eval_users |   precision@5 |   recall@5 |   hit_rate@5 |   map@5 |   ndcg@5 |   precision@10 |   recall@10 |   hit_rate@10 |   map@10 |   ndcg@10 |   precision@20 |   recall@20 |   hit_rate@20 |   map@20 |   ndcg@20 |   diversity@10 |   novelty@10 |   catalog_coverage |
|:----------------------------------|-------------:|--------------:|-----------:|-------------:|--------:|---------:|---------------:|------------:|--------------:|---------:|----------:|---------------:|------------:|--------------:|---------:|----------:|---------------:|-------------:|-------------------:|
| Popularity Baseline               |         1852 |        0.0649 |     0.1013 |       0.2721 |  0.0796 |   0.1136 |         0.0425 |      0.128  |        0.3278 |   0.073  |    0.1129 |         0.0252 |      0.15   |        0.3661 |   0.0736 |    0.1185 |         0.8971 |      16.6096 |             0.0362 |
| Trending Baseline                 |         1852 |        0.0645 |     0.1014 |       0.2711 |  0.0796 |   0.1134 |         0.0424 |      0.127  |        0.3261 |   0.073  |    0.1127 |         0.0258 |      0.1535 |        0.3742 |   0.0739 |    0.1197 |         0.8961 |      16.6096 |             0.0362 |
| Item-Item Collaborative Filtering |         1852 |        0.0501 |     0.0737 |       0.223  |  0.0545 |   0.0819 |         0.0397 |      0.1134 |        0.3105 |   0.0532 |    0.0902 |         0.0335 |      0.1881 |        0.4336 |   0.0589 |    0.1144 |         0.5189 |      16.6007 |             0.9975 |
| ALS Matrix Factorization          |         1852 |        0.0255 |     0.0328 |       0.1134 |  0.0198 |   0.0336 |         0.0254 |      0.0671 |        0.2009 |   0.0214 |    0.0445 |         0.0244 |      0.1308 |        0.3299 |   0.0264 |    0.0656 |         0.4045 |      16.6007 |             1      |
| TruncatedSVD Baseline             |         1852 |        0.0243 |     0.0303 |       0.1112 |  0.0172 |   0.0307 |         0.0239 |      0.0593 |        0.1928 |   0.0185 |    0.04   |         0.0231 |      0.1167 |        0.3072 |   0.0232 |    0.0596 |         0.3081 |      16.6007 |             0.8087 |
| Hybrid Engine                     |         1852 |        0.0532 |     0.0761 |       0.2311 |  0.0646 |   0.093  |         0.0398 |      0.116  |        0.311  |   0.0618 |    0.0993 |         0.0335 |      0.1895 |        0.4401 |   0.0672 |    0.1231 |         0.5856 |      16.6096 |             1      |
| Hybrid (Sparse Cohort)            |           88 |        0.0273 |     0.0909 |       0.1364 |  0.0441 |   0.0614 |         0.0239 |      0.1629 |        0.2273 |   0.0549 |    0.0875 |         0.0148 |      0.2027 |        0.2614 |   0.0586 |    0.0986 |         0.6035 |      16.6096 |             0.7712 |
| Hybrid (Mature Cohort)            |         1764 |        0.0545 |     0.0753 |       0.2358 |  0.0657 |   0.0946 |         0.0406 |      0.1137 |        0.3152 |   0.0621 |    0.0999 |         0.0345 |      0.1888 |        0.449  |   0.0677 |    0.1243 |         0.5847 |      16.6096 |             1      |


## 3. Metric Interpretation & Key Insights

- **Precision@K & Recall@K**: Measure accuracy and coverage of ground-truth future items in top-K recommendations.
- **NDCG@K**: Evaluates ranking quality, heavily penalizing relevant items placed lower in the recommended list.
- **Catalog Coverage**: Proportion of all distinct items in the catalog recommended at least once, highlighting long-tail discovery.
- **Intra-List Diversity**: Measures cross-category diversity within user recommendation sets, preventing single-category concentration.
- **Novelty@10**: Information-theoretic measure of recommendation serendipity vs popular bias.
