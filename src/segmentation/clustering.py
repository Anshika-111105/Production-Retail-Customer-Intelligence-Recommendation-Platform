"""Customer Segmentation model with K-Means and optimal K diagnostics."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

from src.utils.logger import get_logger

logger = get_logger("clustering")


class CustomerClustering:
    """Trains K-Means clustering on customer features and evaluates optimal K."""

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
        n_init: int = 20,
        max_iter: int = 300,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        self.diagnostics_: Dict[str, Any] = {}

    def find_optimal_k(
        self,
        scaled_features: np.ndarray,
        k_range: List[int] = [2, 3, 4, 5, 6, 7, 8],
    ) -> Dict[str, Any]:
        """Evaluate inertia and silhouette scores over a range of K."""
        logger.info(f"Evaluating optimal K over range {k_range}...")
        results = {
            "k_values": [],
            "inertias": [],
            "silhouette_scores": [],
            "davies_bouldin_scores": [],
        }

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init, max_iter=self.max_iter)
            labels = km.fit_predict(scaled_features)
            sil = float(silhouette_score(scaled_features, labels, sample_size=min(5000, len(scaled_features))))
            db = float(davies_bouldin_score(scaled_features, labels))
            
            results["k_values"].append(k)
            results["inertias"].append(float(km.inertia_))
            results["silhouette_scores"].append(sil)
            results["davies_bouldin_scores"].append(db)
            logger.info(f"  K={k}: Inertia={km.inertia_:.2f}, Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}")

        # Best K by silhouette score
        best_k_idx = int(np.argmax(results["silhouette_scores"]))
        best_k = results["k_values"][best_k_idx]
        results["recommended_k_by_silhouette"] = best_k
        self.diagnostics_ = results
        logger.info(f"Recommended K by silhouette score: {best_k} (score: {results['silhouette_scores'][best_k_idx]:.4f})")
        return results

    def fit_predict(self, scaled_features: np.ndarray) -> np.ndarray:
        """Fit K-Means model and return cluster assignments."""
        logger.info(f"Fitting K-Means clustering with K={self.n_clusters}...")
        self.model.fit(scaled_features)
        labels = self.model.labels_

        sil = float(silhouette_score(scaled_features, labels, sample_size=min(5000, len(scaled_features))))
        db = float(davies_bouldin_score(scaled_features, labels))
        self.diagnostics_["final_k"] = self.n_clusters
        self.diagnostics_["final_inertia"] = float(self.model.inertia_)
        self.diagnostics_["final_silhouette"] = sil
        self.diagnostics_["final_davies_bouldin"] = db

        logger.info(f"K-Means fit complete. Inertia: {self.model.inertia_:.2f}, Silhouette: {sil:.4f}")
        return labels

    def predict(self, scaled_features: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new customer feature vectors."""
        return self.model.predict(scaled_features)

    def save(self, filepath: str = "artifacts/models/customer_kmeans.joblib") -> None:
        """Persist trained K-Means model."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved K-Means model to {path}")

    @classmethod
    def load(cls, filepath: str = "artifacts/models/customer_kmeans.joblib") -> "CustomerClustering":
        """Load persisted K-Means model."""
        clustering = cls()
        clustering.model = joblib.load(filepath)
        clustering.n_clusters = clustering.model.n_clusters
        return clustering
