"""Customer segmentation visualization via 2D PCA projection."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.utils.logger import get_logger

logger = get_logger("segment_visualization")


class SegmentVisualizer:
    """Visualizes customer clusters using PCA dimensionality reduction."""

    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def plot_pca_clusters(
        self,
        scaled_features: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_names: Optional[Dict[int, str]] = None,
        output_image_path: str = "outputs/figures/customer_segments_pca.png",
    ) -> str:
        """Project scaled features to 2D PCA coordinates and generate visualization."""
        logger.info("Computing 2D PCA projection for cluster visualization...")
        coords_2d = self.pca.fit_transform(scaled_features)
        var_ratio = self.pca.explained_variance_ratio_

        out_path = Path(output_image_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

        for i, cid in enumerate(unique_clusters):
            mask = cluster_labels == cid
            label_text = cluster_names.get(cid, f"Cluster {cid}") if cluster_names else f"Cluster {cid}"
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=[colors[i]],
                label=f"{cid}: {label_text} (n={mask.sum()})",
                alpha=0.6,
                edgecolors="none",
                s=25,
            )

        # Plot cluster centroids in PCA space
        centroids_pca = np.array([coords_2d[cluster_labels == cid].mean(axis=0) for cid in unique_clusters])
        ax.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            c="black",
            marker="X",
            s=120,
            linewidths=1.5,
            edgecolors="white",
            label="Cluster Centroids",
        )

        ax.set_title(
            f"Customer Segmentation (2D PCA Projection)\n"
            f"Note: Dimensionality reduction for visualization (PC1: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}% var explained)",
            fontsize=12,
            pad=12,
        )
        ax.set_xlabel(f"Principal Component 1 ({var_ratio[0]*100:.1f}% variance)", fontsize=10)
        ax.set_ylabel(f"Principal Component 2 ({var_ratio[1]*100:.1f}% variance)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=9)
        plt.tight_layout()

        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved PCA cluster visualization to {out_path}")
        return str(out_path)
