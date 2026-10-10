"""Deterministic customer segment profiling and automated label generator."""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution or IDE imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("segment_profiling")


class SegmentProfiler:
    """Computes cluster-level statistical profiles and deterministically generates descriptive labels."""

    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or [
            "recency",
            "frequency",
            "monetary_value",
            "average_order_value",
            "average_basket_size",
            "unique_products",
            "unique_categories",
            "category_diversity",
            "discount_usage_rate",
            "active_days",
            "repeat_purchase_rate",
        ]

    def profile_segments(self, df_with_clusters: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """Compute comprehensive cluster metrics and deterministic descriptive labels."""
        logger.info("Computing deterministic cluster profiles and statistics...")
        
        pop_means = df_with_clusters[self.metrics].mean()
        total_customers = len(df_with_clusters)

        profiles = []
        cluster_labels_map = {}

        for cluster_id, group in df_with_clusters.groupby("cluster"):
            n_cust = len(group)
            pct_cust = (n_cust / total_customers) * 100.0

            c_means = group[self.metrics].mean()

            # Deterministic rule-based descriptive label based on relative deviation from population baseline
            r_ratio = c_means["recency"] / max(1e-4, pop_means["recency"])
            f_ratio = c_means["frequency"] / max(1e-4, pop_means["frequency"])
            m_ratio = c_means["monetary_value"] / max(1e-4, pop_means["monetary_value"])
            d_ratio = c_means["discount_usage_rate"] / max(1e-4, pop_means["discount_usage_rate"])

            traits = []
            if m_ratio >= 1.3 and f_ratio >= 1.2:
                traits.append("High-Value Frequent")
            elif m_ratio >= 1.2:
                traits.append("High-Spend")
            elif m_ratio < 0.7:
                traits.append("Low-Spend")

            if r_ratio < 0.7:
                traits.append("Active")
            elif r_ratio > 1.3:
                traits.append("Dormant/At-Risk")

            if d_ratio > 1.3:
                traits.append("Discount-Driven")

            if not traits:
                if f_ratio < 0.8:
                    traits.append("Occasional Casual")
                else:
                    traits.append("Core Mainstream")

            label = " ".join(traits) + " Shoppers"
            cid = int(float(str(cluster_id)))
            cluster_labels_map[cid] = label

            row = {
                "cluster_id": cid,
                "label": label,
                "customer_count": int(n_cust),
                "percentage_of_customers": round(float(pct_cust), 2),
                "avg_recency_days": round(float(c_means["recency"]), 2),
                "avg_frequency": round(float(c_means["frequency"]), 2),
                "avg_monetary_value": round(float(c_means["monetary_value"]), 2),
                "avg_order_value": round(float(c_means["average_order_value"]), 2),
                "avg_basket_size": round(float(c_means["average_basket_size"]), 2),
                "avg_unique_products": round(float(c_means["unique_products"]), 2),
                "avg_unique_categories": round(float(c_means["unique_categories"]), 2),
                "avg_category_diversity": round(float(c_means["category_diversity"]), 3),
                "avg_discount_usage_rate": round(float(c_means["discount_usage_rate"]), 3),
                "avg_active_days": round(float(c_means["active_days"]), 2),
                "avg_repeat_purchase_rate": round(float(c_means["repeat_purchase_rate"]), 3),
            }
            profiles.append(row)

        profiles_df = pd.DataFrame(profiles).sort_values("customer_count", ascending=False).reset_index(drop=True)
        return profiles_df, cluster_labels_map

    def generate_markdown_report(
        self,
        profiles_df: pd.DataFrame,
        output_csv_path: str = "outputs/segment_profiles.csv",
        output_md_path: str = "reports/segment_profiles.md",
    ) -> str:
        """Write segment profiles to CSV and structured Markdown report."""
        out_csv = Path(output_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        profiles_df.to_csv(out_csv, index=False)
        logger.info(f"Saved segment profiles CSV to {out_csv}")

        out_md = Path(output_md_path)
        out_md.parent.mkdir(parents=True, exist_ok=True)

        md_content = [
            "# Customer Segmentation Profiles & Intelligence Report\n",
            "## Executive Summary\n",
            f"Customer segmentation was trained on RFM metrics and multi-dimensional engagement features. "
            f"Identified **{len(profiles_df)} distinct behavioral segments** with deterministic automated interpretations derived directly from measurable group statistics.\n",
            "## Segment Summary Table\n\n",
            profiles_df.to_markdown(index=False),
            "\n\n## In-Depth Segment Profiles\n",
        ]

        for _, row in profiles_df.iterrows():
            cid = row["cluster_id"]
            md_content.append(f"\n### Segment {cid}: {row['label']}\n")
            md_content.append(f"- **Size**: {row['customer_count']:,} customers ({row['percentage_of_customers']}%)")
            md_content.append(f"- **Monetary Spend**: Average ${row['avg_monetary_value']:,.2f} (AOV: ${row['avg_order_value']:,.2f})")
            md_content.append(f"- **Engagement**: Frequency {row['avg_frequency']} events across {row['avg_active_days']} active days")
            md_content.append(f"- **Recency**: {row['avg_recency_days']} days since last interaction")
            md_content.append(f"- **Catalog Exploration**: {row['avg_unique_products']} distinct products in {row['avg_unique_categories']} categories (Diversity: {row['avg_category_diversity']})")
            md_content.append(f"- **Price Sensitivity**: {row['avg_discount_usage_rate']*100:.1f}% discount interaction rate")
            md_content.append(f"- **Loyalty**: {row['avg_repeat_purchase_rate']*100:.1f}% repeat purchase rate\n")

        full_md = "\n".join(md_content)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(full_md)

        logger.info(f"Generated Segment Profiles Markdown report at {out_md}")
        return full_md
