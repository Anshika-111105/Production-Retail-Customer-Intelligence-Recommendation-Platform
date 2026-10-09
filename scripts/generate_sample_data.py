"""
Synthetic Retail Transaction Dataset Generator.
Generates realistic multi-event e-commerce transaction data mimicking RetailRocket and modern e-commerce logs.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_retail_dataset(
    num_customers: int = 3000,
    num_products: int = 800,
    num_interactions: int = 60000,
    start_date: str = "2026-01-01",
    days: int = 180,
    random_seed: int = 42,
    output_dir: str = "data/raw",
) -> Path:
    """Generate and save realistic retail interactions and product catalog."""
    np.random.seed(random_seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic retail data: {num_customers} customers, {num_products} products, target ~{num_interactions} interactions...")

    # 1. Product Catalog
    categories = [
        "Electronics/Computers",
        "Electronics/Smartphones",
        "Electronics/Audio",
        "Home/Kitchen",
        "Home/Furniture",
        "Home/Decor",
        "Apparel/Men",
        "Apparel/Women",
        "Apparel/Footwear",
        "Beauty/Skincare",
        "Sports/Fitness",
        "Books/NonFiction",
    ]
    brands = [
        "AcroTech", "ApexLiving", "AuraGlow", "ByteForce", "CoreGear",
        "EchoSound", "LuxeLine", "NovaWear", "OmniHome", "PeakForm",
        "PrimeStyle", "QuantumLab", "SonicPulse", "UrbanCraft", "Zenith"
    ]

    product_ids = np.array([1000 + i for i in range(num_products)])
    product_categories = np.random.choice(categories, size=num_products)
    product_brands = np.random.choice(brands, size=num_products)
    
    category_price_multipliers = {
        "Electronics/Computers": 800.0,
        "Electronics/Smartphones": 600.0,
        "Electronics/Audio": 120.0,
        "Home/Kitchen": 85.0,
        "Home/Furniture": 350.0,
        "Home/Decor": 45.0,
        "Apparel/Men": 55.0,
        "Apparel/Women": 65.0,
        "Apparel/Footwear": 90.0,
        "Beauty/Skincare": 35.0,
        "Sports/Fitness": 75.0,
        "Books/NonFiction": 22.0,
    }

    base_prices = np.zeros(num_products, dtype=float)
    descriptions = []
    for i in range(num_products):
        cat = product_categories[i]
        mult = category_price_multipliers.get(cat, 50.0)
        price = round(float(np.random.lognormal(mean=np.log(mult), sigma=0.35)), 2)
        base_prices[i] = max(4.99, price)
        descriptions.append(f"Premium {cat.split('/')[-1]} item by {product_brands[i]} with high-performance features and durable design.")

    catalog_df = pd.DataFrame({
        "product_id": product_ids,
        "category": product_categories,
        "brand": product_brands,
        "unit_price": base_prices,
        "description": descriptions,
    })

    catalog_path = out_path / "product_catalog.csv"
    catalog_df.to_csv(catalog_path, index=False)
    logger.info(f"Product catalog saved to {catalog_path} ({len(catalog_df)} items)")

    # 2. Pre-index product attributes for fast vector lookups
    cat_to_indices = {cat: np.where(product_categories == cat)[0] for cat in categories}

    # 3. Customer Archetypes (Latent Personas)
    customer_ids = [20000 + i for i in range(num_customers)]
    persona_assignments = np.random.choice([0, 1, 2, 3], size=num_customers, p=[0.15, 0.25, 0.40, 0.20])
    
    customer_pref_cats = [
        np.random.choice(categories, size=np.random.randint(1, 4), replace=False).tolist()
        for _ in range(num_customers)
    ]

    zipf_ranks = np.random.zipf(a=1.35, size=num_interactions * 2)
    zipf_ranks = zipf_ranks[zipf_ranks <= num_products] - 1
    if len(zipf_ranks) < num_interactions:
        zipf_ranks = np.random.choice(num_products, size=num_interactions)

    start_ts = pd.Timestamp(start_date).timestamp()
    end_ts = start_ts + days * 86400

    # Lists for fast DataFrame assembly
    c_ids_list = []
    p_ids_list = []
    tx_ids_list = []
    ts_list = []
    qty_list = []
    price_list = []
    cat_list = []
    brand_list = []
    desc_list = []
    disc_list = []
    event_list = []

    tx_counter = 500000
    zipf_len = len(zipf_ranks)

    logger.info("Simulating customer journeys (fast vectorized)...")
    for c_idx in range(num_customers):
        cust_id = customer_ids[c_idx]
        persona = persona_assignments[c_idx]
        pref_cats = customer_pref_cats[c_idx]

        if persona == 0:  # High value loyal
            n_events = np.random.randint(25, 55)
            time_bias = 0.7
            discount_prob = 0.15
        elif persona == 1:  # Discount seeker
            n_events = np.random.randint(15, 35)
            time_bias = 0.5
            discount_prob = 0.70
        elif persona == 2:  # Casual
            n_events = np.random.randint(4, 15)
            time_bias = 0.4
            discount_prob = 0.30
        else:  # Dormant / occasional
            n_events = np.random.randint(1, 5)
            time_bias = 0.1
            discount_prob = 0.25

        pref_indices = []
        for cat in pref_cats:
            pref_indices.extend(cat_to_indices[cat])
        pref_indices = np.array(pref_indices) if len(pref_indices) > 0 else np.arange(num_products)

        if time_bias > 0.5:
            u = np.random.power(a=3.0, size=n_events)
        elif time_bias < 0.3:
            u = 1.0 - np.random.power(a=2.5, size=n_events)
        else:
            u = np.random.uniform(0.0, 1.0, size=n_events)

        event_timestamps = np.sort(start_ts + u * (end_ts - start_ts))

        # Vectorized product selection
        use_pref = np.random.rand(n_events) < 0.70
        chosen_p_idx = np.zeros(n_events, dtype=int)
        for i in range(n_events):
            if use_pref[i] and len(pref_indices) > 0:
                chosen_p_idx[i] = pref_indices[np.random.randint(0, len(pref_indices))]
            else:
                chosen_p_idx[i] = zipf_ranks[np.random.randint(0, zipf_len)]

        event_rolls = np.random.rand(n_events)
        discs_rolls = np.random.rand(n_events) < discount_prob

        for i in range(n_events):
            p_idx = chosen_p_idx[i]
            roll = event_rolls[i]
            if roll < 0.60:
                ev = "view"
                q = 0
                d = 0.0
                tx = None
            elif roll < 0.85:
                ev = "add_to_cart"
                q = 1
                d = 0.0
                tx = None
            else:
                ev = "purchase"
                q = int(np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05]))
                d = float(np.random.choice([0.0, 0.10, 0.20, 0.30], p=[0.5, 0.25, 0.15, 0.10])) if discs_rolls[i] else 0.0
                tx_counter += 1
                tx = tx_counter

            c_ids_list.append(cust_id)
            p_ids_list.append(product_ids[p_idx])
            tx_ids_list.append(tx)
            ts_list.append(event_timestamps[i])
            qty_list.append(q)
            price_list.append(base_prices[p_idx])
            cat_list.append(product_categories[p_idx])
            brand_list.append(product_brands[p_idx])
            desc_list.append(descriptions[p_idx])
            disc_list.append(d)
            event_list.append(ev)

    dt_series = pd.to_datetime(ts_list, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    events_df = pd.DataFrame({
        "customer_id": c_ids_list,
        "product_id": p_ids_list,
        "transaction_id": tx_ids_list,
        "timestamp": dt_series,
        "quantity": qty_list,
        "unit_price": price_list,
        "category": cat_list,
        "brand": brand_list,
        "description": desc_list,
        "discount": disc_list,
        "event_type": event_list,
    })

    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)

    events_path = out_path / "retail_events.csv"
    events_df.to_csv(events_path, index=False)
    logger.info(f"Generated {len(events_df)} interaction events across {events_df['customer_id'].nunique()} customers and {events_df['product_id'].nunique()} products.")
    logger.info(f"Saved to {events_path}")

    return events_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic retail dataset.")
    parser.add_argument("--customers", type=int, default=3000)
    parser.add_argument("--products", type=int, default=800)
    parser.add_argument("--interactions", type=int, default=60000)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_retail_dataset(
        num_customers=args.customers,
        num_products=args.products,
        num_interactions=args.interactions,
        days=args.days,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )
