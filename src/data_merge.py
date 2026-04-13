"""
data_merge.py — Data Integration & Feature Engineering
========================================================
Loads all 6 star-schema CSV files, merges them into a single unified
DataFrame via the fact table, cleans data, and engineers new features
(Revenue, Profit, CLV, temporal fields).
"""

import os
import pandas as pd
import numpy as np

# ── Path Configuration ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets")


# ── Profit Margin Mapping ───────────────────────────────────────────
# Industry-standard retail margins by broad category
MARGIN_MAP = {
    "beverage": 0.20,
    "coffee": 0.18,
    "food": 0.15,
    "dishware": 0.10,
    "kitchen": 0.10,
    "gum": 0.25,
    "medicine": 0.12,
}


def _get_margin(category: str) -> float:
    """Return profit margin for a category string based on keyword matching."""
    cat_lower = str(category).lower()
    for keyword, margin in MARGIN_MAP.items():
        if keyword in cat_lower:
            return margin
    return 0.10  # default 10%


# ── Data Loading Functions ──────────────────────────────────────────

def load_customer_dim() -> pd.DataFrame:
    """Load customer dimension table."""
    df = pd.read_csv(
        os.path.join(DATA_DIR, "customer_dim.csv"),
        encoding="latin-1",
    )
    # Clean: fill missing names
    df["name"] = df["name"].fillna("Unknown")
    return df


def load_item_dim() -> pd.DataFrame:
    """Load item/product dimension table."""
    df = pd.read_csv(
        os.path.join(DATA_DIR, "item_dim.csv"),
        encoding="latin-1",
    )
    # Standardize category names (strip whitespace)
    df["desc"] = df["desc"].str.strip()
    # Fill missing unit
    df["unit"] = df["unit"].fillna("pcs")
    return df


def load_store_dim() -> pd.DataFrame:
    """Load store dimension table."""
    return pd.read_csv(os.path.join(DATA_DIR, "store_dim.csv"))


def load_trans_dim() -> pd.DataFrame:
    """Load transaction/payment dimension table."""
    df = pd.read_csv(os.path.join(DATA_DIR, "Trans_dim.csv"))
    # Fill missing bank names for cash transactions
    df["bank_name"] = df["bank_name"].fillna("N/A (Cash)")
    return df


def load_time_dim() -> pd.DataFrame:
    """Load time dimension table."""
    df = pd.read_csv(os.path.join(DATA_DIR, "time_dim.csv"))
    # Parse the date column
    df["date_parsed"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M", errors="coerce")
    return df


def load_fact_table() -> pd.DataFrame:
    """Load the central fact table (1M transactions)."""
    return pd.read_csv(os.path.join(DATA_DIR, "fact_table.csv"))


# ── Main Merge Function ────────────────────────────────────────────

def load_and_merge() -> pd.DataFrame:
    """
    Load all dimension tables and the fact table, merge via star-schema
    joins, clean data, and engineer features.

    Returns
    -------
    pd.DataFrame
        Unified dataset with ~1M rows and all enriched columns.
    """
    # ── Load all tables ─────────────────────────────────────────────
    fact = load_fact_table()
    customers = load_customer_dim()
    items = load_item_dim()
    stores = load_store_dim()
    trans = load_trans_dim()
    time = load_time_dim()

    # ── Star-schema joins ───────────────────────────────────────────
    # Join fact → customer
    df = fact.merge(customers, on="coustomer_key", how="left")

    # Join fact → item (use item-dim price; fact table also has unit_price)
    df = df.merge(
        items.rename(columns={"unit_price": "item_unit_price", "unit": "item_unit"}),
        on="item_key",
        how="left",
    )

    # Join fact → store
    df = df.merge(stores, on="store_key", how="left")

    # Join fact → transaction/payment
    df = df.merge(trans, on="payment_key", how="left")

    # Join fact → time
    df = df.merge(time, on="time_key", how="left")

    # ── Clean ───────────────────────────────────────────────────────
    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Fill any remaining NaN in text columns
    text_cols = df.select_dtypes(include="object").columns
    df[text_cols] = df[text_cols].fillna("Unknown")

    # ── Feature Engineering ─────────────────────────────────────────

    # Revenue (use fact-table unit_price × quantity, which equals total_price)
    df["revenue"] = df["total_price"].astype(float)

    # Profit (assumption-based margin by category)
    df["profit_margin"] = df["desc"].apply(_get_margin)
    df["profit"] = df["revenue"] * df["profit_margin"]

    # Temporal features from the parsed date
    df["transaction_date"] = df["date_parsed"]
    df["trans_month"] = df["date_parsed"].dt.month
    df["trans_year"] = df["date_parsed"].dt.year
    df["trans_quarter"] = df["date_parsed"].dt.quarter
    df["trans_day_of_week"] = df["date_parsed"].dt.day_name()
    df["month_year"] = df["date_parsed"].dt.to_period("M").astype(str)

    # Customer Lifetime Value (CLV) — total revenue per customer
    clv = df.groupby("coustomer_key")["revenue"].sum().reset_index()
    clv.columns = ["coustomer_key", "clv"]
    df = df.merge(clv, on="coustomer_key", how="left")

    # Customer transaction count (for repeat/new classification)
    tx_count = df.groupby("coustomer_key")["time_key"].nunique().reset_index()
    tx_count.columns = ["coustomer_key", "transaction_count"]
    df = df.merge(tx_count, on="coustomer_key", how="left")

    # Rename confusing columns for display
    df = df.rename(columns={
        "desc": "category",
        "division": "region",
        "name": "customer_name",
        "item_name": "product_name",
    })

    return df


# ── Quick Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading and merging datasets...")
    merged = load_and_merge()
    print(f"✅ Merged dataset shape: {merged.shape}")
    print(f"   Columns: {merged.columns.tolist()}")
    print(f"\n   Revenue range: ৳{merged['revenue'].min():,.0f} — ৳{merged['revenue'].max():,.0f}")
    print(f"   Total Revenue: ৳{merged['revenue'].sum():,.0f}")
    print(f"   Unique Customers: {merged['coustomer_key'].nunique():,}")
    print(f"   Date Range: {merged['transaction_date'].min()} to {merged['transaction_date'].max()}")
    print(f"\n   Sample:\n{merged.head(3).to_string()}")
