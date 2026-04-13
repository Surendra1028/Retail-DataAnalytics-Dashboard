"""
analysis.py — Analytics & Business Intelligence Functions
==========================================================
Provides KPI computation, EDA aggregations, customer analytics
(RFM segmentation, repeat/new classification), and optional ML
(KMeans clustering, sales trend prediction).
"""

import pandas as pd
import numpy as np


# =====================================================================
# KPI FUNCTIONS
# =====================================================================

def get_kpis(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPI metrics from the merged dataset.

    Returns dict with: total_revenue, total_profit, total_transactions,
    unique_customers, aov, top_category, best_store, best_region.
    """
    total_revenue = df["revenue"].sum()
    total_profit = df["profit"].sum()
    total_transactions = df["time_key"].nunique()
    unique_customers = df["coustomer_key"].nunique()
    aov = total_revenue / total_transactions if total_transactions > 0 else 0

    # Top category by revenue
    cat_rev = df.groupby("category")["revenue"].sum()
    top_category = cat_rev.idxmax() if len(cat_rev) > 0 else "N/A"

    # Best performing store (district level)
    store_rev = df.groupby("district")["revenue"].sum()
    best_store = store_rev.idxmax() if len(store_rev) > 0 else "N/A"

    # Best region
    region_rev = df.groupby("region")["revenue"].sum()
    best_region = region_rev.idxmax() if len(region_rev) > 0 else "N/A"

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "total_transactions": total_transactions,
        "unique_customers": unique_customers,
        "aov": aov,
        "top_category": top_category,
        "best_store": best_store,
        "best_region": best_region,
    }


# =====================================================================
# EXPLORATORY DATA ANALYSIS
# =====================================================================

def sales_trend(df: pd.DataFrame, granularity: str = "monthly") -> pd.DataFrame:
    """
    Aggregate sales by time period.

    Parameters
    ----------
    granularity : str
        'monthly' or 'yearly'

    Returns
    -------
    pd.DataFrame with columns [period, revenue, profit, transactions]
    """
    if granularity == "yearly":
        group_col = "trans_year"
    else:
        group_col = "month_year"

    result = (
        df.groupby(group_col)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            transactions=("time_key", "nunique"),
            quantity=("quantity", "sum"),
        )
        .reset_index()
        .rename(columns={group_col: "period"})
    )

    # Sort chronologically
    if granularity == "monthly":
        result["sort_key"] = pd.to_datetime(result["period"], format="%Y-%m", errors="coerce")
        result = result.sort_values("sort_key").drop(columns="sort_key")
    else:
        result = result.sort_values("period")

    return result.reset_index(drop=True)


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N products by total revenue."""
    return (
        df.groupby("product_name")
        .agg(
            revenue=("revenue", "sum"),
            quantity_sold=("quantity", "sum"),
            transactions=("time_key", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def category_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue breakdown by product category."""
    result = (
        df.groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            items_sold=("quantity", "sum"),
            transactions=("time_key", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    result["revenue_pct"] = (result["revenue"] / result["revenue"].sum() * 100).round(1)
    return result


def store_performance(df: pd.DataFrame, level: str = "district") -> pd.DataFrame:
    """
    Store performance aggregation.

    Parameters
    ----------
    level : str
        'district', 'upazila', or 'region'
    """
    return (
        df.groupby(level)
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            transactions=("time_key", "nunique"),
            customers=("coustomer_key", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )


def region_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue & metrics by region (division)."""
    return (
        df.groupby("region")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            transactions=("time_key", "nunique"),
            customers=("coustomer_key", "nunique"),
            stores=("store_key", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )


def customer_purchase_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """Purchase frequency distribution of customers."""
    freq = (
        df.groupby("coustomer_key")
        .agg(
            total_transactions=("time_key", "nunique"),
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity", "sum"),
            avg_order_value=("revenue", "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )
    return freq


def sales_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table of revenue by month × year for heatmap."""
    pivot = df.pivot_table(
        values="revenue",
        index="trans_month",
        columns="trans_year",
        aggfunc="sum",
        fill_value=0,
    )
    # Rename month numbers to names
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    pivot.index = pivot.index.map(lambda x: month_names.get(x, x))
    return pivot


# =====================================================================
# CUSTOMER ANALYTICS
# =====================================================================

def repeat_vs_new(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify customers as 'New' (1 transaction) or 'Repeat' (>1).

    Returns summary DataFrame with counts and percentages.
    """
    cust_tx = df.groupby("coustomer_key")["time_key"].nunique().reset_index()
    cust_tx.columns = ["coustomer_key", "tx_count"]
    cust_tx["customer_type"] = cust_tx["tx_count"].apply(
        lambda x: "New (1 purchase)" if x == 1 else "Repeat (2+ purchases)"
    )

    summary = (
        cust_tx.groupby("customer_type")
        .agg(count=("coustomer_key", "count"))
        .reset_index()
    )
    summary["percentage"] = (summary["count"] / summary["count"].sum() * 100).round(1)
    return summary


def high_value_customers(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top N customers by lifetime value (CLV)."""
    return (
        df.groupby(["coustomer_key", "customer_name"])
        .agg(
            clv=("revenue", "sum"),
            total_transactions=("time_key", "nunique"),
            avg_order_value=("revenue", "mean"),
            total_quantity=("quantity", "sum"),
        )
        .reset_index()
        .sort_values("clv", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) scores and segment customers.

    Segments: Champions, Loyal Customers, At Risk, Lost Customers
    """
    # Reference date = max date in the dataset + 1 day
    max_date = df["transaction_date"].max()
    if pd.isna(max_date):
        max_date = pd.Timestamp("2020-12-31")
    reference_date = max_date + pd.Timedelta(days=1)

    rfm = (
        df.groupby("coustomer_key")
        .agg(
            recency=("transaction_date", lambda x: (reference_date - x.max()).days),
            frequency=("time_key", "nunique"),
            monetary=("revenue", "sum"),
        )
        .reset_index()
    )

    # Score each dimension (1-4 quartiles, 4=best)
    rfm["r_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    # Composite RFM score
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # Segment
    def _segment(score):
        if score >= 10:
            return "Champions"
        elif score >= 7:
            return "Loyal Customers"
        elif score >= 5:
            return "At Risk"
        else:
            return "Lost Customers"

    rfm["segment"] = rfm["rfm_score"].apply(_segment)

    return rfm


def rfm_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize RFM segments."""
    return (
        rfm_df.groupby("segment")
        .agg(
            customers=("coustomer_key", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
        .sort_values("avg_monetary", ascending=False)
        .reset_index(drop=True)
    )


# =====================================================================
# OPTIONAL ML
# =====================================================================

def customer_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    KMeans clustering on RFM features.

    Returns RFM DataFrame with an additional 'cluster' column.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    rfm = rfm_segmentation(df)
    features = rfm[["recency", "frequency", "monetary"]].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(scaled)
    rfm["cluster"] = rfm["cluster"].apply(lambda x: f"Cluster {x + 1}")

    return rfm


def sales_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple linear trend prediction on monthly sales.
    Predicts the next 6 months.
    """
    from sklearn.linear_model import LinearRegression

    trend = sales_trend(df, "monthly")
    trend["period_dt"] = pd.to_datetime(trend["period"], format="%Y-%m", errors="coerce")
    trend = trend.dropna(subset=["period_dt"]).sort_values("period_dt")

    # Numeric representation of time
    trend["time_idx"] = np.arange(len(trend))

    model = LinearRegression()
    model.fit(trend[["time_idx"]], trend["revenue"])

    # Predict next 6 months
    last_idx = trend["time_idx"].max()
    last_date = trend["period_dt"].max()
    future_idx = np.arange(last_idx + 1, last_idx + 7).reshape(-1, 1)
    future_revenue = model.predict(future_idx)

    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq="MS")
    forecast = pd.DataFrame({
        "period": future_dates.strftime("%Y-%m"),
        "predicted_revenue": future_revenue,
        "type": "forecast",
    })

    # Combine with actuals
    trend["type"] = "actual"
    trend = trend.rename(columns={"revenue": "predicted_revenue"})
    combined = pd.concat(
        [trend[["period", "predicted_revenue", "type"]], forecast],
        ignore_index=True,
    )
    return combined


# ── Quick Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_merge import load_and_merge

    print("Loading data...")
    df = load_and_merge()

    print("\n📊 KPIs:")
    kpis = get_kpis(df)
    for k, v in kpis.items():
        print(f"   {k}: {v}")

    print(f"\n📈 Top 5 Products:\n{top_products(df, 5).to_string()}")
    print(f"\n🏪 Region Sales:\n{region_sales(df).to_string()}")
    print(f"\n👥 Repeat vs New:\n{repeat_vs_new(df).to_string()}")
    print(f"\n🎯 RFM Segments:\n{rfm_summary(rfm_segmentation(df)).to_string()}")
