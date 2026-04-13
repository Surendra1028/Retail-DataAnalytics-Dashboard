"""
app.py — Retail Analytics Dashboard (Streamlit)
=================================================
Professional, interactive dashboard for retail/e-commerce data
warehouse analytics with dynamic filtering, KPI cards, Plotly charts,
customer segmentation, and ML clustering.
"""

import sys
import os

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_merge import load_and_merge
from src.analysis import (
    get_kpis,
    sales_trend,
    top_products,
    category_revenue,
    store_performance,
    region_sales,
    customer_purchase_behavior,
    sales_heatmap_data,
    repeat_vs_new,
    high_value_customers,
    rfm_segmentation,
    rfm_summary,
    customer_clustering,
    sales_prediction,
)


# =====================================================================
# PAGE CONFIG & CUSTOM CSS
# =====================================================================

st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Dark Theme CSS
st.markdown("""
<style>
/* ── Import Google Font ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Styles ──────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Sidebar Styling ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff;
}

/* ── KPI Card Styling ───────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30, 30, 60, 0.9) 0%, rgba(20, 20, 50, 0.95) 100%);
    border: 1px solid rgba(100, 100, 255, 0.15);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(80, 80, 255, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

div[data-testid="stMetric"] label {
    color: #8888cc !important;
    font-weight: 500;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700;
    font-size: 1.8rem;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #44ddaa !important;
}

/* ── Header Styling ─────────────────────────────────────────── */
.dashboard-header {
    background: linear-gradient(135deg, #1a1a3e 0%, #2d1b69 50%, #1a1a3e 100%);
    border: 1px solid rgba(100, 100, 255, 0.2);
    border-radius: 20px;
    padding: 30px 40px;
    margin-bottom: 24px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    text-align: center;
}

.dashboard-header h1 {
    background: linear-gradient(135deg, #7b68ee 0%, #00bfff 50%, #7b68ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
}

.dashboard-header p {
    color: #9999cc;
    font-size: 1rem;
    margin: 8px 0 0 0;
    font-weight: 400;
}

/* ── Section Headers ────────────────────────────────────────── */
.section-header {
    background: linear-gradient(90deg, rgba(100, 100, 255, 0.1) 0%, transparent 100%);
    border-left: 4px solid #7b68ee;
    padding: 12px 20px;
    border-radius: 0 12px 12px 0;
    margin: 24px 0 16px 0;
}

.section-header h2 {
    color: #d0d0ff;
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0;
}

/* ── Chart Containers ───────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    background: rgba(15, 15, 35, 0.6);
    border: 1px solid rgba(100, 100, 255, 0.1);
    border-radius: 16px;
    padding: 8px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

/* ── Expander Styling ───────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(20, 20, 50, 0.8);
    border-radius: 12px;
    color: #d0d0ff !important;
    font-weight: 600;
}

/* ── Tabs ───────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30, 30, 60, 0.6);
    border-radius: 10px 10px 0 0;
    border: 1px solid rgba(100, 100, 255, 0.15);
    color: #9999cc;
    font-weight: 500;
    padding: 10px 20px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2d1b69, #1a1a3e) !important;
    color: #7b68ee !important;
    border-bottom: 2px solid #7b68ee;
}

/* ── Dataframe Styling ──────────────────────────────────────── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Download Button ────────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #7b68ee, #5b4bc7);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 8px 24px;
    transition: all 0.2s ease;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #8b78ff, #6b5bd7);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(123, 104, 238, 0.3);
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# PLOTLY TEMPLATE (Dark + Premium)
# =====================================================================

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#d0d0ff"),
    title_font=dict(size=18, color="#e0e0ff"),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        bgcolor="rgba(20,20,50,0.6)",
        bordercolor="rgba(100,100,255,0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
)

# Color palette — vibrant gradient
COLORS = [
    "#7b68ee", "#00bfff", "#44ddaa", "#ff6b9d",
    "#ffa726", "#ab47bc", "#26c6da", "#ef5350",
    "#66bb6a", "#ffd54f", "#5c6bc0", "#ec407a",
]


# =====================================================================
# DATA LOADING (CACHED)
# =====================================================================

@st.cache_data(show_spinner="🔄 Loading & merging 1M+ transactions...")
def load_data():
    """Load and cache the merged dataset."""
    return load_and_merge()


@st.cache_data(show_spinner="🧠 Computing RFM segments...")
def compute_rfm(df):
    """Cache RFM computation."""
    return rfm_segmentation(df)


@st.cache_data(show_spinner="🤖 Running KMeans clustering...")
def compute_clusters(df, n):
    """Cache clustering computation."""
    return customer_clustering(df, n)


# =====================================================================
# MAIN APP
# =====================================================================

def main():
    # ── Header ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="dashboard-header">
        <h1>📊 Retail Analytics Dashboard</h1>
        <p>Real-time business intelligence • 1M+ transactions • Star-schema data warehouse</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data ───────────────────────────────────────────────────
    df = load_data()

    # ── Sidebar Filters ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎛️ Filters")
        st.markdown("---")

        # Date Range
        st.markdown("### 📅 Date Range")
        min_date = df["transaction_date"].min()
        max_date = df["transaction_date"].max()
        if pd.isna(min_date):
            min_date = pd.Timestamp("2014-01-01")
        if pd.isna(max_date):
            max_date = pd.Timestamp("2020-12-31")

        date_range = st.date_input(
            "Select range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="date_filter",
        )

        st.markdown("---")

        # Region Filter
        st.markdown("### 🌍 Region (Division)")
        all_regions = sorted(df["region"].unique())
        selected_regions = st.multiselect(
            "Select regions",
            options=all_regions,
            default=all_regions,
            key="region_filter",
        )

        # Dynamic store filter based on region
        st.markdown("### 🏪 District")
        available_districts = sorted(
            df[df["region"].isin(selected_regions)]["district"].unique()
        )
        selected_districts = st.multiselect(
            "Select districts",
            options=available_districts,
            default=available_districts,
            key="district_filter",
        )

        st.markdown("---")

        # Category Filter
        st.markdown("### 📦 Category")
        all_categories = sorted(df["category"].unique())
        selected_categories = st.multiselect(
            "Select categories",
            options=all_categories,
            default=all_categories,
            key="category_filter",
        )

        st.markdown("---")

        # Transaction Type Filter
        st.markdown("### 💳 Payment Type")
        all_trans_types = sorted(df["trans_type"].unique())
        selected_trans_types = st.multiselect(
            "Select payment types",
            options=all_trans_types,
            default=all_trans_types,
            key="trans_type_filter",
        )

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#666; font-size:0.8rem;'>"
            "Built with ❤️ using Streamlit & Plotly</p>",
            unsafe_allow_html=True,
        )

    # ── Apply Filters ───────────────────────────────────────────────
    filtered = df.copy()

    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["transaction_date"].dt.date >= start_date)
            & (filtered["transaction_date"].dt.date <= end_date)
        ]

    # Region
    if selected_regions:
        filtered = filtered[filtered["region"].isin(selected_regions)]

    # District
    if selected_districts:
        filtered = filtered[filtered["district"].isin(selected_districts)]

    # Category
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]

    # Transaction Type
    if selected_trans_types:
        filtered = filtered[filtered["trans_type"].isin(selected_trans_types)]

    # Check if any data remains
    if len(filtered) == 0:
        st.warning("⚠️ No data matches the selected filters. Please adjust your selections.")
        return

    # ── KPI Section ─────────────────────────────────────────────────
    kpis = get_kpis(filtered)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"৳ {kpis['total_revenue']:,.0f}",
            delta=f"Profit: ৳{kpis['total_profit']:,.0f}",
        )
    with col2:
        st.metric(
            label="📦 Total Orders",
            value=f"{kpis['total_transactions']:,}",
            delta=f"Items: {filtered['quantity'].sum():,}",
        )
    with col3:
        st.metric(
            label="👥 Unique Customers",
            value=f"{kpis['unique_customers']:,}",
            delta=f"Best Region: {kpis['best_region']}",
        )
    with col4:
        st.metric(
            label="📊 Avg Order Value",
            value=f"৳ {kpis['aov']:,.1f}",
            delta=f"Top: {kpis['top_category'][:25]}",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Sales Analytics",
        "👥 Customer Analytics",
        "🏪 Store & Region",
        "🤖 Advanced / ML",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1: SALES ANALYTICS
    # ════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
        <div class="section-header"><h2>📈 Sales Trend Analysis</h2></div>
        """, unsafe_allow_html=True)

        # ── Sales Trend Chart ───────────────────────────────────────
        trend_col1, trend_col2 = st.columns([3, 1])
        with trend_col2:
            trend_granularity = st.radio(
                "Granularity", ["monthly", "yearly"], index=0, key="trend_gran"
            )

        trend_data = sales_trend(filtered, trend_granularity)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend_data["period"],
            y=trend_data["revenue"],
            mode="lines+markers",
            name="Revenue",
            line=dict(color="#7b68ee", width=3),
            marker=dict(size=6, color="#7b68ee"),
            fill="tozeroy",
            fillcolor="rgba(123, 104, 238, 0.1)",
        ))
        fig_trend.add_trace(go.Scatter(
            x=trend_data["period"],
            y=trend_data["profit"],
            mode="lines+markers",
            name="Profit",
            line=dict(color="#44ddaa", width=2, dash="dash"),
            marker=dict(size=5, color="#44ddaa"),
        ))
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            title=f"{'Monthly' if trend_granularity == 'monthly' else 'Yearly'} Sales Trend",
            xaxis_title="Period",
            yaxis_title="Amount (৳)",
            hovermode="x unified",
            height=420,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Top Products & Category Analysis ────────────────────────
        st.markdown("""
        <div class="section-header"><h2>🏆 Product & Category Insights</h2></div>
        """, unsafe_allow_html=True)

        prod_col, cat_col = st.columns(2)

        with prod_col:
            n_products = st.slider("Top N Products", 5, 20, 10, key="top_n_slider")
            top_prods = top_products(filtered, n_products)

            fig_prods = go.Figure(go.Bar(
                x=top_prods["revenue"],
                y=top_prods["product_name"],
                orientation="h",
                marker=dict(
                    color=top_prods["revenue"],
                    colorscale=[[0, "#2d1b69"], [0.5, "#7b68ee"], [1, "#00bfff"]],
                    cornerradius=6,
                ),
                text=top_prods["revenue"].apply(lambda x: f"৳{x:,.0f}"),
                textposition="inside",
                textfont=dict(color="white", size=11),
            ))
            fig_prods.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Top {n_products} Products by Revenue",
                xaxis_title="Revenue (৳)",
                yaxis=dict(autorange="reversed"),
                height=450,
            )
            st.plotly_chart(fig_prods, use_container_width=True)

        with cat_col:
            cat_data = category_revenue(filtered)

            fig_cat = go.Figure(go.Pie(
                labels=cat_data["category"],
                values=cat_data["revenue"],
                hole=0.5,
                marker=dict(colors=COLORS),
                textinfo="label+percent",
                textposition="outside",
                textfont=dict(size=10),
                pull=[0.05 if i == 0 else 0 for i in range(len(cat_data))],
            ))
            fig_cat.update_layout(
                **PLOTLY_LAYOUT,
                title="Category-wise Revenue Distribution",
                height=450,
                showlegend=False,
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        # ── Sales Heatmap ───────────────────────────────────────────
        st.markdown("""
        <div class="section-header"><h2>🗓️ Sales Heatmap (Month × Year)</h2></div>
        """, unsafe_allow_html=True)

        heatmap_data = sales_heatmap_data(filtered)
        fig_heatmap = go.Figure(go.Heatmap(
            z=heatmap_data.values,
            x=[str(c) for c in heatmap_data.columns],
            y=heatmap_data.index,
            colorscale=[[0, "#0f0f23"], [0.3, "#2d1b69"], [0.6, "#7b68ee"], [1, "#00bfff"]],
            text=[[f"৳{v:,.0f}" for v in row] for row in heatmap_data.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hoverongaps=False,
            colorbar=dict(title="Revenue"),
        ))
        fig_heatmap.update_layout(
            **PLOTLY_LAYOUT,
            title="Revenue Heatmap — Month vs Year",
            xaxis_title="Year",
            yaxis_title="Month",
            height=400,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # ── Category Details Table ──────────────────────────────────
        with st.expander("📋 Detailed Category Breakdown"):
            cat_data_display = cat_data.copy()
            cat_data_display["revenue"] = cat_data_display["revenue"].apply(lambda x: f"৳{x:,.0f}")
            cat_data_display["profit"] = cat_data_display["profit"].apply(lambda x: f"৳{x:,.0f}")
            st.dataframe(cat_data_display, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2: CUSTOMER ANALYTICS
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
        <div class="section-header"><h2>👥 Customer Segmentation & Behavior</h2></div>
        """, unsafe_allow_html=True)

        # ── Repeat vs New ───────────────────────────────────────────
        cust_col1, cust_col2 = st.columns(2)

        with cust_col1:
            repeat_data = repeat_vs_new(filtered)
            fig_repeat = go.Figure(go.Pie(
                labels=repeat_data["customer_type"],
                values=repeat_data["count"],
                hole=0.6,
                marker=dict(colors=["#7b68ee", "#44ddaa"]),
                textinfo="label+value+percent",
                textfont=dict(size=12),
            ))
            fig_repeat.update_layout(
                **PLOTLY_LAYOUT,
                title="Repeat vs New Customers",
                height=400,
                annotations=[dict(
                    text=f"{repeat_data['count'].sum():,}<br>Total",
                    x=0.5, y=0.5, font_size=16,
                    font_color="#d0d0ff",
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_repeat, use_container_width=True)

        with cust_col2:
            # Purchase frequency distribution
            cust_behavior = customer_purchase_behavior(filtered)
            freq_dist = cust_behavior["total_transactions"].value_counts().sort_index().head(20)

            fig_freq = go.Figure(go.Bar(
                x=freq_dist.index.astype(str),
                y=freq_dist.values,
                marker=dict(
                    color=freq_dist.values,
                    colorscale=[[0, "#2d1b69"], [1, "#00bfff"]],
                    cornerradius=4,
                ),
            ))
            fig_freq.update_layout(
                **PLOTLY_LAYOUT,
                title="Purchase Frequency Distribution",
                xaxis_title="Number of Transactions",
                yaxis_title="Number of Customers",
                height=400,
            )
            st.plotly_chart(fig_freq, use_container_width=True)

        # ── RFM Segmentation ────────────────────────────────────────
        st.markdown("""
        <div class="section-header"><h2>🎯 RFM Segmentation</h2></div>
        """, unsafe_allow_html=True)

        rfm_data = compute_rfm(filtered)
        rfm_summ = rfm_summary(rfm_data)

        rfm_col1, rfm_col2 = st.columns([2, 1])

        with rfm_col1:
            # RFM Scatter Plot
            segment_colors = {
                "Champions": "#44ddaa",
                "Loyal Customers": "#7b68ee",
                "At Risk": "#ffa726",
                "Lost Customers": "#ef5350",
            }
            fig_rfm = px.scatter(
                rfm_data,
                x="frequency",
                y="monetary",
                color="segment",
                size="monetary",
                size_max=20,
                color_discrete_map=segment_colors,
                hover_data=["coustomer_key", "recency", "rfm_score"],
                opacity=0.7,
            )
            fig_rfm.update_layout(
                **PLOTLY_LAYOUT,
                title="RFM Customer Segments",
                xaxis_title="Frequency (# Transactions)",
                yaxis_title="Monetary (Total Revenue ৳)",
                height=450,
            )
            st.plotly_chart(fig_rfm, use_container_width=True)

        with rfm_col2:
            # Segment summary
            st.markdown("#### 📊 Segment Summary")
            for _, row in rfm_summ.iterrows():
                seg = row["segment"]
                color = segment_colors.get(seg, "#888")
                st.markdown(
                    f"<div style='background:rgba(30,30,60,0.8); border-left:4px solid {color}; "
                    f"padding:12px 16px; border-radius:0 10px 10px 0; margin-bottom:10px;'>"
                    f"<strong style='color:{color}'>{seg}</strong><br>"
                    f"<span style='color:#aaa; font-size:0.85rem;'>"
                    f"Customers: {row['customers']:,}<br>"
                    f"Avg Revenue: ৳{row['avg_monetary']:,.0f}<br>"
                    f"Avg Frequency: {row['avg_frequency']:.1f}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

        # ── High-Value Customers ────────────────────────────────────
        st.markdown("""
        <div class="section-header"><h2>💎 High-Value Customers (Top 20)</h2></div>
        """, unsafe_allow_html=True)

        hv_customers = high_value_customers(filtered, 20)
        hv_display = hv_customers.copy()
        hv_display.index = range(1, len(hv_display) + 1)
        hv_display["clv"] = hv_display["clv"].apply(lambda x: f"৳{x:,.0f}")
        hv_display["avg_order_value"] = hv_display["avg_order_value"].apply(lambda x: f"৳{x:,.1f}")
        hv_display.columns = [
            "Customer ID", "Name", "Lifetime Value",
            "Transactions", "Avg Order Value", "Items Purchased"
        ]
        st.dataframe(hv_display, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 3: STORE & REGION
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
        <div class="section-header"><h2>🌍 Regional Performance</h2></div>
        """, unsafe_allow_html=True)

        reg_data = region_sales(filtered)

        # Region comparison
        reg_col1, reg_col2 = st.columns(2)

        with reg_col1:
            fig_region = go.Figure()
            fig_region.add_trace(go.Bar(
                x=reg_data["region"],
                y=reg_data["revenue"],
                name="Revenue",
                marker=dict(
                    color=reg_data["revenue"],
                    colorscale=[[0, "#2d1b69"], [0.5, "#7b68ee"], [1, "#00bfff"]],
                    cornerradius=6,
                ),
                text=reg_data["revenue"].apply(lambda x: f"৳{x:,.0f}"),
                textposition="outside",
                textfont=dict(size=10),
            ))
            fig_region.update_layout(
                **PLOTLY_LAYOUT,
                title="Revenue by Region",
                xaxis_title="Region (Division)",
                yaxis_title="Revenue (৳)",
                height=420,
            )
            st.plotly_chart(fig_region, use_container_width=True)

        with reg_col2:
            fig_reg_pie = go.Figure(go.Pie(
                labels=reg_data["region"],
                values=reg_data["customers"],
                hole=0.5,
                marker=dict(colors=COLORS[:len(reg_data)]),
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig_reg_pie.update_layout(
                **PLOTLY_LAYOUT,
                title="Customer Distribution by Region",
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig_reg_pie, use_container_width=True)

        # ── Drill-Down: District Performance ────────────────────────
        st.markdown("""
        <div class="section-header"><h2>🏪 District Drill-Down</h2></div>
        """, unsafe_allow_html=True)

        drill_region = st.selectbox(
            "Select a region to drill down",
            options=sorted(filtered["region"].unique()),
            key="drill_region",
        )

        district_data = store_performance(
            filtered[filtered["region"] == drill_region], "district"
        ).head(15)

        fig_district = go.Figure(go.Bar(
            x=district_data["revenue"],
            y=district_data["district"],
            orientation="h",
            marker=dict(
                color=district_data["revenue"],
                colorscale=[[0, "#1a1a3e"], [0.5, "#7b68ee"], [1, "#44ddaa"]],
                cornerradius=6,
            ),
            text=district_data["revenue"].apply(lambda x: f"৳{x:,.0f}"),
            textposition="inside",
            textfont=dict(color="white", size=11),
        ))
        fig_district.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Top Districts in {drill_region}",
            xaxis_title="Revenue (৳)",
            yaxis=dict(autorange="reversed"),
            height=450,
        )
        st.plotly_chart(fig_district, use_container_width=True)

        # ── Region Summary Table ────────────────────────────────────
        with st.expander("📋 Full Regional Summary"):
            reg_display = reg_data.copy()
            reg_display["revenue"] = reg_display["revenue"].apply(lambda x: f"৳{x:,.0f}")
            reg_display["profit"] = reg_display["profit"].apply(lambda x: f"৳{x:,.0f}")
            reg_display.columns = [
                "Region", "Revenue", "Profit", "Transactions", "Customers", "Stores"
            ]
            st.dataframe(reg_display, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4: ADVANCED / ML
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("""
        <div class="section-header"><h2>🤖 Machine Learning Insights</h2></div>
        """, unsafe_allow_html=True)

        ml_tab1, ml_tab2 = st.tabs(["🧩 Customer Clustering", "📈 Sales Prediction"])

        # ── KMeans Clustering ───────────────────────────────────────
        with ml_tab1:
            st.markdown("#### KMeans Customer Clustering (on RFM Features)")
            n_clusters = st.slider("Number of clusters", 2, 8, 4, key="n_clusters")

            cluster_data = compute_clusters(filtered, n_clusters)

            fig_cluster = px.scatter_3d(
                cluster_data,
                x="recency",
                y="frequency",
                z="monetary",
                color="cluster",
                color_discrete_sequence=COLORS,
                opacity=0.6,
                hover_data=["coustomer_key", "rfm_score"],
            )
            fig_cluster.update_layout(
                **PLOTLY_LAYOUT,
                title="3D Customer Clusters (Recency × Frequency × Monetary)",
                height=550,
                scene=dict(
                    xaxis_title="Recency (days)",
                    yaxis_title="Frequency",
                    zaxis_title="Monetary (৳)",
                    bgcolor="rgba(15,15,35,0.8)",
                ),
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

            # Cluster summary
            cluster_summary = (
                cluster_data.groupby("cluster")
                .agg(
                    count=("coustomer_key", "count"),
                    avg_recency=("recency", "mean"),
                    avg_frequency=("frequency", "mean"),
                    avg_monetary=("monetary", "mean"),
                )
                .reset_index()
                .sort_values("avg_monetary", ascending=False)
            )
            cluster_summary["avg_monetary"] = cluster_summary["avg_monetary"].apply(
                lambda x: f"৳{x:,.0f}"
            )
            cluster_summary.columns = [
                "Cluster", "Customers", "Avg Recency (days)",
                "Avg Frequency", "Avg Revenue"
            ]
            st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

        # ── Sales Prediction ────────────────────────────────────────
        with ml_tab2:
            st.markdown("#### Linear Trend Sales Prediction (Next 6 Months)")

            pred_data = sales_prediction(filtered)

            fig_pred = go.Figure()
            # Actuals
            actual = pred_data[pred_data["type"] == "actual"]
            forecast = pred_data[pred_data["type"] == "forecast"]

            fig_pred.add_trace(go.Scatter(
                x=actual["period"],
                y=actual["predicted_revenue"],
                mode="lines+markers",
                name="Actual Revenue",
                line=dict(color="#7b68ee", width=2),
                marker=dict(size=5),
            ))
            fig_pred.add_trace(go.Scatter(
                x=forecast["period"],
                y=forecast["predicted_revenue"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#ff6b9d", width=3, dash="dot"),
                marker=dict(size=7, symbol="diamond"),
            ))
            fig_pred.update_layout(
                **PLOTLY_LAYOUT,
                title="Sales Trend with 6-Month Forecast",
                xaxis_title="Period",
                yaxis_title="Revenue (৳)",
                height=450,
                hovermode="x unified",
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            st.info(
                "📌 **Note:** This is a simple linear regression trend forecast. "
                "For production use, consider ARIMA, Prophet, or LSTM models."
            )

    # ── Footer: Download ────────────────────────────────────────────
    st.markdown("---")
    foot_col1, foot_col2, foot_col3 = st.columns([2, 1, 1])

    with foot_col1:
        st.markdown(
            f"<span style='color:#666; font-size:0.85rem;'>"
            f"Showing <strong>{len(filtered):,}</strong> of <strong>{len(df):,}</strong> "
            f"transactions after filters</span>",
            unsafe_allow_html=True,
        )

    with foot_col3:
        csv = filtered.head(10000).to_csv(index=False)
        st.download_button(
            label="⬇️ Download Filtered Data (CSV)",
            data=csv,
            file_name="filtered_retail_data.csv",
            mime="text/csv",
        )


# ── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
