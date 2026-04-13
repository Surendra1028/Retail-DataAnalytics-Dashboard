# 📊 Retail Analytics Dashboard

> A professional, full-stack data analytics project built on a **star-schema data warehouse** with 1M+ retail transactions. Features interactive Streamlit dashboard, customer segmentation (RFM), ML clustering, and comprehensive EDA.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)

---

## 🎯 Project Overview

This project simulates a **real-world retail analytics system** using relational data modeled as a star schema. It integrates 6 datasets (4 dimension tables + 1 fact table + 1 time dimension) to build a unified analytics layer and presents business insights through an interactive Streamlit dashboard.

### Key Features

- **Data Integration**: Star-schema merge of 6 relational datasets (1M+ transactions)
- **Feature Engineering**: Revenue, Profit (margin-based), CLV, temporal features
- **Exploratory Data Analysis**: Sales trends, product rankings, category analysis, regional performance
- **Customer Analytics**: RFM Segmentation, repeat vs new classification, high-value customer identification
- **Machine Learning**: KMeans clustering on RFM features, linear sales prediction
- **Premium Dashboard**: Dark glassmorphism theme, dynamic filters, drill-down analysis

---

## 📂 Dataset Description

| Dataset | Rows | Description |
|---------|------|-------------|
| `customer_dim.csv` | 9,191 | Customer ID, name, contact, NID |
| `item_dim.csv` | 264 | Product info, category, price, supplier |
| `store_dim.csv` | 726 | Store location (division, district, upazila) |
| `Trans_dim.csv` | 39 | Payment type (cash/card), bank name |
| `fact_table.csv` | 1,000,000 | Transaction facts (quantity, price, totals) |
| `time_dim.csv` | ~100K+ | Date/time dimension (date, hour, month, year) |

### Star Schema

```
                    ┌──────────────┐
                    │  time_dim    │
                    └──────┬───────┘
                           │ time_key
┌──────────────┐   ┌──────┴───────┐   ┌──────────────┐
│ customer_dim │───│  fact_table  │───│   item_dim   │
└──────────────┘   └──────┬───────┘   └──────────────┘
       coustomer_key      │ store_key, payment_key
                    ┌─────┴────────┐
               ┌────┴─────┐  ┌────┴─────┐
               │ store_dim│  │Trans_dim │
               └──────────┘  └──────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app/app.py
```

### 3. Run EDA Script (Optional)

```bash
python notebooks/eda.py
```

---

## 📁 Project Structure

```
project/
├── datasets/              # Raw CSV data files
│   ├── customer_dim.csv
│   ├── item_dim.csv
│   ├── store_dim.csv
│   ├── Trans_dim.csv
│   ├── fact_table.csv
│   └── time_dim.csv
├── notebooks/
│   └── eda.py             # Exploratory data analysis script
├── app/
│   └── app.py             # Streamlit dashboard (main app)
├── src/
│   ├── __init__.py
│   ├── data_merge.py      # Data integration & feature engineering
│   └── analysis.py        # Analytics & ML functions
├── requirements.txt
└── README.md
```

---

## 🖥️ Dashboard Features

### Sidebar Filters
- 📅 Date range picker
- 🌍 Region (Division) multi-select
- 🏪 District multi-select (dynamic, filtered by region)
- 📦 Category multi-select
- 💳 Payment type multi-select

### KPI Cards
- 💰 Total Revenue with Profit delta
- 📦 Total Orders with Items count
- 👥 Unique Customers with Best Region
- 📊 Average Order Value with Top Category

### Charts & Visualizations
- 📈 **Line Chart** — Monthly/yearly sales trend with revenue & profit
- 📊 **Bar Chart** — Top N products by revenue
- 🥧 **Donut Chart** — Category-wise revenue distribution
- 🗓️ **Heatmap** — Sales by month × year
- 🎯 **Scatter Plot** — RFM customer segments
- 🧩 **3D Scatter** — KMeans customer clusters
- 📈 **Forecast Chart** — 6-month sales prediction

### Advanced Features
- Dynamic filtering across all charts
- Region → District drill-down analysis
- Downloadable filtered data (CSV)
- Expandable detailed data tables

---

## 🔧 Feature Engineering

| Feature | Formula / Method |
|---------|-----------------|
| Revenue | `total_price` from fact table |
| Profit | `revenue × margin` (15% Food, 20% Beverage, 10% Other) |
| CLV | Sum of all revenue per customer |
| Month/Year/Quarter | Extracted from parsed transaction date |
| Transaction Count | Unique transactions per customer |
| RFM Scores | Quartile-based (1–4) for Recency, Frequency, Monetary |
| Customer Segments | Champions, Loyal, At Risk, Lost |

---

## 👥 Customer Segmentation (RFM)

| Segment | Score Range | Description |
|---------|-------------|-------------|
| 🏆 Champions | 10–12 | Recent, frequent, high spenders |
| 💎 Loyal Customers | 7–9 | Regular buyers with good spending |
| ⚠️ At Risk | 5–6 | Declining engagement |
| 🔴 Lost Customers | 3–4 | Haven't purchased recently |

---

## 🤖 ML Components

1. **KMeans Clustering**: Groups customers based on standardized RFM features (configurable 2–8 clusters)
2. **Sales Prediction**: Linear regression trend with 6-month forecast

---

## 📊 KPI Metrics

- Total Revenue & Profit
- Total Transactions
- Unique Customers
- Average Order Value (AOV)
- Top Category & Best Store
- Best Performing Region

---

## 🛠️ Technologies

- **Python 3.10+**
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Plotly** — Interactive visualizations
- **Streamlit** — Dashboard framework
- **scikit-learn** — ML (KMeans, LinearRegression)

---

## 📝 License

This project is for educational and portfolio purposes.
