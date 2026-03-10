"""
Streamlit app: Zoopla Property Dataset – AI & Data Science (OIM7507-B)
Data overview, quality, distributions, map and model. Supports CSV upload for analysis.
Run from final_v1: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Project root (directory containing app.py)
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "zoopla_raw.csv"
DATA_LABELED = ROOT / "data" / "processed" / "zoopla_labeled.csv"
DATA_QUALITY = ROOT / "reports" / "tables" / "data_quality_summary.csv"
FIG_DIR = ROOT / "reports" / "figures"
MAP_HTML = FIG_DIR / "map_by_city.html"
MODEL_PATH = ROOT / "models" / "price_rf.pkl"

st.set_page_config(page_title="Zoopla Dataset – AI & Data Science", layout="wide")

st.title("🏠 Zoopla UK Property Dataset")
st.caption("OIM7507-B AI and Data Science – Group Assignment | University of Bradford")

# --- Upload CSV (sidebar) ---
st.sidebar.subheader("📤 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file to run all analysis on your data", type=["csv"], help="If provided, all sections below will use this dataset.")
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.session_state["uploaded_df"] = df_uploaded
        st.sidebar.success(f"Loaded **{len(df_uploaded):,}** rows, **{len(df_uploaded.columns)}** columns.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.session_state.pop("uploaded_df", None)
else:
    if "uploaded_df" in st.session_state:
        del st.session_state["uploaded_df"]
    st.sidebar.info("No file uploaded → using default data in `data/`.")

def get_data():
    """Return the active DataFrame: prefer uploaded, else labeled or raw."""
    if "uploaded_df" in st.session_state:
        return st.session_state["uploaded_df"].copy()
    if DATA_LABELED.exists():
        return pd.read_csv(DATA_LABELED)
    if DATA_RAW.exists():
        return pd.read_csv(DATA_RAW)
    return None

def get_data_source_label():
    if "uploaded_df" in st.session_state:
        return "**Source: your uploaded CSV file**"
    if DATA_LABELED.exists():
        return "**Source: data/processed/zoopla_labeled.csv**"
    if DATA_RAW.exists():
        return "**Source: data/raw/zoopla_raw.csv**"
    return ""

# Sidebar: section
section = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Data & Statistics",
        "Data Quality",
        "Distributions & Exploration",
        "Map by City",
        "Price Prediction Model",
    ],
)

df = get_data()
source_label = get_data_source_label()

# --- Overview ---
if section == "Overview":
    st.header("Project Overview")
    if source_label:
        st.info(source_label)
    st.markdown("""
    - **Data source:** Zoopla.co.uk (UK residential for-sale listings), collected via browser extension.
    - **Pipeline:** Raw → Cleaning & Feature Engineering → Annotation (price_category) → ML demo.
    - **Structure:** `data/raw`, `data/cleaned`, `data/processed`, `reports/figures`, `models/`.
    """)
    if df is not None:
        st.subheader("Data – first 5 rows")
        st.dataframe(df.head(), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Columns", ", ".join(df.columns[:5].tolist()) + ("..." if len(df.columns) > 5 else ""))
    else:
        st.warning("No data available. Upload a CSV in the sidebar or ensure `data/raw/zoopla_raw.csv` exists.")

# --- Data & Statistics ---
elif section == "Data & Statistics":
    st.header("Data & Statistics")
    if source_label:
        st.info(source_label)
    if df is None:
        st.warning("No data. Upload a CSV or run notebooks 01→04.")
    else:
        df = df.copy()
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        if "price" in df.columns:
            c3.metric("Mean price (£)", f"{df['price'].mean():,.0f}")
        else:
            c3.metric("Numeric columns", len(df.select_dtypes(include=[np.number]).columns))
        st.subheader("Sample data")
        st.dataframe(df.head(100), use_container_width=True)
        if "price_category" in df.columns:
            st.subheader("price_category distribution")
            st.bar_chart(df["price_category"].value_counts())
        elif "city" in df.columns:
            st.subheader("Distribution by city (top 15)")
            st.bar_chart(df["city"].value_counts().head(15))

# --- Data Quality ---
elif section == "Data Quality":
    st.header("Data Quality Report")
    if source_label:
        st.info(source_label)
    if df is None:
        st.warning("No data. Upload a CSV or run notebook 02.")
    else:
        # Always computed from current df (upload or default)
        q = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null": df.count().values,
            "null_count": df.isna().sum().values,
        })
        q["null_pct"] = (q["null_count"] / len(df) * 100).round(2)
        q["example"] = df.iloc[0].astype(str).values
        st.dataframe(q, use_container_width=True)
        st.subheader("Missing rate by column (%)")
        st.bar_chart(q.set_index("column")["null_pct"])

# --- Distributions & Exploration ---
elif section == "Distributions & Exploration":
    st.header("Distributions & Exploration")
    if source_label:
        st.info(source_label)
    if df is None:
        st.warning("No data. Upload a CSV or run notebooks 01→04.")
    else:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_plot_cols = ["price", "bedrooms", "bathrooms", "area_sqft", "price_per_sqft"] + [c for c in numeric_cols if c not in ["price", "bedrooms", "bathrooms", "area_sqft", "price_per_sqft"]]
        all_plot_cols = [c for c in all_plot_cols if c in df.columns]
        if not all_plot_cols:
            all_plot_cols = numeric_cols[:10] if numeric_cols else list(df.columns)[:5]
        col = st.selectbox("Select variable for histogram", all_plot_cols, index=0)
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(x) > 0:
                if col == "price" and x.max() > 1e6:
                    x = x[x.between(x.quantile(0.01), x.quantile(0.99))]
                st.bar_chart(x.value_counts().sort_index().head(50))
        if "city" in df.columns and "price" in df.columns:
            st.subheader("Price by city (top 10)")
            df["price_num"] = pd.to_numeric(df["price"], errors="coerce")
            df_valid = df.dropna(subset=["price_num"])
            top_cities = df_valid["city"].value_counts().head(10).index.tolist()
            df_top = df_valid[df_valid["city"].isin(top_cities)]
            by_city = df_top.groupby("city")["price_num"].agg(["mean", "count"]).round(0)
            st.dataframe(by_city, use_container_width=True)
        elif "city" in df.columns:
            st.subheader("Count by city (top 10)")
            st.bar_chart(df["city"].value_counts().head(10))

# --- Map ---
elif section == "Map by City":
    st.header("Map by Region (City)")
    if source_label:
        st.info(source_label)
    if df is not None and "city" in df.columns:
        price_col = "price" if "price" in df.columns else None
        if not price_col:
            for c in ["price_num"]:
                if c in df.columns:
                    price_col = c
                    break
        if price_col:
            df_map = df.copy()
            df_map["_price_num"] = pd.to_numeric(df_map[price_col], errors="coerce")
            first_col = df_map.columns[0]
            agg = df_map.groupby("city").agg(mean_price=("_price_num", "mean"), count=(first_col, "count")).reset_index()
            agg = agg[agg["count"] >= 5]
            if len(agg) > 0:
                try:
                    import folium
                    CITY_COORDS = {
                        "London": (51.5074, -0.1278), "Manchester": (53.4808, -2.2426),
                        "Birmingham": (52.4862, -1.8904), "Leeds": (53.8008, -1.5491),
                        "West Yorkshire": (53.8008, -1.5491), "Liverpool": (53.4084, -2.9916),
                        "Cardiff": (51.4816, -3.1791), "Bristol": (51.4545, -2.5879),
                        "Sheffield": (53.3811, -1.4701), "Newcastle upon Tyne": (54.9783, -1.6178),
                        "Nottingham": (52.9548, -1.1581), "Leicester": (52.6369, -1.1398),
                        "Edinburgh": (55.9533, -3.1883), "Glasgow": (55.8642, -4.2518),
                        "Southampton": (50.9097, -1.4044), "Brighton": (50.8225, -0.1372),
                        "Reading": (51.4543, -0.9731), "Surrey": (51.3148, -0.5597),
                        "Essex": (51.5742, 0.4857), "Kent": (51.2787, 0.5217),
                    }
                    UK_CENTRE = (54.0, -2.5)
                    m = folium.Map(location=[54.0, -2.5], zoom_start=6, tiles="CartoDB positron")
                    for _, row in agg.iterrows():
                        lat, lon = CITY_COORDS.get(row["city"], UK_CENTRE)
                        radius = min(50, 10 + row["count"] / 80)
                        color = "darkred" if row["mean_price"] > 500_000 else "orange" if row["mean_price"] > 300_000 else "green"
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=radius,
                            color=color,
                            fill=True,
                            fill_opacity=0.6,
                            popup=f"{row['city']}<br>Listings: {int(row['count'])}<br>Mean price: £{row['mean_price']:,.0f}",
                        ).add_to(m)
                    st.components.v1.html(m._repr_html_(), height=500, scrolling=False)
                except Exception as e:
                    st.warning(f"Could not build map from data: {e}. Using default map.")
                    if MAP_HTML.exists():
                        with open(MAP_HTML, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=500, scrolling=True)
            else:
                st.warning("Not enough cities (need ≥5 listings per city).")
                if MAP_HTML.exists():
                    with open(MAP_HTML, "r", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=500, scrolling=True)
        else:
            st.warning("A price column is required to show mean price by city on the map.")
            if MAP_HTML.exists():
                with open(MAP_HTML, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=500, scrolling=True)
    else:
        if MAP_HTML.exists():
            with open(MAP_HTML, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=500, scrolling=True)
        else:
            st.warning("No map available. Upload a CSV with city and price columns or run notebook 02.")

# --- Model ---
elif section == "Price Prediction Model":
    st.header("Price Prediction Model (Random Forest)")
    if source_label and "uploaded_df" in st.session_state:
        st.info("Model was trained on the default dataset. Feature importance is from the saved model.")
    if not MODEL_PATH.exists():
        st.info("No `models/price_rf.pkl` found. Run notebook 05 to train and save the model.")
    else:
        import joblib
        model = joblib.load(MODEL_PATH)
        st.success("Random Forest model loaded.")
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=[f"f{i}" for i in range(len(model.feature_importances_))])
            st.subheader("Feature importance (top 15)")
            st.bar_chart(imp.nlargest(15))
        st.markdown("See notebook `05_dataset_readiness_demo.ipynb` for RMSE, MAE, R² and predicted vs actual.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Final v1** – Zoopla Dataset")
