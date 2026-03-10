"""
Streamlit app: Zoopla Property Dataset – AI & Data Science (OIM7507-B)
Tổng hợp dữ liệu, chất lượng, phân phối, bản đồ và mô hình.
Chạy từ thư mục final_v1: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Project root (thư mục chứa app.py)
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

# Sidebar: chọn section
section = st.sidebar.radio(
    "Chọn mục",
    [
        "Tổng quan",
        "Dữ liệu & Thống kê",
        "Chất lượng dữ liệu",
        "Phân phối & Khám phá",
        "Bản đồ theo thành phố",
        "Mô hình dự đoán giá",
    ],
)

# --- Tổng quan ---
if section == "Tổng quan":
    st.header("Tổng quan dự án")
    st.markdown("""
    - **Nguồn dữ liệu:** Zoopla.co.uk (bất động sản bán tại UK), thu thập qua browser extension.
    - **Quy trình:** Raw → Cleaning & Feature Engineering → Annotation (price_category) → ML demo.
    - **Cấu trúc:** `data/raw`, `data/cleaned`, `data/processed`, `reports/figures`, `models/`.
    """)
    if DATA_RAW.exists():
        df = pd.read_csv(DATA_RAW, nrows=5)
        st.subheader("Raw data – 5 dòng đầu")
        st.dataframe(df.head(), use_container_width=True)
        n = sum(1 for _ in open(DATA_RAW)) - 1
        st.metric("Số dòng (raw)", f"{n:,}")
    else:
        st.warning("Chưa có file `data/raw/zoopla_raw.csv`. Chạy pipeline hoặc copy CSV vào.")

# --- Dữ liệu & Thống kê ---
elif section == "Dữ liệu & Thống kê":
    st.header("Dữ liệu đã gán nhãn & thống kê")
    if not DATA_LABELED.exists():
        st.warning("Chưa có `data/processed/zoopla_labeled.csv`. Chạy notebook 01→04 trước.")
    else:
        df = pd.read_csv(DATA_LABELED)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        c1, c2, c3 = st.columns(3)
        c1.metric("Số bản ghi", f"{len(df):,}")
        c2.metric("Số cột", len(df.columns))
        c3.metric("Giá trung bình (£)", f"{df['price'].mean():,.0f}")
        st.subheader("Mẫu dữ liệu")
        st.dataframe(df.head(100), use_container_width=True)
        if "price_category" in df.columns:
            st.subheader("Phân bố price_category")
            st.bar_chart(df["price_category"].value_counts())

# --- Chất lượng dữ liệu ---
elif section == "Chất lượng dữ liệu":
    st.header("Báo cáo chất lượng dữ liệu")
    if not DATA_QUALITY.exists():
        st.warning("Chưa có `reports/tables/data_quality_summary.csv`. Chạy notebook 02 trước.")
    else:
        q = pd.read_csv(DATA_QUALITY)
        st.dataframe(q, use_container_width=True)
        st.subheader("Tỷ lệ thiếu theo cột (%)")
        if "null_pct" in q.columns:
            st.bar_chart(q.set_index("column")["null_pct"])

# --- Phân phối & Khám phá ---
elif section == "Phân phối & Khám phá":
    st.header("Phân phối & Khám phá")
    if not DATA_LABELED.exists():
        st.warning("Cần có `zoopla_labeled.csv`. Chạy notebook 01→04 trước.")
    else:
        df = pd.read_csv(DATA_LABELED)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])
        col = st.selectbox("Chọn biến để vẽ histogram", ["price", "bedrooms", "bathrooms", "area_sqft", "price_per_sqft"], index=0)
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            if col == "price":
                x = x[x.between(x.quantile(0.01), x.quantile(0.99))]
            st.bar_chart(x.value_counts().sort_index().head(50))
        st.subheader("Giá theo thành phố (top 10)")
        top_cities = df["city"].value_counts().head(10).index.tolist()
        df_top = df[df["city"].isin(top_cities)]
        by_city = df_top.groupby("city")["price"].agg(["mean", "count"]).round(0)
        st.dataframe(by_city, use_container_width=True)

# --- Bản đồ ---
elif section == "Bản đồ theo thành phố":
    st.header("Bản đồ theo khu vực (thành phố)")
    if not MAP_HTML.exists():
        st.warning("Chưa có `reports/figures/map_by_city.html`. Chạy notebook 02 (ô map) trước.")
    else:
        with open(MAP_HTML, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=500, scrolling=True)

# --- Mô hình ---
elif section == "Mô hình dự đoán giá":
    st.header("Mô hình dự đoán giá (Random Forest)")
    if not MODEL_PATH.exists():
        st.info("Chưa có file `models/price_rf.pkl`. Chạy notebook 05 để train và lưu mô hình.")
    else:
        import joblib
        model = joblib.load(MODEL_PATH)
        st.success("Đã load mô hình Random Forest.")
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=[f"f{i}" for i in range(len(model.feature_importances_))])
            st.subheader("Feature importance (top 15)")
            st.bar_chart(imp.nlargest(15))
        st.markdown("Dùng notebook `05_dataset_readiness_demo.ipynb` để xem RMSE, MAE, R² và predicted vs actual.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Final v1** – Zoopla Dataset")
