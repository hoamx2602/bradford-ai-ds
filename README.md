# Zoopla Property Dataset – AI & Data Science Group Assignment

Project for **OIM7507-B AI and Data Science** (University of Bradford).  
Dataset: UK residential property listings from Zoopla (browser extension collection).

## Structure

```
final_v1/
├── data/
│   ├── raw/          # zoopla_raw.csv (from crawler)
│   ├── cleaned/      # zoopla_cleaned.csv
│   └── processed/    # zoopla_features.csv, zoopla_labeled.csv
├── notebooks/
│   ├── 01_data_acquisition_and_validation.ipynb
│   ├── 02_data_quality_and_visual_exploration.ipynb
│   ├── 03_data_cleaning_and_feature_engineering.ipynb
│   ├── 04_annotation_and_labeling.ipynb
│   └── 05_dataset_readiness_demo.ipynb
├── documentation/
│   ├── dataset_card.md
│   ├── methodology.md
│   ├── responsible_ai.md
│   └── annotation_rules.md
├── reports/
│   ├── figures/      # plots from notebooks
│   └── tables/       # data_quality_summary.csv etc.
├── src/
│   └── data_pipeline/
│       └── impute_area.py
├── models/           # price_rf.pkl
├── report/           # group_report.docx (final report)
├── app.py            # Streamlit overview app
└── requirements.txt
```

## Setup

```bash
cd final_v1
pip install -r requirements.txt
```

Place raw CSV as `data/raw/zoopla_raw.csv` (or copy from `properties_202603071243.csv`).

## Run notebooks

Run in order 01 → 02 → 03 → 04 → 05 from project root (`final_v1`) or from `notebooks/` (paths adjust automatically).

- **01:** Data acquisition and validation; raw metadata.
- **02:** Data quality, missingness, distributions, outliers, correlation; saves figures and `reports/tables/data_quality_summary.csv`.
- **03:** Cleaning, feature engineering; saves `zoopla_cleaned.csv` and `zoopla_features.csv`.
- **04:** Annotation (price_category); saves `zoopla_labeled.csv`.
- **05:** Baseline ML (Linear Regression, Random Forest); saves `models/price_rf.pkl` and figures.

## Run Streamlit app (overview)

```bash
cd final_v1
streamlit run app.py
```

The app includes: overview, data & statistics, data quality, distributions & exploration, map by city, and price prediction model. You can upload your own CSV to run all analysis on it.

## Report

Final group report: `report/group_report.docx`.  
Individual contributions: `documentation/individual_contributions.md` (200–400 words per member).
