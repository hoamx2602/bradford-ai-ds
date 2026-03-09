"""
Impute missing area_sqft using a RandomForest regressor.
Uses numeric and categorical features; intended to be run after basic cleaning.
Usage:
  python -m src.data_pipeline.impute_area [--input path] [--output path]
Or import and call impute_area(df) returning df with area_sqft filled.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def impute_area(df: pd.DataFrame, area_col: str = "area_sqft", random_state: int = 42) -> pd.DataFrame:
    """
    Impute missing area_sqft using RandomForest on price, bedrooms, bathrooms, property_type, city.
    """
    df = df.copy()
    if area_col not in df.columns:
        return df
    missing = df[area_col].isna()
    if not missing.any():
        return df
    # Features for prediction
    df["price_num"] = pd.to_numeric(df["price"], errors="coerce")
    for c in ["bedrooms", "bathrooms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    train_df = df[~missing].dropna(subset=[area_col, "price_num"])
    if len(train_df) < 50:
        # Fallback: median by property_type
        med = df.groupby("property_type")[area_col].transform("median")
        df[area_col] = df[area_col].fillna(med).fillna(df[area_col].median())
        return df
    X_cols = ["price_num", "bedrooms", "bathrooms"]
    for cat in ["property_type", "city"]:
        if cat in df.columns:
            le = LabelEncoder()
            df[cat + "_enc"] = le.fit_transform(df[cat].fillna("Unknown").astype(str))
            X_cols.append(cat + "_enc")
    X_train = train_df[X_cols].fillna(0)
    y_train = train_df[area_col]
    X_miss = df.loc[missing, X_cols].fillna(0)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state)
    model.fit(X_train, y_train)
    df.loc[missing, area_col] = model.predict(X_miss)
    # Drop temporary enc columns if desired (optional)
    for c in ["property_type_enc", "city_enc", "price_num"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    return df


def main():
    p = argparse.ArgumentParser(description="Impute missing area_sqft using RandomForest")
    p.add_argument("--input", type=Path, default=Path("data/cleaned/zoopla_cleaned.csv"))
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    out = args.output or args.input
    df = pd.read_csv(args.input)
    df = impute_area(df)
    df.to_csv(out, index=False)
    print("Saved to", out)


if __name__ == "__main__":
    main()
