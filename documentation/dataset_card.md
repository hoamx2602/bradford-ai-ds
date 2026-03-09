# Dataset Card: Zoopla UK Property Listings

## Summary

- **Purpose:** Support analysis of factors influencing UK residential property prices and training of price prediction / price-category classification models.
- **Composition:** Listings collected from Zoopla (for-sale), including price, location (city, address), property type, bedrooms, bathrooms, living rooms, area (sqft), EPC rating, and free-text description.
- **Collection method:** Browser extension capturing listing pages, with backend aggregation to CSV. See `documentation/methodology.md`.
- **Size:** ~14,400+ records (target ≥12k); 13+ raw columns; expanded with engineered features (price_per_sqft, EPC score, text flags, postcode area).
- **Intended use:** Research and coursework (AI & Data Science); regression (predict price) or classification (predict price_category: affordable / mid_range / luxury).
- **Out-of-scope:** Commercial deployment without further validation; use for valuations or legal decisions.

## Limitations

- **Geographic bias:** Coverage depends on browsing (e.g. London-heavy). Generalisation to underrepresented cities may be poor.
- **Temporal:** Snapshot at capture date; prices and availability change.
- **Missing data:** area_sqft and epc_rating often missing; imputation and missing-indicator strategies applied.
- **Sampling:** Not a random sample of all UK listings; reproducibility depends on documented capture criteria.

## Data Dictionary (key variables)

| Variable         | Type    | Description |
|------------------|---------|-------------|
| id               | int     | Row/listing ID |
| url              | object  | Zoopla listing URL (unique key) |
| city             | object  | City name |
| price            | float   | Price in GBP |
| address          | object  | Full or partial address |
| property_type    | object  | Flat/Apartment, Terraced, Detached, Semi-Detached, etc. |
| bedrooms         | int     | Number of bedrooms |
| bathrooms        | int     | Number of bathrooms |
| living_rooms     | int     | Number of reception/living rooms |
| area_sqft        | float   | Internal area in sq ft (may be imputed) |
| description      | object  | Listing description text |
| created_at       | object  | Capture timestamp |
| epc_rating       | object  | A–G or Unknown |
| price_category   | object  | affordable / mid_range / luxury (after annotation) |
| price_per_sqft   | float   | price / area_sqft (engineered) |
| epc_score        | int     | 7 (A) … 1 (G), 0 (Unknown) |
| has_garden, has_parking, has_balcony, is_renovated | int | 0/1 from description |
| postcode_area    | object  | UK postcode area (e.g. SW11, M27) |

## Provenance

- **Source:** Zoopla.co.uk (for-sale listings).
- **Collection tool:** Browser extension + backend aggregator (see methodology).
- **Processing:** Cleaning and feature engineering in `notebooks/03_*`; annotation in `notebooks/04_*`.

## Ethics & Responsible AI

See `documentation/responsible_ai.md` for bias, fairness, privacy, and governance.
