# Methodology

## Data acquisition

### Source and scope

- **Source:** Zoopla.co.uk (for-sale residential listings).
- **Scope:** Listings captured via a browser extension during browsing sessions; coverage is determined by which pages were visited (cities, filters, date range).

### Justification for collection method

Data was collected using a **browser extension** (`zoopla_extension`) with backend support (`zoopla_crawler_app`):

- **Anti-bot avoidance:** Zoopla employs anti-scraping measures; a real browser session reduces blocking.
- **JavaScript-rendered content:** Listing details are loaded dynamically; the extension runs in the page context and captures the final DOM.
- **Session and cookies:** Uses the same session as a normal user, reducing CAPTCHA and access issues.
- **Realistic traffic:** Requests originate from a real browser.

### Limitations and mitigation

- **Slower collection** and **potential sampling bias** (e.g. London-heavy). Mitigation: document capture dates, log metadata, randomise navigation where possible, and state sampling criteria clearly.
- **Reproducibility:** Exact snapshot depends on date and browsing path; re-runs may differ. Mitigation: version the raw CSV and document capture window.

### Technology and process

- **Stack:** Python, pandas, browser extension (JavaScript), backend aggregator.
- **Flow:** Zoopla pages → extension capture → backend aggregator → CSV (`data/raw/zoopla_raw.csv`).

### Data quality and processing

- **Cleaning:** Preprocessing (price/area numeric, EPC standardised), duplicate removal (url; fallback address+price), missing value handling (median imputation by property_type; EPC → "Unknown" + missing flag), winsorisation of price (1–99%).
- **Feature engineering:** price_per_sqft, total_rooms, size_category, epc_score, text flags (has_garden, has_parking, has_balcony, is_renovated), postcode_area from address.
- **Annotation:** price_category (affordable / mid_range / luxury) by fixed price thresholds; see `documentation/annotation_rules.md`.

### Model and evaluation

- **Task:** Regression (predict price). Optional: classification (predict price_category).
- **Train/test:** 80/20 split; baseline Linear Regression and Random Forest; metrics: RMSE, MAE, R²; visualisation: predicted vs actual, feature importance.
