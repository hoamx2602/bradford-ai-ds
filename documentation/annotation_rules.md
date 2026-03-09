# Annotation Rules: Price Category

## Primary label: `price_category`

| Label        | Condition (price in GBP) |
|-------------|---------------------------|
| affordable  | price < 300,000          |
| mid_range   | 300,000 ≤ price ≤ 800,000 |
| luxury      | price > 800,000          |

- Missing or invalid price → assign `unknown` (excluded from classification evaluation if needed).

## Rationale

- **affordable:** Aligns with first-time buyer and entry-level segment in many UK regions.
- **mid_range:** Covers a large share of family and mid-market transactions.
- **luxury:** Premium segment; often London and high-value areas.

## Subjectivity and limitations

- A £500k property may be mid-range nationally but expensive in smaller cities. For city-relative labelling, use `price_category_by_city` (e.g. city-specific tertiles or quantiles); see optional implementation in notebook 04.
- Thresholds are fixed; inflation or market changes would require periodic review.

## Reproducibility

- Implemented in `notebooks/04_annotation_and_labeling.ipynb` as a deterministic function of `price`.
- Output: `data/processed/zoopla_labeled.csv` with column `price_category`.

## Optional labels from description

- **has_garden,** **has_parking:** Derived from keyword search in description; validated on a small manually checked sample (see notebook 03 / 04).
