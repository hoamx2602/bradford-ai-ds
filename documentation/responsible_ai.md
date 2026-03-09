# Responsible AI, Data Governance, and Communication

## Bias and fairness

### Examples of bias in the dataset

- **City imbalance:** Listings may be dominated by London or a few cities, leading to models that perform poorly or unfairly for underrepresented regions. *Mitigation:* Stratified sampling or city-specific models; report performance by city.
- **Property type imbalance:** Flats may outnumber detached houses. *Mitigation:* Report class/category distribution; consider stratification in train/test splits.
- **Missing EPC data:** EPC is missing for many listings; imputation or “Unknown” can introduce bias. *Mitigation:* Use `epc_missing_flag`; avoid over-interpreting EPC in under-represented segments.

### Use of visualisation

- Distribution and geographic plots (e.g. listings per city, price by city) are used to communicate imbalance and support decisions on stratification and fairness.

---

## Privacy

### Potential PII and sensitive data

- **Exact addresses** and listing identifiers (URLs) could support re-identification.
- **Descriptions** may contain names or contact details in rare cases.

### Mitigation

- **For sharing or publication:** Remove or hash full addresses; retain city and postcode_area only where possible. Retain URLs only for reproducibility in a controlled setting.
- **Data governance:** Access to raw data limited to the project team; processed versions used for modelling and reporting.

---

## Risks and mitigation

| Risk | Mitigation |
|------|------------|
| Location bias in price prediction | Stratify by city/region; report metrics per group; document geographic coverage. |
| Ethical scraping | Data collected via browser extension with human-like usage; terms of use and robots.txt considered; use for education/research. |
| Misuse for valuation or legal decisions | Dataset card and documentation state intended use (research/coursework); not for direct valuation or legal use without further validation. |

---

## Data governance

- **Data dictionary:** See `documentation/dataset_card.md`.
- **Provenance:** Raw data from Zoopla; collection method and date documented in `methodology.md`.
- **Known limitations:** Geographic and temporal coverage, missing EPC/area, sampling bias—documented in dataset card and methodology.
- **Intended use cases:** Research and coursework (AI & Data Science); regression/classification demos; not for commercial valuation or legal reliance.

---

## Communication

- **Dataset card** (`documentation/dataset_card.md`): Summarises purpose, composition, collection method, limitations, and data dictionary.
- **Visualisations** in notebooks and `reports/figures/` communicate data quality, imbalance, and model behaviour to support transparency and responsible use.
