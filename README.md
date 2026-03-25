# Polish Apartment Price Estimator

XGBoost regression model that predicts apartment prices across 15 Polish cities, trained on 28,310 listings scraped from Otodom.pl. Includes a full scraping pipeline, exploratory analysis, SHAP feature importance, and an interactive Streamlit app for price estimation and reverse lookup.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)



---

## Dataset

**Source:** [Otodom.pl](https://www.otodom.pl) — the dominant Polish real estate listings platform
**Scraped:** March 2026
**Method:** `requests` + `BeautifulSoup` — each listing page embeds a `__NEXT_DATA__` JSON block (Next.js SSR) that contains the full structured property data without requiring a headless browser.

| Property | Value |
|---|---|
| Raw listings scraped | 28,934 |
| After cleaning | 28,310 |
| Cities | 15 |
| Property type | Apartments only (`estate == "FLAT"`) |
| Price range (kept) | 50,000 – 5,000,000 PLN |
| Area range (kept) | 15 – 250 m² |
| Price/m² ceiling | 40,000 PLN/m² |

**15 cities:** Białystok, Bydgoszcz, Gdańsk, Katowice, Kraków, Lublin, Łódź, Poznań, Rzeszów, Szczecin, Toruń, Warszawa, Wrocław, Kielce, Olsztyn.

**Scraping strategy:** large cities (Warszawa, Kraków, Wrocław, Łódź, Poznań, Gdańsk) scraped per-district via `topup_districts.py` to bypass Otodom's pagination limits — yields 3–14× more listings than city-level scraping alone.

### Scraping notes

Otodom.pl uses Next.js server-side rendering. Every listing page contains a `<script id="__NEXT_DATA__">` tag with the full JSON payload — no JavaScript execution required. The scraper extracts:

- Price and price per m² directly from the JSON root
- Area (`areaInSquareMeters`) and room count (`roomsNumber`) as structured fields — room count arrives as an enum string (`ONE`, `TWO`, `THREE`...) and floor as `GROUND`, `FIRST`, `SECOND`..., both mapped to integers
- Neighborhood via `location.reverseGeocoding.locations` — filtered by `locationLevel == "district"` for district-level names (Mokotów, Wola, Żoliborz) rather than the city-level fallback that `location.address.district.name` returns for ~95% of listings
- `isPrivateOwner` boolean distinguishing private sellers from agencies

---

## Pipeline

1. **`scrape_otodom.py`** — scrapes Otodom.pl list pages via `__NEXT_DATA__` JSON (no headless browser); outputs `otodom_all.csv`
2. **`topup_districts.py`** — re-scrapes large cities (Warsaw, Kraków, Wrocław, Łódź, Poznań, Gdańsk) at district level to bypass pagination limits; merges and deduplicates by URL
3. **`housing_price.ipynb`** — EDA, cleaning, feature engineering, XGBoost training, SHAP explainability, exports model artefacts
4. **`app.py`** — Streamlit app loading the artefacts; two modes: Estimate Price and Reverse Lookup

---

## Features

| Feature | Description | Null % |
|---|---|---|
| `area_m2` | Apartment area in square metres | 0% |
| `rooms` | Number of rooms (Polish convention: bedrooms + living room) | 1.2% |
| `floor` | Floor number (0 = ground floor) | 3.8% |
| `city_enc` | LabelEncoded city | 0% |
| `neighborhood_enc` | LabelEncoded district-level neighborhood | 0.6% |
| `is_private_owner` | True = private seller, False = agency | 2.1% |

`sub_neighborhood` (residential-level geocoding) was extracted but dropped from the model due to 63.5% null rate. `city` and `neighborhood` string columns are retained in the CSV for display purposes.

---

## Results

### Model performance

| Metric | Value |
|---|---|
| R² | 0.790 |
| MAE | 136,197 PLN |
| MAPE | 15.7% |
| Train / test split | 80 / 20 |
| Target transformation | log1p(price) |

### City-level MAE

| City | N (test) | MAE (PLN) | Notes |
|---|---|---|---|
| Łódź | 490 | 65,962 | Lowest prices — tightest absolute error |
| Lublin | 159 | 67,137 | |
| Kielce | 152 | 78,901 | |
| Poznań | 377 | 124,490 | |
| Wrocław | 744 | 141,039 | |
| Kraków | 913 | 156,279 | |
| Warszawa | 1,485 | 160,473 | High variance across districts |
| Gdańsk | 395 | 211,000 | Jelitkowo premium district inflates error |

3× more training data pushed R² from 0.627 → 0.790 and MAPE from 18.4% → 15.7%.

### SHAP — top features

| Feature | Mean |SHAP| | Direction |
|---|---|---|
| `area_m2` | highest | Larger area → higher price |
| `neighborhood_enc` | high | Premium districts add significant value |
| `city_enc` | high | Warsaw/Kraków baseline is materially higher |
| `rooms` | medium | More rooms → higher price, but collinear with area |
| `floor` | low | Ground floor and top floor show slight discounts |
| `is_private_owner` | low | Agency listings priced slightly higher on average |

---

## Visualizations

### Price distribution by city
![Price distribution by city](assets/price_distribution.png)

*Warsaw and Kraków distributions skew right with fat tails. Łódź and Katowice cluster tightly at lower price/m² values.*

### Price vs area
![Price vs area](assets/price_vs_area.png)

*Clear positive relationship. Warsaw data points (orange) sit consistently above the regression line for other cities.*

### Neighborhood price gaps — top 5 cities
![Neighborhood prices](assets/neighborhood_prices.png)

*District-level price spread within Warsaw reaches 10,000+ PLN/m² between Śródmieście and outer districts. Justifies the neighborhood feature.*

### Actual vs predicted
![Actual vs predicted](assets/actual_vs_predicted.png)

*R² = 0.790. Main source of error: luxury segment above 1.5M PLN where the model systematically underestimates.*

### SHAP — global feature importance
![SHAP bar](assets/shap_importance.png)

*Area dominates. Neighborhood and city encoding together account for most of the location premium.*

### SHAP — beeswarm
![SHAP beeswarm](assets/shap_beeswarm.png)

*High area values (red) push predictions up consistently. Neighborhood encoding shows wide spread — the location effect is non-linear and city-specific.*

---

## Streamlit App

Two modes accessible via tabs at the top of the page.

### Estimate Price

![App — Estimate Price 1/2](assets/app_estimate1.png)

![App — Estimate Price 2/2](assets/app_estimate2.png)

Input parameters in a horizontal bar: city, neighborhood, area (m²), rooms, floor, private seller toggle. After clicking **Estimate Price**:

- **4 KPI metrics** — estimated total price, price/m², city median, % deviation from city median
- **Price distribution histogram** — selected neighborhood overlaid on full city distribution, estimate marked with a vertical line
- **Cross-city benchmark** — the same flat spec predicted across all 15 cities ranked by PLN/m²
- **Neighborhood ranking** — all districts in the selected city ranked by predicted PLN/m² for the given spec
- **Price sensitivity chart** — tornado diagram showing how ±20% area, ±1 room, ±1 floor shifts the estimate
- **Comparable listings table** — 6 closest real listings from the scraped dataset with hyperlinks to the original Otodom.pl pages (links may be inactive after sale)

### Reverse Lookup

![App — Reverse Lookup](assets/app_reverse.png)

Input: budget (PLN), minimum rooms, private seller toggle. After clicking **Find Apartments**:

- **Algorithm's Choice card** — the single city/neighborhood combination with the highest achievable area for the given budget
- **Purchasing power chart** — horizontal bar chart showing the maximum area (m²) achievable per city
- **Results table** — all matching cities with neighborhood, area, estimated price, PLN/m², and percentage of budget used

The reverse lookup uses binary search (20 iterations per city/neighborhood pair) to find the maximum area fitting within the budget.

---

## Key Takeaways

1. **Location is the dominant non-size factor.** Adding district-level neighborhood encoding reduced Warsaw MAE by 40% relative to a city-only model. The price gap between Śródmieście and outer Warsaw districts exceeds the entire price range of Łódź.
2. **Room count adds limited independent signal.** After controlling for area, rooms contribute modestly — the two features are highly correlated (r ≈ 0.72). Polish listings report room count using a convention that excludes kitchen unless open-plan, creating additional noise.
3. **Floor matters less than expected.** Ground floor and top floor discounts are statistically present but economically small (~20–30k PLN effect for a 50m² flat), consistent with a market where elevator availability and building type matter more than floor number alone.
4. **Private seller vs agency has minimal price signal.** The `is_private_owner` feature has near-zero SHAP importance. Agencies may price similarly to private sellers at listing stage; actual transaction prices likely differ.
5. **Log-transforming the target was essential.** Raw price residuals showed severe heteroscedasticity. With log1p transformation, residuals are approximately normal and the model generalises better across the full price range.

---

## Limitations & potential improvements

| Limitation | Why | Potential fix |
|---|---|---|
| No apartment condition | Otodom exposes `stan wykończenia` (`move-in ready` / `needs renovation` / `shell`) only on individual listing pages, not on list view | Scrape each of the 28k listing URLs separately — ~28,000 additional requests, estimated +0.05–0.08 R² |
| No year built | Same — detail page only | Same enrichment pass |
| No building type | (block / tenement / new development) — detail page only | Same enrichment pass |
| Label encoding for city/neighborhood | Ordinal encoding implies false ordering between cities | Replace with target encoding (mean price per city/neighborhood) or one-hot |
| Static dataset | Prices scraped March 2026 — market drifts | Schedule monthly re-scrape via `scrape_otodom.py` + `topup_districts.py` |

The scraping architecture already supports the enrichment pass — each `Listing` object stores the original URL, so a follow-up script can iterate `otodom_all.csv`, fetch individual pages, and join the extra fields without re-scraping the full dataset.

---

## Running Locally

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Scrape fresh data (optional — CSV included in repo)
python scrape_otodom.py

# 3. Run the notebook to retrain (optional — model artefacts included)
# Open housing_price.ipynb in Jupyter / Colab and run all cells

# 4. Launch the app
streamlit run app.py
```

**Python version:** 3.11+
**Key dependencies:** `streamlit`, `xgboost`, `scikit-learn`, `shap`, `pandas`, `numpy`, `matplotlib`, `joblib`, `requests`, `beautifulsoup4`

---

## Deploying to Streamlit Community Cloud (free)

1. Push the repository to GitHub (include `model_artefacts/` and `data/raw/otodom_all.csv`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, branch `main`, main file `app.py`
4. Click Deploy — the app is live at `https://your-app-name.streamlit.app` within ~2 minutes
5. Update the badge URL at the top of this README

No paid infrastructure required. The free tier supports one active app with 1 GB RAM.

---

## File Structure

```
housing-scraper/
├── scrape_otodom.py          # Scraper — requests + BeautifulSoup, 15 cities
├── topup_districts.py        # Augments large cities with district-level scraping + dedup
├── housing_price.ipynb       # EDA, feature engineering, model training, SHAP
├── app.py                    # Streamlit app — Estimate Price + Reverse Lookup
├── requirements.txt
├── data/
│   └── raw/
│       └── otodom_all.csv    # 28,310 cleaned listings
├── model_artefacts/
│   ├── xgb_model.joblib      # Trained XGBoost model (~2.9 MB)
│   ├── le_city.joblib        # LabelEncoder for city
│   ├── le_neighborhood.joblib
│   ├── city_neighborhoods.json   # city → list of neighborhoods mapping
│   └── meta.json             # feature ranges for UI sliders
└── assets/                   # Screenshots for README
```
