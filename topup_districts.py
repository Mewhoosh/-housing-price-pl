"""
topup_districts.py
==================
Augments existing scraped data with district-level listings for large cities
where city-level scraping hits Otodom's bot-detection / pagination limits.

Usage:
    python topup_districts.py

Reads:  data/raw/otodom_all.csv   (output of scrape_otodom.py)
Writes: data/raw/otodom_all.csv   (updated, deduplicated)
        data/raw/otodom_<city>.csv (per-city files, updated)

Why a separate script?
    scrape_otodom.py does a breadth-first pass across all cities.  For large
    cities Otodom returns ≤25 pages (~600-700 listings) before rate-limiting.
    This script goes one level deeper — city → district — so each district
    request is small enough to bypass the limit, typically yielding 3-5× more
    records for Warsaw, Kraków, Wrocław, Poznań, Gdańsk, and Łódź.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import pandas as pd

# Reuse fetch/parse logic from the main scraper — single source of truth.
from scrape_otodom import (
    BASE_URL,
    DELAY_HI,
    DELAY_LO,
    HEADERS,
    MAX_PAGES,
    OUTPUT_DIR,
    fetch_page,
    parse_listings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# District slugs for cities that need augmentation
# City-level slug is included first to capture listings without a district tag.
# ---------------------------------------------------------------------------

DISTRICT_SLUGS: dict[str, list[str]] = {
    "Warszawa": [
        "warszawa",                                 # catch-all (bez dzielnicy)
        "mazowieckie/warszawa/bemowo",
        "mazowieckie/warszawa/bialoleka",
        "mazowieckie/warszawa/bielany",
        "mazowieckie/warszawa/mokotow",
        "mazowieckie/warszawa/ochota",
        "mazowieckie/warszawa/praga-polnoc",
        "mazowieckie/warszawa/praga-poludnie",
        "mazowieckie/warszawa/rembertow",
        "mazowieckie/warszawa/srodmiescie",
        "mazowieckie/warszawa/targowek",
        "mazowieckie/warszawa/ursus",
        "mazowieckie/warszawa/ursynow",
        "mazowieckie/warszawa/wawer",
        "mazowieckie/warszawa/wesola",
        "mazowieckie/warszawa/wilanow",
        "mazowieckie/warszawa/wlochy",
        "mazowieckie/warszawa/wola",
        "mazowieckie/warszawa/zoliborz",
    ],
    "Kraków": [
        "krakow",                                   # catch-all
        "malopolskie/krakow/bienczyce",
        "malopolskie/krakow/biezanow-prokocim",
        "malopolskie/krakow/bronowice",
        "malopolskie/krakow/czyzyny",
        "malopolskie/krakow/debniki",
        "malopolskie/krakow/grzegorzki",
        "malopolskie/krakow/krowodrza",
        "malopolskie/krakow/lagiewniki-borek-falecki",
        "malopolskie/krakow/mistrzejowice",
        "malopolskie/krakow/nowa-huta",
        "malopolskie/krakow/podgorze",
        "malopolskie/krakow/podgorze-duchackie",
        "malopolskie/krakow/pradnik-bialy",
        "malopolskie/krakow/pradnik-czerwony",
        "malopolskie/krakow/stare-miasto",
        "malopolskie/krakow/swoszowice",
        "malopolskie/krakow/wzgorza-krzeslawickie",
        "malopolskie/krakow/zwierzyniec",
    ],
    "Wrocław": [
        "wroclaw",                                  # catch-all (no district tag)
        "dolnoslaskie/wroclaw/fabryczna",
        "dolnoslaskie/wroclaw/krzyki",
        "dolnoslaskie/wroclaw/psie-pole",
        "dolnoslaskie/wroclaw/stare-miasto",
        "dolnoslaskie/wroclaw/srodmiescie",
    ],
    "Poznań": [
        "poznan",
        "wielkopolskie/poznan/grunwald",
        "wielkopolskie/poznan/jezyce",
        "wielkopolskie/poznan/nowe-miasto",
        "wielkopolskie/poznan/stare-miasto",
        "wielkopolskie/poznan/wilda",
    ],
    "Gdańsk": [
        "gdansk",
        "pomorskie/gdansk/srodmiescie",
        "pomorskie/gdansk/wrzeszcz",
        "pomorskie/gdansk/oliwa",
        "pomorskie/gdansk/przymorze-wielkie",
        "pomorskie/gdansk/zaspa-mlyniec",
        "pomorskie/gdansk/morena",
        "pomorskie/gdansk/chelm",
    ],
    "Łódź": [
        "lodz",
        "lodzkie/lodz/baluty",
        "lodzkie/lodz/gorna",
        "lodzkie/lodz/polesie",
        "lodzkie/lodz/srodmiescie",
        "lodzkie/lodz/widzew",
    ],
}


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def scrape_slugs(session, slugs: list[str], city_name: str) -> list[dict]:
    rows: list[dict] = []
    for slug in slugs:
        log.info("[%s] scraping slug: %s", city_name, slug)
        for page in range(1, MAX_PAGES + 1):
            soup = fetch_page(session, slug, page)
            if soup is None:
                log.warning("[%s] slug=%s page=%d fetch failed — stopping slug", city_name, slug, page)
                break
            listings = parse_listings(soup, city_name)
            if not listings:
                log.info("[%s] slug=%s page=%d — no more listings", city_name, slug, page)
                break
            from dataclasses import asdict
            rows.extend(asdict(lst) for lst in listings)
            log.info("[%s] slug=%s page=%d — +%d listings", city_name, slug, page, len(listings))
            time.sleep(random.uniform(DELAY_LO, DELAY_HI))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    existing_path = OUTPUT_DIR / "otodom_all.csv"
    if not existing_path.exists():
        log.error("No existing data found at %s — run scrape_otodom.py first.", existing_path)
        return

    log.info("Loading existing data from %s", existing_path)
    base_df = pd.read_csv(existing_path, encoding="utf-8-sig")
    log.info("Loaded %d rows", len(base_df))

    import requests as req
    session = req.Session()
    session.headers.update(HEADERS)

    new_rows: list[dict] = []
    for city_name, slugs in DISTRICT_SLUGS.items():
        log.info("=== Augmenting: %s (%d slugs) ===", city_name, len(slugs))
        rows = scrape_slugs(session, slugs, city_name)
        log.info("[%s] scraped %d raw rows", city_name, len(rows))
        new_rows.extend(rows)

    if not new_rows:
        log.warning("No new data scraped — nothing to merge.")
        return

    combined = pd.concat([base_df, pd.DataFrame(new_rows)], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    log.info("Merged: %d existing + %d new → %d unique (removed %d duplicates)",
             len(base_df), len(new_rows), len(combined), before - len(combined))

    # Save per-city CSVs
    city_slug_map = {
        "Warszawa": "warszawa", "Kraków": "krakow", "Wrocław": "wroclaw",
        "Poznań": "poznan", "Gdańsk": "gdansk", "Łódź": "lodz",
        "Szczecin": "szczecin", "Bydgoszcz": "bydgoszcz", "Lublin": "lublin",
        "Katowice": "katowice", "Białystok": "bialystok", "Rzeszów": "rzeszow",
        "Kielce": "kielce", "Olsztyn": "olsztyn", "Toruń": "torun",
    }
    for city_name, city_df in combined.groupby("city"):
        slug = city_slug_map.get(city_name, city_name.lower())
        out_path = OUTPUT_DIR / f"otodom_{slug}.csv"
        city_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log.info("[%s] saved %d rows → %s", city_name, len(city_df), out_path)

    combined.to_csv(existing_path, index=False, encoding="utf-8-sig")
    log.info("Saved combined dataset (%d rows) → %s", len(combined), existing_path)


if __name__ == "__main__":
    main()
