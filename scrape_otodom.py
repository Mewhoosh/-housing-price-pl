from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# slug → city display name
# Large cities are scraped per-district to bypass Otodom pagination/bot limits.
# City-level slug is always first to catch listings without a district tag.
CITIES: dict[str, str] = {
    # --- Warszawa (catch-all + 18 dzielnic) ---
    "warszawa":                              "Warszawa",
    "mazowieckie/warszawa/bemowo":           "Warszawa",
    "mazowieckie/warszawa/bialoleka":        "Warszawa",
    "mazowieckie/warszawa/bielany":          "Warszawa",
    "mazowieckie/warszawa/mokotow":          "Warszawa",
    "mazowieckie/warszawa/ochota":           "Warszawa",
    "mazowieckie/warszawa/praga-polnoc":     "Warszawa",
    "mazowieckie/warszawa/praga-poludnie":   "Warszawa",
    "mazowieckie/warszawa/rembertow":        "Warszawa",
    "mazowieckie/warszawa/srodmiescie":      "Warszawa",
    "mazowieckie/warszawa/targowek":         "Warszawa",
    "mazowieckie/warszawa/ursus":            "Warszawa",
    "mazowieckie/warszawa/ursynow":          "Warszawa",
    "mazowieckie/warszawa/wawer":            "Warszawa",
    "mazowieckie/warszawa/wesola":           "Warszawa",
    "mazowieckie/warszawa/wilanow":          "Warszawa",
    "mazowieckie/warszawa/wlochy":           "Warszawa",
    "mazowieckie/warszawa/wola":             "Warszawa",
    "mazowieckie/warszawa/zoliborz":         "Warszawa",
    # --- Kraków (catch-all + 18 dzielnic) ---
    "krakow":                                        "Kraków",
    "malopolskie/krakow/bienczyce":                  "Kraków",
    "malopolskie/krakow/biezanow-prokocim":          "Kraków",
    "malopolskie/krakow/bronowice":                  "Kraków",
    "malopolskie/krakow/czyzyny":                    "Kraków",
    "malopolskie/krakow/debniki":                    "Kraków",
    "malopolskie/krakow/grzegorzki":                 "Kraków",
    "malopolskie/krakow/krowodrza":                  "Kraków",
    "malopolskie/krakow/lagiewniki-borek-falecki":   "Kraków",
    "malopolskie/krakow/mistrzejowice":              "Kraków",
    "malopolskie/krakow/nowa-huta":                  "Kraków",
    "malopolskie/krakow/podgorze":                   "Kraków",
    "malopolskie/krakow/podgorze-duchackie":         "Kraków",
    "malopolskie/krakow/pradnik-bialy":              "Kraków",
    "malopolskie/krakow/pradnik-czerwony":           "Kraków",
    "malopolskie/krakow/stare-miasto":               "Kraków",
    "malopolskie/krakow/swoszowice":                 "Kraków",
    "malopolskie/krakow/wzgorza-krzeslawickie":      "Kraków",
    "malopolskie/krakow/zwierzyniec":                "Kraków",
    # --- Wrocław (catch-all + 5 dzielnic) ---
    "wroclaw":                               "Wrocław",
    "dolnoslaskie/wroclaw/fabryczna":        "Wrocław",
    "dolnoslaskie/wroclaw/krzyki":           "Wrocław",
    "dolnoslaskie/wroclaw/psie-pole":        "Wrocław",
    "dolnoslaskie/wroclaw/stare-miasto":     "Wrocław",
    "dolnoslaskie/wroclaw/srodmiescie":      "Wrocław",
    # --- Poznań (catch-all + 5 dzielnic) ---
    "poznan":                                "Poznań",
    "wielkopolskie/poznan/grunwald":         "Poznań",
    "wielkopolskie/poznan/jezyce":           "Poznań",
    "wielkopolskie/poznan/nowe-miasto":      "Poznań",
    "wielkopolskie/poznan/stare-miasto":     "Poznań",
    "wielkopolskie/poznan/wilda":            "Poznań",
    # --- Gdańsk (catch-all + 7 dzielnic) ---
    "gdansk":                                "Gdańsk",
    "pomorskie/gdansk/chelm":                "Gdańsk",
    "pomorskie/gdansk/morena":               "Gdańsk",
    "pomorskie/gdansk/oliwa":                "Gdańsk",
    "pomorskie/gdansk/przymorze-wielkie":    "Gdańsk",
    "pomorskie/gdansk/srodmiescie":          "Gdańsk",
    "pomorskie/gdansk/wrzeszcz":             "Gdańsk",
    "pomorskie/gdansk/zaspa-mlyniec":        "Gdańsk",
    # --- Łódź (catch-all + 5 dzielnic) ---
    "lodz":                                  "Łódź",
    "lodzkie/lodz/baluty":                   "Łódź",
    "lodzkie/lodz/gorna":                    "Łódź",
    "lodzkie/lodz/polesie":                  "Łódź",
    "lodzkie/lodz/srodmiescie":              "Łódź",
    "lodzkie/lodz/widzew":                   "Łódź",
    # --- Pozostałe miasta (scraping na poziomie miasta) ---
    "szczecin":   "Szczecin",
    "bydgoszcz":  "Bydgoszcz",
    "lublin":     "Lublin",
    "katowice":   "Katowice",
    "bialystok":  "Białystok",
    "rzeszow":    "Rzeszów",
    "kielce":     "Kielce",
    "olsztyn":    "Olsztyn",
    "torun":      "Toruń",
}

BASE_URL    = "https://www.otodom.pl/pl/oferty/sprzedaz/mieszkanie/{city}"
MAX_PAGES   = 25          # safety cap per city (~72 listings/page → ~1800 max)
DELAY_LO    = 2.0         # seconds
DELAY_HI    = 4.5
OUTPUT_DIR  = Path(__file__).parent / "data" / "raw"

ROOMS_MAP: dict[str, int] = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
    "FIVE": 5, "SIX_OR_MORE": 6,
}
FLOOR_MAP: dict[str, int] = {
    "GROUND": 0, "FIRST": 1, "SECOND": 2, "THIRD": 3,
    "FOURTH": 4, "FIFTH": 5, "SIXTH": 6, "SEVENTH": 7,
    "EIGHTH": 8, "NINTH": 9, "TENTH_OR_ABOVE": 10,
}

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.otodom.pl/",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Listing:
    city:             str
    price:            int | None
    price_per_m2:     float | None
    area_m2:          float | None
    rooms:            int | None
    floor:            int | None
    neighborhood:     str | None   # district level: Mokotów, Wola, Żoliborz
    sub_neighborhood: str | None   # residential level: Koło, Stary Mokotów
    is_private_owner: bool | None
    url:              str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(d: dict[str, Any], *keys: str) -> Any:
    """Traverse nested dict safely; return None if any key is missing."""
    node: Any = d
    for k in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(k)
    return node


def _get_rev_geo(locations: list[dict], level: str) -> str | None:
    """Extract named location from reverseGeocoding list by locationLevel."""
    for loc in locations or []:
        if loc.get("locationLevel") == level:
            return loc.get("name")
    return None


def _to_int(val: Any) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _to_float(val: Any) -> float | None:
    try:
        return float(str(val).replace(",", ".").replace(" ", ""))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_page(
    session: requests.Session,
    city_slug: str,
    page: int,
) -> BeautifulSoup | None:
    url = BASE_URL.format(city=city_slug)
    params: dict[str, Any] = {"page": page}

    for attempt in range(2):
        try:
            resp = session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            log.warning(
                "HTTP %d for %s page %d (attempt %d)",
                resp.status_code, city_slug, page, attempt + 1,
            )
        except requests.RequestException as exc:
            log.warning("Request error: %s (attempt %d)", exc, attempt + 1)

        time.sleep(random.uniform(5.0, 9.0))  # longer delay on failure

    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_listings(soup: BeautifulSoup, city_name: str) -> list[Listing]:
    tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if not tag or not tag.string:
        return []

    try:
        data: dict[str, Any] = json.loads(tag.string)
    except json.JSONDecodeError:
        log.error("Failed to parse __NEXT_DATA__ JSON")
        return []

    # path may change with Otodom deploys — log if empty so we can adapt
    items: list[dict] = (
        _safe_get(data, "props", "pageProps", "data", "searchAds", "items") or []
    )

    results: list[Listing] = []
    for item in items:
        # only process flat listings; investment bundles have no individual data
        if item.get("estate") != "FLAT":
            continue

        price        = _to_int(_safe_get(item, "totalPrice", "value"))
        price_per_m2 = _to_float(_safe_get(item, "pricePerSquareMeter", "value"))
        area_m2      = _to_float(item.get("areaInSquareMeters"))
        rooms        = ROOMS_MAP.get(item.get("roomsNumber") or "", None)
        floor        = FLOOR_MAP.get(item.get("floorNumber") or "", None)

        rev_geo          = _safe_get(item, "location", "reverseGeocoding", "locations") or []
        neighborhood     = _get_rev_geo(rev_geo, "district")
        sub_neighborhood = _get_rev_geo(rev_geo, "residential")
        is_private_owner = item.get("isPrivateOwner")

        slug        = item.get("slug") or ""
        listing_url = f"https://www.otodom.pl/pl/oferta/{slug}" if slug else None

        results.append(Listing(
            city=city_name,
            price=price,
            price_per_m2=price_per_m2,
            area_m2=area_m2,
            rooms=rooms,
            floor=floor,
            neighborhood=neighborhood,
            sub_neighborhood=sub_neighborhood,
            is_private_owner=is_private_owner,
            url=listing_url,
        ))

    return results


# ---------------------------------------------------------------------------
# City scraper
# ---------------------------------------------------------------------------

def scrape_city(
    session: requests.Session,
    city_slug: str,
    city_name: str,
    max_pages: int = MAX_PAGES,
) -> list[Listing]:
    all_listings: list[Listing] = []

    for page in range(1, max_pages + 1):
        soup = fetch_page(session, city_slug, page)
        if soup is None:
            log.warning("[%s] page %d fetch failed — stopping city", city_name, page)
            break

        listings = parse_listings(soup, city_name)
        if not listings:
            log.info("[%s] page %d returned 0 listings — end of results", city_name, page)
            break

        all_listings.extend(listings)
        log.info("[%s] page %d/%d — %d listings (+%d total)",
                 city_name, page, max_pages, len(listings), len(all_listings))

        time.sleep(random.uniform(DELAY_LO, DELAY_HI))

    return all_listings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CITY_SLUG_MAP = {
    "Warszawa": "warszawa", "Kraków": "krakow",  "Wrocław": "wroclaw",
    "Poznań":   "poznan",   "Gdańsk": "gdansk",  "Łódź":    "lodz",
    "Szczecin": "szczecin", "Bydgoszcz": "bydgoszcz", "Lublin": "lublin",
    "Katowice": "katowice", "Białystok": "bialystok", "Rzeszów": "rzeszow",
    "Kielce":   "kielce",   "Olsztyn": "olsztyn", "Toruń": "torun",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows: list[dict] = []

    for slug, name in CITIES.items():
        log.info("=== [%s] slug: %s ===", name, slug)
        listings = scrape_city(session, slug, name)

        if not listings:
            log.warning("[%s] no listings scraped — skipping", name)
            continue

        rows = [asdict(lst) for lst in listings]
        all_rows.extend(rows)
        log.info("[%s] +%d rows (running total: %d)", name, len(rows), len(all_rows))

    if all_rows:
        combined = pd.DataFrame(all_rows)

        # deduplicate cross-district overlaps
        before = len(combined)
        combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
        log.info("Deduplicated: %d → %d rows (removed %d duplicates)",
                 before, len(combined), before - len(combined))

        # save per-city CSVs (grouped by city name, not by slug)
        for city_name, city_df in combined.groupby("city"):
            file_slug = _CITY_SLUG_MAP.get(city_name, city_name.lower())
            out_path = OUTPUT_DIR / f"otodom_{file_slug}.csv"
            city_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            log.info("[%s] saved %d rows → %s", city_name, len(city_df), out_path)

        combined_path = OUTPUT_DIR / "otodom_all.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        log.info("Combined: %d rows → %s", len(combined), combined_path)

        null_pct = combined.isna().mean().mul(100).round(1)
        print("\nNull % per column:")
        print(null_pct.to_string())
        print(f"\nTotal rows: {len(combined)}")
        print(f"Cities    : {combined['city'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
