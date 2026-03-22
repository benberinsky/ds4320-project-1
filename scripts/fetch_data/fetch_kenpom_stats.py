"""
fetch_kenpom_stats.py
Fetches season-level stats from KenPom API endpoints:
ratings, four-factors, height, and point distribution.
Saves raw data to data/raw/.
"""

import time
import pandas as pd
from utils import (
    setup_logger, get_session, save,
    BASE_URL, START_YEAR, END_YEAR, DELAY, RAW_DIR
)

ENDPOINTS = {
    "ratings":       "ratings",
    "four-factors":  "four_factors",
    "height":        "height",
    "pointdist":     "point_dist",
}


def fetch_one(session, endpoint, year, logger, retries=3):
    """
    Fetches a single endpoint for a single year with retry logic.

    Args:
        session:  Authenticated requests session
        endpoint: KenPom endpoint name
        year:     Season ending year
        logger:   Logger instance
        retries:  Max retry attempts
    Returns:
        DataFrame or None on failure
    """
    params = {"endpoint": endpoint, "y": year}

    for attempt in range(1, retries + 1):
        try:
            r = session.get(BASE_URL, params=params)

            if r.status_code != 200:
                logger.error(f"{endpoint} y={year} | HTTP {r.status_code}")
                return None

            df = pd.DataFrame(r.json())
            df["year"] = year
            logger.info(f"  ✓ {endpoint} | {year} | {len(df)} teams")
            time.sleep(DELAY)
            return df

        except Exception as e:
            wait = DELAY * attempt * 2
            logger.warning(f"  Retry {attempt}/{retries}: {endpoint} y={year} | {e}")
            time.sleep(wait)

    logger.error(f"  ✗ FAILED: {endpoint} y={year}")
    return None


def main():
    """Pulls all KenPom stat endpoints across all years."""
    logger = setup_logger("fetch_kenpom_stats")
    session = get_session()

    for endpoint, filename in ENDPOINTS.items():
        logger.info(f"{'─'*50}")
        logger.info(f"Pulling: {endpoint}")

        frames = []
        for year in range(START_YEAR, END_YEAR + 1):
            df = fetch_one(session, endpoint, year, logger)
            if df is not None:
                frames.append(df)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            save(combined, filename, RAW_DIR, logger)


if __name__ == "__main__":
    main()