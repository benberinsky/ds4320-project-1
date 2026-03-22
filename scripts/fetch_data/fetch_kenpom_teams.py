"""
fetch_kenpom_teams.py
Fetches team info from KenPom API including team name,
ID, conference, coach, and arena info for each season.
Saves raw data to data/raw/.
"""

import time
import pandas as pd
from utils import (
    setup_logger, get_session, save,
    BASE_URL, START_YEAR, END_YEAR, DELAY, RAW_DIR
)


def fetch_one(session, year, logger, retries=3):
    """
    Fetches teams data for a single year.

    Args:
        session:  Authenticated requests session
        year:     Season ending year
        logger:   Logger instance
        retries:  Max retry attempts
    Returns:
        DataFrame or None on failure
    """
    params = {"endpoint": "teams", "y": year}

    for attempt in range(1, retries + 1):
        try:
            r = session.get(BASE_URL, params=params)

            if r.status_code != 200:
                logger.error(f"y={year} | HTTP {r.status_code}")
                return None

            df = pd.DataFrame(r.json())
            logger.info(f"  ✓ {year} | {len(df)} teams")
            time.sleep(DELAY)
            return df

        except Exception as e:
            wait = DELAY * attempt * 2
            logger.warning(
                f"  Retry {attempt}/{retries}: y={year} | {e}"
            )
            time.sleep(wait)

    logger.error(f"  ✗ FAILED: y={year}")
    return None


def main():
    """Pulls KenPom teams data across all years."""
    logger = setup_logger("fetch_kenpom_teams")

    logger.info("═" * 50)
    logger.info("KenPom Teams Pull")
    logger.info(f"Years: {START_YEAR} → {END_YEAR}")
    logger.info("═" * 50)

    session = get_session()
    logger.info("Session authenticated")

    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        df = fetch_one(session, year, logger)
        if df is not None:
            frames.append(df)

    if not frames:
        logger.error("No data collected — exiting")
        return None

    combined = pd.concat(frames, ignore_index=True)

    logger.info("═" * 50)
    logger.info(
        f"Total: {len(combined)} rows | "
        f"{combined.shape[1]} cols | "
        f"{combined['Season'].min()}–{combined['Season'].max()}"
    )
    logger.info("═" * 50)

    save(combined, "teams", RAW_DIR, logger)

    return combined


if __name__ == "__main__":
    main()