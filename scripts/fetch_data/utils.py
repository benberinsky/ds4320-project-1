"""
utils.py
Shared utilities for KenPom and Kaggle data pipeline.
Provides logging, authentication, and save functions
used across all fetch and cleaning scripts.
"""

import requests
import pandas as pd
import logging
import os
from datetime import datetime
from getpass import getpass

# ── Configuration ──────────────────────────────────────────
BASE_URL   = "https://kenpom.com/api.php"
START_YEAR = 2002
END_YEAR   = 2025
DELAY      = 1.5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR   = os.path.join(PROJECT_ROOT, "data", "raw")
CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "clean")


def setup_logger(name: str) -> logging.Logger:
    """
    Configures logger to write to both console and 
    timestamped log file.

    Args:
        name: Logger name, used for log filename
    Returns:
        logging.Logger: Configured logger instance
    """
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(f"logs/{name}_{ts}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_session() -> requests.Session:
    """
    Prompts for API key securely and returns 
    an authenticated session.

    Returns:
        requests.Session: Session with Bearer token set
    """
    api_key = getpass("Enter your KenPom API key: ")
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}"
    })
    return session


def save(df: pd.DataFrame,
         name: str,
         output_dir: str,
         logger: logging.Logger) -> None:
    """
    Saves DataFrame as both CSV and Parquet.

    Args:
        df:         DataFrame to save
        name:       Base filename (no extension)
        output_dir: Directory to save to
        logger:     Logger instance
    """
    if df.empty:
        logger.warning(f"Empty DataFrame — skipping save for {name}")
        return

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{name}.csv")
    parquet_path = os.path.join(output_dir, f"{name}.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    csv_size = os.path.getsize(csv_path) / 1024
    parquet_size = os.path.getsize(parquet_path) / 1024

    logger.info(
        f"Saved {name} | CSV: {csv_size:.1f}KB | "
        f"Parquet: {parquet_size:.1f}KB"
    )