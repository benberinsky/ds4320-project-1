"""
clean.py
Cleans and standardizes raw data from KenPom and Kaggle sources.
Applies team name mapping, derives upset flags, assigns tournament
rounds, and outputs analysis-ready tables to data/clean/.
"""

# Kaggle files required in data/raw/:
#   Source: https://www.kaggle.com/competitions/march-machine-learning-mania-2025
#   Files:
#     - MNCAATourneyDetailedResults.csv
#     - MNCAATourneySeeds.csv  
#     - MTeams.csv

import pandas as pd
import os
from utils import setup_logger, save, RAW_DIR, CLEAN_DIR

# maps Kaggle team names to KenPom team names for consistency across datasets
TEAM_NAME_MAP = {
    # ── Abbreviation → Full Name ──────────────────────────────
    "Abilene Chr":          "Abilene Christian",
    "Ark Little Rock":      "Little Rock",
    "Ark Pine Bluff":       "Arkansas Pine Bluff",
    "C Michigan":           "Central Michigan",
    "Central Conn":         "Central Connecticut",
    "Coastal Car":          "Coastal Carolina",
    "E Kentucky":           "Eastern Kentucky",
    "E Washington":         "Eastern Washington",
    "FL Atlantic":          "Florida Atlantic",
    "G Washington":         "George Washington",
    "N Colorado":           "Northern Colorado",
    "N Dakota St":          "North Dakota St.",
    "N Kentucky":           "Northern Kentucky",
    "S Dakota St":          "South Dakota St.",
    "S Illinois":           "Southern Illinois",
    "W Michigan":           "Western Michigan",

    # ── St vs St. ─────────────────────────────────────────────
    "Alabama St":           "Alabama St.",
    "Alcorn St":            "Alcorn St.",
    "Appalachian St":       "Appalachian St.",
    "Arizona St":           "Arizona St.",
    "Boise St":             "Boise St.",
    "Cleveland St":         "Cleveland St.",
    "Colorado St":          "Colorado St.",
    "Coppin St":            "Coppin St.",
    "Delaware St":          "Delaware St.",
    "Florida St":           "Florida St.",
    "Fresno St":            "Fresno St.",
    "Georgia St":           "Georgia St.",
    "Grambling":            "Grambling St.",
    "Indiana St":           "Indiana St.",
    "Iowa St":              "Iowa St.",
    "Jackson St":           "Jackson St.",
    "Jacksonville St":      "Jacksonville St.",
    "Kansas St":            "Kansas St.",
    "Kent":                 "Kent St.",
    "Long Beach St":        "Long Beach St.",
    "McNeese St":           "McNeese St.",
    "Michigan St":          "Michigan St.",
    "Mississippi St":       "Mississippi St.",
    "Montana St":           "Montana St.",
    "Morehead St":          "Morehead St.",
    "Morgan St":            "Morgan St.",
    "Murray St":            "Murray St.",
    "New Mexico St":        "New Mexico St.",
    "Norfolk St":           "Norfolk St.",
    "Ohio St":              "Ohio St.",
    "Oklahoma St":          "Oklahoma St.",
    "Oregon St":            "Oregon St.",
    "Penn St":              "Penn St.",
    "Portland St":          "Portland St.",
    "Sam Houston St":       "Sam Houston St.",
    "San Diego St":         "San Diego St.",
    "Tennessee St":         "Tennessee St.",
    "Utah St":              "Utah St.",
    "Washington St":        "Washington St.",
    "Weber St":             "Weber St.",
    "Wichita St":           "Wichita St.",
    "Wright St":            "Wright St.",

    # ── Saint vs St. ──────────────────────────────────────────
    "St Bonaventure":       "St. Bonaventure",
    "St Francis PA":        "St. Francis PA",
    "St John's":            "St. John's",
    "St Joseph's PA":       "Saint Joseph's",
    "St Louis":             "Saint Louis",
    "St Mary's CA":         "Saint Mary's",
    "St Peter's":           "Saint Peter's",
    "Mt St Mary's":         "Mount St. Mary's",

    # ── Acronyms ──────────────────────────────────────────────
    "ETSU":                 "East Tennessee St.",
    "FGCU":                 "Florida Gulf Coast",
    "MTSU":                 "Middle Tennessee",
    "NC A&T":               "North Carolina A&T",
    "NC Central":           "North Carolina Central",
    "NC State":             "N.C. State",
    "NE Omaha":             "Nebraska Omaha",
    "WKU":                  "Western Kentucky",
    "WI Green Bay":         "Green Bay",
    "WI Milwaukee":         "Milwaukee",

    # ── University Systems ────────────────────────────────────
    "CS Bakersfield":       "Cal St. Bakersfield",
    "CS Fullerton":         "Cal St. Fullerton",
    "CS Northridge":        "Cal St. Northridge",
    "SUNY Albany":          "Albany",
    "TX Southern":          "Texas Southern",
    "SE Louisiana":         "Southeastern Louisiana",
    "SE Missouri St":       "Southeast Missouri St.",
    "IL Chicago":           "Illinois Chicago",
    "S Carolina St":        "South Carolina St.",
    "UT San Antonio":       "UTSA",

    # ── Unique Names ──────────────────────────────────────────
    "American Univ":        "American",
    "Boston Univ":          "Boston University",
    "Col Charleston":       "College of Charleston",
    "F Dickinson":          "Fairleigh Dickinson",
    "Kennesaw":             "Kennesaw St.",
    "Loyola-Chicago":       "Loyola Chicago",
    "Monmouth NJ":          "Monmouth",
    "MS Valley St":         "Mississippi Valley St.",
    "Northwestern LA":      "Northwestern St.",
    "Prairie View":         "Prairie View A&M",
    "Queens NC":            "Queens",
    "SF Austin":            "Stephen F. Austin",
    "Southern Univ":        "Southern",
    "TAM C. Christi":       "Texas A&M Corpus Chris",
}

# loading the raw CSV files from data/raw/ with error handling and logging
def load_raw(filename: str, logger) -> pd.DataFrame:
    """
    Loads a raw CSV from data/raw/.

    Args:
        filename: CSV filename (without path)
        logger: Logger instance
    Returns:
        DataFrame or empty DataFrame on failure
    """
    path = os.path.join(RAW_DIR, filename)
    try:
        df = pd.read_csv(path)
        logger.info(f"  ✓ Loaded {filename} | {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"  ✗ Not found: {path}")
        return pd.DataFrame()


def apply_team_mapping(df: pd.DataFrame,
                       name_cols: list,
                       logger) -> pd.DataFrame:
    """
    Applies team name mapping to one or more columns.
    Standardizes Kaggle names to match KenPom conventions.

    Args:
        df:        DataFrame containing team name columns
        name_cols: List of column names to map
        logger:    Logger instance
    Returns:
        DataFrame with standardized team names
    """
    df = df.copy()
    for col in name_cols:
        before = df[col].nunique()
        df[col] = df[col].replace(TEAM_NAME_MAP)
        after = df[col].nunique()
        logger.info(
            f"  Mapped {col}: {before} unique → {after} unique"
        )
    return df

def assign_rounds(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Assigns tournament round based on game order within each season.
    Handles three tournament formats:
        - 2002:      63 games (no play-in)
        - 2003-2010: 1 play-in + 63 bracket = 64 games
        - 2011+:     4 First Four + 63 bracket = 67 games
        - 2021:      3 First Four + 63 bracket = 66 games (COVID)

    Args:
        df:     Tournament games DataFrame with Season and DayNum
        logger: Logger instance
    Returns:
        DataFrame with Round column assigned
    """
    df = df.sort_values(["Season", "DayNum"]).copy()

    bracket_63 = (
        ["R64"] * 32 +
        ["R32"] * 16 +
        ["S16"] * 8 +
        ["E8"] * 4 +
        ["F4"] * 2 +
        ["Championship"] * 1
    )

    rounds = []
    for season, group in df.groupby("Season"):
        n_games = len(group)

        if n_games == 63:
            rounds.extend(bracket_63)
        elif n_games == 64:
            rounds.extend(["Play-In"] * 1 + bracket_63)
        elif n_games == 67:
            rounds.extend(["First Four"] * 4 + bracket_63)
        elif n_games == 66:
            rounds.extend(["First Four"] * 3 + bracket_63)
        else:
            rounds.extend(["Unknown"] * n_games)
            logger.warning(f"  ⚠️ Season {season}: {n_games} games — unknown format")

    df["Round"] = rounds
    return df

def flag_upsets(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Flags games as upsets when the winning team's seed is
    meaningfully lower (higher number) than the losing team's seed.
    Requires seed difference > 1 to exclude coin-flip matchups
    like 8 vs 9.

    Args:
        df:     Tournament games DataFrame with WSeed and LSeed
        logger: Logger instance
    Returns:
        DataFrame with Upset and SeedDiff columns
    """
    df = df.copy()
    df["SeedDiff"] = abs(df["WSeed"] - df["LSeed"])
    df["Upset"] = (
        (df["WSeed"] > df["LSeed"]) &
        (df["SeedDiff"] > 1)
    ).astype(int)

    total_games = len(df)
    total_upsets = df["Upset"].sum()
    logger.info(
        f"  Upsets: {total_upsets}/{total_games} "
        f"({total_upsets/total_games*100:.1f}%)"
    )
    return df

def parse_seed(seed_str: str) -> int:
    """
    Extracts numeric seed from Kaggle seed format.
    e.g., 'W01' → 1, 'X16a' → 16

    Args:
        seed_str: Raw seed string from Kaggle
    Returns:
        Integer seed value (1-16)
    """
    return int(seed_str[1:3])


def clean_tournament_games(logger) -> pd.DataFrame:
    """
    Builds clean tournament games table from raw Kaggle data.
    Merges seeds and team names, assigns rounds, flags upsets,
    and applies team name mapping.

    Args:
        logger: Logger instance
    Returns:
        Clean tournament games DataFrame
    """
    logger.info("Cleaning tournament games...")

    # Load raw files
    results = load_raw("MNCAATourneyDetailedResults.csv", logger)
    seeds = load_raw("MNCAATourneySeeds.csv", logger)
    teams = load_raw("MTeams.csv", logger)

    # Filter to 2002+
    results = results[results["Season"] >= 2002].copy()
    seeds = seeds[seeds["Season"] >= 2002].copy()

    # Parse seeds
    seeds["SeedNum"] = seeds["Seed"].apply(parse_seed)
    seeds["Region"] = seeds["Seed"].str[0]

    # Merge winner seed + name
    games = results.merge(
        seeds[["Season", "TeamID", "SeedNum"]],
        left_on=["Season", "WTeamID"],
        right_on=["Season", "TeamID"],
        how="left"
    ).rename(columns={"SeedNum": "WSeed"}).drop(columns=["TeamID"])

    games = games.merge(
        teams[["TeamID", "TeamName"]],
        left_on="WTeamID", right_on="TeamID", how="left"
    ).rename(columns={"TeamName": "WTeamName"}).drop(columns=["TeamID"])

    # Merge loser seed + name
    games = games.merge(
        seeds[["Season", "TeamID", "SeedNum"]],
        left_on=["Season", "LTeamID"],
        right_on=["Season", "TeamID"],
        how="left"
    ).rename(columns={"SeedNum": "LSeed"}).drop(columns=["TeamID"])

    games = games.merge(
        teams[["TeamID", "TeamName"]],
        left_on="LTeamID", right_on="TeamID", how="left"
    ).rename(columns={"TeamName": "LTeamName"}).drop(columns=["TeamID"])

    # Assign rounds
    games = assign_rounds(games, logger)

    # Flag upsets
    games = flag_upsets(games, logger)

    # Apply team name mapping
    games = apply_team_mapping(games, ["WTeamName", "LTeamName"], logger)

    # Add composite keys
    games = add_team_season_key(games, ["WTeamName", "LTeamName"], logger=logger)
    
    # Keep only analysis-relevant columns
    games = games[["Season", "WTeamID", "WScore", "LTeamID",
                   "LScore", "WSeed", "WTeamName", "LSeed",
                   "LTeamName", "Round", "SeedDiff", "Upset",
                   "WTeamSeason", "LTeamSeason"]]

    logger.info(f"  Tournament games: {games.shape}")
    return games
def add_team_season_key(df: pd.DataFrame,
                        name_cols: list,
                        season_col: str = "Season",
                        logger=None) -> pd.DataFrame:
    """
    Creates composite TeamSeason key(s) by concatenating
    TeamName and Season (e.g., 'Virginia_2025').

    For KenPom/seeds tables: name_cols = ["TeamName"]
        → produces "TeamSeason"
    For tournament games: name_cols = ["WTeamName", "LTeamName"]
        → produces "WTeamSeason", "LTeamSeason"

    Args:
        df:         DataFrame
        name_cols:  Team name column(s) to combine with season
        season_col: Name of the season column
        logger:     Logger instance
    Returns:
        DataFrame with new TeamSeason key column(s)
    """
    df = df.copy()
    for col in name_cols:
        prefix = col.replace("TeamName", "")  # "" for TeamName, "W" for WTeamName, etc.
        key_col = f"{prefix}TeamSeason"
        df[key_col] = df[col] + "_" + df[season_col].astype(str)
        if logger:
            logger.info(f"  Added key: {key_col} ({df[key_col].nunique()} unique)")
    return df

def clean_tournament_seeds(logger) -> pd.DataFrame:
    """
    Builds clean tournament seeds table from raw Kaggle data.
    Parses seed numbers, merges team names, applies mapping.

    Args:
        logger: Logger instance
    Returns:
        Clean tournament seeds DataFrame
    """
    logger.info("Cleaning tournament seeds...")

    seeds = load_raw("MNCAATourneySeeds.csv", logger)
    teams = load_raw("MTeams.csv", logger)

    seeds = seeds[seeds["Season"] >= 2002].copy()
    seeds["SeedNum"] = seeds["Seed"].apply(parse_seed)
    seeds["Region"] = seeds["Seed"].str[0]

    seeds = seeds.merge(
        teams[["TeamID", "TeamName"]],
        on="TeamID", how="left"
    )

    seeds = seeds[["Season", "TeamID", "TeamName", "SeedNum", "Region"]]

    # Apply team name mapping
    seeds = apply_team_mapping(seeds, ["TeamName"], logger)

    # Add composite key
    seeds = add_team_season_key(seeds, ["TeamName"], logger=logger)
    seeds = seeds[["Season", "TeamID", "TeamName", "SeedNum", "Region", "TeamSeason"]]

    logger.info(f"  Tournament seeds: {seeds.shape}")
    return seeds


def verify_mapping(tournament_df: pd.DataFrame,
                   logger) -> None:
    """
    Checks for team names in tournament data that don't
    exist in KenPom data. Flags any remaining mismatches.

    Args:
        tournament_df: Clean tournament games or seeds DataFrame
        logger:        Logger instance
    """
    kenpom = load_raw("four_factors.csv", logger)
    kenpom_names = set(kenpom["TeamName"].unique())

    # Check all name columns that exist
    name_cols = [c for c in ["TeamName", "WTeamName", "LTeamName"]
                 if c in tournament_df.columns]

    all_names = set()
    for col in name_cols:
        all_names.update(tournament_df[col].unique())

    unmatched = all_names - kenpom_names
    if unmatched:
        logger.warning(f"  ⚠️ {len(unmatched)} unmatched names:")
        for name in sorted(unmatched):
            logger.warning(f"    {name}")
    else:
        logger.info("  ✓ All team names match KenPom")

def save_mapping(logger) -> None:
    """
    Saves the team name mapping dictionary as a CSV
    for documentation and provenance tracking.

    Args:
        logger: Logger instance
    """
    mapping_df = pd.DataFrame(
        list(TEAM_NAME_MAP.items()),
        columns=["kaggle_name", "kenpom_name"]
    )
    save(mapping_df, "team_name_mapping", CLEAN_DIR, logger)

def clean_kenpom(logger) -> dict:
    """
    Loads and column-filters the four KenPom tables.
    """
    logger.info("Cleaning KenPom tables...")

    four_factors = load_raw("four_factors.csv", logger)
    four_factors = four_factors.drop(columns=[
        "DataThrough", "ConfOnly", "eFG_Pct", "TO_Pct", "OR_Pct",
        "FT_Rate", "DeFG_Pct", "DTO_Pct", "DOR_Pct", "DFT_Rate",
        "OE", "DE", "Tempo", "AdjOE", "AdjDE", "AdjTempo", "year"
    ])
    four_factors = add_team_season_key(four_factors, ["TeamName"], logger=logger)

    height = load_raw("height.csv", logger)
    height = height.drop(columns=[
        "DataThrough", "AvgHgt", "HgtEff", "Hgt5", "Hgt4", "Hgt3",
        "Hgt2", "Hgt1", "Exp", "Bench", "Continuity", "year"
    ])
    height = add_team_season_key(height, ["TeamName"], logger=logger)

    ratings = load_raw("ratings.csv", logger)
    ratings = ratings.drop(columns=[
        "DataThrough", "AdjEM", "Pythag", "AdjOE", "OE", "AdjDE",
        "DE", "Tempo", "AdjTempo", "Luck", "SOS", "SOSO", "SOSD",
        "NCSOS", "APL_Off", "APL_Def", "ConfAPL_Off", "ConfAPL_Def", "year",
        # duplicates already in four_factors
        "RankOE", "RankDE", "RankTempo", "RankAdjOE", "RankAdjDE", "RankAdjTempo"
    ])
    ratings = add_team_season_key(ratings, ["TeamName"], logger=logger)

    teams = load_raw("teams.csv", logger)
    teams = teams.drop(columns=["Coach", "Arena", "ArenaCity", "ArenaState"])
    teams = add_team_season_key(teams, ["TeamName"], logger=logger)

    return {
        "four_factors": four_factors,
        "height": height,
        "ratings": ratings,
        "teams": teams,
    }


def main():
    """
    Main cleaning pipeline:
        1. Clean tournament games (rounds, upsets, name mapping)
        2. Clean tournament seeds (parse seeds, name mapping)
        3. Verify all names match KenPom
        4. Save clean versions + mapping documentation
    """
    logger = setup_logger("clean")

    logger.info("═" * 50)
    logger.info("Data Cleaning Pipeline")
    logger.info("═" * 50)

    # Clean
    games = clean_tournament_games(logger)
    seeds = clean_tournament_seeds(logger)
    kenpom = clean_kenpom(logger) 

    # Verify
    logger.info("─" * 50)
    logger.info("Verifying team name mappings...")
    verify_mapping(games, logger)
    verify_mapping(seeds, logger)

    # Save
    logger.info("─" * 50)
    logger.info("Saving clean data...")
    save(games, "tournament_games", CLEAN_DIR, logger)
    save(seeds, "tournament_seeds", CLEAN_DIR, logger)
    for name, df in kenpom.items():                          
        save(df, name, CLEAN_DIR, logger)
    save_mapping(logger)

    # Summary
    logger.info("═" * 50)
    logger.info("CLEANING COMPLETE")
    logger.info(f"  Games:   {games.shape}")
    logger.info(f"  Seeds:   {seeds.shape}")
    logger.info(f"  Mapping: {len(TEAM_NAME_MAP)} entries")
    logger.info("═" * 50)

    return games, seeds


if __name__ == "__main__":
    main()