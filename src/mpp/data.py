"""Download and preprocess historical international football results."""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_FILE = DATA_DIR / "results.csv"

# Pre-tournament warm-up competitions: coaches rotate squads, results are not representative.
WARMUP_TOURNAMENTS = {
    "FIFA Series",
    "CONCACAF Series",
    "Baltic Cup",
    "Tri-Nations Cup",
    "Unity Cup",
    "Kirin Cup",
    "King's Cup",
    "Soccer Ashes",
    "Al Ain International Cup",
    "MSG Prime Minister's Cup",
    "Diamond Jubilee International Football Tournament",
    "Morocco, Capital of African Football",
    "Outrigger Challenge Cup",
}


def download_data(force: bool = False) -> Path:
    """Download historical results CSV if not already present."""
    DATA_DIR.mkdir(exist_ok=True)
    if not RESULTS_FILE.exists() or force:
        print("Downloading historical match data...")
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        RESULTS_FILE.write_bytes(response.content)
        print(f"Saved {len(response.content) // 1024} KB to {RESULTS_FILE}")
    return RESULTS_FILE


def load_matches(
    min_year: int = 2016,
    reference_date: date | None = None,
    xi: float = 0.004,
    include_friendlies: bool = False,
    friendly_weight: float = 0.3,
    min_matches_per_team: int = 15,
) -> pd.DataFrame:
    """Load and preprocess matches with exponential time-decay weights.

    Args:
        min_year: Ignore matches before this year.
        reference_date: Decay reference point (default: today).
        xi: Daily decay rate. Default 0.004 ≈ half-life of ~173 days.
        include_friendlies: Whether to include friendly matches.
        friendly_weight: Weight multiplier applied to friendly matches when
            include_friendlies=True. Values < 1 reduce their influence relative
            to competitive matches while keeping them in the dataset.
        min_matches_per_team: Drop matches involving teams with fewer than N
            appearances in the dataset. Removes tiny nations (Anguilla, Cuba…)
            whose extreme defense parameters inflate opponents' attack ratings
            (confederation-schedule bias).

    Returns:
        DataFrame with columns: date, home_team, away_team, home_score,
        away_score, neutral, weight.
    """
    if reference_date is None:
        reference_date = date.today()

    download_data()
    df = pd.read_csv(
        RESULTS_FILE,
        parse_dates=["date"],
        true_values=["True", "true"],
        false_values=["False", "false"],
    )
    df = df[df["date"].dt.year >= min_year].copy()

    # Only keep matches up to the reference date
    df = df[df["date"].dt.date <= reference_date]

    excluded = {"Friendly"} | WARMUP_TOURNAMENTS
    if include_friendlies:
        excluded = WARMUP_TOURNAMENTS
    df = df[~df["tournament"].isin(excluded)]

    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Ensure neutral is boolean
    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].astype(str).str.lower() == "true"
    df["neutral"] = df["neutral"].astype(bool)

    # Remove matches involving nations with too few appearances.
    # These tiny nations have extreme (very negative) defense parameters that
    # artificially inflate the attack ratings of their opponents.
    if min_matches_per_team > 0:
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        qualified = counts[counts >= min_matches_per_team].index
        df = df[df["home_team"].isin(qualified) & df["away_team"].isin(qualified)]

    ref = pd.Timestamp(reference_date)
    days_ago = (ref - df["date"]).dt.days.clip(lower=0)
    df["weight"] = days_ago.apply(lambda d: math.exp(-xi * d))

    if include_friendlies and friendly_weight != 1.0:
        is_friendly = df["tournament"] == "Friendly"
        df.loc[is_friendly, "weight"] *= friendly_weight

    cols = ["date", "home_team", "away_team", "home_score", "away_score", "neutral", "weight"]
    return df[cols].reset_index(drop=True)


def load_all_matches(
    min_year: int = 1960,
    reference_date: date | None = None,
) -> pd.DataFrame:
    """Load the full historical dataset including friendlies and the tournament column.

    Used for Elo computation: deeper history with all tournament types gives
    better inter-confederation calibration than the filtered training window.

    Returns:
        DataFrame with columns: date, home_team, away_team, home_score,
        away_score, neutral, tournament.
    """
    if reference_date is None:
        reference_date = date.today()

    download_data()
    df = pd.read_csv(
        RESULTS_FILE,
        parse_dates=["date"],
        true_values=["True", "true"],
        false_values=["False", "false"],
    )
    df = df[df["date"].dt.year >= min_year].copy()
    df = df[df["date"].dt.date <= reference_date]
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].astype(str).str.lower() == "true"
    df["neutral"] = df["neutral"].astype(bool)

    cols = ["date", "home_team", "away_team", "home_score", "away_score", "neutral", "tournament"]
    return df[cols].reset_index(drop=True)
