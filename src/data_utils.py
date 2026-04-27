import pandas as pd

# List of features that will be used as inputs for the model
FEATURE_COLS = [
    "home",
    "rest_days",
    "back_to_back",

    "team_avg_point_diff",
    "team_avg_scored",
    "team_win_rate",
    "recent_form",
    "last_game_result",

    "opp_win_rate",
    "opp_avg_point_diff",
    "opp_avg_scored",
    "opp_recent_form",

    "strength_diff",
    "scoring_diff",
]

# Load NBA dataset and convert to team-level data
def load_nba(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path)

    # Create dataset from away team perspective
    away = pd.DataFrame({
        "date": raw["Date"],
        "team": raw["Away Team"],
        "opponent": raw["Home Team"],
        "home": 0,
        "points_scored": raw["Away Points"],
        "points_allowed": raw["Home Points"],
    })

    # Create dataset from home team perspective
    home = pd.DataFrame({
        "date": raw["Date"],
        "team": raw["Home Team"],
        "opponent": raw["Away Team"],
        "home": 1,
        "points_scored": raw["Home Points"],
        "points_allowed": raw["Away Points"],
    })

    # Combine both so each game becomes 2 rows (one per team)
    return pd.concat([away, home], ignore_index=True)


# Load and clean IE dataset
def load_ie(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Rename columns to match NBA format
    df = df.rename(columns={
        "Team": "team",
        "Date": "date",
        "Opponent": "opponent",
        "HomeAway": "_home_away",
        "IEScore": "points_scored",
        "OppScore": "points_allowed",
        "IEWin/Loss": "_win_loss_raw",
    })

    # Convert home/away text to numeric (1 = home, 0 = away)
    df["home"] = (
        df["_home_away"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("home")
        .astype(int)
    )

    # Remove unused columns
    df = df.drop(columns=["_home_away", "_win_loss_raw"], errors="ignore")
    return df

# Clean and standardize date column
def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # First attemp: asssume day-month-year formate
    parsed = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    # If some dates failed, try agin with default format
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(df.loc[mask, "date"], errors="coerce")

    df["date"] = parsed
    return df

# Feature engineering: create new features based on existing data to help the model learn patterns
def engineer_features(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    df = df.copy()
    # Clean dates and sort chronologically per team
    df = _parse_dates(df)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    # Basic features: point difference and win/loss result
    df["point_diff"] = df["points_scored"] - df["points_allowed"]
    df["result"] = (df["point_diff"] > 0).astype(int)

    # Rest days between games
    df["rest_days"] = (
        df.groupby("team")["date"]
        .diff()
        .dt.days
        .clip(lower=0)   
    )

    # Back to back games
    df["back_to_back"] = (df["rest_days"].fillna(0) == 1).astype(int)

    # Rolling stats (last 5 games, only past data)
    df["team_avg_point_diff"] = (
        df.groupby("team")["point_diff"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0)
    )

    df["team_avg_scored"] = (
        df.groupby("team")["points_scored"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(df["points_scored"].mean())
    )

    df["team_win_rate"] = (
        df.groupby("team")["result"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.5)
    )

    df["recent_form"] = (
        df.groupby("team")["result"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.5)
    )

    # Result of previous game
    df["last_game_result"] = (
        df.groupby("team")["result"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return df

# Build team strength (used as opponent strength later)
def build_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        df.groupby("team")
        .agg(
            opp_win_rate=("result", "mean"),
            opp_avg_point_diff=("point_diff", "mean"),
            opp_avg_scored=("points_scored", "mean"),
        )
        .reset_index()
    )
    return profile

# Build IE opponent strength from the opponent's perspective (using IE game data)
def build_ie_opponent_strength(ie_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build opponent profiles from IE game data using the OPPONENT's perspective.
 
    From each IE row:
      - opponent scored    = ie points_allowed
      - opponent allowed   = ie points_scored
      - opponent won       = ie lost (1 - result)
 
    Returns a df with column 'opponent' (not 'team') so it merges directly.
    This gives 21 distinct opponent rows instead of just Men/Women.
    """
    opp = ie_df.copy()
    opp["opp_point_diff"] = opp["points_allowed"] - opp["points_scored"]
    opp["opp_won"]        = (1 - opp["result"]).astype(int)
 
    profile = (
        opp.groupby("opponent")
        .agg(
            opp_win_rate       =("opp_won",        "mean"),
            opp_avg_point_diff =("opp_point_diff", "mean"),
            opp_avg_scored     =("points_allowed", "mean"),  # opp scored = IE allowed
        )
        .reset_index()
        # column is already named 'opponent' — ready to merge directly, no rename needed
    )
    return profile
 
# Attach opponent strength to main dataset
def attach_opponent_strength(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    opp_cols = ["opp_win_rate", "opp_avg_point_diff", "opp_avg_scored"]
 
    # Remove old columns to avoid duplicates
    df = df.drop(columns=[c for c in opp_cols if c in df.columns], errors="ignore")
 
    # profile from build_nba_opponent_strength has column 'team' → rename to 'opponent'
    # profile from build_ie_opponent_strength already has column 'opponent' → no rename needed
    merge_col = "opponent" if "opponent" in profile.columns else "team"
    profile_ready = profile.rename(columns={merge_col: "opponent"})
    
    # Merge opponent stats into main dataset
    merged = df.merge(profile_ready, on="opponent", how="left")
    
    # Fill missing values with averages
    fallback = profile[opp_cols].mean()
    for col in opp_cols:
        merged[col] = merged[col].fillna(fallback[col])
    
    # Opponent recent form (last 5 games)
    merged["opp_recent_form"] = (
        merged.groupby("opponent")["result"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.5)
    )
    # Comparison features
    merged["strength_diff"] = merged["team_avg_point_diff"] - merged["opp_avg_point_diff"]
    merged["scoring_diff"]  = merged["team_avg_scored"]     - merged["opp_avg_scored"]
 
    return merged
