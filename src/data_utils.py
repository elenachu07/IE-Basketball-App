import pandas as pd

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


def load_nba(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path)

    away = pd.DataFrame({
        "date": raw["Date"],
        "team": raw["Away Team"],
        "opponent": raw["Home Team"],
        "home": 0,
        "points_scored": raw["Away Points"],
        "points_allowed": raw["Home Points"],
    })

    home = pd.DataFrame({
        "date": raw["Date"],
        "team": raw["Home Team"],
        "opponent": raw["Away Team"],
        "home": 1,
        "points_scored": raw["Home Points"],
        "points_allowed": raw["Away Points"],
    })

    return pd.concat([away, home], ignore_index=True)


def load_ie(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    df = df.rename(columns={
        "Team": "team",
        "Date": "date",
        "Opponent": "opponent",
        "HomeAway": "_home_away",
        "IEScore": "points_scored",
        "OppScore": "points_allowed",
        "IEWin/Loss": "_win_loss_raw",
    })

    df["home"] = (
        df["_home_away"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("home")
        .astype(int)
    )

    df = df.drop(columns=["_home_away", "_win_loss_raw"], errors="ignore")
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    parsed = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(df.loc[mask, "date"], errors="coerce")

    df["date"] = parsed
    return df


def engineer_features(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    df = df.copy()
    df = _parse_dates(df)
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    df["point_diff"] = df["points_scored"] - df["points_allowed"]
    df["result"] = (df["point_diff"] > 0).astype(int)

    df["rest_days"] = (
        df.groupby("team")["date"]
        .diff()
        .dt.days
        .clip(lower=0)   
    )

    df["back_to_back"] = (df["rest_days"].fillna(0) == 1).astype(int)

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

    df["last_game_result"] = (
        df.groupby("team")["result"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return df


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
 
 
def attach_opponent_strength(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    opp_cols = ["opp_win_rate", "opp_avg_point_diff", "opp_avg_scored"]
 
    # drop stale opp columns to avoid _x/_y duplicates on repeated calls
    df = df.drop(columns=[c for c in opp_cols if c in df.columns], errors="ignore")
 
    # profile from build_nba_opponent_strength has column 'team' → rename to 'opponent'
    # profile from build_ie_opponent_strength already has column 'opponent' → no rename needed
    merge_col = "opponent" if "opponent" in profile.columns else "team"
    profile_ready = profile.rename(columns={merge_col: "opponent"})
 
    merged = df.merge(profile_ready, on="opponent", how="left")
 
    fallback = profile[opp_cols].mean()
    for col in opp_cols:
        merged[col] = merged[col].fillna(fallback[col])
 
    merged["opp_recent_form"] = (
        merged.groupby("opponent")["result"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.5)
    )
 
    merged["strength_diff"] = merged["team_avg_point_diff"] - merged["opp_avg_point_diff"]
    merged["scoring_diff"]  = merged["team_avg_scored"]     - merged["opp_avg_scored"]
 
    return merged
