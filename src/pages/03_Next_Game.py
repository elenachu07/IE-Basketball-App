import pandas as pd
import numpy as np
import streamlit as st
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings

from data_utils import (
    load_nba, load_ie, engineer_features,
    build_ie_opponent_strength, attach_opponent_strength,
    FEATURE_COLS,
)

warnings.filterwarnings("ignore")

NBA_FILE = "data/NBA_Season_2024_25_Dataset.xlsx"
IE_FILE  = "data/IE_Basketball_Dataset.xlsx"


# ── model training (same as predictions.py) ───────────────────────────────────

@st.cache_resource
def load_and_train():
    nba_raw = load_nba(NBA_FILE)
    ie_raw  = load_ie(IE_FILE)

    nba = engineer_features(nba_raw, "NBA")
    ie  = engineer_features(ie_raw,  "IE")

    opp_profile = build_ie_opponent_strength(ie)
    nba = attach_opponent_strength(nba, opp_profile)
    ie  = attach_opponent_strength(ie,  opp_profile)

    X_nba = nba[FEATURE_COLS].fillna(0)
    y_cls = nba["result"]
    y_reg = nba["point_diff"]

    X_train, X_test, y_cls_train, _ = train_test_split(X_nba, y_cls, test_size=0.2, random_state=42)
    _, _,            y_reg_train, _ = train_test_split(X_nba, y_reg, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_tr_sc, y_cls_train)

    rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_cls_train)

    lin_model = LinearRegression()
    lin_model.fit(X_tr_sc, y_reg_train)

    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    rf_reg.fit(X_train, y_reg_train)

    return log_model, rf_model, lin_model, rf_reg, scaler, ie, opp_profile


def _team_recent_state(ie: pd.DataFrame, team: str) -> dict:
    """
    Extract rolling stats from the last 5 games for a given team.
    These mirror exactly what engineer_features computes historically.
    """
    g = ie[ie["team"] == team].sort_values("date")
    last5 = g.tail(5)

    return {
        "team_avg_point_diff": round(last5["point_diff"].mean(), 2),
        "team_avg_scored":     round(last5["points_scored"].mean(), 2),
        "team_win_rate":       round(last5["result"].mean(), 3),
        "recent_form":         round(last5["result"].mean(), 3),
        "last_game_result":    int(g["result"].iloc[-1]) if len(g) else 0,
        "last_game_date":      g["date"].iloc[-1] if len(g) else None,
        "avg_scored":          round(g["points_scored"].mean(), 2),  # full season avg for score prediction
    }


def _build_feature_row(
    home: int,
    rest_days: float,
    team_state: dict,
    opp_row: pd.Series,
) -> pd.DataFrame:
    """Assemble a single-row DataFrame with all FEATURE_COLS."""
    row = {
        "home":                home,
        "rest_days":           rest_days,
        "back_to_back":        int(rest_days == 1),
        "team_avg_point_diff": team_state["team_avg_point_diff"],
        "team_avg_scored":     team_state["team_avg_scored"],
        "team_win_rate":       team_state["team_win_rate"],
        "recent_form":         team_state["recent_form"],
        "last_game_result":    team_state["last_game_result"],
        "opp_win_rate":        opp_row["opp_win_rate"],
        "opp_avg_point_diff":  opp_row["opp_avg_point_diff"],
        "opp_avg_scored":      opp_row["opp_avg_scored"],
        "opp_recent_form":     0.5,   # unknown for future opponent
        "strength_diff":       team_state["team_avg_point_diff"] - opp_row["opp_avg_point_diff"],
        "scoring_diff":        team_state["team_avg_scored"]     - opp_row["opp_avg_scored"],
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def _predict_score(team_state: dict, opp_row: pd.Series, pred_point_diff: float) -> tuple[int, int]:
    """
    Estimate the actual scoreline from the predicted point differential.
    Uses the team's season avg scored as the anchor, then derives opponent score.
    """
    team_score = round(team_state["avg_scored"])
    opp_score  = round(team_score - pred_point_diff)
    return int(team_score), int(max(opp_score, 0))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("Next Game Prediction")
    st.write("Predict the outcome of IE Basketball's next game.")

    log_model, rf_model, lin_model, rf_reg, scaler, ie, opp_profile = load_and_train()

    all_opponents = sorted(opp_profile["opponent"].str.strip().unique().tolist())
    teams         = sorted(ie["team"].unique().tolist())

    # ── input form ────────────────────────────────────────────────────────────
    st.subheader("Game Details")

    col1, col2 = st.columns(2)
    with col1:
        team     = st.selectbox("IE Team", teams)
        location = st.selectbox("Location", ["Home", "Away"])
    with col2:
        opponent     = st.selectbox("Opponent", all_opponents + ["New opponent (not in history)"])
        game_date    = st.date_input("Game Date", value=date.today())

    # rest days: calculate from last known game or let user override
    team_state    = _team_recent_state(ie, team)
    last_date     = team_state["last_game_date"]

    if last_date is not None:
        auto_rest = (pd.Timestamp(game_date) - pd.Timestamp(last_date)).days
        auto_rest = max(auto_rest, 0)
        st.caption(f"Last game: **{last_date.date() if hasattr(last_date, 'date') else last_date}** "
                   f"→ auto-calculated rest days: **{auto_rest}**")
        rest_days = st.number_input("Rest Days (override if needed)", min_value=0,
                                    max_value=365, value=int(auto_rest))
    else:
        rest_days = st.number_input("Rest Days", min_value=0, max_value=365, value=7)

    predict_btn = st.button("Predict", type="primary")

    if not predict_btn:
        return

    # ── resolve opponent strength ─────────────────────────────────────────────
    opponent_clean = opponent.strip()
    opp_match = opp_profile[opp_profile["opponent"].str.strip() == opponent_clean]

    if len(opp_match) == 0:
        # unknown opponent → use league average
        opp_row = opp_profile[["opp_win_rate", "opp_avg_point_diff", "opp_avg_scored"]].mean()
        st.info("Opponent not in history — using league average strength.")
    else:
        opp_row = opp_match.iloc[0]

    # ── build feature row & predict ───────────────────────────────────────────
    home      = 1 if location == "Home" else 0
    X         = _build_feature_row(home, rest_days, team_state, opp_row)
    X_scaled  = scaler.transform(X)

    log_prob      = log_model.predict_proba(X_scaled)[0, 1]
    rf_prob       = rf_model.predict_proba(X)[0, 1]
    avg_win_prob  = (log_prob + rf_prob) / 2

    log_pred      = log_model.predict(X_scaled)[0]
    rf_pred       = rf_model.predict(X)[0]
    # ensemble: both must agree for high-confidence prediction
    final_result  = 1 if avg_win_prob >= 0.5 else 0
    result_label  = "Win" if final_result == 1 else "Loss"

    lin_diff      = lin_model.predict(X_scaled)[0]
    rf_diff       = rf_reg.predict(X)[0]
    avg_diff      = (lin_diff + rf_diff) / 2

    team_score, opp_score = _predict_score(team_state, opp_row, avg_diff)

    # ── display results ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Prediction")

    # headline result
    color = "🟢" if final_result == 1 else "🔴"
    st.markdown(f"## {color} Predicted Result: **{result_label}**")

    # 4 key metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Win Probability",    f"{avg_win_prob:.1%}")
    m2.metric("Expected Point Diff", f"{avg_diff:+.1f}")
    m3.metric(f"{team} Score",      team_score)
    m4.metric(f"{opponent_clean} Score", opp_score)

    # confidence bar
    st.markdown("**Win Probability**")
    st.progress(float(avg_win_prob))

    # model breakdown
    with st.expander("Model breakdown"):
        st.markdown(f"""
        | Model | Prediction | Win Probability |
        |---|---|---|
        | Logistic Regression | {'Win' if log_pred == 1 else 'Loss'} | {log_prob:.1%} |
        | Random Forest       | {'Win' if rf_pred  == 1 else 'Loss'} | {rf_prob:.1%} |
        | **Ensemble (avg)**  | **{result_label}** | **{avg_win_prob:.1%}** |
        """)

        st.markdown("**Features used for this prediction:**")
        st.dataframe(X.T.rename(columns={0: "value"}).round(3), use_container_width=True)

    # context
    with st.expander("Team context (last 5 games)"):
        last5 = ie[ie["team"] == team].sort_values("date").tail(5)
        st.dataframe(
            last5[["date", "opponent", "home", "points_scored", "points_allowed", "result"]]
            .assign(result=last5["result"].map({1: "Win", 0: "Loss"}))
            .reset_index(drop=True),
            use_container_width=True,
        )


main()