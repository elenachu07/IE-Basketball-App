import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

from data_utils import (
    load_nba, load_ie, engineer_features,
    build_opponent_strength, attach_opponent_strength,
    FEATURE_COLS,
)

warnings.filterwarnings("ignore")

NBA_FILE = "data/NBA_Season_2024_25_Dataset.xlsx"
IE_FILE  = "data/IE_Basketball_Dataset.xlsx"


def train_logistic(nba: pd.DataFrame):
    X = nba[FEATURE_COLS].fillna(0)
    y = nba["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y_train)

    return model, scaler, accuracy_score(y_train, model.predict(X_tr)), accuracy_score(y_test, model.predict(X_te))


def train_random_forest(nba: pd.DataFrame):
    nba = nba.sort_values("date").reset_index(drop=True)

    split_idx = int(len(nba) * 0.8)

    train_df = nba.iloc[:split_idx]
    test_df = nba.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df["result"]

    X_test = test_df[FEATURE_COLS].fillna(0)
    y_test = test_df["result"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    return model, train_acc, test_acc

def train_linear(nba: pd.DataFrame):
    nba = nba.sort_values("date").reset_index(drop=True)

    split_idx = int(len(nba) * 0.8)

    train_df = nba.iloc[:split_idx]
    test_df = nba.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df["point_diff"]

    X_test = test_df[FEATURE_COLS].fillna(0)
    y_test = test_df["point_diff"]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_tr, y_train)

    train_r2 = r2_score(y_train, model.predict(X_tr))
    test_r2 = r2_score(y_test, model.predict(X_te))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_te)))

    return model, scaler, train_r2, test_r2, rmse


def predict_ie(ie, log_model, log_scaler, rf_model, lin_model, lin_scaler) -> pd.DataFrame:
    X      = ie[FEATURE_COLS].fillna(0)
    X_sc   = log_scaler.transform(X)

    log_pred   = log_model.predict(X_sc)
    log_prob   = log_model.predict_proba(X_sc)[:, 1]
    rf_pred    = rf_model.predict(X)
    rf_prob    = rf_model.predict_proba(X)[:, 1]
    pred_diff  = lin_model.predict(lin_scaler.transform(X))

    results = ie[["date", "team", "opponent", "result"]].copy()
    results["actual"]           = results["result"].map({1: "Win", 0: "Loss"})
    results["logistic_pred"]    = pd.Series(log_pred).map({1: "Win", 0: "Loss"}).values
    results["logistic_prob"]    = np.round(log_prob, 3)
    results["rf_pred"]          = pd.Series(rf_pred).map({1: "Win", 0: "Loss"}).values
    results["rf_prob"]          = np.round(rf_prob, 3)
    results["pred_pt_diff"]     = np.round(pred_diff, 1)
    results["log_correct"]      = np.where(results["result"] == log_pred, "✓", "✗")
    results["rf_correct"]       = np.where(results["result"] == rf_pred,  "✓", "✗")

    return results


def main():
    st.title("Predictions")
    st.write("NBA-trained model predictions for IE Basketball")

    nba_raw = load_nba(NBA_FILE)
    ie_raw  = load_ie(IE_FILE)

    nba = engineer_features(nba_raw, "NBA")
    ie  = engineer_features(ie_raw,  "IE")

    opp_profile = build_opponent_strength(nba)
    nba = attach_opponent_strength(nba, opp_profile)
    ie  = attach_opponent_strength(ie,  opp_profile)

    log_model, log_scaler, log_tr, log_te          = train_logistic(nba)
    rf_model,  rf_tr, rf_te                        = train_random_forest(nba)
    lin_model, lin_scaler, lin_tr, lin_te, lin_rmse = train_linear(nba)

    # ── Model performance ─────────────────────────────────────────────────────
    st.subheader("Model Performance (trained on NBA 2024-25)")

    with st.expander("Features used"):
        st.markdown("""
        | Feature | Why it helps |
        |---|---|
        | `home` | Home court advantage is one of the strongest signals in basketball |
        | `opp_win_rate` | Stronger opponents → harder to win |
        | `opp_avg_point_diff` | Captures both offense and defense quality in one number |
        | `opp_avg_scored` | How much the opponent puts up — tests your defense |
        | `rest_days` | Fatigue has a real effect on performance |
        | `back_to_back` | Playing on consecutive days (rest_days=1) is a nonlinear fatigue spike |
        | `recent_form` | Rolling 5-game win rate — teams in good form tend to stay in form |
        """)

    st.markdown("**Logistic Regression**")
    c1, c2 = st.columns(2)
    c1.metric("Train Accuracy", f"{log_tr:.2%}")
    c2.metric("Test Accuracy",  f"{log_te:.2%}")

    st.markdown("**Random Forest** *(usually more accurate — captures nonlinear patterns)*")
    c3, c4 = st.columns(2)
    c3.metric("Train Accuracy", f"{rf_tr:.2%}")
    c4.metric("Test Accuracy",  f"{rf_te:.2%}")

    st.markdown("**Linear Regression** *(point differential)*")
    c5, c6, c7 = st.columns(3)
    c5.metric("Train R²",  f"{lin_tr:.3f}")
    c6.metric("Test R²",   f"{lin_te:.3f}")
    c7.metric("RMSE",      f"{lin_rmse:.2f}")

    # ── IE predictions ────────────────────────────────────────────────────────
    results = predict_ie(ie, log_model, log_scaler, rf_model, lin_model, lin_scaler)

    st.subheader("IE Prediction Results")
    st.dataframe(results.drop(columns="result"), use_container_width=True)

    st.subheader("Accuracy Summary")
    log_acc = (results["result"] == results["logistic_pred"].map({"Win": 1, "Loss": 0})).mean()
    rf_acc  = (results["result"] == results["rf_pred"].map({"Win": 1, "Loss": 0})).mean()

    s1, s2, s3 = st.columns(3)
    s1.metric("Logistic Accuracy",       f"{log_acc:.2%}")
    s2.metric("Random Forest Accuracy",  f"{rf_acc:.2%}")
    s3.metric("Avg Predicted Pt Diff",   f"{results['pred_pt_diff'].mean():+.1f}")

    # ── Feature importance (Random Forest) ───────────────────────────────────
    st.subheader("Feature Importance (Random Forest)")
    fi = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    st.bar_chart(fi.set_index("feature")["importance"])


main()