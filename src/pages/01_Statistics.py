import pandas as pd
import streamlit as st

from data_utils import (
    load_ie, load_nba, engineer_features,
    build_ie_opponent_strength, attach_opponent_strength,
)

IE_FILE  = "src/data/IE_Basketball_Dataset.xlsx"
NBA_FILE = "src/data/NBA_Season_2024_25_Dataset.xlsx"


# ── helpers ──────────────────────────────────────────────────────────────────

def _current_streaks(group: pd.DataFrame) -> dict:
    results = group.sort_values("date")["result"].tolist()
    win_streak = loss_streak = 0
    for r in reversed(results):
        if r == 1:
            if loss_streak == 0:
                win_streak += 1
            else:
                break
        else:
            if win_streak == 0:
                loss_streak += 1
            else:
                break
    return {"win_streak": win_streak, "loss_streak": loss_streak}


def _rec(grp) -> tuple:
    """Return (record_str, win_rate, avg_pt_diff, avg_scored, avg_allowed) for a sub-group."""
    if len(grp) == 0:
        return "—", None, None, None, None
    w  = int(grp["result"].sum())
    l  = len(grp) - w
    return (
        f"{w}W – {l}L",
        round(w / len(grp) * 100, 1),
        round(grp["point_diff"].mean(), 1),
        round(grp["points_scored"].mean(), 1),
        round(grp["points_allowed"].mean(), 1),
    )


# ── sections ─────────────────────────────────────────────────────────────────

def section_overview(ie: pd.DataFrame):
    st.subheader("Overview")
    total  = len(ie)
    wins   = int(ie["result"].sum())
    losses = total - wins
    wr     = wins / total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Games",      total)
    c2.metric("Total Wins",       wins)
    c3.metric("Total Losses",     losses)
    c4.metric("Overall Win Rate", f"{wr:.1%}")


def section_record_and_location(ie: pd.DataFrame):
    """Record, scoring, home/away breakdown — all in one table."""
    st.subheader("Record & Scoring")

    rows = []
    for team, g in ie.groupby("team"):
        home_g = g[g["home"] == 1]
        away_g = g[g["home"] == 0]

        overall_rec, wr, apd, asc, aal = _rec(g)
        home_rec,   hwr, hpd, hsc, hal = _rec(home_g)
        away_rec,   awr, wpd, wsc, wal = _rec(away_g)

        rows.append({
            "Team":              team,
            "Overall Record":    overall_rec,
            "Win Rate (%)":      wr,
            "Avg Scored":        asc,
            "Avg Allowed":       aal,
            "Avg Point Diff":    apd,
            "Home Record":       home_rec,
            "Home Win Rate (%)": hwr,
            "Home Avg Pt Diff":  hpd,
            "Away Record":       away_rec,
            "Away Win Rate (%)": awr,
            "Away Avg Pt Diff":  wpd,
        })

    st.dataframe(pd.DataFrame(rows).set_index("Team"), use_container_width=True)

    # Home vs Away win rate bar chart
    st.markdown("**Home vs Away Win Rate (%)**")
    chart_data = (
        ie.groupby(["team", "home"])["result"]
        .mean().mul(100).round(1).reset_index()
    )
    chart_data["location"] = chart_data["home"].map({1: "Home", 0: "Away"})
    pivot = chart_data.pivot(index="team", columns="location", values="result").fillna(0)
    st.bar_chart(pivot)


def section_total_scoring(ie: pd.DataFrame):
    st.subheader("Total Points")
    rows = []
    for team, g in ie.groupby("team"):
        rows.append({
            "Team":           team,
            "Total Scored":   int(g["points_scored"].sum()),
            "Total Allowed":  int(g["points_allowed"].sum()),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Team"), use_container_width=True)


def section_rest_opponent(ie: pd.DataFrame):
    st.subheader("Rest Days & Opponent Strength")
    st.caption("Opponent stats are calculated from IE game results — "
               "each opponent's scoring and win rate as seen across all IE games.")

    rows = []
    for team, g in ie.groupby("team"):
        rows.append({
            "Team":                    team,
            "Avg Rest Days":           round(g["rest_days"].mean(), 1),
            "Min Rest Days":           int(g["rest_days"].min()),
            "Back-to-Back Games":      int((g["rest_days"] == 1).sum()),
            "Avg Opp Win Rate (%)":    round(g["opp_win_rate"].mean() * 100, 1),
            "Avg Opp Point Diff":      round(g["opp_avg_point_diff"].mean(), 1),
            "Avg Opp Points Scored":   round(g["opp_avg_scored"].mean(), 1),
        })

    st.dataframe(pd.DataFrame(rows).set_index("Team"), use_container_width=True)


def section_highlights(ie: pd.DataFrame):
    st.subheader("Performance Highlights")

    biggest_win = ie.loc[ie["point_diff"].idxmax()]
    worst_loss  = ie.loc[ie["point_diff"].idxmin()]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Biggest Win**")
        st.markdown(f"- **Date:** {biggest_win['date']}")
        st.markdown(f"- **Team:** {biggest_win['team']}")
        st.markdown(f"- **Opponent:** {biggest_win['opponent']}")
        st.markdown(f"- **Margin:** +{biggest_win['point_diff']}")
    with col2:
        st.markdown("**Worst Loss**")
        st.markdown(f"- **Date:** {worst_loss['date']}")
        st.markdown(f"- **Team:** {worst_loss['team']}")
        st.markdown(f"- **Opponent:** {worst_loss['opponent']}")
        st.markdown(f"- **Margin:** {worst_loss['point_diff']}")

    st.markdown("**Current Streaks**")
    streak_rows = []
    for team, g in ie.groupby("team"):
        s = _current_streaks(g)
        streak_rows.append({
            "Team":                team,
            "Current Win Streak":  f"{s['win_streak']}W" if s["win_streak"] else "—",
            "Current Loss Streak": f"{s['loss_streak']}L" if s["loss_streak"] else "—",
        })
    st.dataframe(pd.DataFrame(streak_rows).set_index("Team"), use_container_width=True)


def section_men_vs_women(ie: pd.DataFrame):
    st.subheader("Men vs Women Comparison")

    rows = []
    for team, g in ie.groupby("team"):
        n = len(g)
        w = int(g["result"].sum())
        rows.append({
            "Team":                  team,
            "Games":                 n,
            "Win Rate (%)":          round(w / n * 100, 1),
            "Avg Scored":            round(g["points_scored"].mean(), 1),
            "Avg Allowed":           round(g["points_allowed"].mean(), 1),
            "Avg Point Diff":        round(g["point_diff"].mean(), 1),
            "Avg Opp Win Rate (%)":  round(g["opp_win_rate"].mean() * 100, 1),
            "Avg Opp Point Diff":    round(g["opp_avg_point_diff"].mean(), 1),
            "Avg Rest Days":         round(g["rest_days"].mean(), 1),
        })

    st.dataframe(pd.DataFrame(rows).set_index("Team"), use_container_width=True)

    st.markdown("**Average Points Scored vs Allowed**")
    score_comp = pd.DataFrame(rows).set_index("Team")[["Avg Scored", "Avg Allowed"]]
    st.bar_chart(score_comp)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("Statistics")
    st.write("IE University Basketball — performance analysis")

    ie_raw = load_ie(IE_FILE)
    ie     = engineer_features(ie_raw, "IE")

    # Build opponent strength from IE data itself (not NBA)
    opp_profile = build_ie_opponent_strength(ie)
    ie = attach_opponent_strength(ie, opp_profile)

    section_overview(ie)
    section_record_and_location(ie)
    section_total_scoring(ie)
    section_rest_opponent(ie)
    section_highlights(ie)
    section_men_vs_women(ie)


main()