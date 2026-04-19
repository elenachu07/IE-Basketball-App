import streamlit as st

st.set_page_config(
    page_title="IE Basketball App",
    page_icon="🏀",
    layout="wide",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        [data-testid="stSidebarHeader"] { display: none; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── sidebar title ─────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='font-size: 22px; font-weight: bold; padding: 10px 0px;'>🏀 IE Basketball</div>",
    unsafe_allow_html=True,
)

# ── session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_to(page: str):
    st.session_state.page = page
    st.rerun()

# ── pages ─────────────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "Home":
    st.title("🏀 IE University Basketball")
    st.markdown("### Welcome to the IE Basketball Analysis App")
    st.write(
        "This app provides performance analysis and predictions "
        "for the IE University Men's and Women's basketball teams, "
        "using models trained on NBA 2024-25 season data."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Statistics")
        st.write("Explore team records, scoring, rest days, opponent strength, and Men vs Women comparisons.")
        if st.button("Go to Statistics", use_container_width=True):
            st.switch_page("pages/01_Statistics.py")

    with col2:
        st.markdown("### 🤖 Predictions")
        st.write("See how the NBA-trained model predicts IE game outcomes using Logistic Regression and Random Forest.")
        if st.button("Go to Predictions", use_container_width=True):
            st.switch_page("pages/02_Predictions.py")

    with col3:
        st.markdown("### 🔮 Next Game")
        st.write("Predict the result, score, win probability, and point difference for an upcoming game.")
        if st.button("Go to Next Game", use_container_width=True):
            st.switch_page("pages/03_Next_Game.py")

    st.markdown("---")
    st.markdown(
        """
        **How it works:**
        - Models are trained on the full NBA 2024-25 season (~2,400 team-game rows)
        - Features include home/away, rest days, recent form, and opponent strength
        - Opponent strength for IE games is calculated directly from IE game history
        - Predictions use an ensemble of Logistic Regression and Random Forest
        """
    )

elif page == "Statistics":
    import statistics
    statistics.main()

elif page == "Predictions":
    import predictions
    predictions.main()

elif page == "Next Game":
    import next_game
    next_game.main()







