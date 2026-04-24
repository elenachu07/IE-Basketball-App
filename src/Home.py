import streamlit as st

# Configure the streamlit app layout and metadata
st.set_page_config(
    page_title="IE Basketball App",
    page_icon="🏀",
    layout="wide",
)

# Sidebar title
st.sidebar.title("🏀 IE Basketball")

# Home page content
st.title("🏀 IE University Basketball")
st.markdown("### Welcome to the IE Basketball Analysis App")

# Description of the application
st.write(
    "This app provides performance analysis and predictions "
    "for the IE University Men's and Women's basketball teams, "
    "using models trained on NBA 2024-25 season data."
)

# Divider for visual separation
st.markdown("---")

# Create a 3-column layout for navigation buttons
col1, col2, col3 = st.columns(3)

# ── Statistics Section ───────────────────────────────────────
with col1:
    st.markdown("### 📊 Statistics")

    # Button to navitage to Statistics page
    st.write("Explore team records, scoring, rest days, opponent strength, and Men vs Women comparisons.")
    if st.button("Go to Statistics", use_container_width=True):
        st.switch_page("pages/01_Statistics.py")
        
# ── Predictions Section ──────────────────────────────────────
with col2:
    st.markdown("### 🤖 Predictions")
    # Button to navitage to Predictions page
    st.write("See how the NBA-trained model predicts IE game outcomes using Logistic Regression and Random Forest.")
    if st.button("Go to Predictions", use_container_width=True):
        st.switch_page("pages/02_Predictions.py")

# ── Next Game Section ────────────────────────────────────────
with col3:
    st.markdown("### 🔮 Next Game")
    # Button to navitage to Next Game prediction page
    st.write("Predict the result, score, win probability, and point difference for an upcoming game.")
    if st.button("Go to Next Game", use_container_width=True):
        st.switch_page("pages/03_Next_Game.py")

# Divider before explanation section
st.markdown("---")

# Explanation of how the system works
st.markdown(
    """
    **How it works:**
    - Models are trained on the full NBA 2024-25 season (~2,400 team-game rows)
    - Features include home/away, rest days, recent form, and opponent strength
    - Opponent strength for IE games is calculated directly from IE game history
    - Predictions use an ensemble of Logistic Regression and Random Forest
    """
)



