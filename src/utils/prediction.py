# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings("ignore")


# # ── file paths (update if needed) ────────────────────────────────────────────
# NBA_FILE = "data/NBA_Season_2024_25_Dataset.xlsx"
# IE_FILE  = "data/IE_Basketball_Dataset.xlsx"

# # ── features used for modelling ──────────────────────────────────────────────
# FEATURE_COLS = ["home", "opponent_strength", "rest_days"]


# # ═════════════════════════════════════════════════════════════════════════════
# # 1. LOAD & RESHAPE NBA DATA
# #    Raw NBA: one row per game  →  reshape to one row per team per game
# # ═════════════════════════════════════════════════════════════════════════════

# def load_nba(path: str) -> pd.DataFrame:
#     """
#     Load NBA game-level data and reshape to team-level.
#     """
#     # Load the data
#     nba_data = pd.read_excel(path)
    
#     # Reshape the data
#     reshaped_data = nba_data.melt(id_vars=["game_id", "date"], 
#                                    value_vars=["home_team", "away_team"],
#                                    var_name="team_type", 
#                                    value_name="team")
    
#     # Further processing can be done here
#     return reshaped_data

# def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
#     """
#     Preprocess the data for modeling.
#     """
#     # Example preprocessing steps
#     data['home'] = data['team_type'].apply(lambda x: 1 if x == 'home_team' else 0)
#     # Add more preprocessing as needed
#     return data

# def make_predictions(model, X: pd.DataFrame) -> np.ndarray:
#     """
#     Make predictions using the trained model.
#     """
#     return model.predict(X)

# def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
#     """
#     Evaluate the model's performance.
#     """
#     accuracy = accuracy_score(y_true, y_pred)
#     mse = mean_squared_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     return {"accuracy": accuracy, "mse": mse, "r2": r2}