# IE Basketball Statistics and Analysis App

This project is a data analysis and machine learning application designed to analyze and predict the performance of the IE University Men's and Women's basketball teams. The system uses real IE match data and utilizes datasets from the NBA 2024-2025 to generate predictions.

## Project Structure

```
basketball-stats-app
├── .devcontainer
    ├── devcontainer.json      #Defines a development environment for the project so it runs the same everywhere
├── src
│   ├── Home.py                # Main entry point for the Streamlit application
|   ├── data_utils.py          #Functions for loading and processing basketball data
│   ├── pages
│   │   ├── 01_Statistics.py           # Home page with an introduction to the app
│   │   ├── 02_Predictions.py      # Page displaying basketball match statistics
│   │   └── 03_Next_Game.py      # Page for making predictions based on match data
│   └── data
│       ├── NBA_Season_2024_25_Dataset.xlsx  # NBA match results dataset
│       └── IE_Basketball_Dataset.xlsx        # Additional basketball match results dataset
├── .streamlit
│   └── config.toml          # Configuration settings for the Streamlit app
├── .gitattributes            #Rules for how Git should treat files in repository
├── requirements.txt          # Required Python packages for the project
└── README.md                 # Documentation for the project
```

## 🚀 Features

- 📊 **Statistics Dashboard**

  - Team records (wins/losses)
  - Scoring averages
  - Home vs Away performance
  - Rest days
  - Opponent strength analysis
  - Men vs Women comparison

- 🤖 **Prediction System**

  - Predict match outcome (Win/Loss)
  - Estimate win probability
  - Predict point difference
  - Compare Logistic Regression and Random Forest models

- 🔮 **Next Game Prediction**
  - Simulate future match outcomes based on input conditions

## How to run the project

1. Clone the repository:

   ```
   git clone https://github.com/elenachu07/IE-Basketball-App.git
   cd basketball-stats-app
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/Home.py
   ```
