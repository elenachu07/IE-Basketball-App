# IE Basketball Statistics and Analysis App

This project is a data analysis and machine learning application designed to analyze and predict the performance of the IE University Men's and Women's basketball teams. The system uses real IE match data and utilizes datasets from the NBA 2024-2025 to generate predictions.

## Project Structure

```
basketball-stats-app
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   ├── pages
│   │   ├── home.py           # Home page with an introduction to the app
│   │   ├── statistics.py      # Page displaying basketball match statistics
│   │   └── predictions.py      # Page for making predictions based on match data
│   ├── utils
│   │   └── prediction.py      # Functions for loading and processing basketball data
│   └── data
│       ├── NBA_Season_2024_25_Dataset.xlsx  # NBA match results dataset
│       └── IE_Basketball_Dataset.xlsx        # Additional basketball match results dataset
├── requirements.txt          # Required Python packages for the project
├── .streamlit
│   └── config.toml          # Configuration settings for the Streamlit app
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
