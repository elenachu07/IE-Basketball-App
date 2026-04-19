# Basketball Statistics App

This project is a Streamlit application designed to display statistics and predictions related to basketball match results. It utilizes datasets from the NBA and other basketball leagues to provide insights and analytics.

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

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd basketball-stats-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

## Usage

- Navigate to the home page for an overview of the application.
- Visit the statistics page to view various basketball match statistics.
- Use the predictions page to make predictions based on the loaded datasets.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you'd like to add.