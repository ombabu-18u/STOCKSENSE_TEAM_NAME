# StockSense - Predict Demand. Prevent Stock-out. Power Better Decisions (NovaMart Retail Pvt. Ltd.)

## 🎯 Hackathon Mission & Overview
This repository contains the IntelliData 2026 Hackathon submission for NovaMart Retail Pvt. Ltd. The mission of this project is to build an end-to-end intelligent data pipeline capable of reducing out-of-stock incidents while minimizing excess inventory costs.

The project fulfills the 3 mandatory Hackathon rounds:
1. **Round 1 (Data Quality & EDA):** Robust data ingestion, cleaning anomalies, mapping categorical discrepancies, and exploratory statistical analysis for business insights.
2. **Round 2 (Feature Engineering & Predictive Modeling):** Developing a time-aware Machine Learning pipeline strictly utilizing historical and lag features. Two models are built: a 7-day Demand Forecast Model and a Stock-out Risk Classifier.
3. **Round 3 (Business Action & Dashboard Prototype):** Translating predictive outputs into a Business Action Layer and showcasing the intelligence through an interactive, Streamlit-based dashboard for retail managers.

## 📁 Solution Architecture & Folder Structure

```
STOCKSENSE_TEAM_NAME/
│
├── data/
│   ├── raw/                  # Original generated sample data
│   └── processed/            # Cleaned, aggregated master analytics dataset
│
├── models/                   # Serialized ML models (.pkl)
│   ├── demand_forecast_model.pkl
│   └── stockout_risk_model.pkl
│
├── reports/                  # Markdown reports and output CSVs
│   ├── eda_charts/           # EDA visualizations (PNG)
│   ├── data_quality_report.md
│   ├── eda_and_statistics_summary.md
│   └── manager_action_recommendations.csv
│
├── setup_and_clean.py        # Generates, cleans, and structures data (Round 1)
├── eda_and_statistics.py     # Exploratory analysis & hypothesis testing (Round 1/2)
├── ml_pipeline.py            # Feature engineering and model training (Round 2)
├── app.py                    # Streamlit prototype dashboard and Action Layer (Round 3)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🚀 How to Run the Project

**1. Install Dependencies**
Ensure you have Python 3 installed, then run:
```bash
pip install -r requirements.txt
```

**2. Execute the Data Pipeline**
Run the scripts sequentially to generate data, analyze it, and train the models:
```bash
python3 setup_and_clean.py
python3 eda_and_statistics.py
python3 ml_pipeline.py
```

**3. Launch the Dashboard**
Once the models are generated and saved in `models/`, launch the interactive Streamlit prototype:
```bash
streamlit run app.py
```

## 📊 Model Performance & Business Impact Summary

**Machine Learning Architecture:**
- **Demand Forecasting (Model 1):** Utilizes `RandomForestRegressor` to predict demand across the next 7 days based on robust lagged features, 7-day rolling statistics, and seasonality indicators.
- **Stock-out Risk Classification (Model 2):** Utilizes `RandomForestClassifier` weighted for class imbalances to assign high, medium, or low risk flags for impending stock-outs. 

**Business Action Layer:**
The pipeline translates mathematical probabilities into actionable retail decisions.
- **Safety Stock Calculation:** A 20% safety buffer is dynamically appended to forecasted demand.
- **Reorder Quantity Logic:** Algorithmically computes exactly how much stock is needed by subtracting current on-hand inventory from recommended stock thresholds.
- **Manager Actions:** Outputs a direct, prioritized list of units to be reordered for the manager to review, streamlining supply chain efficiency and reducing revenue leakages from empty shelves.
