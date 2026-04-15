# Financial Analytics Project

A Streamlit-based financial analytics app for data preprocessing, exploratory analysis, simple forecasting, and anomaly detection.

## Project Structure

- `app.py` - Main Streamlit app
- `data/sample_data.csv` - Example dataset
- `src/preprocess.py` - Data loading and cleaning
- `src/analysis.py` - KPI and chart helpers
- `src/prediction.py` - Simple trend forecasting
- `src/anomaly.py` - Isolation Forest anomaly detection
- `utils/helpers.py` - Shared utility functions

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Expected Data Columns

The sample workflow assumes these columns:
- `date`
- `revenue`
- `expenses`

You can upload your own CSV with compatible columns.
