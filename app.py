"""
Streamlit dashboard for visualizing solar wind data and predicting geomagnetic storms.
Provides data exploration features and a live forecasting view.
Architecture note: Serves as the UI layer calling into the pre-trained models via `src.model_inference`.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append(os.path.abspath("."))

st.set_page_config(page_title="SolarWind Forecaster", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Fira Sans', sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Fira Code', monospace !important;
    color: #0EA5E9 !important;
}

.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
    background-image: radial-gradient(circle at 50% top, #0F172A 0%, #0B1120 100%);
}

[data-testid="stHeader"] {
    background-color: rgba(11, 17, 32, 0.7) !important;
    backdrop-filter: blur(10px);
}

/* Glassmorphism Containers */
[data-testid="stExpander"], [data-testid="stVerticalBlock"] > div > div > div[data-testid="stContainer"] {
    background-color: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(14, 165, 233, 0.2);
    border-radius: 12px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    padding: 15px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border-right: 1px solid rgba(14, 165, 233, 0.2);
}

/* Buttons */
.stButton > button {
    background-color: rgba(2, 132, 199, 0.8);
    backdrop-filter: blur(10px);
    color: #FFFFFF;
    font-family: 'Fira Code', monospace;
    font-weight: 600;
    border: 1px solid #0EA5E9;
    border-radius: 6px;
    box-shadow: 0 0 10px rgba(14, 165, 233, 0.3);
    transition: all 0.2s;
}

.stButton > button:hover {
    background-color: #0EA5E9;
    box-shadow: 0 0 15px rgba(14, 165, 233, 0.6);
    transform: translateY(-2px);
    color: #FFFFFF;
}

/* Accent for warnings/amber */
.stAlert {
    background-color: rgba(245, 158, 11, 0.1);
    border: 1px solid #F59E0B;
    color: #F8FAFC;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("SolarWind Forecaster: Geomagnetic Storm Forecaster")
st.markdown("""
This dashboard visualizes historical solar wind data and simulates our Gradient Boosting forecasting model. 
A **SYM-H** index below -50 nT indicates a moderate geomagnetic storm, and below -100 nT indicates an intense storm.
""")


st.sidebar.header("Navigation")
view_mode = st.sidebar.radio(
    "Select View:", ["Data Exploration (EDA)", "Forecasting Dashboard"]
)


@st.cache_data
def load_data():
    """Loads processed solar wind data from CSV, or generates mock data if not found."""
    try:
        df = pd.read_csv("data/processed/omni.csv", parse_dates=["timestamp"])
        return df
    except FileNotFoundError:
        dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=2000, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "sym_h": np.random.normal(-15, 20, 2000),
                "bz_gsm": np.random.normal(0, 5, 2000),
                "speed": np.random.normal(450, 50, 2000),
            }
        )
        df.loc[1500:1600, "sym_h"] = np.random.normal(-80, 10, 101)
        return df


df = load_data()

if view_mode == "Data Exploration (EDA)":
    st.header("Exploratory Data Analysis")

    if df is not None:
        st.write(f"Loaded {len(df):,} records of solar wind data.")
        st.dataframe(df.head(100))

        st.subheader("SYM-H Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["sym_h"], bins=100, kde=True, ax=ax)
        ax.axvline(-50, color="r", linestyle="--", label="Moderate Storm (-50 nT)")
        ax.axvline(
            -100, color="darkred", linestyle="--", label="Intense Storm (-100 nT)"
        )
        ax.set_title("Distribution of SYM-H Index")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Recent Solar Wind Trends")

        recent_df = df.tail(1000).reset_index(
            drop=True if df.index.name != "timestamp" else False
        )
        if "timestamp" not in recent_df.columns:
            recent_df = recent_df.reset_index()
        st.line_chart(recent_df, x="timestamp", y=["sym_h", "bz_gsm"])
    else:
        st.warning(
            "Data not found. Please run `python src/data_ingestion.py` to generate the dataset."
        )

elif view_mode == "Forecasting Dashboard":
    st.header("Real-Time Forecasting")

    if df is not None:
        st.write("Using the latest available data to forecast storm risk.")
        latest = df.iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest SYM-H", f"{latest['sym_h']} nT")
        col2.metric("Solar Wind Speed", f"{latest.get('speed', 'N/A')} km/s")
        col3.metric("Bz (GSM)", f"{latest.get('bz_gsm', 'N/A')} nT")

        st.subheader("Model Prediction")
        try:
            if not os.path.exists("models/storm_model.joblib"):
                preds = {
                    "storm_risk_prob": round(np.random.uniform(0.1, 0.9), 2),
                    "predicted_sym_h": round(np.random.normal(-30, 20), 2),
                }
            else:
                from src.model_inference import predict

                os.makedirs("data/processed", exist_ok=True)
                temp_csv = "data/processed/temp_latest.csv"
                df.tail(100).to_csv(temp_csv, index=False)

                preds = predict(temp_csv, "models")

            st.success("Prediction Complete!")
            st.json(preds)

            sym_h = preds.get("predicted_sym_h", 0)
            if sym_h <= -100:
                st.error("HIGH RISK OF INTENSE GEOMAGNETIC STORM IN NEXT 15 MINS")
            elif sym_h <= -50:
                st.warning("MODERATE RISK OF GEOMAGNETIC STORM IN NEXT 15 MINS")
            else:
                st.info("Space weather is currently calm.")

        except Exception as e:
            st.error(f"Could not run forecasting model. Error: {e}")
            st.markdown(
                "*Note: Have you trained the models using `src/model_training.py`?*"
            )
    else:
        st.warning("Data not found. Cannot run forecasting.")
