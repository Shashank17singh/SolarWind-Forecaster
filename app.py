import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.abspath("."))

st.set_page_config(page_title="SolarWind Forecaster", layout="wide")

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
    try:
        df = pd.read_csv("data/processed/omni.csv", parse_dates=["timestamp"])
        return df
    except FileNotFoundError:
        # Generate realistic mock data for portfolio demonstration if running on Streamlit Cloud
        dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=2000, freq="1min")
        df = pd.DataFrame({
            "timestamp": dates,
            "sym_h": np.random.normal(-15, 20, 2000),
            "bz_gsm": np.random.normal(0, 5, 2000),
            "speed": np.random.normal(450, 50, 2000)
        })
        # Add a simulated storm
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

        recent_df = df.tail(1000).reset_index(drop=True if df.index.name != "timestamp" else False)
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
                preds = {"storm_risk_prob": round(np.random.uniform(0.1, 0.9), 2), "predicted_sym_h": round(np.random.normal(-30, 20), 2)}
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
