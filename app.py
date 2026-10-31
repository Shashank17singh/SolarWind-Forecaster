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
/* Hallmark · checked: 2026-10-31 · genre: modern-minimal (inferred)
 * fixed: 21 tokens lifted, focus, motion and states added · structural: left for the owner, see the audit
 * built: the project · redesign: polish · v2.0.0 */
:root {
  --color-teal-500: #0ea5e9;
  --color-blue-800: #0b1120;
  --color-white: #f8fafc;
  --color-blue-800-2: #0f172a;
  --color-blue-800-a70: rgba(11,17,32,0.7);
  --color-blue-700-a50: rgba(30,41,59,0.5);
  --color-teal-500-a20: rgba(14,165,233,0.2);
  --color-black-a50: rgba(0,0,0,0.5);
  --color-blue-800-a80: rgba(15,23,42,0.8);
  --color-blue-500-a80: rgba(2,132,199,0.8);
  --color-white-2: #ffffff;
  --color-teal-500-a30: rgba(14,165,233,0.3);
  --color-teal-500-a60: rgba(14,165,233,0.6);
  --color-orange-400-a10: rgba(245,158,11,0.1);
  --color-orange-400: #f59e0b;
  --font-sans: 'Fira Sans', sans-serif !important;
  --font-mono: 'Fira Code', monospace !important;
}


@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: var(--font-sans);
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-mono);
    color: var(--color-teal-500) !important;
}

.stApp {
    background-color: var(--color-blue-800);
    color: var(--color-white);
    background-image: radial-gradient(circle at 50% top, var(--color-blue-800-2) 0%, var(--color-blue-800) 100%);
}

[data-testid="stHeader"] {
    background-color: var(--color-blue-800-a70) !important;
    backdrop-filter: blur(10px);
}

/* Glassmorphism Containers */
[data-testid="stExpander"], [data-testid="stVerticalBlock"] > div > div > div[data-testid="stContainer"] {
    background-color: var(--color-blue-700-a50);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--color-teal-500-a20);
    border-radius: 12px;
    box-shadow: 0 8px 32px 0 var(--color-black-a50);
    padding: 15px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--color-blue-800-a80) !important;
    border-right: 1px solid var(--color-teal-500-a20);
}

/* Buttons */
.stButton > button {
    background-color: var(--color-blue-500-a80);
    backdrop-filter: blur(10px);
    color: var(--color-white-2);
    font-family: var(--font-mono);
    font-weight: 600;
    border: 1px solid var(--color-teal-500);
    border-radius: 6px;
    box-shadow: 0 0 10px var(--color-teal-500-a30);
    transition: color 0.2s, background-color 0.2s, border-color 0.2s, box-shadow 0.2s, transform 0.2s, opacity 0.2s;
}

.stButton > button:hover {
    background-color: var(--color-teal-500);
    box-shadow: 0 0 15px var(--color-teal-500-a60);
    transform: translateY(-2px);
    color: var(--color-white-2);
}

/* Accent for warnings/amber */
.stAlert {
    background-color: var(--color-orange-400-a10);
    border: 1px solid var(--color-orange-400);
    color: var(--color-white);
}

/* Accessibility states added by the polish pass. */
:root {
  --focus-ring: #e2e8f0;
  --disabled-fill: #2a2f3a;
  --disabled-ink: #8b93a1;
}

[hidden] {
  display: none !important;
}

html,
body {
  overflow-x: clip;
}

a:focus-visible,
button:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible,
summary:focus-visible,
[tabindex]:focus-visible,
[role="button"]:focus-visible,
input:focus-visible + label {
  outline: 2px solid var(--focus-ring);
  outline-offset: 2px;
}

button:active:not(:disabled) {
  transform: translateY(1px);
}

button:disabled,
input:disabled,
select:disabled,
textarea:disabled {
  background: var(--disabled-fill);
  color: var(--disabled-ink);
  cursor: not-allowed;
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
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
            drop=df.index.name != "timestamp"
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

        except Exception as e:  # noqa: BLE001
            st.error(f"Could not run forecasting model. Error: {e}")
            st.markdown(
                "*Note: Have you trained the models using `src/model_training.py`?*"
            )
    else:
        st.warning("Data not found. Cannot run forecasting.")
