# Project Report

## Space Weather Monitoring & Solar Storm Risk Predictor

---

## 1. Executive Summary

Space Weather Sentinel is a real-time machine learning system that monitors solar activity and forecasts geomagnetic storm risk, satellite drag, solar flare probability, and radiation hazards. It ingests live data from NASA/NOAA satellite feeds, runs multiple trained ML models, and presents all predictions on a live web dashboard accessible through any browser.

The system addresses a critical operational need — space weather events cause billions of dollars in damage to satellites, power grids, GPS infrastructure, and communication systems every year. By providing 15-minute to 30-day forecasts across multiple hazard types, this project gives satellite operators, power companies, and space agencies actionable early warning.

---

## 2. Problem Statement

The Sun constantly emits charged particles, magnetic fields, and radiation toward Earth. When this activity intensifies — during solar flares, coronal mass ejections (CMEs), or geomagnetic storms — it can:

- Increase atmospheric drag on low-Earth orbit satellites, causing orbit decay
- Damage satellite electronics through radiation
- Disrupt GPS positioning by up to 50 meters
- Cause blackouts in power grids through geomagnetically induced currents (GIC)
- Endanger astronauts with elevated radiation doses
- Disrupt HF radio and satellite communications

Existing tools from NOAA/NASA provide forecasts but require expert interpretation, have limited operational dashboards, and are not easily deployable for private operators. This project bridges that gap.

---

## 3. Objectives

- Build an end-to-end ML pipeline trained on 30 years of real NASA solar wind data
- Forecast geomagnetic storm risk and intensity at a 15-minute horizon
- Provide satellite-specific drag and anomaly risk estimates
- Deliver all predictions through a live, real-time web dashboard
- Keep the system lightweight enough to run on a single server without GPU

---

## 4. Data Sources

| Dataset | Source | Coverage |
|---|---|---|
| OMNI High-Resolution Solar Wind (1-min) | NASA GSFC SPDF | 1995 – 2025 |
| X-Ray Flare Reports (XRS) | NOAA NGDC | 1975 – 2016 |
| Kyoto Dst Index (hourly) | Kyoto WDC | 1957 – present |
| JB2008 Thermospheric Indices | Space Environment Technologies | 2000 – present |
| Spacecraft Anomaly Records (NCEI) | NOAA NCEI | 1971 – 2020 |
| Live Solar Wind (real-time) | NOAA SWPC / ACE / DSCOVR | Real-time |
| Live Kp Index | NOAA SWPC | Real-time |
| Live X-ray Flux (GOES) | NOAA SWPC | Real-time |
| Live Proton Flux (SEP) | NOAA SWPC | Real-time |
| CME Events (DONKI) | NASA CCMC | Real-time |
| Aurora Oval Images | NOAA SWPC OVATION | Real-time |
| WSA-Enlil Solar Wind Forecast | NOAA NCEP | 1–4 day forecast |

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DATA LAYER                        │
│  NASA OMNI (30yr)  +  NOAA Live Feeds (real-time)   │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              FEATURE ENGINEERING                    │
│  parse_omni.py → build_dataset.py                   │
│  Rolling windows: 15-min, 60-min averages           │
│  Derived: Bz_south, V×Bz, flow pressure, etc.       │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│               ML MODEL LAYER                        │
│  storm_model     → LightGBM classifier              │
│  symh_model      → LightGBM regressor               │
│  flare_model     → LightGBM classifier              │
│  drag_model      → LightGBM regressor               │
│  sat_impact_model→ LightGBM classifier              │
│  dst_lstm        → LSTM + Attention (Keras)         │
│  cme_impact_model→ LightGBM (event-based)           │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              API SERVER (Python)                    │
│  src/server.py  — ThreadingHTTPServer               │
│  /api/metrics   /api/kp     /api/series             │
│  /api/alerts    /api/aurora /api/dst                │
│  /api/cme       /api/enlil  /api/health             │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│             WEB DASHBOARD (Frontend)                │
│  Static HTML/CSS/JS  +  Next.js (React) app         │
│  Canvas charts, live cards, aurora imagery          │
│  GPS / SatCom impact panels                         │
└─────────────────────────────────────────────────────┘
```

---

## 6. ML Models

### 6.1 Storm Risk Model
- **Type:** LightGBM binary classifier
- **Target:** P(SYM/H ≤ −50 nT within next 15 minutes)
- **Features:** 20 base solar wind features + 15-min and 60-min rolling averages
- **Training data:** OMNI 1-min, 1995–2023
- **Output:** Calibrated probability 0–1

### 6.2 SYM/H Regression Model
- **Type:** LightGBM regressor
- **Target:** SYM/H value at +15 minutes (nT)
- **Use:** Quantifies storm intensity, not just yes/no
- **Output:** Predicted SYM/H in nanoTesla

### 6.3 Flare Risk Model
- **Type:** LightGBM binary classifier
- **Target:** P(M or X class flare within 15 minutes)
- **Fallback:** X-ray flux proxy from GOES satellite (A/B/C/M/X class mapping)
- **Output:** Probability + flare class label

### 6.4 Dst LSTM + Attention Model
- **Type:** LSTM with attention mechanism (Keras/TensorFlow)
- **Target:** Dst index at +1 hour
- **Input:** 48-step sequence (48 hours of hourly data)
- **Features:** 49 features including IMF components, plasma parameters, geomagnetic indices
- **Trained:** Up to 2023, validated 2023–2025
- **Output:** Forecast Dst in nT + 72-hour forecast series

### 6.5 Satellite Drag Model
- **Type:** LightGBM regressor
- **Target:** JB2008 DTC (thermospheric temperature correction) at +3 hours
- **Use:** Higher DTC = denser atmosphere = more drag on satellites
- **Output:** DTC value → translated to drag acceleration (m/s²) per satellite

### 6.6 Satellite Anomaly / Impact Model
- **Type:** LightGBM binary classifier
- **Target:** P(satellite anomaly within next 6 hours)
- **Training data:** NCEI spacecraft anomaly records + OMNI solar wind
- **Output:** Probability + Low/Elevated/High risk level

### 6.7 CME Impact Model
- **Type:** LightGBM classifier + regressor (two models)
- **Target:** P(CME hits Earth) + transit time (hours)
- **Input:** CME speed, width, latitude, longitude, halo flag
- **Data source:** NASA DONKI real-time CME catalog
- **Output:** Impact probability + estimated arrival time (ETA)

---

## 7. Feature Engineering

Key engineered features used across models:

| Feature | Description |
|---|---|
| `bz_gsm` | North-south magnetic field — primary storm driver |
| `bz_south` | max(0, −Bz) — only southward component |
| `vbz_south` | V × Bz_south — reconnection proxy |
| `flow_speed` | Solar wind velocity (km/s) |
| `flow_pressure` | Dynamic pressure (nPa) |
| `proton_density` | Particle density (cm⁻³) |
| `electric_field` | Interplanetary electric field (mV/m) |
| `plasma_beta` | Ratio of plasma to magnetic pressure |
| `alfven_mach` | Alfvén Mach number |
| `ae`, `al`, `au` | Auroral electrojet indices |
| `sym_h` | Current geomagnetic disturbance level |
| Rolling 15-min avg | Short-term trend of all above |
| Rolling 60-min avg | Medium-term trend of all above |

---

## 8. Tech Stack

### Backend / ML
| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| LightGBM | Primary ML model (storm, flare, drag, impact, CME) |
| Keras / TensorFlow | LSTM + Attention model for Dst forecasting |
| Scikit-learn | Preprocessing, calibration, pipelines |
| Pandas | Data loading, cleaning, feature engineering |
| NumPy | Numerical operations, array handling |
| Joblib | Model serialization (.joblib files) |
| PyArrow | Parquet shards for large dataset training |
| SciPy | Statistical utilities |
| Matplotlib | Evaluation plots and figures |

### API Server
| Tool | Purpose |
|---|---|
| Python `http.server` | Lightweight ThreadingHTTPServer |
| urllib | Live data fetching from NOAA/NASA APIs |
| JSON | API response format |
| Rotating file logs | Request logging with size limits |

### Frontend
| Tool | Purpose |
|---|---|
| HTML5 / CSS3 / JavaScript | Static web dashboard (`web/`) |
| Next.js 14 + React 18 | App router version (`app/`) |
| Canvas API | Custom chart rendering (no chart library needed) |
| Space Grotesk + JetBrains Mono | Google Fonts for UI typography |
| NOAA SWPC image feeds | Live aurora, solar wind, geospace imagery |

### Deployment
| Tool | Purpose |
|---|---|
| Vercel | Frontend deployment (Next.js) |
| Nginx | Reverse proxy config (`configs/nginx.conf`) |
| Cron | Scheduled live data updates (every 1 minute) |
| Docker / Kubernetes | Container deployment (production ops) |

---

## 9. Data Pipeline

```
Step 1: Download
  scripts/download_omni.sh         → data/omni/
  scripts/download_flare_reports.sh→ data/flare_reports/
  src/fetch_jb2008_indices.py      → data/indices/jb2008/
  src/fetch_kyoto_dst.py           → data/indices/kyoto/

Step 2: Parse & Process
  src/parse_omni.py                → data/processed/omni.csv
  src/build_dataset.py             → data/processed/dataset.csv
  src/build_drag_dataset.py        → data/processed/drag_dataset.csv
  src/build_satellite_impact_dataset.py → data/processed/sat_impact_dataset.csv
  src/build_cme_dataset.py         → data/processed/cme_impact_dataset.csv

Step 3: Train
  src/train_lgbm.py                → models/storm_model.joblib
                                   → models/symh_model.joblib
  src/train_dst_lstm_attention.py  → models/dst_lstm_attention.keras
  src/train_drag_lgbm.py           → models/drag_model.joblib
  src/train_sat_impact_lgbm.py     → models/sat_impact_model.joblib
  src/train_cme_impact_lgbm.py     → models/cme_impact_model.txt

Step 4: Live Update (every 1 min via cron)
  src/update_live_omni.py          → data/processed/omni_live.csv

Step 5: Serve
  src/server.py                    → http://localhost:8000
```

---

## 10. API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/metrics` | Main predictions: storm risk, SYM/H, flare, drag, SEP |
| `GET /api/kp` | Latest Kp index from NOAA |
| `GET /api/series` | Time series: SYM/H, Bz, solar wind speed |
| `GET /api/dst` | Historical Dst observed vs predicted |
| `GET /api/dst-forecast` | 72-hour Dst forecast |
| `GET /api/dst-outlook` | 30-day geomagnetic outlook |
| `GET /api/solar-wind-ml` | 7-day ML solar wind forecast |
| `GET /api/enlil` | WSA-Enlil 4-day physics forecast |
| `GET /api/cme` | Latest CME event + impact prediction |
| `GET /api/cme-climo` | CME climatology by month |
| `GET /api/cme-scenario` | Custom CME scenario predictor |
| `GET /api/alerts` | Live NOAA space weather alerts |
| `GET /api/aurora` | Aurora nowcast + forecast |
| `GET /api/satellites` | Satellite preset configurations |
| `GET /api/health` | System health, uptime, data quality |

---

## 11. Dashboard Features

The live web dashboard includes:

- **Alert Banner** — Real-time storm risk level (Normal / Elevated / Severe)
- **Storm Risk Gauge** — Animated probability meter with color coding
- **Kp Index Card** — Current planetary K-index with progress bar
- **Dst / SYM-H Card** — Current value + 15-min forecast + trend arrow
- **Flare Risk Card** — M/X probability + current X-ray class + flux
- **SEP / Radiation Card** — S-scale radiation storm level + proton flux
- **Satellite Drag Card** — Per-satellite drag acceleration with custom inputs
- **Satellite Impact Card** — Anomaly probability for selected satellite
- **Live Solar Wind Card** — Real-time Bz, speed, density
- **Time Series Charts** — Last 12 hours of SYM/H, Bz, solar wind speed
- **CME Forecast Panel** — Live event prediction + climatology + scenario tool
- **Dst Forecast Panel** — Historical comparison + 72-hour + 30-day outlook
- **Solar Wind Forecast** — ML 7-day + WSA-Enlil physics model
- **Aurora Panel** — NOAA OVATION north/south nowcast and forecast images
- **Alerts Panel** — Live NOAA bulletins (watches, warnings, alerts)
- **GPS Impact Panel** — Positioning accuracy degradation estimates
- **SatCom Panel** — UHF/SHF link quality and outage probability
- **Solar Image Gallery** — Live SDO, ACE, CCOR-1, magnetometer feeds
- **Deep Dive Links** — Detailed pages for solar wind, flares, protons, aurora, magnetometers

---

## 12. Key Results & Capabilities

| Capability | Value |
|---|---|
| Forecast horizon (storm) | 15 minutes |
| Forecast horizon (Dst) | 72 hours |
| Forecast horizon (drag) | 3 hours |
| Forecast horizon (satellite impact) | 6 hours |
| Long-range outlook | 30 days |
| Training data span | 30 years (1995–2025) |
| Inference latency | < 10 milliseconds |
| Data refresh rate | Every 1 minute (live) |
| Model cache TTL | Configurable (default ~60 sec) |
| Runs without GPU | Yes — tree models only |
| Satellite configurations | 5 presets + fully custom |

---

## 13. Comparison with NASA/IBM Surya

| Aspect | NASA/IBM Surya | This Project |
|---|---|---|
| Released | August 2025 | 2026 |
| Input data | Solar imagery (SDO photos) | Solar wind in-situ sensor data |
| Where it looks | At the Sun (source) | At Earth's doorstep (L1 point) |
| Forecast horizon | 2 hours (flares), 4 days (wind) | 15 min – 30 days (multi-model) |
| Architecture | Spatiotemporal transformer (foundation model) | LightGBM + LSTM ensemble |
| Compute needed | Large GPU cluster | Single CPU server |
| Satellite-specific outputs | No | Yes (drag, anomaly per satellite) |
| Operational dashboard | No (research model) | Yes (full live web UI) |
| Deployment | Research / HuggingFace | Self-hosted / Vercel |
| Complementary role | Early warning (days ahead) | Last-mile alarm (minutes ahead) |

Both models are complementary. Surya watches the Sun and gives days of warning. This project watches the solar wind arriving at Earth and gives the precise 15-minute operational alarm.

---

## 14. Limitations

- Storm risk model is calibrated for SYM/H ≤ −50 nT — moderate storm threshold. Extreme events (SYM/H < −200 nT) are rare in training data and may be under-predicted.
- GPS/GNSS and SatCom impact panels are heuristic mappings from storm/Kp/SEP levels, not independently trained models.
- Flare prediction accuracy depends on availability of real-time X-ray flux; model falls back to flux proxy when the trained flare model is unavailable.
- Data latency from NOAA/NASA upstream feeds (typically 1–5 minutes) limits true real-time accuracy.
- The LSTM Dst model requires 48 hours of continuous hourly history — gaps in data reduce forecast quality.

---

## 15. Future Scope

- Integrate NASA Surya output as an upstream feature (combine solar image predictions with in-situ data)
- Add GIC (Geomagnetically Induced Current) risk model for power grid operators
- Train a dedicated ionospheric TEC model for GPS accuracy forecasting
- Add push notifications / alerting via email or SMS on storm threshold breach
- Multi-satellite simultaneous tracking with orbit propagation
- Mobile-responsive PWA (Progressive Web App) version of the dashboard
- Deploy on Kubernetes with auto-scaling for high-availability production use

---

## 16. Conclusion

Space Weather Sentinel is a complete, production-ready space weather forecasting system built on 30 years of real NASA data. It combines classical ML (LightGBM), deep learning (LSTM + Attention), and real-time data ingestion into a single deployable package. The live dashboard gives satellite operators, power grid managers, and space agencies actionable risk information across multiple hazard types — all running on a single server with no GPU requirement and millisecond inference latency.

---

*Report generated: June 2026*
*Project: Space Weather Sentinel*
*Data: NASA OMNI, NOAA SWPC, Kyoto WDC, NCEI*
