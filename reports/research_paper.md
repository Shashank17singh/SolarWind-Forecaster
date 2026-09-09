# Space Weather Monitoring & Solar Storm Risk Predictor
**A Multi-Model, Physics-Informed, Real-Time Forecasting System for Geomagnetic Storms, Solar Wind, and Satellite Impact**

**Author:** Project Team (Hackathon Submission)  
**Date:** 2026-02-08  
**Version:** 1.0

---

## Abstract
This paper documents a full-stack space-weather forecasting system that combines real-time solar wind ingestion, machine-learning models, and physics-informed outlooks to forecast geomagnetic activity and satellite-impact risk. The system predicts Dst/SYM-H dynamics, storm risk probability, solar wind state, and satellite anomaly likelihoods. It includes an operational backend, a live dashboard, health monitoring, dataset pipelines, model training code, and evaluation tooling. We provide full algorithmic details, equations, activation functions, feature engineering, and the physics models currently implemented. We also highlight the project’s unique contributions, newly unlocked features, and future directions for scientific and operational use.

**Keywords:** space weather, geomagnetic storms, Dst, SYM-H, solar wind, LSTM, attention, LightGBM, satellite drag, satellite anomaly, WSA-Enlil, operational forecasting

---

## Table of Contents
1. Introduction
2. System Overview
3. Data Sources
4. Data Quality and Preprocessing
5. Feature Engineering
6. Machine Learning Models
   - 6.1 Dst LSTM + Attention
   - 6.2 Solar Wind Multi-Target LSTM
   - 6.3 Storm Risk Classifier
   - 6.4 SYM-H Regressor
   - 6.5 Flare Probability Classifier
   - 6.6 Satellite Impact Classifier
7. Training Protocols
8. Evaluation Metrics
9. Physics-Based Components
   - 9.1 Derived Solar Wind Physics
   - 9.2 30-Day Geomagnetic Outlook (Recurrence + Climatology)
   - 9.3 WSA-Enlil Physics Feed (Operational Integration)
10. Frontend Visualization and User Experience
11. System Health Monitoring
12. What Is Special About This Model
13. Newly Unlocked Features
14. Practical Deployment Considerations
15. Limitations
16. Future Scope
17. Mathematical Appendix
18. Machine Learning Concepts Used
19. Reproducibility Checklist
20. Conclusion

---

## 1. Introduction
Space weather directly impacts satellite operations, navigation, HF communication, and power systems. Rapid geomagnetic disturbances often follow solar wind shocks, coronal mass ejections, or magnetic reconnection in the heliosphere. This project aims to provide a comprehensive, real-time forecasting system that integrates:

- **Short-horizon ML forecasting** (minutes to hours) using LSTM+Attention architectures.
- **Operational risk indicators** for satellite anomalies using historical event data.
- **Physics-informed recurrence/climatology forecasts** for 30-day outlooks.
- **A real-time dashboard** with overlays of observed and predicted values.

The design is modular: ingestion, feature engineering, modeling, and visualization can operate independently. This architecture supports rapid prototyping and iterative improvement.

---

## 2. System Overview
The system is composed of the following components:

- **Ingestion Pipeline**: Pulls real-time solar wind magnetometer/plasma streams and stores them in `omni_live.csv`.
- **Dataset Pipeline**: Builds training datasets with rolling statistics and labels for storms and flares.
- **Model Training**: Includes LSTM+Attention for Dst and solar wind, and tree-based models for classification tasks.
- **Prediction API**: A lightweight HTTP server serves live predictions and cached outputs.
- **Dashboard UI**: Interactive plots and health dashboards for real-time monitoring.

Key operational tasks:
- Update live data every minute.
- Refresh model outputs on short intervals.
- Continuously display model vs observed overlays.

---

## 3. Data Sources
The system integrates the following data sources:

1. **NOAA SWPC real-time solar wind**
   - Magnetometer: `mag-1-day.json`
   - Plasma: `plasma-1-day.json`

2. **OMNI-derived historical solar wind data**
   - Offline training dataset
   - Long timespan for robust feature statistics

3. **Kyoto Dst hourly index**
   - Used for Dst/SYM-H labels and evaluation

4. **GOES X-ray flux (1-day)**
   - Real-time flare classification proxy

5. **Satellite anomaly datasets**
   - NCEI anomaly tables
   - GOES-16/17 EXIS event lists
   - TDRS-1 SEU anomaly records

6. **Physics feed (WSA-Enlil)**
   - NOAA operational solar wind forecast for 1–4 day outlook

Each source is validated for time consistency and merged into a unified time grid for model features.

---

## 4. Data Quality and Preprocessing
### 4.1 Timestamp Handling
- All timestamps are normalized to UTC without timezone offsets.
- Real-time streams are merged by minute and resampled to hourly means.

### 4.2 Missing Value Handling
There are two strategies:
- **Interpolation** (time-based, with 24h limits), for stable, slowly varying inputs.
- **Drop or forward-fill** for real-time safety to avoid numeric artifacts.

### 4.3 Outlier Filtering
We remove extreme outliers:
- `flow_speed < 0` or `flow_speed > 5000` set to NaN
- `|bz_gsm| > 200` set to NaN
- `|sym_h| > 2000` set to NaN

### 4.4 Quality Scores
A quality score is computed per ingestion window to show if the data is usable. The dashboard reflects this via the **health status badge**.

---

## 5. Feature Engineering
The base feature set comes from `FEATURE_COLS` in the dataset pipeline:

- **Magnetic Field**: `bx_gse, by_gse, bz_gse, by_gsm, bz_gsm`
- **Plasma**: `flow_speed, proton_density, temperature`
- **Velocity Components**: `vx_gse, vy_gse, vz_gse`
- **Derived**: `flow_pressure, electric_field, plasma_beta, alfven_mach`
- **Indices**: `ae, al, au, sym_h, asy_h`

### 5.1 Rolling Statistics
For each feature, rolling windows (15 and 60 timesteps) are computed:

- Mean
- Standard deviation
- Minimum
- Maximum
- Delta (current minus past)

This captures short and medium-term variability without explicitly adding derivative features.

### 5.2 Labels
The dataset is labeled for multiple tasks:

- **Storm risk**: `storm_risk = 1` if `symh_future <= -50 nT`.
- **SYM-H regression**: direct target of `symh_future`.
- **Flare classification**: label if M/X-class event occurs within horizon.

---

## 6. Machine Learning Models

### 6.1 Dst LSTM + Attention
**Architecture** (sequence length 48):

- Bidirectional LSTM (150 units) + Attention + Dropout + LayerNorm
- Bidirectional LSTM (170 units) + Attention + Dropout + LayerNorm
- Global Average Pooling
- Dense output (1)

The attention layers allow the model to weigh different time steps, improving storm onset detection.

**LSTM equations** (per timestep):

- Forget gate:  
  `f_t = σ(W_f [h_{t-1}, x_t] + b_f)`
- Input gate:  
  `i_t = σ(W_i [h_{t-1}, x_t] + b_i)`
- Candidate:  
  `g_t = tanh(W_g [h_{t-1}, x_t] + b_g)`
- Cell state:  
  `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`
- Output gate:  
  `o_t = σ(W_o [h_{t-1}, x_t] + b_o)`
- Hidden state:  
  `h_t = o_t ⊙ tanh(c_t)`

**Attention (dot-product)**:

- `score_t = Q · K_t`
- `α_t = softmax(score_t)`
- `context = Σ α_t V_t`


### 6.2 Solar Wind Multi-Target LSTM
**Purpose**: Predict multiple solar wind targets 24h or 6h ahead.

- Bidirectional LSTM (128) + Attention + Dropout + LayerNorm
- Bidirectional LSTM (128) + Attention + Dropout + LayerNorm
- GlobalAveragePooling
- Dense (128, ReLU)
- Dense output with `n_targets`

Targets include:
- `b_mag, bx_gse, by_gse, bz_gse`
- `flow_speed, proton_density, temperature`
- `flow_pressure, electric_field, plasma_beta`
- `alfven_mach, magnetosonic_mach`


### 6.3 Storm Risk Classifier
We train a time-split classifier for `storm_risk`:

- **Model**: Histogram Gradient Boosting Classifier
- **Loss**: Log-loss (binary cross-entropy)


### 6.4 SYM-H Regressor
We train a time-split regression model for `symh_future`:

- **Model**: Histogram Gradient Boosting Regressor
- **Loss**: Mean Absolute Error


### 6.5 Flare Probability Classifier
Trained when flare labels are present:

- **Model**: Histogram Gradient Boosting Classifier
- **Fallback**: Real-time GOES X-ray flux converted to probability bands


### 6.6 Satellite Impact Classifier
Uses historical anomaly events (NCEI + GOES EXIS + TDRS SEU):

- **Model**: LightGBM binary classifier
- **Features**: Solar wind + geomagnetic context
- **Class imbalance**: handled by `scale_pos_weight`
- **Metrics**: ROC-AUC and PR-AUC where possible

---

## 7. Training Protocols

### 7.1 Learning Rate Schedule
For LSTM models:

```
LR = 1e-3 for epochs 0–4
LR = 1e-4 for epochs 5–9
(repeat every 5 epochs)
```

This alternation helps escape local minima without destabilizing training.

### 7.2 Loss Functions
- **Regression**: MSE / RMSE
- **Classification**: Binary log-loss

### 7.3 GPU Settings
GPU memory growth is enabled to avoid preallocation:

```
tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 8. Evaluation Metrics

### 8.1 Regression
- **MAE**: `MAE = (1/N) Σ |y - ŷ|`
- **RMSE**: `RMSE = sqrt((1/N) Σ (y - ŷ)^2)`

### 8.2 Classification
- **Log loss**: `- (y log p + (1-y) log (1-p))`
- **ROC-AUC**: ranking quality of probabilities
- **PR-AUC**: robust under class imbalance

Metrics are computed on time-based splits (train/val/test).

---

## 9. Physics-Based Components

### 9.1 Derived Solar Wind Physics
These formulas are computed during ingestion (`update_live_omni.py`):

**Dynamic pressure (nPa)**  
`P_dyn = 1.6726e-6 * n * V^2`

**Convective electric field (mV/m)**  
`E = -V * Bz * 1e-3`

**Plasma beta**  
`β = (2 μ0 P_th) / B^2`

**Alfven Mach number**  
`M_A = V / V_A`, where `V_A = B / sqrt(μ0 ρ)`

These fields serve as physics-based derived features for ML models.

### 9.2 30-Day Geomagnetic Outlook (Recurrence + Climatology)
We produce a 30-day Dst-min forecast using:

- **Recurrence** at 27 days: assume solar rotation repeats structures.
- **Climatology**: median and percentile bands by day-of-year.

If recurrence data is missing, climatology is used. The forecast output includes:

- `dst_min_pred`
- `climo_p25, climo_p75`
- `storm_prob` from historical frequency

### 9.3 WSA-Enlil Physics Feed
We integrate NOAA’s WSA-Enlil operational forecast for 1–4 days for:

- Solar wind speed
- Density
- Bz (if provided)

The dashboard renders these in separate panels with labeled physics provenance.

---

## 10. Frontend Visualization and User Experience
The UI is designed for operational clarity:

- Live panels for Dst forecast vs observed
- Solar wind ML forecasts and physics outlook side by side
- Satellite impact probability with anomaly labels
- 30-day geomagnetic outlook with climatology band
- Health dashboard showing data/model/API readiness

Key visual choices:
- Dual-line overlays for observation vs prediction
- Legend + hover tooltips with exact values
- Range selectors to change time horizon

---

## 11. System Health Monitoring
A dedicated `/api/health_full` endpoint reports:

- Model file status
- Data file status
- Last update timestamps
- Error rates and request latency

The dashboard displays a health card for operational readiness.

---

## 12. What Is Special About This Model
1. **Multi-model integration**: LSTM attention + tree models + physics outlooks in one system.
2. **Real-time operation**: ingestion pipeline and API built for live forecasting.
3. **Satellite impact risk**: anomaly classifier integrated with space-weather context.
4. **User-centric UI**: overlay forecasts and health monitoring for fast interpretation.

---

## 13. Newly Unlocked Features
- **Satellite anomaly classifier** using multi-source event labels.
- **30-day outlook** using recurrence + climatology.
- **Live update system** with data quality tracking.
- **Interactive overlays** for observed vs predicted series.

---

## 14. Practical Deployment Considerations
- Real-time inference requires up-to-date `omni_live.csv`.
- API caching prevents overload.
- GPU memory growth avoids allocation failures.
- Downstream users need clear uncertainty reporting.

---

## 15. Limitations
- Predictions are constrained by data latency.
- Extreme events remain rare and hard to learn.
- Satellite impact labels are noisy and imbalanced.
- Long-horizon forecasting (weeks to months) is fundamentally uncertain.

---

## 16. Future Scope
1. Add probabilistic forecasts with uncertainty bands.
2. Implement ensemble LSTM + transformer hybrids.
3. Expand solar flare labels to include C-class impacts.
4. Add geomagnetic indices beyond SYM-H/Dst (Kp, AE).
5. Deploy on cloud with automatic retraining.

---

## 17. Mathematical Appendix

### 17.1 Activation Functions
- **Sigmoid:** `σ(x) = 1 / (1 + e^{-x})`
- **Tanh:** `tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})`
- **ReLU:** `ReLU(x) = max(0, x)`
- **Linear:** `f(x) = x`

### 17.2 Loss Functions
- **MSE**: `MSE = (1/N) Σ (y - ŷ)^2`
- **RMSE**: `RMSE = sqrt(MSE)`
- **MAE**: `MAE = (1/N) Σ |y - ŷ|`
- **Binary Log Loss**:  
  `L = -[y log p + (1-y) log (1-p)]`

### 17.3 Attention Summary
`Attention(Q, K, V) = softmax(QK^T) V`

---

## 18. Machine Learning Concepts Used
- Supervised learning
- Regression and classification
- Time-series forecasting
- Time-based train/val/test splits
- Feature scaling (implicit by model normalization)
- Class imbalance handling (pos weighting)
- Model calibration (isotonic regression in tree models)
- Overfitting mitigation via dropout and regularization
- Evaluation with AUC, PR-AUC, MAE, RMSE

---

## 19. Reproducibility Checklist
- Data ingestion script: `src/update_live_omni.py`
- Dataset builder: `src/build_dataset.py`
- LSTM models: `src/train_dst_lstm_attention.py`, `src/train_solar_wind_lstm.py`
- Tree models: `src/train.py`, `src/train_lgbm.py`
- Satellite impact: `src/build_satellite_impact_dataset.py`, `src/train_sat_impact_lgbm.py`
- Dashboard: `web/index.html`, `web/app.js`
- Server: `src/server.py`

---

## 20. Conclusion
This project delivers a full operational pipeline for space weather forecasting. It combines physics-informed features with modern deep learning and gradient-boosting models and exposes results through a live dashboard and API. The system is modular, extensible, and suitable for real-time operational monitoring. While uncertainties remain unavoidable in space weather, the system provides a practical framework to improve forecasting reliability and situational awareness.

---

**End of Paper**

---

# Appendix A: Detailed Algorithm Descriptions

## A.1 LSTM + Attention for Dst Forecasting
The Dst/SYM-H model ingests the last 48 hours of hourly solar wind and geomagnetic features. The key idea is to learn temporal dependencies and allow the network to focus on informative intervals, such as sudden southward IMF Bz turns.

### A.1.1 Sequence Construction
Let `X_t` denote the feature vector at hour `t` with dimension `F`. A sequence of length `L` is:

`S_t = [X_{t-L+1}, X_{t-L+2}, ..., X_t]`

The label for the horizon `H` is:

`y_t = Dst_{t+H}`

Training pairs: `(S_t, y_t)`.

### A.1.2 Attention Interpretation
Attention weights `α_t` can be used to interpret which past hours were most relevant for the predicted Dst. In practice, attention often highlights intervals with strong Bz southward changes or density spikes.


## A.2 Multi-Target Solar Wind Forecasting
The solar wind model predicts multiple outputs simultaneously. This is a multi-task regression setup where a single sequence encoder outputs a vector of targets. The loss is the mean of per-target MSE.

`L = (1/K) Σ_k (1/N) Σ_i (y_{i,k} - ŷ_{i,k})^2`

Where `K` is number of targets and `N` is number of samples.


## A.3 Histogram Gradient Boosting (HGB)
HGB is a tree ensemble optimized for speed on large datasets:

- Continuous features are binned.
- Trees are built on binned gradients.
- It supports high-dimensional feature sets efficiently.

This is used for storm classification and SYM-H regression when training from CSV datasets.


## A.4 LightGBM (LGBM) for Satellite Impact
LightGBM uses gradient-based one-side sampling and exclusive feature bundling to train efficient tree ensembles. It is robust on imbalanced data when combined with `scale_pos_weight`.

Pseudo-code:

```
Initialize prediction with class prior.
For each boosting round:
  Compute gradients and hessians.
  Build a tree to reduce residuals.
  Update predictions.
```

---

# Appendix B: Physics Models and Formulas

## B.1 Satellite Drag (Core Physics)
Satellite drag is driven by atmospheric density and relative velocity:

`F_d = 0.5 * ρ * v^2 * C_d * A`

Where:
- `ρ` is atmospheric density
- `v` is relative velocity
- `C_d` is drag coefficient
- `A` is cross-sectional area

The drag acceleration is:

`a_d = F_d / m`

This is foundational for satellite orbit decay and anomaly risk modeling.


## B.2 Geomagnetic Indices
- **Dst**: globally averaged ring current index. Negative Dst indicates stronger geomagnetic storms.
- **SYM-H**: high-resolution (1 min) analog of Dst.
- **AE, AL, AU**: auroral electrojet indices.


## B.3 Dynamic Pressure and IMF Coupling
Storm strength is linked to the solar wind dynamic pressure and IMF orientation.

- `P_dyn` increases with density and speed.
- Southward IMF (`Bz < 0`) enhances reconnection.

These variables are explicitly encoded in our model features.


## B.4 Recurrence Forecasting
The 27-day recurrence assumption uses the solar rotation period. If a coronal hole persists, similar geomagnetic effects may repeat after ~27 days. This is used in our 30-day outlook when data exists.

---

# Appendix C: Dataset Schema

## C.1 Solar Wind Dataset (`omni.csv` or `omni_live.csv`)
Columns include:

- `time` (timestamp)
- Magnetic field vectors: `bx_gse, by_gse, bz_gse, by_gsm, bz_gsm`
- Velocity: `flow_speed, vx_gse, vy_gse, vz_gse`
- Plasma: `proton_density, temperature`
- Derived: `flow_pressure, electric_field, plasma_beta, alfven_mach`
- Indices: `ae, al, au, sym_h, asy_h`

## C.2 ML Dataset
The ML dataset adds rolling window statistics:

- `*_w15_mean, *_w15_std, *_w15_min, *_w15_max, *_w15_delta`
- `*_w60_*` for medium-term features

And labels:

- `storm_risk` (binary)
- `symh_future` (regression)
- `flare_mx_next_15m` (binary, optional)

## C.3 Satellite Impact Dataset
The satellite impact dataset aligns anomaly events with solar wind conditions and labels:

- `sat_impact_next_6h` (binary)

---

# Appendix D: Training and Evaluation Details

## D.1 Train/Val/Test Splits
We use **time-based splits** to avoid leakage:

- Train: historical window
- Validation: subsequent window
- Test: latest window

This is critical for time-series forecasts.


## D.2 Class Imbalance
Storm and anomaly events are rare. To mitigate imbalance:

- LightGBM uses `scale_pos_weight`
- Metrics include PR-AUC (more reliable for rare positives)


## D.3 Calibration
When needed, probabilities can be calibrated using isotonic regression:

`p_cal = f_iso(p_raw)`

This improves interpretability of probabilistic outputs.

---

# Appendix E: Dashboard and UX Logic

## E.1 Overlays
Prediction lines are plotted over observed lines for:

- Dst
- SYM-H

This provides immediate visual assessment of forecast quality.

## E.2 Live Status
The system tracks two ages:

- **Data age**: time since last real solar wind observation
- **API age**: time since last API refresh

This avoids confusion when upstream data is delayed.

---

# Appendix F: Future Research Extensions

- **Transformers for long-range temporal dependencies**
- **Data assimilation** with physics-based models
- **Probabilistic ensembles** for uncertainty quantification
- **Joint solar wind + Dst forecasting** with multi-task learning

---

# Appendix G: Activation Functions Used

- **tanh**: LSTM internal gating
- **sigmoid**: LSTM gates
- **ReLU**: Dense intermediate layers
- **linear**: final regression outputs

---

# Appendix H: Model Differentiators

1. **End-to-end integration** from ingestion to UI
2. **Hybrid modeling** combining ML + physics-based outlooks
3. **Operational readiness** with health monitoring and caching
4. **Satellite-specific risk estimation**

---

# Appendix I: Mathematical Symbols (Reference)

- `x_t`: input feature vector at time `t`
- `y_t`: target value
- `L`: sequence length
- `H`: forecast horizon
- `σ`: sigmoid
- `⊙`: element-wise multiplication

---

**This extended appendix is intended to complete the requested detailed, 20-page equivalent documentation.**

---

# Appendix J: CME Detection and Plasma Cloud Forecasting Model

## J.1 Motivation
Coronal Mass Ejections (CMEs) drive interplanetary plasma clouds (ICMEs) that can cause geomagnetic storms when they reach Earth. Predicting whether a CME will produce an Earth-impacting plasma cloud, and estimating its transit time, is critical for long-horizon operational forecasting.

## J.2 Data Sources
This module uses:

- **NASA DONKI CME catalog** (event properties and analysis inputs)
- **HELCATS ICMECAT** (observed ICME arrival times at Earth/L1)

These sources are merged to create labels for **Earth impact** and **transit time**.

## J.3 Features
For each CME event:

- `speed` (km/s)
- `half_angle`, `width`
- `latitude`, `longitude` (source location)
- `is_halo` (width >= 360)
- `active_region` number
- `catalog` and `cme_type` (encoded)

## J.4 Labels
- **Earth impact**: 1 if ICME arrival observed within 10–120 hours after CME start.
- **Transit time**: hours between CME start and ICME arrival (for positive events).

## J.5 Models
Two LightGBM models are trained:

1. **CME impact classifier**
   - Output: probability CME will reach Earth.
2. **Transit-time regressor**
   - Output: predicted arrival time in hours (only for positives).

## J.6 Evaluation
- **Classification**: ROC-AUC, PR-AUC
- **Regression**: MAE (hours)

## J.7 Operational Usage
The output provides:
- Probability of Earth impact
- Predicted arrival time window

This supports long-horizon planning for satellite operators and communications.

---
