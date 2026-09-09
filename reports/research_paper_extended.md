# Space Weather Forecasting System: A Comprehensive Machine Learning Approach
# Extended Research Paper - Part 2

<!-- Continuation from main paper -->

# 3. Data Sources and Acquisition

## 3.1 OMNI Solar Wind Database

The primary data source for this research is the OMNI (Operating Missions as Nodes on the Internet) high-resolution solar wind database maintained by NASA's Space Physics Data Facility (SPDF). OMNI provides:

- **Temporal Coverage**: 1995-present (30+ years)
- **Temporal Resolution**: 1-minute cadence
- **Spatial Location**: L1 Lagrange point (~1.5 million km upstream of Earth)
- **Data Sources**: Multiple spacecraft (ACE, Wind, DSCOVR, IMP-8, Geotail)
- **Parameters**: 50+ solar wind, IMF, and geomagnetic indices

### 3.1.1 Key Measurements

**Interplanetary Magnetic Field (IMF)**:
- Magnetic field magnitude (B)
- Components in GSE coordinates (Bx, By, Bz)
- Components in GSM coordinates (By_GSM, Bz_GSM)
- Standard deviations and RMS values

**Solar Wind Plasma**:
- Bulk velocity (V) and components (Vx, Vy, Vz)
- Proton density (n_p)
- Proton temperature (T_p)
- Flow pressure (P_flow)

**Derived Parameters**:
- Electric field (E = -V × B)
- Plasma beta (β)
- Alfvén Mach number (M_A)
- Magnetosonic Mach number (M_ms)

**Geomagnetic Indices**:
- SYM-H (1-minute Dst proxy)
- ASY-H, ASY-D (asymmetric disturbance)
- AE, AL, AU (auroral electrojet indices)
- PCN (polar cap index)

### 3.1.2 Data Quality Indicators

OMNI includes quality metadata:
- Spacecraft ID (IMF and plasma sources)
- Number of points in average (npts)
- Percentage of interpolated data
- Time shift to bow shock nose
- RMS time shift and phase front normal

These quality indicators enable filtering and uncertainty quantification.

## 3.2 Kyoto Dst Index

The Kyoto World Data Center provides the definitive Dst index:

- **Temporal Coverage**: 1957-present
- **Temporal Resolution**: 1-hour values
- **Spatial Coverage**: Global (4 mid-latitude stations)
- **Stations**: Hermanus, Kakioka, Honolulu, San Juan
- **Processing**: Baseline removal, quiet-day correction

We use Kyoto Dst as the ground truth target for model training and evaluation, as it represents the community standard for geomagnetic storm intensity.

## 3.3 Satellite Anomaly Data

Historical satellite anomaly data comes from multiple sources:

### 3.3.1 NOAA NCEI Spacecraft Anomaly Database
- **Coverage**: 1971-2022
- **Records**: 10,000+ anomaly events
- **Satellites**: 200+ spacecraft
- **Anomaly Types**: Single event upsets, component failures, charging events

### 3.3.2 GOES EXIS Space Weather Events
- **Coverage**: 2016-present
- **Focus**: GOES-16/17 anomalies
- **Detail**: High-resolution event characterization

### 3.3.3 TDRS Anomaly Reports
- **Coverage**: 1983-2016
- **Focus**: Communication satellite impacts
- **Detail**: Operational impact assessment

## 3.4 Solar Flare Reports

NOAA NGDC X-ray flare reports provide:
- **Coverage**: 1975-2016
- **Source**: GOES X-ray sensors (XRS)
- **Classification**: A, B, C, M, X classes
- **Timing**: Start, peak, end times
- **Location**: Active region coordinates

## 3.5 Data Download and Storage

### 3.5.1 Automated Download Scripts

```bash
# OMNI high-resolution data (1995-2025)
scripts/download_omni.sh

# GOES XRS flare reports
scripts/download_flare_reports.sh

# GOES XRS flux time series (optional, large)
scripts/download_goes_xrs.sh
```

### 3.5.2 Storage Architecture

```
data/
├── omni/                    # Raw OMNI files (30 GB)
├── indices/
│   ├── kyoto/              # Dst hourly data
│   └── jb2008/             # Thermospheric indices
├── flare_reports/          # GOES flare event lists
├── impact/
│   └── ncei/               # Satellite anomaly tables
└── processed/
    ├── omni.csv            # Parsed solar wind (5 GB)
    ├── dataset.csv         # ML-ready features (10 GB)
    └── parquet/            # Sharded training data
        ├── train/
        ├── val/
        └── test/
```

## 3.6 Data Preprocessing Pipeline

### 3.6.1 OMNI Parsing

The `parse_omni.py` script processes raw OMNI files:

1. **Format Detection**: Handles multiple OMNI format versions
2. **Column Mapping**: Maps format-specific columns to standard names
3. **Fill Value Handling**: Replaces 9999.99, 999.9, etc. with NaN
4. **Timestamp Parsing**: Converts year/day/hour/minute to datetime
5. **Coordinate Transformations**: Ensures GSE and GSM coordinates
6. **Quality Filtering**: Flags low-quality measurements

Output: `data/processed/omni.csv` with standardized columns and timestamps.

### 3.6.2 Hourly Aggregation

For Dst prediction, we resample to hourly means:

```python
df = df.set_index('time').sort_index()
df_hourly = df.resample('1h').mean()
```

This reduces noise while matching Dst temporal resolution.

### 3.6.3 Missing Data Strategies

Two approaches are supported:

**Strategy 1: Cubic Spline Interpolation**
- Method: 3rd-order spline
- Limit: 24 hours maximum gap
- Area: Inside only (no extrapolation)
- Use case: Training on historical data

**Strategy 2: Drop Missing**
- Method: Remove rows with any NaN
- Use case: Strict evaluation, no imputation bias

**Strategy 3: Mean Fill (Operational)**
- Method: Fill with training set mean
- Use case: Real-time inference when features unavailable

### 3.6.4 Data Quality Assessment

Quality metrics computed per time window:

- **Completeness**: Fraction of non-missing values
- **Gap Distribution**: Histogram of gap lengths
- **Outlier Detection**: Values beyond 5σ from rolling mean
- **Drift Detection**: KL divergence from baseline distribution

---

# 4. Feature Engineering

## 4.1 Feature Categories

The complete feature set consists of 49 features across 6 categories:

### 4.1.1 Temporal Features (4 features)
- **year**: Annual cycle, solar cycle proxy
- **doy**: Day of year (1-366), seasonal variation
- **hour**: Hour of day (0-23), diurnal variation
- **minute**: Minute of hour (0-59), sub-hourly timing

### 4.1.2 Data Quality Metadata (13 features)
- **imf_sc_id**: IMF spacecraft identifier
- **sw_sc_id**: Solar wind spacecraft identifier
- **imf_npts**: Number of IMF measurements in average
- **sw_npts**: Number of plasma measurements in average
- **pct_interp**: Percentage of interpolated data
- **timeshift_sec**: Propagation time to bow shock
- **rms_timeshift_sec**: RMS variation in time shift
- **rms_phase_front_norm**: Phase front normal RMS
- **dbot_sec**: Time difference bow shock to observation
- **rms_sd_b**: RMS standard deviation of B
- **rms_sd_bvec**: RMS standard deviation of B vector
- **sc_x_gse, sc_y_gse, sc_z_gse**: Spacecraft position
- **bsn_x, bsn_y, bsn_z**: Bow shock nose position

### 4.1.3 Magnetic Field Features (9 features)
- **b_mag**: Total field magnitude
- **bx_gse, by_gse, bz_gse**: GSE components
- **by_gsm, bz_gsm**: GSM components (critical for coupling)
- **bz_south**: min(Bz_GSM, 0) - southward component only
- **bz_abs**: |Bz_GSM| - magnitude regardless of direction

### 4.1.4 Solar Wind Plasma Features (10 features)
- **flow_speed**: Bulk velocity magnitude
- **vx_gse, vy_gse, vz_gse**: Velocity components
- **proton_density**: Number density
- **temperature**: Proton temperature
- **flow_pressure**: Dynamic pressure
- **electric_field**: Convective E-field
- **plasma_beta**: Thermal/magnetic pressure ratio
- **alfven_mach**: V/V_A
- **magnetosonic_mach**: V/V_ms

### 4.1.5 Derived Coupling Features (5 features)
- **v_np**: V × n_p (momentum flux)
- **v2_np**: V² × n_p (energy flux)
- **vbz_south**: V × min(Bz, 0) (southward coupling)

### 4.1.6 Geomagnetic Indices (8 features)
- **ae**: Auroral electrojet index
- **al**: Lower envelope of AE
- **au**: Upper envelope of AE
- **sym_d**: Symmetric disturbance D-component
- **sym_h**: Symmetric disturbance H-component (Dst proxy)
- **asy_d**: Asymmetric disturbance D-component
- **asy_h**: Asymmetric disturbance H-component
- **pcn**: Polar cap north index

## 4.2 Feature Computation

### 4.2.1 Derived Features

Computed in `train_dst_lstm_attention.py`:

```python
def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # Momentum and energy flux
    df['v_np'] = df['flow_speed'] * df['proton_density']
    df['v2_np'] = (df['flow_speed'] ** 2) * df['proton_density']
    
    # Southward IMF coupling
    df['bz_south'] = np.minimum(df['bz_gsm'], 0.0)
    df['bz_abs'] = np.abs(df['bz_gsm'])
    df['vbz_south'] = df['flow_speed'] * np.minimum(df['bz_gsm'], 0.0)
    
    return df
```

### 4.2.2 Rolling Window Features (Legacy System)

The original storm risk model (`build_dataset.py`) includes rolling statistics:

**15-minute windows**:
- Mean, std, min, max for each base feature
- Delta (current - 15min ago)

**60-minute windows**:
- Mean, std, min, max for each base feature
- Delta (current - 60min ago)

This creates 200+ features for the LightGBM models.

### 4.2.3 Feature Normalization

For LSTM models, features are standardized using training set statistics:

```python
mean = train_df[feature_cols].mean().fillna(0.0)
std = train_df[feature_cols].std().replace(0, 1.0).fillna(1.0)
df_normalized = (df[feature_cols] - mean) / std
```

Normalization parameters are saved in model metadata for inference.

## 4.3 Feature Selection and Importance

### 4.3.1 Physical Motivation

Feature selection is guided by solar wind-magnetosphere coupling physics:

**Critical Features** (highest importance):
1. **bz_gsm**: Primary coupling parameter
2. **flow_speed**: Energy input driver
3. **proton_density**: Momentum flux
4. **electric_field**: Convective coupling
5. **sym_h**: Recent geomagnetic state

**Supporting Features**:
- Plasma parameters (beta, Mach numbers)
- Field variability (RMS values)
- Temporal context (hour, doy)

### 4.3.2 Correlation Analysis

Feature correlation with Dst reveals:
- **Bz_GSM**: r = 0.65 (strongest single predictor)
- **Electric field**: r = 0.58
- **SYM-H**: r = 0.95 (autoregressive component)
- **Flow speed**: r = 0.42
- **Density**: r = 0.38

### 4.3.3 Feature Ablation Studies

Removing key features impacts performance:
- Without Bz_GSM: RMSE increases 40%
- Without velocity: RMSE increases 25%
- Without density: RMSE increases 15%
- Without temporal features: RMSE increases 10%

## 4.4 Feature Drift Monitoring

Operational deployment requires monitoring feature distributions:

### 4.4.1 Baseline Computation

```bash
python3 src/compute_feature_baseline.py \
  --omni-csv data/processed/omni.csv \
  --out-json configs/feature_baseline.json
```

Computes for each feature:
- Mean, std, min, max, median
- Percentiles (1, 5, 25, 75, 95, 99)
- Histogram bins

### 4.4.2 Drift Detection

Real-time drift score:

```python
def compute_drift_score(current_dist, baseline_dist):
    # KL divergence
    kl_div = np.sum(current_dist * np.log(current_dist / baseline_dist))
    return kl_div
```

Alert thresholds:
- **Warning**: KL divergence > 0.1
- **Critical**: KL divergence > 0.5

---

# 5. Model Architecture

## 5.1 Dst LSTM with Attention

### 5.1.1 Architecture Overview

The core Dst prediction model is a bidirectional LSTM with self-attention:

```
Input: (batch, 48, 49)
  ↓
BiLSTM(150 units, return_sequences=True)
  ↓
Self-Attention
  ↓
Dropout(0.1)
  ↓
LayerNormalization
  ↓
BiLSTM(170 units, return_sequences=True)
  ↓
Self-Attention
  ↓
Dropout(0.2)
  ↓
LayerNormalization
  ↓
GlobalAveragePooling1D
  ↓
Dense(1, linear activation)
  ↓
Output: Dst prediction
```

**Total Parameters**: ~450,000
**Inference Time**: <10ms on CPU, <2ms on GPU

### 5.1.2 LSTM Cell Equations

For each time step t, the LSTM cell computes:

**Forget Gate**:
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

**Input Gate**:
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

**Candidate Cell State**:
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)

**Cell State Update**:
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

**Output Gate**:
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

**Hidden State**:
h_t = o_t ⊙ tanh(c_t)

where:
- σ = sigmoid activation
- ⊙ = element-wise multiplication
- W, b = learnable weights and biases

**Bidirectional Processing**:
- Forward LSTM: processes t=1 to T
- Backward LSTM: processes t=T to 1
- Concatenation: h_t = [h_t^forward; h_t^backward]

### 5.1.3 Self-Attention Mechanism

The attention layer computes:

**Query, Key, Value**:
Q = H · W_Q
K = H · W_K
V = H · W_V

where H = [h_1, ..., h_T] is the sequence of hidden states.

**Attention Scores**:
A = softmax((Q · K^T) / √d_k)

**Attention Output**:
Z = A · V

This allows the model to focus on critical time steps (e.g., sudden Bz southward turns).

### 5.1.4 Regularization

**Dropout**: Applied after each attention layer
- Layer 1: 10% dropout
- Layer 2: 20% dropout
- Prevents overfitting to training sequences

**Layer Normalization**: Applied after dropout
- Stabilizes training
- Reduces internal covariate shift

**Early Stopping**: Monitors validation RMSE
- Patience: 5 epochs
- Restores best weights

### 5.1.5 Loss Function

Mean Squared Error (MSE):

L(θ) = (1/N) Σ(Dst_true - Dst_pred)²

Optimized using Adam optimizer with learning rate schedule.

## 5.2 Solar Wind Forecasting Model

### 5.2.1 Multi-Output Architecture

Similar to Dst model but with 17 outputs:

```
Input: (batch, 48, 49)
  ↓
BiLSTM(128 units, return_sequences=True)
  ↓
Self-Attention
  ↓
Dropout(0.1)
  ↓
LayerNormalization
  ↓
BiLSTM(128 units, return_sequences=True)
  ↓
Self-Attention
  ↓
Dropout(0.2)
  ↓
LayerNormalization
  ↓
GlobalAveragePooling1D
  ↓
Dense(128, ReLU)
  ↓
Dense(17, linear)
  ↓
Output: [B, Bx, By, Bz, V, Vx, Vy, Vz, n_p, T, P, E, β, M_A, M_ms, By_GSM, Bz_GSM]
```

### 5.2.2 Multi-Output Loss

L(φ) = (1/N) Σ_i Σ_j (y_ij - ŷ_ij)²

where j indexes the 17 output features.

### 5.2.3 Autoregressive Forecasting

For horizons beyond training (e.g., 24h from 6h model):

```python
def forecast_autoregressive(model, initial_window, steps):
    predictions = []
    window = initial_window.copy()
    
    for step in range(steps):
        pred = model.predict(window)
        predictions.append(pred)
        
        # Shift window and append prediction
        window = np.roll(window, -1, axis=1)
        window[0, -1, :] = pred
    
    return predictions
```

## 5.3 LightGBM Models

### 5.3.1 Storm Risk Classifier

Gradient-boosted decision trees for binary classification:

**Objective**: Binary log loss
**Learning Rate**: 0.05
**Num Leaves**: 64
**Feature Fraction**: 0.8
**Bagging Fraction**: 0.8
**Scale Pos Weight**: Auto-computed from class imbalance

**Training**: Incremental over Parquet shards
- 10 boosting rounds per shard
- Continues from previous model
- Enables out-of-core training

### 5.3.2 SYM-H Regressor

**Objective**: L1 (MAE)
**Learning Rate**: 0.05
**Num Leaves**: 64
**Feature Fraction**: 0.8
**Bagging Fraction**: 0.8

### 5.3.3 Flare Classifier

**Objective**: Binary log loss
**Learning Rate**: 0.05
**Num Leaves**: 64
**Scale Pos Weight**: Auto-computed

### 5.3.4 Isotonic Calibration

Post-training calibration for probability outputs:

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
calibrator.fit(val_probs, val_labels)

calibrated_probs = calibrator.transform(test_probs)
```

Ensures predicted probabilities match empirical frequencies.

---

# 6. Training Methodology

## 6.1 Data Splitting Strategy

### 6.1.1 Temporal Splits

**Training Set**: 1995-01-01 to 2023-12-31 (29 years)
**Validation Set**: 2024-01-01 to 2025-12-31 (2 years)
**Test Set**: 2026-01-01 onwards (future data)

Temporal splitting prevents data leakage and simulates operational deployment.

### 6.1.2 Alternative Splits for Evaluation

**Split A** (2023 evaluation):
- Train: 1995-2022
- Val: 2023
- Test: 2024-2025

**Split B** (Recent performance):
- Train: 1995-2023
- Val: 2024
- Test: 2025

## 6.2 Learning Rate Schedule

### 6.2.1 Periodic Schedule

```python
def lr_schedule(epoch):
    block = (epoch // 5) % 2
    return 1e-3 if block == 0 else 1e-4
```

**Rationale**:
- High LR (1e-3): Epochs 0-4, 10-14, 20-24, ...
- Low LR (1e-4): Epochs 5-9, 15-19, 25-29, ...
- Alternation helps escape local minima
- Inspired by cyclical learning rates

### 6.2.2 Alternative Schedules Tested

**Exponential Decay**:
lr = lr_0 × 0.95^epoch

**Cosine Annealing**:
lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × epoch / T))

**Result**: Periodic schedule performed best for this problem.

## 6.3 Training Procedure

### 6.3.1 Dst Model Training

```bash
python3 -u src/train_dst_lstm_attention.py \
  --omni-csv /path/to/omni.csv \
  --dst-csv /path/to/dst_hourly.csv \
  --model-dir models \
  --train-end 2023-12-31 \
  --val-end 2025-12-31 \
  --seq-len 48 \
  --horizon-hours 1 \
  --epochs 50 \
  --batch-size 48 \
  --no-interp \
  --checkpoint-dir models/checkpoints \
  --early-stop-patience 5
```

**Training Time**: ~6 hours on NVIDIA RTX 3090
**Memory Usage**: ~8 GB GPU RAM

### 6.3.2 Solar Wind Model Training

```bash
python3 -u src/train_solar_wind_lstm.py \
  --omni-csv /path/to/omni.csv \
  --model-dir models \
  --train-end 2025-12-31 \
  --val-end 2025-12-31 \
  --seq-len 48 \
  --horizon-hours 6 \
  --epochs 40 \
  --batch-size 64 \
  --checkpoint-dir models/checkpoints
```

**Training Time**: ~8 hours (17 outputs vs 1)
**Memory Usage**: ~10 GB GPU RAM

### 6.3.3 LightGBM Training

```bash
python3 src/train_lgbm.py \
  --parquet-dir data/processed/parquet \
  --model-dir models \
  --rounds-per-shard 10 \
  --max-eval-rows 300000
```

**Training Time**: ~2 hours on 32-core CPU
**Memory Usage**: ~16 GB RAM (out-of-core)

## 6.4 Hyperparameter Tuning

### 6.4.1 Grid Search Results

**Sequence Length**:
- Tested: 24, 48, 72, 96 hours
- Best: 48 hours (balance of context and overfitting)

**LSTM Units**:
- Tested: (64, 64), (128, 128), (150, 170), (256, 256)
- Best: (150, 170) asymmetric

**Dropout Rates**:
- Tested: (0.0, 0.0), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)
- Best: (0.1, 0.2)

**Batch Size**:
- Tested: 16, 32, 48, 64, 128
- Best: 48 (stability vs speed tradeoff)

### 6.4.2 Ablation Studies

**Without Attention**:
- RMSE: 6.8 nT (+33% vs 5.1 nT)
- Correlation: 0.92 (vs 0.96)

**Without Bidirectional**:
- RMSE: 6.2 nT (+22%)
- Correlation: 0.93

**Without Layer Normalization**:
- RMSE: 5.9 nT (+16%)
- Training instability

**Single LSTM Layer**:
- RMSE: 6.5 nT (+27%)
- Correlation: 0.93

## 6.5 Reproducibility

### 6.5.1 Random Seeds

All training scripts use fixed seeds:
- Python: `random.seed(42)`
- NumPy: `np.random.seed(42)`
- TensorFlow: `tf.random.set_seed(42)`

### 6.5.2 Model Metadata

Saved with each trained model:
- Feature list and order
- Normalization parameters (mean, std)
- Sequence length and horizon
- Training/validation split dates
- Hyperparameters
- Training history (loss per epoch)
- Test metrics

### 6.5.3 Model Registry

`models_deploy/registry.json` tracks all trained models:

```json
{
  "models": [
    {
      "name": "dst_lstm_attention",
      "version": "20260207094145",
      "artifact_path": "/path/to/model.keras",
      "created_at": "2026-02-07T09:41:45",
      "metrics": {"rmse": 5.117, "mae": 3.880, "correlation": 0.9586},
      "metadata": {
        "train_end": "2023-12-31",
        "val_end": "2025-12-31",
        "seq_len": 48,
        "horizon_hours": 1,
        "feature_cols": [...]
      }
    }
  ]
}
```

---

