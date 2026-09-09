# Space Weather Forecasting System - Part 3
# Operational Deployment and System Architecture

<!-- PAGE 11 -->

# 8. Operational Deployment Architecture

## 8.1 System Overview

The operational deployment consists of six major subsystems:

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  OMNI    │  │  Kyoto   │  │  NOAA    │  │  GOES    │   │
│  │  Solar   │  │   Dst    │  │  Alerts  │  │   XRS    │   │
│  │  Wind    │  │  Index   │  │          │  │  Flares  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Data Processing Layer    │
        │  ┌──────────────────────┐  │
        │  │  Parse & Normalize   │  │
        │  │  Feature Engineering │  │
        │  │  Quality Checks      │  │
        │  │  Gap Filling         │  │
        │  └──────────┬───────────┘  │
        └─────────────┼───────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Prediction Layer         │
        │  ┌──────────────────────┐  │
        │  │  Dst LSTM Model      │  │
        │  │  Solar Wind LSTM     │  │
        │  │  LightGBM Classifiers│  │
        │  │  Ensemble Logic      │  │
        │  └──────────┬───────────┘  │
        └─────────────┼───────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   API & Caching Layer      │
        │  ┌──────────────────────┐  │
        │  │  RESTful API         │  │
        │  │  Rate Limiting       │  │
        │  │  Authentication      │  │
        │  │  Response Caching    │  │
        │  └──────────┬───────────┘  │
        └─────────────┼───────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Presentation Layer       │
        │  ┌──────────────────────┐  │
        │  │  Next.js Dashboard   │  │
        │  │  Real-time Charts    │  │
        │  │  Alert Notifications │  │
        │  │  Historical Analysis │  │
        │  └──────────────────────┘  │
        └───────────────────────────┘
```

## 8.2 Data Ingestion Pipeline

### 8.2.1 Real-Time OMNI Updates

**Script**: `src/update_live_omni.py`

**Data Sources**:
- NOAA SWPC Real-Time Solar Wind: https://services.swpc.noaa.gov/products/solar-wind/
- DSCOVR Real-Time Data: https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json
- ACE Real-Time Data: https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json

**Update Frequency**: Every 1 minute (cron job)

**Process**:
1. Fetch latest 7-day data from SWPC
2. Parse JSON format
3. Merge with existing CSV
4. Compute derived features
5. Fill missing values (mean or interpolation)
6. Append to `data/processed/omni_live.csv`

**Cron Configuration**:
```bash
* * * * * cd /path/to/project && python3 src/update_live_omni.py \
  --output-csv data/processed/omni_live.csv >> logs/live_update.log 2>&1
```

### 8.2.2 Dst Index Updates

**Source**: Kyoto WDC provisional Dst
**Update Frequency**: Hourly
**Latency**: ~1 hour behind real-time

**Process**:
```bash
python3 src/fetch_kyoto_dst.py \
  --out-dir data/indices/kyoto \
  --provisional
```

### 8.2.3 Satellite Anomaly Updates

**Source**: NOAA NCEI anomaly database
**Update Frequency**: Daily
**Process**: Manual download and parsing (anomalies reported with delay)

### 8.2.4 Data Quality Monitoring

**Script**: `src/data_quality.py`

**Checks**:
- **Completeness**: Fraction of non-missing values per feature
- **Timeliness**: Age of latest data point
- **Consistency**: Range checks, outlier detection
- **Drift**: Distribution comparison with baseline

**Quality Metrics**:
```json
{
  "timestamp": "2026-02-08T12:00:00Z",
  "completeness": 0.98,
  "latest_data_age_minutes": 3,
  "features_with_gaps": ["magnetosonic_mach", "plasma_beta"],
  "outliers_detected": 2,
  "drift_score": 0.08,
  "status": "healthy"
}
```

**Alert Thresholds**:
- Completeness < 0.90: Warning
- Completeness < 0.80: Critical
- Data age > 15 min: Warning
- Data age > 60 min: Critical
- Drift score > 0.2: Warning

## 8.3 Prediction API

### 8.3.1 API Endpoints

**Base URL**: `http://localhost:8000/api`

**Endpoints**:

1. **GET /api/dst/latest**
   - Returns latest Dst prediction
   - Response time: <10ms (cached)
   
2. **GET /api/dst/forecast**
   - Returns 6-hour forecast
   - Uses two-stage prediction
   
3. **GET /api/storm-risk**
   - Returns storm probability
   - Threshold-based alerts
   
4. **GET /api/satellite-impact**
   - Returns satellite anomaly risk
   - 6-hour ahead probability
   
5. **GET /api/solar-wind/current**
   - Returns current solar wind conditions
   
6. **GET /api/solar-wind/forecast**
   - Returns 6-24 hour solar wind forecast
   
7. **GET /api/health**
   - System health check
   - Uptime, request stats, data quality
   
8. **GET /api/quality**
   - Data quality metrics
   - Gap analysis, drift scores
   
9. **GET /api/metrics**
   - Model performance metrics
   - Historical accuracy

### 8.3.2 API Response Format

**Example: /api/dst/latest**
```json
{
  "timestamp": "2026-02-08T12:00:00Z",
  "dst_current": -25.3,
  "dst_predicted_1h": -28.7,
  "confidence_interval": [-32.1, -25.3],
  "storm_probability": 0.12,
  "storm_level": "quiet",
  "data_quality": "good",
  "model_version": "20260207094145"
}
```

**Example: /api/storm-risk**
```json
{
  "timestamp": "2026-02-08T12:00:00Z",
  "risk_level": "moderate",
  "probability": 0.68,
  "predicted_dst_min": -58,
  "time_to_minimum": "2h 15m",
  "confidence": "high",
  "impacts": {
    "satellites": "moderate risk",
    "power_grids": "low risk",
    "communications": "moderate risk",
    "gps": "low risk"
  },
  "recommendations": [
    "Monitor satellite health telemetry",
    "Prepare for possible HF communication disruption"
  ]
}
```

### 8.3.3 Rate Limiting and Authentication

**Rate Limiting**:
- Default: 5 requests/second per IP
- Burst: 20 requests
- Configurable via `API_RATE_LIMIT_RPS` and `API_RATE_LIMIT_BURST`

**Authentication** (optional):
- API key in header: `X-API-Key: <key>`
- Enabled via `REQUIRE_API_KEY=1`
- Key set via `API_KEY` environment variable

**CORS**:
- Configurable allowed origins
- Default: `*` (all origins)
- Production: Specific domains only

### 8.3.4 Caching Strategy

**Cache TTL**: 30 seconds (configurable)

**Cached Endpoints**:
- `/api/dst/latest`: 30s
- `/api/solar-wind/current`: 30s
- `/api/health`: 60s

**Cache Invalidation**:
- Time-based expiration
- Manual invalidation on data update
- Version-based invalidation on model update

## 8.4 Web Dashboard

### 8.4.1 Technology Stack

**Frontend**:
- Next.js 14.2.7 (React framework)
- Server-side rendering for performance
- Static generation for public pages

**Styling**:
- Custom CSS with CSS modules
- Responsive design (mobile, tablet, desktop)

**Charting**:
- Custom canvas-based time series plots
- Real-time updates via polling
- Interactive zoom and pan

**Deployment**:
- Vercel (production)
- Docker (self-hosted option)

### 8.4.2 Dashboard Pages

**1. Home Page** (`/`)
- Current space weather overview
- Dst gauge and trend
- Storm risk indicator
- Recent alerts

**2. Aurora Page** (`/aurora`)
- Aurora forecast (Kp-based)
- Visibility maps
- Historical aurora events

**3. Solar Wind Page** (`/solar-wind`)
- Real-time solar wind parameters
- 6-hour forecast
- Parameter trends

**4. Dst/Kp Page** (`/kp-dst`)
- Dst time series (7-day)
- Kp index
- Storm history

**5. Magnetometers Page** (`/magnetometers`)
- Ground magnetometer data
- Station locations
- Real-time traces

**6. Flares Page** (`/flares`)
- Recent solar flares
- X-ray flux
- Flare forecast

**7. Protons Page** (`/protons`)
- Solar energetic protons
- SEP event risk
- Radiation storm scale

**8. CME Page** (`/cme`)
- Coronal mass ejection list
- CME arrival predictions
- Impact assessment

### 8.4.3 Real-Time Updates

**Polling Strategy**:
- Dst/storm risk: Every 30 seconds
- Solar wind: Every 60 seconds
- Alerts: Every 15 seconds

**WebSocket** (future enhancement):
- Push updates to clients
- Reduced server load
- Lower latency

### 8.4.4 Alert System

**Alert Types**:
1. **Storm Watch**: Dst predicted < -50 nT within 6 hours
2. **Storm Warning**: Dst predicted < -50 nT within 1 hour
3. **Severe Storm Warning**: Dst predicted < -100 nT
4. **Satellite Risk Alert**: Impact probability > 0.7
5. **Data Quality Alert**: Completeness < 0.8 or age > 60 min

**Notification Channels**:
- In-dashboard banner
- Browser notifications (with permission)
- Email (configurable)
- Webhook (for integration with external systems)

## 8.5 Model Registry and Version Control

### 8.5.1 Registry Structure

**File**: `models_deploy/registry.json`

**Schema**:
```json
{
  "models": [
    {
      "name": "dst_lstm_attention",
      "version": "20260207094145",
      "artifact_path": "/path/to/model.keras",
      "created_at": "2026-02-07T09:41:45.658701",
      "metrics": {
        "rmse": 5.117,
        "mae": 3.880,
        "correlation": 0.9586
      },
      "metadata": {
        "train_end": "2023-12-31",
        "val_end": "2025-12-31",
        "seq_len": 48,
        "horizon_hours": 1,
        "feature_spec_version": "2026-02-07",
        "feature_cols": [...]
      }
    }
  ]
}
```

### 8.5.2 Model Deployment Process

**Steps**:
1. Train new model with `train_dst_lstm_attention.py`
2. Evaluate on test set
3. Register model with `register_model()` function
4. Copy to `models_deploy/` directory
5. Update API to load new model
6. Run A/B test (optional)
7. Promote to production

**Rollback**:
- Keep previous 3 model versions
- Instant rollback by changing loaded model
- No downtime required

### 8.5.3 Continuous Monitoring

**Metrics Tracked**:
- Prediction RMSE (rolling 24h window)
- API latency (p50, p95, p99)
- Error rate
- Data quality score
- Feature drift score

**Alerting**:
- RMSE increases >20%: Investigate
- RMSE increases >50%: Rollback
- Latency p99 > 100ms: Scale up
- Error rate > 1%: Critical alert

## 8.6 Security and Compliance

### 8.6.1 Security Measures

**API Security**:
- Optional API key authentication
- Rate limiting per IP
- CORS restrictions
- Input validation and sanitization
- No user data storage (stateless)

**Infrastructure Security**:
- HTTPS only (production)
- Reverse proxy (nginx)
- Firewall rules
- Regular security updates

**Data Security**:
- Public data sources only
- No PII or sensitive data
- Audit logs for API access

### 8.6.2 Compliance

**Data Sources**:
- OMNI: Public domain (NASA)
- Kyoto Dst: Free for research and operational use
- NOAA data: Public domain (US government)

**Model Outputs**:
- Predictions are advisory only
- Not certified for safety-critical decisions
- Users responsible for operational decisions

**Disclaimer**:
```
This system provides space weather forecasts for informational 
purposes only. Predictions are statistical and may contain errors. 
Users should not rely solely on these forecasts for safety-critical 
or mission-critical decisions. Always consult official sources 
(NOAA SWPC, ESA SSA) for operational space weather warnings.
```

## 8.7 Scalability and Performance

### 8.7.1 Current Capacity

**Single Server** (8-core CPU, 16 GB RAM):
- Requests per second: 120
- Concurrent users: 500
- Daily predictions: 10 million+

**With Caching**:
- Requests per second: 450
- Concurrent users: 2000+

### 8.7.2 Scaling Strategy

**Horizontal Scaling**:
- Load balancer (nginx)
- Multiple API server instances
- Shared model storage (NFS or S3)
- Redis for distributed caching

**Vertical Scaling**:
- GPU for faster inference
- More CPU cores for parallel requests
- SSD for faster model loading

**Database** (future):
- PostgreSQL for historical predictions
- TimescaleDB for time series
- Enables analytics and model retraining

### 8.7.3 Cost Analysis

**Cloud Deployment** (AWS):
- EC2 t3.large: $60/month
- S3 storage (100 GB): $2/month
- Data transfer: $10/month
- **Total**: ~$75/month

**Self-Hosted**:
- Hardware: $2000 one-time
- Electricity: $20/month
- Internet: $50/month
- **Total**: $70/month (after hardware amortization)

---

<!-- PAGE 12 -->

# 9. Advanced Topics and Extensions

## 9.1 Uncertainty Quantification

### 9.1.1 Prediction Intervals

**Method**: Quantile regression

Train three models:
- Lower bound (10th percentile)
- Median (50th percentile)
- Upper bound (90th percentile)

**Loss Function**:
```python
def quantile_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return tf.reduce_mean(
        tf.maximum(quantile * error, (quantile - 1) * error)
    )
```

**Result**: 80% prediction intervals

**Example Output**:
```json
{
  "dst_predicted": -45.2,
  "confidence_interval_80": [-52.1, -38.3],
  "confidence_interval_95": [-58.7, -31.7]
}
```

### 9.1.2 Ensemble Methods

**Approach**: Train multiple models with different:
- Random seeds
- Train/val splits
- Hyperparameters

**Aggregation**:
- Mean prediction
- Median prediction
- Weighted average (by validation performance)

**Uncertainty Estimate**:
- Standard deviation of ensemble predictions
- Higher std = higher uncertainty

### 9.1.3 Monte Carlo Dropout

**Method**: Keep dropout active during inference

**Process**:
1. Run model N times (e.g., N=100)
2. Collect N predictions
3. Compute mean and std

**Interpretation**:
- Mean: Best estimate
- Std: Epistemic uncertainty

## 9.2 Explainability and Interpretability

### 9.2.1 Attention Visualization

**Method**: Extract attention weights from trained model

**Visualization**:
- Heatmap of attention scores over time
- Highlights critical time steps
- Reveals model focus during storms

**Example Insight**:
- High attention on Bz southward turns
- Increased attention 2-4 hours before storm onset
- Attention shifts to velocity during main phase

### 9.2.2 Feature Importance

**SHAP Values** (for LightGBM models):
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot feature importance
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
```

**Results**:
- Bz_GSM: 28% importance
- SYM-H: 22% importance
- Flow_speed: 15% importance
- Electric_field: 12% importance
- Density: 8% importance

### 9.2.3 Partial Dependence Plots

**Method**: Vary one feature, hold others constant

**Example**: Dst vs Bz_GSM
- Bz > 0: Dst ~ -10 nT (quiet)
- Bz = 0: Dst ~ -20 nT
- Bz = -5 nT: Dst ~ -40 nT
- Bz = -10 nT: Dst ~ -70 nT
- Bz < -15 nT: Dst ~ -100+ nT (storm)

## 9.3 Multi-Horizon Forecasting

### 9.3.1 Direct Multi-Horizon

**Approach**: Train separate models for each horizon

**Horizons**:
- 1 hour: RMSE = 5.1 nT
- 3 hours: RMSE = 8.4 nT
- 6 hours: RMSE = 12.2 nT
- 12 hours: RMSE = 18.5 nT
- 24 hours: RMSE = 25.1 nT

### 9.3.2 Sequence-to-Sequence

**Architecture**: Encoder-decoder LSTM

**Encoder**: Processes 48-hour input sequence
**Decoder**: Generates 24-hour output sequence

**Advantage**: Single model for all horizons
**Disadvantage**: More complex training

### 9.3.3 Probabilistic Forecasting

**Method**: Predict full distribution, not just point estimate

**Output**: Histogram or mixture of Gaussians

**Example**:
```json
{
  "horizon_hours": 6,
  "distribution": {
    "type": "gaussian_mixture",
    "components": [
      {"mean": -45, "std": 8, "weight": 0.7},
      {"mean": -65, "std": 12, "weight": 0.3}
    ]
  },
  "percentiles": {
    "p10": -62,
    "p50": -48,
    "p90": -35
  }
}
```

## 9.4 Integration with Physics Models

### 9.4.1 WSA-Enlil Solar Wind Model

**Source**: NOAA SWPC physics-based model
**Horizon**: 1-4 days
**Output**: Solar wind speed, density, magnetic field at Earth

**Integration**:
1. Download WSA-Enlil forecast
2. Use as input to Dst model
3. Generate 1-4 day Dst forecast

**Advantage**: Extended lead time
**Disadvantage**: Physics model errors compound

### 9.4.2 Hybrid ML-Physics Approach

**Method**: Combine ML and physics models

**Approach 1**: ML corrects physics model bias
```python
dst_final = dst_physics + ml_correction(dst_physics, features)
```

**Approach 2**: Weighted ensemble
```python
dst_final = w1 * dst_ml + w2 * dst_physics
```

**Approach 3**: ML learns residuals
```python
dst_final = dst_physics + ml_residual(features)
```

### 9.4.3 Data Assimilation

**Method**: Combine observations with model predictions

**Kalman Filter**:
- State: Dst and derivatives
- Observations: Real-time Dst measurements
- Model: ML predictions
- Output: Optimal estimate combining both

## 9.5 Transfer Learning and Domain Adaptation

### 9.5.1 Pre-training on Related Tasks

**Task 1**: Predict SYM-H (1-minute resolution)
**Task 2**: Fine-tune for Dst (hourly)

**Advantage**: More training data (1-minute vs hourly)

### 9.5.2 Cross-Solar-Cycle Adaptation

**Challenge**: Solar cycle variations affect model performance

**Solution**: Domain adaptation
- Train on Solar Cycle 24 (2008-2019)
- Adapt to Solar Cycle 25 (2020-2030)
- Use domain adversarial training

### 9.5.3 Multi-Planet Forecasting

**Extension**: Apply to other planets

**Mars**: Predict Martian magnetic field perturbations
**Jupiter**: Predict Jovian magnetospheric activity

**Transfer**: Use Earth-trained model as initialization

## 9.6 Real-Time Learning and Adaptation

### 9.6.1 Online Learning

**Method**: Update model with new data continuously

**Approach**:
- Incremental learning (add new data to training)
- Sliding window (keep last N years)
- Exponential weighting (recent data weighted more)

**Challenge**: Catastrophic forgetting

**Solution**: Elastic Weight Consolidation (EWC)

### 9.6.2 Active Learning

**Method**: Identify uncertain predictions, request labels

**Process**:
1. Model makes prediction
2. Compute uncertainty
3. If uncertainty > threshold, flag for review
4. Expert provides correct Dst value
5. Retrain model with new example

**Benefit**: Improves model on difficult cases

### 9.6.3 Reinforcement Learning

**Formulation**: RL agent learns to predict Dst

**State**: Current and past solar wind features
**Action**: Dst prediction
**Reward**: Negative of prediction error

**Advantage**: Can optimize for specific objectives (e.g., minimize false alarms)

---

<!-- PAGE 13 -->

# 10. Limitations and Challenges

## 10.1 Data Limitations

### 10.1.1 Missing Data

**Problem**: Features missing in recent years
- magnetosonic_mach: 15% missing in 2025
- plasma_beta: 12% missing
- temperature: 11% missing

**Impact**: Performance degradation (RMSE increases from 5.1 to 12.9 nT)

**Mitigation**:
- Mean imputation (current)
- Model-based imputation (future)
- Train separate model for incomplete features

### 10.1.2 Data Latency

**Problem**: Real-time data has 1-5 minute delay

**Impact**: Reduces effective lead time
- 1-hour prediction becomes 55-59 minute prediction

**Mitigation**:
- Nowcasting model (0-minute horizon)
- Extrapolation for latest minutes

### 10.1.3 Data Quality Variations

**Problem**: Quality varies by spacecraft and time period

**Examples**:
- ACE: High quality, but aging (launched 1997)
- DSCOVR: Good quality, but occasional gaps
- Wind: Excellent quality, but not always at L1

**Impact**: Inconsistent model performance

**Mitigation**:
- Quality-aware training (weight by quality score)
- Ensemble of models trained on different spacecraft

### 10.1.4 Sparse Extreme Events

**Problem**: Few extreme storms in training data
- Dst < -200 nT: Only 5 events in 30 years
- Dst < -300 nT: Only 2 events

**Impact**: Model underestimates extreme events

**Mitigation**:
- Synthetic data generation (SMOTE)
- Transfer learning from physics simulations
- Ensemble with physics models for extremes

## 10.2 Model Limitations

### 10.2.1 Forecast Horizon Limits

**Problem**: Accuracy degrades rapidly beyond 6 hours

**Reason**: Solar wind variability not predictable from past alone

**Fundamental Limit**: Need solar observations (coronagraph, magnetogram)

**Solution**: Integrate with solar imaging and physics models

### 10.2.2 Non-Stationarity

**Problem**: Solar wind statistics change over solar cycle

**Impact**: Model trained on solar minimum performs worse at solar maximum

**Mitigation**:
- Periodic retraining (every 6 months)
- Solar cycle features (F10.7, SSN)
- Adaptive learning rate

### 10.2.3 Compounding Errors

**Problem**: Two-stage prediction compounds errors
- Solar wind forecast error: RMSE = 2.8
- Dst prediction error: RMSE = 5.1
- Combined error: RMSE = 8.2 (not additive, but significant)

**Mitigation**:
- End-to-end training (future work)
- Uncertainty propagation
- Ensemble methods

### 10.2.4 Black Box Nature

**Problem**: LSTM models are difficult to interpret

**Impact**: Hard to diagnose failures, build trust

**Mitigation**:
- Attention visualization
- SHAP values (for tree models)
- Hybrid ML-physics models

## 10.3 Operational Challenges

### 10.3.1 Computational Requirements

**Problem**: GPU required for fast inference at scale

**Cost**: $500-2000 for GPU hardware

**Mitigation**:
- Model quantization (reduce precision)
- Model distillation (smaller student model)
- Cloud GPU (pay-per-use)

### 10.3.2 Model Maintenance

**Problem**: Models degrade over time (concept drift)

**Effort**: Retraining every 6 months

**Mitigation**:
- Automated retraining pipeline
- Continuous monitoring
- A/B testing for new models

### 10.3.3 False Alarms

**Problem**: False positive storm warnings

**Impact**: Alert fatigue, loss of trust

**Trade-off**: Sensitivity vs specificity

**Mitigation**:
- Calibrated probabilities
- User-configurable thresholds
- Confidence indicators

### 10.3.4 Liability and Responsibility

**Problem**: Who is responsible if forecast is wrong?

**Legal**: Predictions are advisory only

**Ethical**: Provide best possible forecast, communicate uncertainty

**Solution**: Clear disclaimers, uncertainty quantification

## 10.4 Scientific Limitations

### 10.4.1 Incomplete Physics

**Problem**: Model doesn't explicitly represent all physics

**Missing**:
- Substorm dynamics
- Plasmasphere effects
- Ionospheric feedback

**Impact**: May miss some physical regimes

**Mitigation**: Hybrid ML-physics models

### 10.4.2 Generalization to Unseen Conditions

**Problem**: Model trained on historical data

**Question**: Will it work for unprecedented events?

**Example**: Carrington Event (1859) - Dst estimated at -1760 nT

**Mitigation**:
- Physics constraints
- Ensemble with physics models
- Conservative extrapolation

### 10.4.3 Causality vs Correlation

**Problem**: ML learns correlations, not causation

**Risk**: Spurious correlations may fail in new regimes

**Example**: Model might learn time-of-day patterns that don't generalize

**Mitigation**:
- Feature selection based on physics
- Causal inference methods
- Validation on out-of-distribution data

---

