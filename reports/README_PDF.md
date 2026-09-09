# Research Paper PDF Documentation

## Generated PDF

**File**: `research_paper_full.pdf`
**Pages**: 70
**Size**: 166 KB
**Format**: A4
**Created**: February 8, 2026

## Contents

The comprehensive research paper includes:

### Part 1: Core Research (research_paper.md)
- Title Page and Executive Summary
- Abstract
- Introduction (Background, Objectives, Contributions)
- Problem Statement and Physical Background
  - Formal problem definition
  - Solar wind-magnetosphere coupling physics
  - Dynamic pressure, convective electric field
  - Ring current dynamics
  - Plasma parameters and derived coupling functions
- Data and Feature Engineering
- Preprocessing and Data Quality
- Model Architecture (Dst LSTM with Attention)
  - Core architecture
  - LSTM equations
  - Attention mechanism
- Training Strategy
- Evaluation Results (Full-year 2023, Windows, 2025 Forward)
  - RMSE: 5.117 nT
  - Correlation: 0.9586
  - Comparison with baselines and literature
- Solar Wind Forecasting Model
- Computational Performance

### Part 2: Extended Technical Details (research_paper_extended.md)
- Data Sources and Acquisition
  - OMNI Solar Wind Database
  - Kyoto Dst Index
  - Satellite Anomaly Data
  - Solar Flare Reports
  - Data download and storage architecture
- Data Preprocessing Pipeline
  - OMNI parsing
  - Hourly aggregation
  - Missing data strategies
  - Data quality assessment
- Feature Engineering (49 features across 6 categories)
  - Temporal features
  - Data quality metadata
  - Magnetic field features
  - Solar wind plasma features
  - Derived coupling features
  - Geomagnetic indices
- Feature computation and normalization
- Feature selection and importance
- Feature drift monitoring
- Detailed Model Architecture
  - Dst LSTM with attention (450K parameters)
  - LSTM cell equations
  - Self-attention mechanism
  - Regularization techniques
- Solar Wind Forecasting Model (multi-output)
- LightGBM Models (storm risk, SYM-H, flare)
- Training Methodology
  - Data splitting strategy
  - Learning rate schedule
  - Training procedures
  - Hyperparameter tuning
  - Ablation studies
- Reproducibility (seeds, metadata, model registry)

### Part 3: Operational Deployment (research_paper_part3.md)
- Operational Deployment Architecture
  - System overview diagram
  - Data ingestion layer
  - Data processing layer
  - Prediction layer
  - API & caching layer
  - Presentation layer
- Data Ingestion Pipeline
  - Real-time OMNI updates
  - Dst index updates
  - Satellite anomaly updates
  - Data quality monitoring
- Prediction API
  - 9 API endpoints
  - Response formats
  - Rate limiting and authentication
  - Caching strategy
- Web Dashboard
  - Technology stack (Next.js)
  - 8 dashboard pages
  - Real-time updates
  - Alert system
- Model Registry and Version Control
- Security and Compliance
- Scalability and Performance
  - Current capacity (120 rps)
  - Scaling strategy
  - Cost analysis ($75/month cloud)
- Advanced Topics
  - Uncertainty quantification
  - Explainability (attention visualization, SHAP)
  - Multi-horizon forecasting
  - Integration with physics models
  - Transfer learning
  - Real-time learning
- Limitations and Challenges
  - Data limitations (missing data, latency, quality)
  - Model limitations (horizon limits, non-stationarity)
  - Operational challenges (compute, maintenance, false alarms)
  - Scientific limitations (incomplete physics, causality)

## Key Highlights

### Performance Metrics
- **RMSE**: 5.117 nT (state-of-the-art)
- **MAE**: 3.880 nT
- **Correlation**: 0.9586
- **Prediction Efficiency**: 0.918
- **Storm Detection Rate**: 91.7% (moderate storms)
- **Inference Latency**: <10ms CPU, <2ms GPU

### Model Architecture
- Bidirectional LSTM with self-attention
- 48-hour sequence length
- 49 input features
- 450,000 parameters
- Periodic learning rate schedule

### Operational System
- Real-time data ingestion (1-minute updates)
- RESTful API (9 endpoints)
- Web dashboard (8 pages)
- 120 requests/second capacity
- Sub-second inference latency

### Data Coverage
- 30+ years of solar wind data (1995-2025)
- 10,000+ satellite anomaly events
- Multiple spacecraft sources (ACE, Wind, DSCOVR)
- Hourly Dst index from Kyoto WDC

## How to Regenerate PDF

```bash
# Install dependencies
pip3 install --user markdown weasyprint

# Run conversion script
python3 reports/convert_to_pdf.py
```

## Source Files

1. `research_paper.md` - Main paper with core results
2. `research_paper_extended.md` - Technical details and methodology
3. `research_paper_part3.md` - Operational deployment and advanced topics
4. `convert_to_pdf.py` - PDF generation script

## Citation

If using this research, please cite:

```
Space Weather Research Team (2026). Space Weather Forecasting System: 
A Comprehensive Machine Learning Approach to Geomagnetic Storm Prediction 
and Satellite Impact Assessment. Technical Report, Version 2.0.
```

## Contact

For questions about this research or the operational system, please refer to the project README.md.
