# FinShield: Automated Fraud Monitoring Pipeline

FinShield is an MLOps-driven financial fraud detection system that automates the monitoring and maintenance of models in production.

## Features
- **Real-time Serving:** FastAPI-based inference engine with integrated Prometheus metrics.
- **Drift Detection:** Automated analysis using EvidentlyAI to detect feature and target drift in financial transaction streams.
- **CI/CD Retraining:** GitHub Actions workflow that automatically triggers model retraining when precision drops below 95%.
- **Observability:** Custom Grafana dashboards for monitoring inference latency and fraud distribution.

## Tech Stack
- **Languages/Tools:** Python, Docker, GitHub Actions
- **Libraries:** XGBoost, Scikit-learn, FastAPI, EvidentlyAI
- **Monitoring:** Prometheus, Grafana
