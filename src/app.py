import time
import pandas as pd
import joblib
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, make_asgi_app

app = FastAPI()

# Load pre-trained XGBoost model
model = joblib.load("models/fraud_model.pkl")

# Prometheus Metrics
PREDICTION_COUNTER = Counter("fraud_predictions_total", "Total predictions made")
LATENCY_HISTOGRAM = Histogram("inference_latency_seconds", "Time spent processing prediction")
FRAUD_DETECTED = Counter("fraud_detected_total", "Total fraud cases identified")

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.post("/predict")
async def predict(request: Request):
    start_time = time.time()
    data = await request.json()
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Inference
    prediction = model.predict(df)[0]
    
    # Update Metrics
    PREDICTION_COUNTER.inc()
    if prediction == 1:
        FRAUD_DETECTED.inc()
    
    latency = time.time() - start_time
    LATENCY_HISTOGRAM.observe(latency)
    
    return {"fraud": int(prediction), "latency": latency}
