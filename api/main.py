from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI()

# -------------------------------
# ENABLE CORS (VERY IMPORTANT 🔥)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# LOAD MODEL SAFELY
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "job_risk_model.pkl")
features_path = os.path.join(BASE_DIR, "..", "models", "features.pkl")

model = joblib.load(model_path)
features = joblib.load(features_path)


@app.get("/")
def home():
    return {"message": "AI Job Risk API Running"}


@app.post("/predict")
def predict(data: dict):

    # Create full input with 0
    input_data = {col: 0 for col in features}

    # Numeric inputs
    for key, value in data.items():
        if key != "job_role":
            input_data[key] = value

    # Handle job role
    job_role = data.get("job_role")
    job_col = f"job_role_{job_role}"

    if job_col in input_data:
        input_data[job_col] = 1

    # DataFrame
    df = pd.DataFrame([input_data])
    df = df[features]

    prediction = model.predict(df)[0]

    return {"ai_job_risk": round(float(prediction), 2)}