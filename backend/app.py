from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from io import BytesIO
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from google.oauth2 import service_account
from google.cloud import storage
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load credentials for Google Cloud Storage
credentials = service_account.Credentials.from_service_account_file(
    r"C:\Users\Kuwul\Documents\Personal\Projects\React\principal-lane-436113-h7-d598f11aa3a4.json"
)
client = storage.Client(credentials=credentials)

# Create FastAPI instance
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to models directory in Google Cloud Storage
BUCKET_NAME = "backend-api"

def load_models_and_scalers(city_name: str):
    """
    Load models and scalers for a specific city from Google Cloud Storage.
    """
    try:
        bucket = client.get_bucket(BUCKET_NAME)
        models = {}
        scalers = {}

        # Load models
        for model_type in ["linear_regression", "knn", "random_forest", "decision_tree"]:
            blob = bucket.blob(f"models/{city_name}_{model_type}_model.pkl")
            model_data = blob.download_as_bytes()
            models[model_type] = joblib.load(BytesIO(model_data))

        # Load scalers
        for scaler_type in ["scaler_X", "scaler_y"]:
            blob = bucket.blob(f"models/{city_name}_{scaler_type}.pkl")
            scaler_data = blob.download_as_bytes()
            scalers[scaler_type] = joblib.load(BytesIO(scaler_data))

        return models, scalers
    except Exception as e:
        raise FileNotFoundError(f"Error loading models or scalers for city '{city_name}': {e}")

def get_prediction(models, scalers, X_input):
    """
    Get predictions from all models and return the consensus or mean.
    """
    X_input_df = pd.DataFrame(X_input, columns=["Year", "Month", "Day"])
    X_scaled = scalers["scaler_X"].transform(X_input_df)

    predictions = {
        "linear_regression": scalers["scaler_y"].inverse_transform(
            models["linear_regression"].predict(X_scaled).reshape(-1, 1)
        ).flatten(),
        "knn": scalers["scaler_y"].inverse_transform(
            models["knn"].predict(X_scaled).reshape(-1, 1)
        ).flatten(),
        "random_forest": scalers["scaler_y"].inverse_transform(
            models["random_forest"].predict(X_scaled).reshape(-1, 1)
        ).flatten(),
        "decision_tree": scalers["scaler_y"].inverse_transform(
            models["decision_tree"].predict(X_scaled).reshape(-1, 1)
        ).flatten(),
    }

    # Calculate consensus or mean
    values, counts = np.unique(np.concatenate(list(predictions.values())), return_counts=True)
    if np.any(counts >= 2):
        final_prediction = round(values[counts >= 2][0])
    else:
        final_prediction = round(np.mean(np.concatenate(list(predictions.values()))))

    return final_prediction

@app.get("/buckets")
def list_buckets():
    """
    Endpoint to list all buckets in the Google Cloud project.
    """
    try:
        buckets = client.list_buckets()
        bucket_names = [bucket.name for bucket in buckets]
        return {"buckets": bucket_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching buckets: {e}")

@app.get("/forecast/today/{city_name}")
def get_today_forecast(city_name: str):
    """
    Endpoint to get today's forecast for a city.
    """
    try:
        models, scalers = load_models_and_scalers(city_name)
        today = datetime.now()
        X_input = np.array([[today.year, today.month, today.day]])
        prediction = get_prediction(models, scalers, X_input)
        return {"date": today.strftime("%Y-%m-%d"), "predicted_value": prediction}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/7days/{city_name}")
def get_7day_forecast(city_name: str):
    """
    Endpoint to get the 7-day forecast for a city.
    """
    try:
        models, scalers = load_models_and_scalers(city_name)
        today = datetime.now()
        dates = [today + timedelta(days=i) for i in range(1, 8)]
        predictions = []

        for date in dates:
            X_input = np.array([[date.year, date.month, date.day]])
            prediction = get_prediction(models, scalers, X_input)
            predictions.append({"date": date.strftime("%Y-%m-%d"), "predicted_value": prediction})

        return predictions
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Hello, Vercel!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
