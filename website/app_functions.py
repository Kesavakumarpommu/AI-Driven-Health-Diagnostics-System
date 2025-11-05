import joblib
import numpy as np
import pandas as pd

def ValuePredictor(disease, to_predict_list):
    """
    Predicts disease result based on disease type and input list.
    """
    disease = disease.lower()
    base_path = "./website/app_models"

    # Convert all inputs safely to floats
    numeric_input = []
    for val in to_predict_list:
        try:
            numeric_input.append(float(val))
        except ValueError:
            numeric_input.append(0.0)

    X = np.array(numeric_input).reshape(1, -1)

    # Load model and scaler based on disease
    if disease == "kidney":
        model = joblib.load(f"{base_path}/kidney_model.pkl")
        scaler = joblib.load(f"{base_path}/kidney_scaler.pkl")

    elif disease == "liver":
        model = joblib.load(f"{base_path}/liver_model.pkl")
        scaler = joblib.load(f"{base_path}/liver_scaler.pkl")

    elif disease == "heart":
        model = joblib.load(f"{base_path}/heart_model.pkl")
        scaler = joblib.load(f"{base_path}/heart_scaler.pkl")

    elif disease == "stroke":
        model = joblib.load(f"{base_path}/stroke_model.pkl")
        scaler = joblib.load(f"{base_path}/stroke_scaler.pkl")

    elif disease == "diabetes":
        model = joblib.load(f"{base_path}/diabetes_model.pkl")
        scaler = joblib.load(f"{base_path}/diabetes_scaler.pkl")

    else:
        raise ValueError(f"Unsupported disease type: {disease}")

    # Scale and predict
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)

    return int(pred[0]), disease
