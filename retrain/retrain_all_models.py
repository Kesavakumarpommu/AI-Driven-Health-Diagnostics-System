import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Folder paths
DATASET_DIR = "datasets"
MODEL_DIR = "../website/app_models"

os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(disease, label_column, model_type="logistic"):
    """
    Trains a model for a given disease and saves model + scaler
    """
    dataset_path = os.path.join(DATASET_DIR, f"{disease}.csv")

    try:
        df = pd.read_csv(dataset_path, encoding='utf-8', encoding_errors='replace')
    except Exception as e:
        print(f"⚠️ Failed to read {disease}.csv: {e}")
        return

    print(f"📊 Columns in {disease}.csv:", df.columns.tolist())

    if label_column not in df.columns:
        print(f"❌ Label column '{label_column}' not found in {disease}.csv")
        return

    # Drop ID column if present
    df = df.drop(columns=["id"], errors="ignore")

    # Define selected features for kidney
    kidney_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc',
                       'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    # Split features and target
    if disease == "kidney":
        X = df[kidney_features]
    else:
        X = df.drop(label_column, axis=1)

    y = df[label_column].astype(str).str.strip().replace({'ckd': 1, 'notckd': 0, 'yes': 1, 'no': 0})


    # Encode categorical columns
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str).str.strip()
            X[col] = X[col].astype('category').cat.codes

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "randomforest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train_scaled, y_train)

    # Evaluate
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"✅ {disease.capitalize()} model trained | Accuracy: {acc:.2f}")

    # Save model and scaler
    model_file = os.path.join(MODEL_DIR, f"{disease}_model.pkl")
    scaler_file = os.path.join(MODEL_DIR, f"{disease}_scaler.pkl")
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"🎉 Saved {model_file} and {scaler_file}\n")


# ---------- Train All Models ----------
if __name__ == "__main__":
    try:
        train_model("diabetes", label_column="Outcome", model_type="logistic")
        train_model("heart", label_column="target", model_type="logistic")
        train_model("kidney", label_column="classification", model_type="randomforest")
        train_model("liver", label_column="Result", model_type="logistic")
        train_model("stroke", label_column="stroke", model_type="logistic")
        print("✅ All models trained and saved successfully!")
    except Exception as e:
        print(f"❌ Error during training: {e}")
