import joblib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from preprocess import preprocess, get_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_model(y_test, preds, model_name):
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n--- {model_name} ---")
    print(f"RMSE:  {rmse:.4f} (${rmse*100000:,.0f} average error)")
    print(f"MAE:   {mae:.4f} (${mae*100000:,.0f} average error)")
    print(f"R²:    {r2:.4f} ({r2*100:.1f}% variance explained)")

    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def train_and_evaluate():
    print("="*50)
    print("  Training models on California Housing dataset")
    print("="*50)

    X_train, X_test, y_train, y_test = preprocess()

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=100, random_state=42, eval_metric="rmse"
        )
    }

    results = {}
    best_r2 = -999
    best_model = None
    best_name = ""

    for name, reg in models.items():
        pipeline = Pipeline([
            ("preprocessor", get_pipeline()),
            ("regressor", reg)
        ])

        # train
        pipeline.fit(X_train, y_train)

        # predict
        preds = pipeline.predict(X_test)

        # evaluate
        metrics = evaluate_model(y_test, preds, name)
        results[name] = metrics

        # track best model using R² score
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_model = pipeline
            best_name = name

    # save best model
    os.makedirs("models", exist_ok=True)
    model_path = "models/house_price_best_model.pkl"
    joblib.dump(best_model, model_path)

    # save feature names for Streamlit app
    X_train, X_test, y_train, y_test = preprocess()
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, "models/feature_names.pkl")

    print(f"\n✅ Best model: {best_name} (R²: {best_r2*100:.1f}%)")
    print(f"✅ Saved to: {model_path}")

    # save results
    results["best_model"] = best_name
    results["best_r2"] = round(best_r2, 4)
    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_model, feature_names


if __name__ == "__main__":
    train_and_evaluate()