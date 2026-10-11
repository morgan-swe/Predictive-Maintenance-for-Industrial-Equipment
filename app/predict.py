import joblib
import pandas as pd
from pathlib import Path

# Project root, resolved from this file's location rather than the
# working directory, so paths work no matter where we launch from.
ROOT = Path(__file__).resolve().parent.parent
MODELS_FOLDER = ROOT / "models"
PROCESSED_FEATURES = ROOT / "data" / "processed" / "train_FD001_features.csv"
MODEL_RESULTS = ROOT / "dashboard" / "data" / "model_results.csv"

# Available models, keyed by the name we show in the UI
MODELS = {
    "Random Forest (tuned)": "random_forest_tuned.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Linear Regression": "linear_regression_model.pkl",
}


# The Model column in model_results.csv is spelled differently to the
# UI names above, so map between them explicitly.
RESULTS_NAMES = {
    "Random Forest (tuned)": "Random Forest (Tuned)",
    "Decision Tree": "Decision Tree",
    "Linear Regression": "Linear Regression",
}


# Reusable model loading function
def load_model(model_name="Random Forest (tuned)"):
    model_path = MODELS_FOLDER / MODELS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


# The models were trained on 88 engineered features in a specific order.
# Passing columns in any other order gives silently wrong predictions,
# so we always reindex against the names stored on the model itself.
def align_features(model, df):
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} feature(s) missing from input, "
            f"e.g. {missing[:5]}"
        )
    return df[expected]


# Evaluation metrics recorded during training, one row per model
def load_metrics(model_name):
    results = pd.read_csv(MODEL_RESULTS)
    row = results[results["Model"] == RESULTS_NAMES[model_name]]
    if row.empty:
        raise ValueError(f"No recorded metrics for {model_name}")
    return row.iloc[0]


# Load the engineered feature table produced by the notebooks
def load_features():
    return pd.read_csv(PROCESSED_FEATURES)


# Predict remaining useful life for every row given
def predict_rul(model, df):
    return model.predict(align_features(model, df))


# Predict RUL for one engine's most recent cycle
def predict_for_engine(model, df, unit_number):
    engine_rows = df[df["unit_number"] == unit_number]
    if engine_rows.empty:
        raise ValueError(f"No data for engine {unit_number}")
    latest = engine_rows.sort_values("time_in_cycles").tail(1)
    return float(predict_rul(model, latest)[0])


# Temp
if __name__ == "__main__":
    model = load_model()
    data = load_features()
    print(type(model))
    print("Engine 1 predicted RUL:", predict_for_engine(model, data, 1))
