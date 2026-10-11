import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

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


# The 20 engines held out from training. Uses the same split as
# notebooks/03_model_training.ipynb, so the app only shows engines
# the model has never seen.
def held_out_engines(df):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(splitter.split(df, groups=df["unit_number"]))
    return sorted(df["unit_number"].iloc[test_idx].unique())


# Predict RUL for one engine at a given cycle (default: its last cycle).
# Training engines run to failure, so the last cycle is always ~0 RUL;
# pick an earlier cycle to see the model mid-life.
def predict_for_engine(model, df, unit_number, cycle=None):
    engine_rows = df[df["unit_number"] == unit_number]
    if engine_rows.empty:
        raise ValueError(f"No data for engine {unit_number}")
    if cycle is None:
        row = engine_rows.sort_values("time_in_cycles").tail(1)
    else:
        row = engine_rows[engine_rows["time_in_cycles"] == cycle]
        if row.empty:
            raise ValueError(f"Engine {unit_number} has no cycle {cycle}")
    return float(predict_rul(model, row)[0])


# Temp
if __name__ == "__main__":
    model = load_model()
    data = load_features()
    print(type(model))
    print("Engine 1 predicted RUL:", predict_for_engine(model, data, 1))
