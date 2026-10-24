"""MPG prediction: preprocessing pipeline, model comparison and selection."""
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
TARGET = "mpg"


def load_data(source) -> pd.DataFrame:
    """Read the CSV, normalise column names and clean the target and numeric columns."""
    df = pd.read_csv(source, na_values=["?", "NA", ""])
    df.columns = [re.sub(r"[\s\-]+", "_", c.strip().lower()) for c in df.columns]
    missing = [c for c in NUMERIC + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    for c in NUMERIC + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def feature_columns(df):
    """Numeric features, plus origin (categorical) when the file has it."""
    cats = ["origin"] if "origin" in df.columns else []
    return NUMERIC, cats


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    """Median imputation + scaling for numbers, imputation + one-hot for categories."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    parts = [("num", numeric, num_cols)]
    if cat_cols:
        categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        parts.append(("cat", categorical, cat_cols))
    return ColumnTransformer(parts)


def candidate_models():
    return {
        "Linear regression (ridge)": Ridge(alpha=1.0),
        "K-nearest neighbours": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "Random forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient boosting": GradientBoostingRegressor(random_state=42),
        "Neural network (MLP)": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000,
                                             random_state=42),
    }


def train_and_select(df: pd.DataFrame):
    """Train every model, compare on a held-out test set, and return the best one.

    The winner is chosen by 5-fold cross-validated R2 on the training split (so the
    test set stays untouched for honest reporting), then refit on all rows.
    """
    num_cols, cat_cols = feature_columns(df)
    cols = num_cols + cat_cols
    X, y = df[cols], df[TARGET]
    if cat_cols:
        X = X.assign(origin=X["origin"].astype("Int64").astype(str).replace("<NA>", np.nan))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rows, fitted = [], {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor(num_cols, cat_cols)), ("model", model)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({"Model": name, "CV R2": cv, "Test R2": r2_score(y_te, pred),
                     "Test RMSE": mean_squared_error(y_te, pred) ** 0.5,
                     "Test MAE": mean_absolute_error(y_te, pred)})
        fitted[name] = pipe
    results = pd.DataFrame(rows).sort_values("CV R2", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    final = Pipeline(fitted[best_name].steps).fit(X, y)  # refit on all data for deployment
    return final, results, best_name, cols


def predict_mpg(model, cols, record: dict) -> float:
    """Predict MPG for one car given as a dict of feature values."""
    row = pd.DataFrame([record])[cols]
    if "origin" in cols:
        row["origin"] = row["origin"].astype(str)
    return float(max(model.predict(row)[0], 0))


def save_model(model, cols, path="mpg_model.joblib"):
    joblib.dump({"model": model, "cols": cols}, path)


def load_model(path="mpg_model.joblib"):
    d = joblib.load(path)
    return d["model"], d["cols"]
