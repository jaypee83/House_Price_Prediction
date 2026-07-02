import pandas as pd
import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_preprocessing import (
    load_data,
    preprocess_data
)

# Load data
df = load_data()

# Preprocess
preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)

print(X_train.columns)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Models (OPTIMIZED FOR SMALL SIZE)
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
}

results = []

for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, rmse, r2])

    # ✔ Save only model part
    joblib.dump(
        pipeline.named_steps["model"],
        f"models/{name}.pkl"
    )

# Save preprocessor separately
joblib.dump(preprocessor, "models/preprocessor.pkl")

# Results table
results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2 Score"]
)

print("\nModel Comparison")
print(results_df)

