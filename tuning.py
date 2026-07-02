import os
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    load_data,
    preprocess_data
)

# -------------------------------
# Load Dataset
# -------------------------------
df = load_data()

preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)

# Create models folder
os.makedirs("models", exist_ok=True)

# -------------------------------
# Pipeline
# -------------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

# -------------------------------
# Hyperparameters
# -------------------------------
param_dist = {
    "model__n_estimators": [100, 150, 200],
    "model__max_depth": [10, 12, 15],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

# -------------------------------
# Random Search
# -------------------------------
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=2,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=True
)

print("Tuning Started...\n")

search.fit(X_train, y_train)

print("\nBest Parameters")
print(search.best_params_)

print("\nBest Score")
print(search.best_score_)

# -------------------------------
# Save SMALL model
# -------------------------------
best_pipeline = search.best_estimator_

best_model = best_pipeline.named_steps["model"]
best_preprocessor = best_pipeline.named_steps["preprocessor"]

joblib.dump(best_model, "models/best_model.pkl", compress=3)
joblib.dump(best_preprocessor, "models/preprocessor.pkl", compress=3)

print("\nFiles Saved Successfully!")
print("models/best_model.pkl")
print("models/preprocessor.pkl")
