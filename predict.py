import pandas as pd
import joblib
import os

# ----------------------------
# Load Model and Preprocessor
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ----------------------------
# New House Data
# ----------------------------

new_house = pd.DataFrame([
    {
        "POSTED_BY": "Owner",
        "UNDER_CONSTRUCTION": 0,
        "RERA": 1,
        "BHK_NO.": 3,
        "BHK_OR_RK": "BHK",
        "SQUARE_FT": 1500,
        "READY_TO_MOVE": 1,
        "RESALE": 0,
        "ADDRESS": "Chennai",
        "LONGITUDE": 80.2707,
        "LATITUDE": 13.0827
    }
])

# Ensure the same column order as training
new_house = new_house[
    [
        "POSTED_BY",
        "UNDER_CONSTRUCTION",
        "RERA",
        "BHK_NO.",
        "BHK_OR_RK",
        "SQUARE_FT",
        "READY_TO_MOVE",
        "RESALE",
        "ADDRESS",
        "LONGITUDE",
        "LATITUDE"
    ]
]

# ----------------------------
# Preprocess Input
# ----------------------------

new_house_processed = preprocessor.transform(new_house)

# ----------------------------
# Predict
# ----------------------------

prediction = model.predict(new_house_processed)

print("\nPredicted House Price:")
print(f"{prediction[0]:.2f} Lakhs")
