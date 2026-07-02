import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------
# Load Model & Preprocessor
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

st.title("🏠 House Price Prediction")
st.write("Enter house details to predict house price")

# -------------------------
# User Inputs
# -------------------------

posted_by = st.selectbox(
    "Posted By",
    ["Owner", "Dealer", "Builder"]
)

under_construction = st.selectbox(
    "Under Construction",
    [0, 1]
)

rera = st.selectbox(
    "RERA Approved",
    [0, 1]
)

bhk = st.number_input(
    "BHK Number",
    min_value=1,
    max_value=10,
    value=2
)

bhk_type = st.selectbox(
    "Property Type",
    ["BHK", "RK"]
)

square_ft = st.number_input(
    "Square Feet",
    min_value=100,
    value=1000
)

ready = st.selectbox(
    "Ready To Move",
    [0, 1]
)

resale = st.selectbox(
    "Resale",
    [0, 1]
)

address = st.text_input(
    "Address",
    "Chennai"
)

longitude = st.number_input(
    "Longitude",
    value=80.2707,
    format="%.6f"
)

latitude = st.number_input(
    "Latitude",
    value=13.0827,
    format="%.6f"
)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Price"):

    data = pd.DataFrame([{
        "POSTED_BY": posted_by,
        "UNDER_CONSTRUCTION": under_construction,
        "RERA": rera,
        "BHK_NO.": bhk,
        "BHK_OR_RK": bhk_type,
        "SQUARE_FT": square_ft,
        "READY_TO_MOVE": ready,
        "RESALE": resale,
        "ADDRESS": address,
        "LONGITUDE": longitude,
        "LATITUDE": latitude
    }])

    # Keep the same column order used during training
    data = data[[
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
    ]]

    # Preprocess the input
    data_processed = preprocessor.transform(data)

    # Predict
    prediction = model.predict(data_processed)

    st.success(f"🏠 Estimated House Price: ₹ {prediction[0]:,.2f} Lakhs")
