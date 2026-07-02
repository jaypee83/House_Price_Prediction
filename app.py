import streamlit as st
import pandas as pd
import joblib
import os

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("🏠 House Price Prediction")
st.write("Enter house details to predict price")

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
    value=80.2707
)

latitude = st.number_input(
    "Latitude",
    value=13.0827
)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Price"):

    data = pd.DataFrame([{
        "POSTED_BY": str(posted_by),
        "UNDER_CONSTRUCTION": int(under_construction),
        "RERA": int(rera),
        "BHK_NO.": int(bhk),
        "BHK_OR_RK": str(bhk_type),
        "SQUARE_FT": float(square_ft),
        "READY_TO_MOVE": int(ready),
        "RESALE": int(resale),
        "ADDRESS": str(address),
        "LONGITUDE": float(longitude),
        "LATITUDE": float(latitude)
    }])

    # FIX COLUMN ORDER (VERY IMPORTANT)
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

    prediction = model.predict(data)

    st.success(f"Estimated Price: {prediction[0]:.2f} Lakhs")
