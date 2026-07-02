import streamlit as st
import pandas as pd
import joblib


# Load model

model = joblib.load(
    "models/best_model.pkl"
)


st.title("🏠 House Price Prediction")

st.write(
    "Enter house details to predict price"
)



# User inputs


posted_by = st.selectbox(
    "Posted By",
    [
        "Owner",
        "Dealer",
        "Builder"
    ]
)


under_construction = st.selectbox(
    "Under Construction",
    [0,1]
)


rera = st.selectbox(
    "RERA Approved",
    [0,1]
)


bhk = st.number_input(
    "BHK Number",
    min_value=1,
    max_value=10,
    value=2
)


bhk_type = st.selectbox(
    "Property Type",
    [
        "BHK",
        "RK"
    ]
)


square_ft = st.number_input(
    "Square Feet",
    value=1000
)


ready = st.selectbox(
    "Ready To Move",
    [0,1]
)


resale = st.selectbox(
    "Resale",
    [0,1]
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



# Prediction button

if st.button("Predict Price"):


    data = pd.DataFrame(
    [
        {
        "POSTED_BY":posted_by,

        "UNDER_CONSTRUCTION":under_construction,

        "RERA":rera,

        "BHK_NO.":bhk,

        "BHK_OR_RK":bhk_type,

        "SQUARE_FT":square_ft,

        "READY_TO_MOVE":ready,

        "RESALE":resale,

        "ADDRESS":address,

        "LONGITUDE":longitude,

        "LATITUDE":latitude
        }
    ]
    )



    prediction = model.predict(data)



    st.success(
        f"Estimated Price: {prediction[0]:.2f} Lakhs"
    )
