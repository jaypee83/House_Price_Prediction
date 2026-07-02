import pandas as pd

import joblib



# Load trained model

model = joblib.load(
    "models/best_house_price_model.pkl"
)



# New house data

new_house = pd.DataFrame(
[
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
]
)



# Prediction

prediction = model.predict(new_house)



print(
    "Predicted House Price:"
)

print(
    prediction[0],
    "Lakhs"
)