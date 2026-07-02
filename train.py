import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


from data_preprocessing import (
    load_data,
    preprocess_data
)



# Load data

df = load_data()



# Preprocess

preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)



# Define models

models = {


    "Linear Regression":

    LinearRegression(),



    "Random Forest":

    RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),



    "XGBoost":

    XGBRegressor(
        random_state=42
    )

}



results = []



for name, model in models.items():


    print("\nTraining:", name)



    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )



    # Train

    pipeline.fit(
        X_train,
        y_train
    )



    # Prediction

    predictions = pipeline.predict(
        X_test
    )



    # Metrics


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    r2 = r2_score(
        y_test,
        predictions
    )



    results.append(
        [
            name,
            mae,
            rmse,
            r2
        ]
    )



    # Save model

    joblib.dump(
        pipeline,
        f"models/{name}.pkl"
    )




# Results table


results_df = pd.DataFrame(

    results,

    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2 Score"
    ]

)



print("\nModel Comparison")

print(results_df)