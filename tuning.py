import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

import joblib


from data_preprocessing import (
    load_data,
    preprocess_data
)



# Load data

df = load_data()



preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)



# Random Forest

rf = RandomForestRegressor(
    random_state=42
)



pipeline = Pipeline(

[
    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        rf
    )
]

)



# Parameters

parameters = {


"model__n_estimators":

[100,200],


"model__max_depth":

[10,20,None]


}



grid = GridSearchCV(

pipeline,

parameters,

cv=3,

scoring="r2",

n_jobs=-1

)



print("Tuning started...")


grid.fit(

X_train,

y_train

)



print("\nBest Parameters:")

print(grid.best_params_)



print("\nBest Score:")

print(grid.best_score_)



# Save best model

joblib.dump(

grid.best_estimator_,

"models/best_house_price_model.pkl"

)

print("\nBest model saved!")