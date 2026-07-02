import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



def load_data():

    df = pd.read_csv(
        "data/House Price.csv"
    )

    return df



def preprocess_data(df):

    # Remove duplicates

    df = df.drop_duplicates()

    df = df.reset_index(drop=True)


    # Separate input and target

    X = df.drop(
        columns=["TARGET(PRICE_IN_LACS)"]
    )

    y = df["TARGET(PRICE_IN_LACS)"]



    # Identify columns

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns


    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns



    # Numerical preprocessing

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )


    # Categorical preprocessing

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )



    # Combine preprocessing

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns
            ),

            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )



    # Train test split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    return (
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )




if __name__ == "__main__":


    df = load_data()


    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)



    print("Training data:")
    print(X_train.shape)


    print("\nTesting data:")
    print(X_test.shape)


    print("\nTarget training:")
    print(y_train.shape)