import pandas as pd


# Load dataset
def load_data(path):

    df = pd.read_csv(path)

    return df


# Preprocess dataset
def preprocess_data(df):

    df = df.copy()

    # -------------------------------
    # Drop ID column
    # -------------------------------

    if "job_id" in df.columns:
        df.drop(columns=["job_id"], inplace=True)


    # -------------------------------
    # Handle Missing Values
    # -------------------------------

    # Numeric columns → fill with median
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)


    # Categorical columns → fill with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)


    # -------------------------------
    # Encode Categorical Variables
    # -------------------------------

    df = pd.get_dummies(df, drop_first=False)



    # -------------------------------
    # Remove Duplicate Rows
    # -------------------------------

    df.drop_duplicates(inplace=True)


    # -------------------------------
    # Reset Index
    # -------------------------------

    df.reset_index(drop=True, inplace=True)

    return df