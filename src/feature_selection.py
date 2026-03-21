import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from preprocessing import load_data, preprocess_data


def select_features(df, target_column="ai_replacement_score", top_n=15):

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train Random Forest for feature importance
    model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X, y)

    # Get feature importance
    importances = model.feature_importances_

    feature_importance = pd.Series(importances, index=X.columns)

    # Select top N features
    top_features = feature_importance.sort_values(ascending=False).head(top_n).index

    return list(top_features)


# --------- TEST RUN ---------
if __name__ == "__main__":

    df = load_data("data/ai_job_replacement_2020_2026_v2.csv")

    df = preprocess_data(df)

    selected_features = select_features(df)

    print("Top Selected Features:\n")
    print(selected_features)