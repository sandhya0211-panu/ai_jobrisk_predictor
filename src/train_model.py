import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from preprocessing import load_data, preprocess_data
from feature_selection import select_features


# -------------------------------
# 1️⃣ Create models folder (fix error)
# -------------------------------
os.makedirs("models", exist_ok=True)


# -------------------------------
# 2️⃣ Load Dataset 
# -------------------------------
df = load_data("data/ai_job_replacement_2020_2026_v2.csv")


# -------------------------------
# 3️⃣ Preprocess Data
# -------------------------------
df = preprocess_data(df)


# -------------------------------
# 4️⃣ Feature Selection
# -------------------------------
selected_features = select_features(df)

# Save selected features
joblib.dump(selected_features, "models/features.pkl")

X = df[selected_features]
y = df["ai_replacement_score"]


# -------------------------------
# 5️⃣ Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# 6️⃣ Define Models
# -------------------------------
models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        verbosity=0
    )
}


best_model = None
best_score = 0
best_model_name = ""


print("\n===== MODEL COMPARISON =====\n")


# -------------------------------
# 7️⃣ Train & Evaluate
# -------------------------------
for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)

    print(f"{name}")
    print("R2 Score:", round(r2, 4))
    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("---------------------------")

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name


# -------------------------------
# 8️⃣ Save Best Model
# -------------------------------
joblib.dump(best_model, "models/job_risk_model.pkl")


print("\n✅ Best Model:", best_model_name)
print("✅ Best R2 Score:", round(best_score, 4))
print("✅ Model Saved Successfully!")