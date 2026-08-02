"""
TASK 3: Car Price Prediction with Machine Learning
----------------------------------------------------
Goal: Predict a used car's Selling_Price from features like age, brand,
present price, kms driven, fuel type, seller type, transmission, owner count.

If you downloaded the real dataset from the CodeAlpha link, replace
data/car_data.csv with it — just make sure column names match, or edit
the COLUMN NAMES section below to match your file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sns.set_style("whitegrid")

# -----------------------------
# STEP 1: Load the data
# -----------------------------
df = pd.read_csv("data/car_data.csv")
print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Standardize column names across dataset variants (some versions use
# Kms_Driven/Seller_Type, others Driven_kms/Selling_type)
df = df.rename(columns={"Driven_kms": "Kms_Driven", "Selling_type": "Seller_Type"})

# -----------------------------
# STEP 2: Feature engineering
# -----------------------------
# Cars are easier to reason about by AGE rather than raw Year
current_year = 2021
df["Car_Age"] = current_year - df["Year"]
df.drop(columns=["Year"], inplace=True)

print("\nAfter feature engineering:\n", df.head())

# -----------------------------
# STEP 3: Quick EDA
# -----------------------------
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Car_Age", y="Selling_Price", hue="Fuel_Type")
plt.title("Car Age vs Selling Price")
plt.tight_layout()
plt.savefig("output/1_age_vs_price.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Transmission", y="Selling_Price")
plt.title("Selling Price by Transmission")
plt.tight_layout()
plt.savefig("output/2_price_by_transmission.png", dpi=120)
plt.close()

# -----------------------------
# STEP 4: Encode categorical variables
# -----------------------------
# One-hot encode categorical columns (drop the car name, too many unique values
# for a small dataset — brand goodwill is a bonus feature you can add if you
# have more data, e.g. mapping brand -> average resale value)
df_model = pd.get_dummies(
    df.drop(columns=["Car_Name"]),
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True
)

X = df_model.drop(columns=["Selling_Price"])
y = df_model["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 5: Train two models and compare
# -----------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Performance:")
    print(f"  MAE  : {mae:.3f} lakhs")
    print(f"  RMSE : {rmse:.3f} lakhs")
    print(f"  R^2  : {r2:.3f}")
    return r2

r2_lin = evaluate("Linear Regression", y_test, lin_pred)
r2_rf = evaluate("Random Forest", y_test, rf_pred)

best_name, best_pred = ("Random Forest", rf_pred) if r2_rf > r2_lin else ("Linear Regression", lin_pred)
print(f"\nBest model: {best_name}")

# -----------------------------
# STEP 6: Feature importance (Random Forest)
# -----------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop features driving price (Random Forest):\n", importances.head(6))

plt.figure(figsize=(7, 5))
importances.head(8).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Feature Importance for Car Price")
plt.tight_layout()
plt.savefig("output/3_feature_importance.png", dpi=120)
plt.close()

# -----------------------------
# STEP 7: Actual vs Predicted plot
# -----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Selling Price (lakhs)")
plt.ylabel("Predicted Selling Price (lakhs)")
plt.title(f"Actual vs Predicted ({best_name})")
plt.tight_layout()
plt.savefig("output/4_actual_vs_predicted.png", dpi=120)
plt.close()

print("\nDone. Charts saved in the output/ folder.")
