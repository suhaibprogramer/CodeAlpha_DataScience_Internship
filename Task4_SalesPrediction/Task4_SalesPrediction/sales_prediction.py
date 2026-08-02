"""
TASK 4: Sales Prediction using Python
--------------------------------------
Goal: Predict Sales based on advertising spend on TV, Radio, and Newspaper.
Dataset columns: TV, Radio, Newspaper (spend in $1000s), Sales (in thousands of units)

If you downloaded the real "Advertising.csv" from the CodeAlpha link, just replace
the file at data/Advertising.csv with it — this script works either way, as long
as the column names match (TV, Radio, Newspaper, Sales).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sns.set_style("whitegrid")

# -----------------------------
# STEP 1: Load the data
# -----------------------------
df = pd.read_csv("data/Advertising.csv")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # drop stray index column if present
print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary stats:\n", df.describe())

# -----------------------------
# STEP 2: Explore relationships (EDA)
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["TV", "Radio", "Newspaper"]):
    sns.scatterplot(data=df, x=col, y="Sales", ax=ax)
    ax.set_title(f"{col} vs Sales")
plt.tight_layout()
plt.savefig("output/1_scatter_relationships.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/2_correlation_heatmap.png", dpi=120)
plt.close()

print("\nCorrelation with Sales:\n", df.corr()["Sales"].sort_values(ascending=False))

# -----------------------------
# STEP 3: Prepare data for modeling
# -----------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 4: Train a Linear Regression model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# -----------------------------
# STEP 5: Evaluate the model
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance on Test Data:")
print(f"  MAE  : {mae:.3f}")
print(f"  RMSE : {rmse:.3f}")
print(f"  R^2  : {r2:.3f}  (closer to 1.0 is better)")

# -----------------------------
# STEP 6: Visualize predictions vs actual
# -----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig("output/3_actual_vs_predicted.png", dpi=120)
plt.close()

# -----------------------------
# STEP 7: Business insight
# -----------------------------
importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("\nBusiness takeaway:")
print(f"  Every extra $1000 spent on '{importance.index[0]}' adds ~{importance.iloc[0]:.3f}k units of sales — the biggest driver.")
print(f"  '{importance.index[-1]}' has the weakest effect on sales — likely the first budget to cut if trimming spend.")

print("\nDone. Charts saved in the output/ folder.")
