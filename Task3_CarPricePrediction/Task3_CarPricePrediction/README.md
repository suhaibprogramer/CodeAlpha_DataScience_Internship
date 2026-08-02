# CodeAlpha_CarPricePrediction

Car Price Prediction with Machine Learning — CodeAlpha Data Science Internship (Task 3)

## Overview
Predicts a used car's **Selling Price** based on features like present (showroom) price, age, kilometers driven, fuel type, seller type, transmission, and ownership history.

## Dataset
`car_data.csv` — 301 records with columns:
- `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Driven_kms`
- `Fuel_Type` (Petrol/Diesel/CNG), `Selling_type` (Dealer/Individual), `Transmission` (Manual/Automatic), `Owner`

## Approach
1. Loaded and cleaned the data (checked for missing values)
2. Feature engineering: converted `Year` into `Car_Age` (more intuitive for pricing)
3. Explored relationships (age vs. price, price by transmission type)
4. One-hot encoded categorical features (Fuel_Type, Seller_Type, Transmission)
5. Trained and compared two models: **Linear Regression** and **Random Forest Regressor**
6. Evaluated both with MAE, RMSE, and R²
7. Extracted feature importance from the Random Forest model

## Results
| Model | R² | MAE (lakhs) | RMSE (lakhs) |
|---|---|---|---|
| Linear Regression | 0.849 | 1.216 | 1.866 |
| **Random Forest** | **0.958** | **0.641** | **0.981** |

**Key insight:** `Present_Price` (the car's current showroom price) is by far the strongest predictor of resale value (~89% of feature importance), followed by `Car_Age` and `Kms_Driven`. Fuel type and transmission have only minor effects.

## Files
```
car_price_prediction.py   # main script
data/car_data.csv         # dataset
output/                   # generated charts (age vs price, feature importance, actual vs predicted)
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python car_price_prediction.py
```

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (Linear Regression, Random Forest)

---
*Part of the CodeAlpha Data Science Internship*
