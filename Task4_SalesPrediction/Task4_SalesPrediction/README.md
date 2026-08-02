# CodeAlpha_SalesPrediction

Sales Prediction using Python — CodeAlpha Data Science Internship (Task 4)

## Overview
Predicts product **Sales** based on advertising spend across three channels: **TV**, **Radio**, and **Newspaper**. Built as part of the CodeAlpha Data Science internship.

## Dataset
`Advertising.csv` — 200 records with columns:
- `TV`, `Radio`, `Newspaper` — advertising spend ($ thousands)
- `Sales` — units sold (thousands)

## Approach
1. Loaded and explored the data (summary stats, missing values check)
2. Visualized relationships between each channel and Sales (scatter plots + correlation heatmap)
3. Split data into train/test sets (80/20)
4. Trained a **Linear Regression** model
5. Evaluated with MAE, RMSE, and R²
6. Plotted actual vs. predicted sales to visually check fit

## Results
| Metric | Score |
|---|---|
| R² | 0.899 |
| MAE | 1.461 |
| RMSE | 1.782 |

**Key insight:** Radio spend has the strongest effect on sales per dollar spent (coefficient ≈ 0.189), followed by TV (≈ 0.045). Newspaper spend has almost no measurable impact (≈ 0.003) — it's the first line item to cut if trimming an ad budget.

## Files
```
sales_prediction.py     # main script
data/Advertising.csv    # dataset
output/                 # generated charts (scatter plots, heatmap, actual vs predicted)
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python sales_prediction.py
```

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---
*Part of the CodeAlpha Data Science Internship*
