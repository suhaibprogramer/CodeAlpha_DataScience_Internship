# CodeAlpha_UnemploymentAnalysis

Unemployment Analysis with Python — CodeAlpha Data Science Internship (Task 2)

## Overview
Explores unemployment rate trends across Indian states, investigates the impact of Covid-19, checks for seasonal patterns, and summarizes findings relevant to economic/social policy.

## Dataset
Two combined files:
- `Unemployment_in_India.csv` — May 2019 to Jun 2020, includes Rural/Urban split
- `Unemployment_Rate_upto_11_2020.csv` — Jan 2020 to Oct 2020, includes region Zone (North/South/East/West/Northeast) and coordinates

## Approach
1. Cleaned both datasets (stripped whitespace/BOM from column names, parsed dates, dropped incomplete rows)
2. Plotted the national monthly unemployment trend with the Covid-19 lockdown period highlighted
3. Compared unemployment before vs. during Covid (Mar–Jun 2020)
4. Compared Rural vs. Urban impact
5. Identified the hardest-hit states during the Covid period
6. Checked for seasonal monthly patterns across the full period
7. Used the second dataset to extend the trend through October 2020, broken down by zone, to see the recovery

## Results
- Unemployment rose **~87%** during Mar–Jun 2020 compared to the pre-Covid average
- **Urban** areas (19.3%) were hit harder than **Rural** areas (16.2%) during the Covid window
- **Puducherry**, Jharkhand, and Haryana were the hardest-hit states
- The national rate **peaked at 23.24% in May 2020**, then eased to **8.03% by October 2020** — a gradual, partial recovery

## Files
```
unemployment_analysis.py                    # main script
data/Unemployment_in_India.csv              # dataset 1
data/Unemployment_Rate_upto_11_2020.csv     # dataset 2
output/                                     # generated charts (trend, rural vs urban, hardest-hit states, seasonality, zone recovery)
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn
python unemployment_analysis.py
```

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn

---
*Part of the CodeAlpha Data Science Internship*
