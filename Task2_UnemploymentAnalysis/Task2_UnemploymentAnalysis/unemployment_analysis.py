"""
TASK 2: Unemployment Analysis with Python
--------------------------------------------
Goal: Explore unemployment rate trends, look at the Covid-19 impact,
find seasonal / regional patterns, and summarize policy-relevant insights.

Uses two real CodeAlpha/Kaggle files:
  - data/Unemployment_in_India.csv         (May 2019 - Jun 2020, has Rural/Urban split)
  - data/Unemployment_Rate_upto_11_2020.csv (Jan 2020 - Oct 2020, has region Zone + lat/long)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

RATE = "Estimated Unemployment Rate (%)"
EMPLOYED = "Estimated Employed"
PARTICIPATION = "Estimated Labour Participation Rate (%)"


def load_clean(path, rename_cols=None):
    df = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig strips the BOM
    df.columns = [c.strip() for c in df.columns]
    if rename_cols:
        df = df.rename(columns=rename_cols)
    # strip whitespace from all string/object columns (this dataset has stray spaces, e.g. " Rural")
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", RATE]).reset_index(drop=True)
    return df

# -----------------------------
# STEP 1: Load & clean both files
# -----------------------------
df1 = load_clean("data/Unemployment_in_India.csv")               # has Area: Rural/Urban
df2 = load_clean("data/Unemployment_Rate_upto_11_2020.csv",
                  rename_cols={"Region.1": "Zone"})               # has Zone: North/South/etc.

print("Unemployment_in_India.csv  -> shape:", df1.shape, "| range:", df1["Date"].min().date(), "to", df1["Date"].max().date())
print("Unemployment_Rate_upto_11_2020.csv -> shape:", df2.shape, "| range:", df2["Date"].min().date(), "to", df2["Date"].max().date())

print("\nFirst 5 rows (India dataset):\n", df1.head())
print("\nMissing values (India dataset):\n", df1.isnull().sum())

# -----------------------------
# STEP 2: National monthly trend (India dataset, May19-Jun20)
# -----------------------------
monthly_avg = df1.groupby("Date")[RATE].mean().reset_index()

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg["Date"], monthly_avg[RATE], marker="o")
plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"),
            color="red", alpha=0.15, label="Covid-19 lockdown period")
plt.title("Average Unemployment Rate Over Time (India)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig("output/1_national_trend.png", dpi=120)
plt.close()

# -----------------------------
# STEP 3: Covid-19 before/after comparison
# -----------------------------
pre_covid = df1[df1["Date"] < "2020-03-01"][RATE].mean()
covid_period = df1[(df1["Date"] >= "2020-03-01") & (df1["Date"] <= "2020-06-30")][RATE].mean()
increase_pct = (covid_period - pre_covid) / pre_covid * 100

print(f"\nAverage unemployment rate BEFORE Covid (pre-Mar 2020): {pre_covid:.2f}%")
print(f"Average unemployment rate DURING Covid (Mar-Jun 2020): {covid_period:.2f}%")
print(f"Increase during Covid: {increase_pct:.1f}%")

# -----------------------------
# STEP 4: Rural vs Urban impact
# -----------------------------
area_covid = df1[(df1["Date"] >= "2020-03-01") & (df1["Date"] <= "2020-06-30")].groupby("Area")[RATE].mean()
print("\nAverage unemployment rate during Covid, by Area:\n", area_covid)

plt.figure(figsize=(7, 5))
for area in df1["Area"].dropna().unique():
    sub = df1[df1["Area"] == area].groupby("Date")[RATE].mean()
    plt.plot(sub.index, sub.values, marker="o", label=area)
plt.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"), color="red", alpha=0.15)
plt.title("Unemployment Rate: Rural vs Urban")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig("output/2_rural_vs_urban.png", dpi=120)
plt.close()

# -----------------------------
# STEP 5: Which states/regions were hit hardest?
# -----------------------------
region_covid = (
    df1[(df1["Date"] >= "2020-03-01") & (df1["Date"] <= "2020-06-30")]
    .groupby("Region")[RATE].mean()
    .sort_values(ascending=False)
)
print("\nTop 5 hardest-hit states during Covid period:\n", region_covid.head())

plt.figure(figsize=(8, 6))
region_covid.head(10).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 10 States by Unemployment Rate (Covid period)")
plt.xlabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("output/3_hardest_hit_regions.png", dpi=120)
plt.close()

# -----------------------------
# STEP 6: Seasonal pattern check
# -----------------------------
df1["Month"] = df1["Date"].dt.month
seasonal = df1.groupby("Month")[RATE].mean()
print("\nAverage unemployment rate by calendar month (all years combined):\n", seasonal)

plt.figure(figsize=(8, 4))
seasonal.plot(kind="bar")
plt.title("Seasonal Pattern: Avg Unemployment Rate by Month")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("output/4_seasonal_pattern.png", dpi=120)
plt.close()

# -----------------------------
# STEP 7: Extended recovery trend using the second file (through Oct 2020), by Zone
# -----------------------------
zone_trend = df2.groupby(["Date", "Zone"])[RATE].mean().reset_index()

plt.figure(figsize=(10, 5))
for zone in zone_trend["Zone"].unique():
    sub = zone_trend[zone_trend["Zone"] == zone]
    plt.plot(sub["Date"], sub[RATE], marker="o", label=zone)
plt.title("Unemployment Rate Recovery by Zone (Jan-Oct 2020)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig("output/5_zone_recovery_trend.png", dpi=120)
plt.close()

peak_month = df2.groupby("Date")[RATE].mean().idxmax()
peak_rate = df2.groupby("Date")[RATE].mean().max()
oct_rate = df2[df2["Date"] == df2["Date"].max()][RATE].mean()
print(f"\nNational peak (2020 dataset) was {peak_rate:.2f}% in {peak_month.strftime('%B %Y')}.")
print(f"By {df2['Date'].max().strftime('%B %Y')}, average rate had eased to {oct_rate:.2f}%.")

# -----------------------------
# STEP 8: Policy-relevant summary
# -----------------------------
worse_area = area_covid.idxmax()
print("\n--- Summary for policymakers ---")
print(f"1. Unemployment rose ~{increase_pct:.0f}% during the Mar-Jun 2020 Covid window vs. the prior period.")
print(f"2. {worse_area} areas were hit harder than the other area type during the pandemic.")
print(f"3. {region_covid.index[0]} recorded the highest average unemployment rate during the Covid window — a candidate for targeted relief.")
print(f"4. Nationally, the rate peaked at {peak_rate:.2f}% in {peak_month.strftime('%B %Y')} and had eased to {oct_rate:.2f}% by {df2['Date'].max().strftime('%B %Y')}, suggesting a gradual recovery rather than a full one.")
print("5. Rates should keep being tracked monthly by zone/state, since recovery speed clearly varies by region.")

print("\nDone. Charts saved in the output/ folder.")
