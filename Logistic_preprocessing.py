"""
Week 2: Data Collection, Cleaning, and Preprocessing for Logistics Analysis
Reference: DataCo SMART Supply Chain dataset
Input: DataCoSupplyChainDataset.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

INPUT_FILE = Path("DataCoSupplyChainDataset.csv")
OUTPUT_DIR = Path("week2_output")
OUTPUT_DIR.mkdir(exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        " DataCoSupplyChainDataset.csv "
    )

df = pd.read_csv(INPUT_FILE, encoding="latin1")
print("Raw shape:", df.shape)
print(df.head())
print(df.info())

# 1. Initial quality audit
quality = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "missing_count": df.isna().sum().values,
    "missing_percent": (df.isna().mean() * 100).round(2).values,
    "unique_count": df.nunique(dropna=True).values
})
quality.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)
print("Duplicates:", df.duplicated().sum())

# 2. Remove exact duplicates
df = df.drop_duplicates().copy()

# 3. Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_", regex=False)
)

# 4. Parse dates where present
for col in ["shipping_date_DateOrders", "order_date_DateOrders",
            "Order_Date", "Shipping_Date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# 5. Convert selected numeric fields
numeric_candidates = [
    "Sales", "Order_Item_Quantity", "Order_Item_Discount_Rate",
    "Order_Item_Product_Price", "Order_Item_Total",
    "Benefit_per_order", "Days_for_shipping_real",
    "Days_for_shipment_scheduled"
]
for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. Missing values: median for numeric, mode/Unknown for categorical
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object", "category"]).columns:
    if df[col].isna().any():
        mode = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

# 7. Basic logical validation
for col in ["Sales", "Order_Item_Quantity",
            "Order_Item_Product_Price", "Order_Item_Total"]:
    if col in df.columns:
        df = df[df[col] >= 0].copy()

# 8. IQR outlier report (flag first; do not blindly delete legitimate orders)
outlier_rows = []
for col in ["Sales", "Order_Item_Quantity",
            "Order_Item_Product_Price", "Order_Item_Total",
            "Benefit_per_order"]:
    if col not in df.columns:
        continue
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    outlier_rows.append({
        "column": col, "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_bound": lower, "upper_bound": upper,
        "outlier_count": int(mask.sum()),
        "outlier_percent": round(mask.mean() * 100, 2)
    })
pd.DataFrame(outlier_rows).to_csv(
    OUTPUT_DIR / "outlier_summary.csv", index=False
)

# 9. Create scaled feature file for ML/clustering
features = [c for c in [
    "Sales", "Order_Item_Quantity", "Order_Item_Discount_Rate",
    "Order_Item_Product_Price", "Order_Item_Total", "Benefit_per_order"
] if c in df.columns]

if features:
    feature_df = df[features].copy()

    standard = StandardScaler().fit_transform(feature_df)
    standard_df = pd.DataFrame(
        standard, columns=[c + "_standardized" for c in features]
    )

    minmax = MinMaxScaler().fit_transform(feature_df)
    minmax_df = pd.DataFrame(
        minmax, columns=[c + "_minmax" for c in features]
    )

    pd.concat([feature_df.reset_index(drop=True),
               standard_df, minmax_df], axis=1).to_csv(
        OUTPUT_DIR / "DataCo_scaled_features.csv", index=False
    )

# 10. Save cleaned dataset
df.to_csv(OUTPUT_DIR / "DataCo_cleaned.csv", index=False)

print("Final cleaned shape:", df.shape)
print("Remaining missing values:", int(df.isna().sum().sum()))
print("Outputs saved in:", OUTPUT_DIR.resolve())

