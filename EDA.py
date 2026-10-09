import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = "gcp_final_approved_dataset.csv"
df = pd.read_csv(file_path)

output_dir = "EDA_Output"
os.makedirs(output_dir, exist_ok=True)

# Dataset dates are in DD-MM-YYYY HH:MM format
df["Usage Start Date"] = pd.to_datetime(
    df["Usage Start Date"], dayfirst=True, errors="coerce"
)
df["Usage End Date"] = pd.to_datetime(
    df["Usage End Date"], dayfirst=True, errors="coerce"
)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

print("DATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

overview = pd.DataFrame({
    "Metric": [
        "Number of Rows",
        "Number of Columns",
        "Duplicate Rows",
        "Total Missing Values",
        "Usage Start Date",
        "Usage End Date"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        df.duplicated().sum(),
        df.isnull().sum().sum(),
        df["Usage Start Date"].min(),
        df["Usage End Date"].max()
    ]
})
overview.to_csv(output_dir + "/dataset_overview.csv", index=False)

# Descriptive statistics
desc_stats = df[numeric_cols].describe().T
desc_stats["median"] = df[numeric_cols].median()
desc_stats["variance"] = df[numeric_cols].var()
desc_stats["standard_deviation"] = df[numeric_cols].std()
desc_stats = desc_stats[
    ["count", "mean", "median", "std",
     "variance", "min", "25%", "50%", "75%", "max"]
]
desc_stats.to_csv(output_dir + "/descriptive_statistics.csv")

print("\nDESCRIPTIVE STATISTICS")
print(desc_stats)

# Distribution analysis
distribution = pd.DataFrame(index=numeric_cols)
distribution["Mean"] = df[numeric_cols].mean()
distribution["Median"] = df[numeric_cols].median()
distribution["Variance"] = df[numeric_cols].var()
distribution["Standard Deviation"] = df[numeric_cols].std()
distribution["Skewness"] = df[numeric_cols].skew()
distribution["Kurtosis"] = df[numeric_cols].kurtosis()

def skew_interpretation(value):
    if abs(value) < 0.5:
        return "Approximately symmetric"
    elif abs(value) < 1:
        return "Moderately skewed"
    else:
        return "Highly skewed"

distribution["Distribution Interpretation"] = distribution["Skewness"].apply(
    skew_interpretation
)

distribution.to_csv(output_dir + "/distribution_analysis.csv")

print("\nDISTRIBUTION ANALYSIS")
print(distribution)

# Outlier detection using IQR
outlier_results = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]

    outlier_results.append({
        "Column": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_limit,
        "Upper Bound": upper_limit,
        "Number of Outliers": len(outliers),
        "Outlier Percentage": (len(outliers) / len(df)) * 100
    })

outlier_df = pd.DataFrame(outlier_results)
outlier_df.to_csv(output_dir + "/outlier_analysis.csv", index=False)

print("\nOUTLIER ANALYSIS")
print(outlier_df)

# Correlation
correlation = df[numeric_cols].corr(method="pearson")
correlation.to_csv(output_dir + "/correlation_matrix.csv")

print("\nCORRELATION MATRIX")
print(correlation)

# Service-wise cost
service_cost = (
    df.groupby("Service Name")["Total Cost (INR)"]
    .agg(["count", "mean", "sum", "median"])
    .sort_values("sum", ascending=False)
)
service_cost.to_csv(output_dir + "/service_cost_analysis.csv")

# Region-wise cost
region_cost = (
    df.groupby("Region/Zone")["Total Cost (INR)"]
    .agg(["count", "mean", "sum", "median"])
    .sort_values("sum", ascending=False)
)
region_cost.to_csv(output_dir + "/region_cost_analysis.csv")

# Usage unit analysis
unit_analysis = (
    df.groupby("Usage Unit")["Total Cost (INR)"]
    .agg(["count", "mean", "sum", "median"])
    .sort_values("sum", ascending=False)
)
unit_analysis.to_csv(output_dir + "/usage_unit_analysis.csv")

# Monthly cost analysis
df["Month"] = df["Usage Start Date"].dt.to_period("M").astype(str)

monthly_cost = (
    df.groupby("Month")["Total Cost (INR)"]
    .agg(["count", "sum", "mean", "median"])
)

monthly_cost.to_csv(output_dir + "/monthly_cost_analysis.csv")

# Histogram plots
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=20, edgecolor="black")
    axes[i].set_title(col)

for i in range(len(numeric_cols), len(axes)):
    axes[i].axis("off")

fig.suptitle("Distribution of Numerical Variables", fontsize=16)
plt.tight_layout()
plt.savefig(output_dir + "/01_histograms.png", dpi=300, bbox_inches="tight")
plt.show()

# Boxplots
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(col)
    axes[i].set_ylabel("Value")

for i in range(len(numeric_cols), len(axes)):
    axes[i].axis("off")

fig.suptitle("Boxplots for Outlier Detection", fontsize=16)
plt.tight_layout()
plt.savefig(output_dir + "/02_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()

# Service cost
plt.figure(figsize=(14, 10))
plt.barh(service_cost.index[::-1], service_cost["sum"][::-1])
plt.title("Total Cost by GCP Service")
plt.xlabel("Total Cost (INR)")
plt.ylabel("Service Name")
plt.tight_layout()
plt.savefig(output_dir + "/03_service_cost.png", dpi=300, bbox_inches="tight")
plt.show()

# Region cost
plt.figure(figsize=(10, 7))
plt.bar(region_cost.index, region_cost["sum"])
plt.title("Total Cost by Region/Zone")
plt.xlabel("Region/Zone")
plt.ylabel("Total Cost (INR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir + "/04_region_cost.png", dpi=300, bbox_inches="tight")
plt.show()

# Usage units
plt.figure(figsize=(8, 6))
plt.bar(unit_analysis.index, unit_analysis["count"])
plt.title("Number of Records by Usage Unit")
plt.xlabel("Usage Unit")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(output_dir + "/05_usage_units.png", dpi=300, bbox_inches="tight")
plt.show()

# CPU vs total cost
plt.figure(figsize=(9, 6))
plt.scatter(
    df["CPU Utilization (%)"],
    df["Total Cost (INR)"],
    alpha=0.6
)
plt.title("CPU Utilization vs Total Cost")
plt.xlabel("CPU Utilization (%)")
plt.ylabel("Total Cost (INR)")
plt.tight_layout()
plt.savefig(output_dir + "/06_cpu_vs_cost.png", dpi=300, bbox_inches="tight")
plt.show()

# Memory vs total cost
plt.figure(figsize=(9, 6))
plt.scatter(
    df["Memory Utilization (%)"],
    df["Total Cost (INR)"],
    alpha=0.6
)
plt.title("Memory Utilization vs Total Cost")
plt.xlabel("Memory Utilization (%)")
plt.ylabel("Total Cost (INR)")
plt.tight_layout()
plt.savefig(output_dir + "/07_memory_vs_cost.png", dpi=300, bbox_inches="tight")
plt.show()

# Network inbound vs outbound
plt.figure(figsize=(9, 6))
plt.scatter(
    df["Network Inbound Data (Bytes)"],
    df["Network Outbound Data (Bytes)"],
    alpha=0.6
)
plt.title("Network Inbound vs Network Outbound Data")
plt.xlabel("Network Inbound Data (Bytes)")
plt.ylabel("Network Outbound Data (Bytes)")
plt.tight_layout()
plt.savefig(output_dir + "/08_network_in_vs_out.png", dpi=300, bbox_inches="tight")
plt.show()

# Correlation heatmap using matplotlib
plt.figure(figsize=(12, 9))
plt.imshow(correlation.values, aspect="auto")
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)
plt.yticks(
    range(len(correlation.index)),
    correlation.index
)

for i in range(correlation.shape[0]):
    for j in range(correlation.shape[1]):
        plt.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    output_dir + "/09_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Monthly cost trend
plt.figure(figsize=(12, 6))
plt.plot(
    monthly_cost.index,
    monthly_cost["sum"],
    marker="o"
)
plt.title("Monthly Total Cloud Cost")
plt.xlabel("Month")
plt.ylabel("Total Cost (INR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    output_dir + "/10_monthly_cost.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY")
print("All tables and graphs are saved inside:", output_dir)
