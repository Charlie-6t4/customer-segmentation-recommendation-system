# -*- coding: utf-8 -*-

"""
Customer Segmentation and Recommendation System
Converted for Python editor / Visual Studio Code
"""

# =========================
# 1. Import Libraries
# =========================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go

from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import Counter
from tabulate import tabulate


# =========================
# 2. Load Dataset
# =========================

df = pd.read_csv(r"C:\Users\8180\Desktop\PRACTICE_PYTHON\Online Retail.csv", encoding="ISO-8859-1")

# Remove extra spaces from column names

df.columns = df.columns.str.strip()

df.rename(columns={"ï»¿InvoiceNo": "InvoiceNo"}, inplace=True)

# Check column names
print("Columns in dataset:")
print(df.columns.tolist())

print("\nInitial Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nNumerical Summary:")
print(df.describe().T)

print("\nCategorical Summary:")
print(df.describe(include="object").T)


# =========================
# 3. Missing Values
# =========================

missing_data = df.isnull().sum()
missing_percentage = (missing_data[missing_data > 0] / df.shape[0]) * 100
missing_percentage.sort_values(ascending=True, inplace=True)

plt.figure(figsize=(15, 4))
plt.barh(missing_percentage.index, missing_percentage, color="#ff6200")

for i, (value, name) in enumerate(zip(missing_percentage, missing_percentage.index)):
    plt.text(value + 0.5, i, f"{value:.2f}%", ha="left", va="center",
             fontweight="bold", fontsize=12, color="black")

plt.title("Percentage of Missing Values", fontweight="bold", fontsize=16)
plt.xlabel("Percentage (%)")
plt.show()

# Remove rows with missing CustomerID or Description
df = df.dropna(subset=["CustomerID", "Description"])

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# =========================
# 4. Remove Duplicate Rows
# =========================

print(f"\nDuplicate rows found: {df.duplicated().sum()}")

df.drop_duplicates(inplace=True)

print(f"Rows after removing duplicates: {df.shape[0]}")


# =========================
# 5. Treat Cancelled Transactions
# =========================

df["Transaction_Status"] = np.where(
    df["InvoiceNo"].astype(str).str.startswith("C"),
    "Cancelled",
    "Completed"
)

cancelled_transactions = df[df["Transaction_Status"] == "Cancelled"]

cancelled_percentage = (cancelled_transactions.shape[0] / df.shape[0]) * 100

print(f"\nCancelled Transactions Percentage: {cancelled_percentage:.2f}%")


# =========================
# 6. Correct StockCode Anomalies
# =========================

unique_stock_codes = df["StockCode"].unique()

numeric_char_counts = pd.Series(unique_stock_codes).apply(
    lambda x: sum(c.isdigit() for c in str(x))
).value_counts()

print("\nNumeric character count in StockCode:")
print(numeric_char_counts)

anomalous_stock_codes = [
    code for code in unique_stock_codes
    if sum(c.isdigit() for c in str(code)) in (0, 1)
]

print("\nAnomalous Stock Codes:")
for code in anomalous_stock_codes:
    print(code)

percentage_anomalous = (df["StockCode"].isin(anomalous_stock_codes).sum() / len(df)) * 100
print(f"\nPercentage of anomalous stock codes: {percentage_anomalous:.2f}%")

df = df[~df["StockCode"].isin(anomalous_stock_codes)]


# =========================
# 7. Clean Description Column
# =========================

description_counts = df["Description"].value_counts()
top_30_descriptions = description_counts.head(30)

plt.figure(figsize=(12, 8))
plt.barh(top_30_descriptions.index[::-1], top_30_descriptions.values[::-1], color="#ff6200")
plt.xlabel("Number of Occurrences")
plt.ylabel("Description")
plt.title("Top 30 Most Frequent Descriptions")
plt.show()

service_related_descriptions = ["Next Day Carriage", "High Resolution Image"]

service_related_percentage = (
    df[df["Description"].isin(service_related_descriptions)].shape[0] / df.shape[0]
) * 100

print(f"\nService related description percentage: {service_related_percentage:.2f}%")

df = df[~df["Description"].isin(service_related_descriptions)]

df["Description"] = df["Description"].str.upper()


# =========================
# 8. Treat Zero Unit Prices
# =========================

print("\nUnitPrice Summary:")
print(df["UnitPrice"].describe())

df = df[df["UnitPrice"] > 0]

df.reset_index(drop=True, inplace=True)

print(f"\nRows after UnitPrice cleaning: {df.shape[0]}")


# =========================
# 9. Feature Engineering
# =========================

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True)

# Extract invoice date only
df["InvoiceDay"] = df["InvoiceDate"].dt.date
df["InvoiceDay"] = pd.to_datetime(df["InvoiceDay"])

# -------------------------
# Recency Feature
# -------------------------

customer_data = df.groupby("CustomerID")["InvoiceDay"].max().reset_index()

most_recent_date = df["InvoiceDay"].max()

customer_data["Days_Since_Last_Purchase"] = (
    most_recent_date - customer_data["InvoiceDay"]
).dt.days

customer_data.drop(columns=["InvoiceDay"], inplace=True)


# -------------------------
# Frequency Features
# -------------------------

total_transactions = df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
total_transactions.rename(columns={"InvoiceNo": "Total_Transactions"}, inplace=True)

total_products_purchased = df.groupby("CustomerID")["Quantity"].sum().reset_index()
total_products_purchased.rename(columns={"Quantity": "Total_Products_Purchased"}, inplace=True)

customer_data = customer_data.merge(total_transactions, on="CustomerID", how="left")
customer_data = customer_data.merge(total_products_purchased, on="CustomerID", how="left")


# -------------------------
# Monetary Features
# -------------------------

df["Total_Spend"] = df["UnitPrice"] * df["Quantity"]

total_spend = df.groupby("CustomerID")["Total_Spend"].sum().reset_index()

average_transaction_value = total_spend.merge(total_transactions, on="CustomerID")
average_transaction_value["Average_Transaction_Value"] = (
    average_transaction_value["Total_Spend"] /
    average_transaction_value["Total_Transactions"]
)

customer_data = customer_data.merge(total_spend, on="CustomerID", how="left")
customer_data = customer_data.merge(
    average_transaction_value[["CustomerID", "Average_Transaction_Value"]],
    on="CustomerID",
    how="left"
)


# -------------------------
# Product Diversity
# -------------------------

unique_products_purchased = df.groupby("CustomerID")["StockCode"].nunique().reset_index()
unique_products_purchased.rename(
    columns={"StockCode": "Unique_Products_Purchased"},
    inplace=True
)

customer_data = customer_data.merge(unique_products_purchased, on="CustomerID", how="left")


# -------------------------
# Behavioral Features
# -------------------------

df["Day_Of_Week"] = df["InvoiceDate"].dt.dayofweek
df["Hour"] = df["InvoiceDate"].dt.hour

days_between_purchases = df.sort_values(["CustomerID", "InvoiceDay"]).groupby("CustomerID")["InvoiceDay"].diff().dt.days

df["Days_Between_Purchases"] = days_between_purchases

average_days_between_purchases = df.groupby("CustomerID")["Days_Between_Purchases"].mean().reset_index()
average_days_between_purchases.rename(
    columns={"Days_Between_Purchases": "Average_Days_Between_Purchases"},
    inplace=True
)

average_days_between_purchases["Average_Days_Between_Purchases"].fillna(0, inplace=True)

favorite_shopping_day = df.groupby(["CustomerID", "Day_Of_Week"]).size().reset_index(name="Count")
favorite_shopping_day = favorite_shopping_day.loc[
    favorite_shopping_day.groupby("CustomerID")["Count"].idxmax()
][["CustomerID", "Day_Of_Week"]]

favorite_shopping_hour = df.groupby(["CustomerID", "Hour"]).size().reset_index(name="Count")
favorite_shopping_hour = favorite_shopping_hour.loc[
    favorite_shopping_hour.groupby("CustomerID")["Count"].idxmax()
][["CustomerID", "Hour"]]

customer_data = customer_data.merge(average_days_between_purchases, on="CustomerID", how="left")
customer_data = customer_data.merge(favorite_shopping_day, on="CustomerID", how="left")
customer_data = customer_data.merge(favorite_shopping_hour, on="CustomerID", how="left")


# -------------------------
# Geographic Feature
# -------------------------

customer_country = df.groupby(["CustomerID", "Country"]).size().reset_index(name="Number_of_Transactions")

customer_main_country = customer_country.sort_values(
    "Number_of_Transactions",
    ascending=False
).drop_duplicates("CustomerID")

customer_main_country["Is_UK"] = customer_main_country["Country"].apply(
    lambda x: 1 if x == "United Kingdom" else 0
)

customer_data = customer_data.merge(
    customer_main_country[["CustomerID", "Is_UK"]],
    on="CustomerID",
    how="left"
)


# -------------------------
# Cancellation Features
# -------------------------

cancelled_transactions = df[df["Transaction_Status"] == "Cancelled"]

cancellation_frequency = cancelled_transactions.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
cancellation_frequency.rename(
    columns={"InvoiceNo": "Cancellation_Frequency"},
    inplace=True
)

customer_data = customer_data.merge(cancellation_frequency, on="CustomerID", how="left")
customer_data["Cancellation_Frequency"].fillna(0, inplace=True)

customer_data = customer_data.merge(
    total_transactions,
    on="CustomerID",
    how="left",
    suffixes=("", "_Check")
)

customer_data["Cancellation_Rate"] = (
    customer_data["Cancellation_Frequency"] /
    customer_data["Total_Transactions"]
)


# -------------------------
# Seasonality and Trends
# -------------------------

df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month

monthly_spending = df.groupby(["CustomerID", "Year", "Month"])["Total_Spend"].sum().reset_index()

seasonal_buying_patterns = monthly_spending.groupby("CustomerID")["Total_Spend"].agg(
    ["mean", "std"]
).reset_index()

seasonal_buying_patterns.rename(
    columns={
        "mean": "Monthly_Spending_Mean",
        "std": "Monthly_Spending_Std"
    },
    inplace=True
)

seasonal_buying_patterns["Monthly_Spending_Std"].fillna(0, inplace=True)


def calculate_trend(spend_data):
    if len(spend_data) > 1:
        x = np.arange(len(spend_data))
        slope, _, _, _, _ = linregress(x, spend_data)
        return slope
    else:
        return 0


spending_trends = monthly_spending.groupby("CustomerID")["Total_Spend"].apply(
    calculate_trend
).reset_index()

spending_trends.rename(columns={"Total_Spend": "Spending_Trend"}, inplace=True)

customer_data = customer_data.merge(seasonal_buying_patterns, on="CustomerID", how="left")
customer_data = customer_data.merge(spending_trends, on="CustomerID", how="left")

customer_data["CustomerID"] = customer_data["CustomerID"].astype(str)

print("\nCustomer-level dataset preview:")
print(customer_data.head())

print("\nCustomer-level dataset info:")
print(customer_data.info())


# =========================
# 10. Outlier Detection
# =========================

model = IsolationForest(contamination=0.05, random_state=0)

customer_data_numeric = customer_data.drop(columns=["CustomerID"])

customer_data["Outlier_Scores"] = model.fit_predict(customer_data_numeric.to_numpy())

customer_data["Is_Outlier"] = customer_data["Outlier_Scores"].apply(
    lambda x: 1 if x == -1 else 0
)

outlier_percentage = customer_data["Is_Outlier"].value_counts(normalize=True) * 100

plt.figure(figsize=(12, 4))
outlier_percentage.plot(kind="barh", color="#ff6200")

for index, value in enumerate(outlier_percentage):
    plt.text(value + 0.5, index, f"{value:.2f}%", fontsize=12)

plt.title("Percentage of Inliers and Outliers")
plt.xlabel("Percentage (%)")
plt.ylabel("Is Outlier")
plt.show()

outliers_data = customer_data[customer_data["Is_Outlier"] == 1]

customer_data_cleaned = customer_data[customer_data["Is_Outlier"] == 0]

customer_data_cleaned = customer_data_cleaned.drop(
    columns=["Outlier_Scores", "Is_Outlier"]
)

customer_data_cleaned.reset_index(drop=True, inplace=True)

print(f"\nRows after outlier removal: {customer_data_cleaned.shape[0]}")


# =========================
# 11. Correlation Analysis
# =========================

sns.set_style("whitegrid")

corr = customer_data_cleaned.drop(columns=["CustomerID"]).corr()

colors_map = ["#ff6200", "#ffcaa8", "white", "#ffcaa8", "#ff6200"]
my_cmap = LinearSegmentedColormap.from_list("custom_map", colors_map, N=256)

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, k=1)] = True

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    mask=mask,
    cmap=my_cmap,
    annot=True,
    center=0,
    fmt=".2f",
    linewidths=2
)

plt.title("Correlation Matrix", fontsize=14)
plt.show()


# =========================
# 12. Feature Scaling
# =========================

scaler = StandardScaler()

columns_to_exclude = ["CustomerID", "Is_UK", "Day_Of_Week"]

columns_to_scale = customer_data_cleaned.columns.difference(columns_to_exclude)

customer_data_scaled = customer_data_cleaned.copy()

customer_data_scaled[columns_to_scale] = scaler.fit_transform(
    customer_data_scaled[columns_to_scale]
)

customer_data_scaled.set_index("CustomerID", inplace=True)

print("\nScaled data preview:")
print(customer_data_scaled.head())


# =========================
# =========================
# 13. PCA
# =========================

# Check missing values before PCA
print("\nMissing values before PCA:")
print(customer_data_scaled.isnull().sum())
# Replace infinite values with NaN
customer_data_scaled = customer_data_scaled.replace([np.inf, -np.inf], np.nan)
# Fill NaN values with 0
customer_data_scaled = customer_data_scaled.fillna(0)
# Confirm no missing values left
print("\nMissing values after fixing:")
print(customer_data_scaled.isnull().sum().sum())
pca_full = PCA()
pca_full.fit(customer_data_scaled)

explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

optimal_components = 6

plt.figure(figsize=(14, 7))

sns.barplot(
    x=list(range(1, len(explained_variance_ratio) + 1)),
    y=explained_variance_ratio,
    color="#fcc36d",
    alpha=0.8
)

plt.plot(
    range(0, len(cumulative_explained_variance)),
    cumulative_explained_variance,
    marker="o",
    linestyle="--",
    color="#ff6200",
    linewidth=2
)

plt.axvline(
    optimal_components - 1,
    color="red",
    linestyle="--",
    label=f"Optimal Components = {optimal_components}"
)

plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")
plt.title("Cumulative Variance vs Number of Components")
plt.legend()
plt.grid()

plt.show()

pca = PCA(n_components=optimal_components)

customer_data_pca = pca.fit_transform(customer_data_scaled)

customer_data_pca = pd.DataFrame(
    customer_data_pca,
    columns=[f"PC{i+1}" for i in range(optimal_components)],
    index=customer_data_scaled.index
)

print("\nPCA data preview:")
print(customer_data_pca.head())


# =========================
# 14. Determine Optimal K
# =========================

inertia_values = []
silhouette_scores = []

k_range = range(2, 13)

for k in k_range:
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=100,
        random_state=0
    )

    labels = km.fit_predict(customer_data_pca)

    inertia_values.append(km.inertia_)
    silhouette_scores.append(silhouette_score(customer_data_pca, labels))

plt.figure(figsize=(12, 5))
plt.plot(k_range, inertia_values, marker="o", color="#ff6200")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(k_range, silhouette_scores, marker="o", color="#ff6200")
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid()
plt.show()

best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nBest K based on Silhouette Score: {best_k}")


# =========================
# 15. K-Means Clustering
# =========================

final_k = 3

kmeans = KMeans(
    n_clusters=final_k,
    init="k-means++",
    n_init=18,
    max_iter=200,
    random_state=0
)

kmeans.fit(customer_data_pca)

cluster_labels = kmeans.labels_

customer_data_cleaned["cluster"] = cluster_labels
customer_data_pca["cluster"] = cluster_labels

print("\nClustered Customer Data:")
print(customer_data_cleaned.head())


# =========================
# 16. 3D Cluster Visualization
# =========================

cluster_colors = ["#e8000b", "#1ac938", "#023eff"]

fig = go.Figure()

for cluster in sorted(customer_data_pca["cluster"].unique()):
    cluster_data = customer_data_pca[customer_data_pca["cluster"] == cluster]

    fig.add_trace(
        go.Scatter3d(
            x=cluster_data["PC1"],
            y=cluster_data["PC2"],
            z=cluster_data["PC3"],
            mode="markers",
            marker=dict(
                color=cluster_colors[cluster],
                size=5,
                opacity=0.4
            ),
            name=f"Cluster {cluster}"
        )
    )

fig.update_layout(
    title=dict(text="3D Visualization of Customer Clusters in PCA Space", x=0.5),
    scene=dict(
        xaxis=dict(backgroundcolor="#fcf0dc", gridcolor="white", title="PC1"),
        yaxis=dict(backgroundcolor="#fcf0dc", gridcolor="white", title="PC2"),
        zaxis=dict(backgroundcolor="#fcf0dc", gridcolor="white", title="PC3"),
    ),
    width=900,
    height=800
)

fig.show()


# =========================
# 17. Cluster Distribution
# =========================

cluster_percentage = (
    customer_data_pca["cluster"].value_counts(normalize=True) * 100
).reset_index()

cluster_percentage.columns = ["Cluster", "Percentage"]
cluster_percentage.sort_values(by="Cluster", inplace=True)

plt.figure(figsize=(10, 4))

sns.barplot(
    x="Percentage",
    y="Cluster",
    data=cluster_percentage,
    orient="h",
    palette=cluster_colors
)

for index, value in enumerate(cluster_percentage["Percentage"]):
    plt.text(value + 0.5, index, f"{value:.2f}%")

plt.title("Distribution of Customers Across Clusters")
plt.xlabel("Percentage (%)")
plt.show()


# =========================
# 18. Evaluation Metrics
# =========================

X = customer_data_pca.drop("cluster", axis=1)
clusters = customer_data_pca["cluster"]

sil_score = silhouette_score(X, clusters)

table_data = [
    ["Number of Observations", len(customer_data_pca)],
    ["Silhouette Score", sil_score]
]

print("\nCluster Evaluation Metrics:")
print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty"))


# =========================
# 19. Cluster Profiling
# =========================

features = customer_data_cleaned.columns[1:-1]
clusters_unique = sorted(customer_data_cleaned["cluster"].unique())

n_rows = len(features)
n_cols = len(clusters_unique)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))

for i, feature in enumerate(features):
    for j, cluster in enumerate(clusters_unique):
        data = customer_data_cleaned[
            customer_data_cleaned["cluster"] == cluster
        ][feature]

        axes[i, j].hist(
            data,
            bins=20,
            color=cluster_colors[j],
            edgecolor="white",
            alpha=0.7
        )

        axes[i, j].set_title(f"Cluster {cluster} - {feature}", fontsize=10)
        axes[i, j].set_xlabel("")
        axes[i, j].set_ylabel("")

plt.tight_layout()
plt.show()


# =========================
# 20. Recommendation System
# =========================

outlier_customer_ids = outliers_data["CustomerID"].astype(float).unique()

df_filtered = df[~df["CustomerID"].isin(outlier_customer_ids)]

customer_data_cleaned["CustomerID"] = customer_data_cleaned["CustomerID"].astype(float)

merged_data = df_filtered.merge(
    customer_data_cleaned[["CustomerID", "cluster"]],
    on="CustomerID",
    how="inner"
)

best_selling_products = merged_data.groupby(
    ["cluster", "StockCode", "Description"]
)["Quantity"].sum().reset_index()

best_selling_products = best_selling_products.sort_values(
    by=["cluster", "Quantity"],
    ascending=[True, False]
)

top_products_per_cluster = best_selling_products.groupby("cluster").head(10)

customer_purchases = merged_data.groupby(
    ["CustomerID", "cluster", "StockCode"]
)["Quantity"].sum().reset_index()

recommendations = []

for cluster in top_products_per_cluster["cluster"].unique():

    top_products = top_products_per_cluster[
        top_products_per_cluster["cluster"] == cluster
    ]

    customers_in_cluster = customer_data_cleaned[
        customer_data_cleaned["cluster"] == cluster
    ]["CustomerID"]

    for customer in customers_in_cluster:

        customer_purchased_products = customer_purchases[
            (customer_purchases["CustomerID"] == customer) &
            (customer_purchases["cluster"] == cluster)
        ]["StockCode"].tolist()

        top_products_not_purchased = top_products[
            ~top_products["StockCode"].isin(customer_purchased_products)
        ]

        top_3_products = top_products_not_purchased.head(3)

        recommendation_row = [customer, cluster]

        for _, row in top_3_products.iterrows():
            recommendation_row.extend([row["StockCode"], row["Description"]])

        while len(recommendation_row) < 8:
            recommendation_row.extend([None, None])

        recommendations.append(recommendation_row)

recommendations_df = pd.DataFrame(
    recommendations,
    columns=[
        "CustomerID",
        "cluster",
        "Rec1_StockCode",
        "Rec1_Description",
        "Rec2_StockCode",
        "Rec2_Description",
        "Rec3_StockCode",
        "Rec3_Description"
    ]
)

customer_data_with_recommendations = customer_data_cleaned.merge(
    recommendations_df,
    on=["CustomerID", "cluster"],
    how="right"
)

print("\nSample Recommendations:")
print(
    customer_data_with_recommendations[
        [
            "CustomerID",
            "cluster",
            "Rec1_StockCode",
            "Rec1_Description",
            "Rec2_StockCode",
            "Rec2_Description",
            "Rec3_StockCode",
            "Rec3_Description"
        ]
    ].sample(10, random_state=0)
)


# =========================
# 21. Save Final Output Files
# =========================

customer_data_cleaned.to_csv("customer_segmented_data.csv", index=False)

customer_data_with_recommendations.to_csv(
    "customer_recommendations.csv",
    index=False
)

print("\nFiles saved successfully:")
print("1. customer_segmented_data.csv")
print("2. customer_recommendations.csv")