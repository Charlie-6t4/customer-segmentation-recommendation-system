import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# =========================
# File Paths
# =========================

BASE_DIR = Path(__file__).parent

segmented_file = BASE_DIR / "data" / "customer_segmented_data.csv"
recommendation_file = BASE_DIR / "data" / "customer_recommendations.csv"
# =========================
# Load Data
# =========================

if not segmented_file.exists():
    st.error("customer_segmented_data.csv file not found. Please place it in the same folder as app.py.")
    st.stop()

if not recommendation_file.exists():
    st.error("customer_recommendations.csv file not found. Please place it in the same folder as app.py.")
    st.stop()

segmented_data = pd.read_csv(segmented_file)
recommendations = pd.read_csv(recommendation_file)

# =========================
# Title
# =========================

st.title("Customer Segmentation and Recommendation Dashboard")

st.write(
    "This interactive dashboard presents customer segmentation results, cluster analysis, "
    "and personalized product recommendations based on purchasing behavior."
)

# =========================
# Sidebar Navigation
# =========================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Cluster Analysis",
        "Customer Recommendations",
        "Raw Data"
    ]
)

# =========================
# Overview Page
# =========================

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", segmented_data.shape[0])

    with col2:
        st.metric("Total Features", segmented_data.shape[1])

    with col3:
        st.metric("Total Clusters", segmented_data["cluster"].nunique())

    st.subheader("Customer Segmented Data Preview")
    st.dataframe(segmented_data.head(10), use_container_width=True)

    st.subheader("Cluster Distribution")

    cluster_counts = segmented_data["cluster"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cluster_counts.index.astype(str), cluster_counts.values)

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customers per Cluster")

    st.pyplot(fig)

# =========================
# Cluster Analysis Page
# =========================

elif page == "Cluster Analysis":

    st.header("Cluster Analysis")

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(segmented_data["cluster"].unique())
    )

    cluster_data = segmented_data[
        segmented_data["cluster"] == selected_cluster
    ]

    st.subheader(f"Cluster {selected_cluster} Summary")

    st.write("Number of customers in this cluster:", cluster_data.shape[0])

    numeric_cols = cluster_data.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if "cluster" in numeric_cols:
        numeric_cols.remove("cluster")

    st.subheader("Statistical Summary")
    st.dataframe(cluster_data[numeric_cols].describe().T, use_container_width=True)

    st.subheader("Feature Distribution")

    selected_feature = st.selectbox(
        "Select Feature",
        numeric_cols
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cluster_data[selected_feature].dropna(), bins=20)

    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{selected_feature} Distribution in Cluster {selected_cluster}")

    st.pyplot(fig)

# =========================
# Customer Recommendations Page
# =========================

elif page == "Customer Recommendations":

    st.header("Customer Product Recommendations")

    customer_ids = recommendations["CustomerID"].unique()

    selected_customer = st.selectbox(
        "Select Customer ID",
        customer_ids
    )

    customer_rec = recommendations[
        recommendations["CustomerID"] == selected_customer
    ]

    st.subheader(f"Recommendations for Customer ID: {selected_customer}")

    display_columns = [
        "CustomerID",
        "cluster",
        "Rec1_StockCode",
        "Rec1_Description",
        "Rec2_StockCode",
        "Rec2_Description",
        "Rec3_StockCode",
        "Rec3_Description"
    ]

    available_columns = [
        col for col in display_columns if col in customer_rec.columns
    ]

    st.dataframe(customer_rec[available_columns], use_container_width=True)

# =========================
# Raw Data Page
# =========================

elif page == "Raw Data":

    st.header("Raw Output Data")

    data_option = st.radio(
        "Choose Data",
        [
            "Segmented Customer Data",
            "Recommendation Data"
        ]
    )

    if data_option == "Segmented Customer Data":
        st.dataframe(segmented_data, use_container_width=True)

    else:
        st.dataframe(recommendations, use_container_width=True)