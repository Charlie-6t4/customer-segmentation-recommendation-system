# Customer Segmentation and Product Recommendation System

## Project Overview

This project focuses on customer segmentation for an online retail business using transactional data. The objective is to transform raw transaction-level data into a customer-level analytical dataset, segment customers using machine learning, and generate personalized product recommendations.

The project includes data cleaning, feature engineering, outlier detection, dimensionality reduction, K-Means clustering, cluster profiling, and an interactive Streamlit dashboard.

---

## Business Problem

Online retail businesses serve customers with different purchasing behaviors. Treating all customers the same can reduce marketing effectiveness.

This project helps to:

- Identify different customer segments.
- Understand customer purchasing behavior.
- Support targeted marketing.
- Recommend relevant products to customers.
- Present insights through an interactive dashboard.

---

## Tools and Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PCA
- K-Means Clustering
- Isolation Forest
- Plotly
- Streamlit

---

## Project Workflow

1. Data Cleaning
2. Feature Engineering
3. Outlier Detection using Isolation Forest
4. Feature Scaling
5. Dimensionality Reduction using PCA
6. Customer Segmentation using K-Means
7. Cluster Evaluation
8. Product Recommendation System
9. Interactive Streamlit Dashboard

---

## Result Preview

<img width="1280" height="696" alt="Figure_4" src="https://github.com/user-attachments/assets/a806bfb1-9365-49b1-9808-ef4e8f8c360d" />

<img width="1200" height="500" alt="Figure_6" src="https://github.com/user-attachments/assets/ada7c31c-b47a-422a-b666-3e2d59f6d329" />

<img width="1280" height="696" alt="Figure_5" src="https://github.com/user-attachments/assets/d2b2b79e-4039-4a33-b427-0b3c85a6ba65" />

<img width="1200" height="500" alt="Figure_7" src="https://github.com/user-attachments/assets/c22996da-4709-47ea-9935-d94364b2a680" />

<img width="1000" height="400" alt="Figure_8" src="https://github.com/user-attachments/assets/fe27ef8b-da8c-4215-b937-9f113edeaee2" />

---

## Dashboard Features

- Customer overview
- Cluster distribution
- Cluster-wise analysis
- Feature distribution
- Product recommendations by customer
- Raw output data view

---

## Dashboard Preview

<img width="1911" height="1135" alt="dashboard_preview1" src="https://github.com/user-attachments/assets/76047369-e14d-4f6b-9c2b-0da839a8eff5" />

<img width="1894" height="1116" alt="dashboard_preview2" src="https://github.com/user-attachments/assets/42450d29-0da0-4882-bf08-84ed93826945" />

---

## Project Structure

```text
customer_segmentation_project
│
├── app.py
├── Customer_Segmentation.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data
│   ├── customer_segmented_data.csv
│   └── customer_recommendations.csv
