# Online Retail Transaction Clustering & Anomaly Detection

An unsupervised machine learning system for e-commerce transaction segmentation and outlier detection using **K-Means** and **DBSCAN**.

## Overview
- **Dataset:** UK Online Retail (500k+ transactional records)
- **Feature Engineering:** Log-transformed `Quantity`, `UnitPrice`, and derived `Total_Spend`
- **Algorithms Evaluated:**
  - **K-Means ($k=3$):** Silhouette Score ~0.42, Davies-Bouldin Index ~0.83 (Optimal for segmenting transaction tiers)
  - **DBSCAN (`eps=0.5`):** Silhouette Score ~0.55 (Used for density-based anomaly and wholesale outlier detection)

## Deployment
Live interactive deployment via Streamlit Cloud. Input an order quantity and unit price to classify transactions in real time.
