from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Retail Transaction Classifier",
    page_icon="🛒",
    layout="centered",
)

st.title("🛒 Retail Transaction Segmentation & Anomaly Tool")
st.markdown(
    "Classify orders into customer purchase tiers and flag wholesale/pricing"
    " anomalies using **K-Means**."
)

# 1. Resolve path and load package
BASE_DIR = Path(__file__).parent
pkg_file = BASE_DIR / "retail_cluster_package.pkl"

if not pkg_file.exists():
  st.error(
      "Missing 'retail_cluster_package.pkl' in the repository root directory."
  )
  st.stop()

pkg = joblib.load(pkg_file)
kmeans = pkg["kmeans_model"]
scaler = pkg["scaler"]
cluster_names = pkg.get(
    "cluster_names",
    {
        2: "Budget / Low-Cost Single Items",
        0: "Premium Specialty Items (High unit price, low qty)",
        1: "Wholesale / Bulk Orders (High volume)",
    },
)

# 2. Input Form
with st.form("transaction_form"):
  col1, col2 = st.columns(2)

  with col1:
    quantity = st.number_input(
        "Item Quantity:",
        min_value=1,
        max_value=10000,
        value=6,
        step=1,
        help="Total units of this item purchased.",
    )

  with col2:
    unit_price = st.number_input(
        "Unit Price (£):",
        min_value=0.01,
        max_value=2000.0,
        value=2.50,
        step=0.5,
        help="Price per individual unit.",
    )

  submit = st.form_submit_button(
      "Classify Transaction", type="primary", use_container_width=True
  )

# 3. Predict & Display
if submit:
  total_spend = quantity * unit_price

  # Build DataFrame with exact feature names used in training
  raw_df = pd.DataFrame([{
      "Quantity": quantity,
      "UnitPrice": unit_price,
      "Total_Spend": total_spend,
  }])

  # Log transform and scale
  scaled_features = scaler.transform(np.log1p(raw_df))

  # Predict cluster ID
  cluster_id = int(kmeans.predict(scaled_features)[0])
  cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

  # Distance to centroid (anomaly detection check)
  centroid = kmeans.cluster_centers_[cluster_id]
  dist_to_centroid = float(np.linalg.norm(scaled_features - centroid))

  st.write("---")
  st.subheader("Order Classification Result")

  col_m1, col_m2, col_m3 = st.columns(3)
  with col_m1:
    st.metric("Total Order Spend", f"£{total_spend:.2f}")
  with col_m2:
    st.metric("Assigned Tier", f"Cluster {cluster_id}")
  with col_m3:
    st.metric("Centroid Distance", f"{dist_to_centroid:.2f}")

  # Segment profile badge
  st.success(f"📦 **Segment:** {cluster_label}")

  # Density/Outlier Alert
  if dist_to_centroid > 2.5:
    st.warning(
        "⚠️ **Outlier Alert:** This order exhibits unusual volume/price ratios"
        " compared to standard retail density. Review for wholesale discount"
        " or manual verification."
    )
  else:
    st.info(
        "ℹ️ Standard retail density verified. Fits normal shopping behavior."
    )
