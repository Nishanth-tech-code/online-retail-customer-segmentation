from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Retail Transaction Clustering",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 E-Commerce Transaction Clustering")
st.markdown("Classify orders into customer purchase tiers and flag anomalies using **K-Means**.")

# 1. Path resolution
BASE_DIR = Path(__file__).parent
pkg_file = BASE_DIR / "retail_cluster_package.pkl"

if not pkg_file.exists():
    st.error("Missing 'retail_cluster_package.pkl' in the repository root directory.")
    st.stop()

pkg = joblib.load(pkg_file)
kmeans = pkg["kmeans_model"]
scaler = pkg["scaler"]
cluster_names = pkg.get("cluster_names", {})

# 2. Input Form
with st.form("cluster_form"):
    col1, col2 = st.columns(2)
    with col1:
        qty = st.number_input("Item Quantity:", min_value=1, max_value=10000, value=6, step=1)
    with col2:
        price = st.number_input("Unit Price (£):", min_value=0.01, max_value=2000.0, value=3.50, step=0.5)

    submit = st.form_submit_button("Classify Order", type="primary", use_container_width=True)

# 3. Predict & Display
if submit:
    spend = qty * price
    input_df = pd.DataFrame([{'Quantity': qty, 'UnitPrice': price, 'Total_Spend': spend}])
    
    scaled_vec = scaler.transform(np.log1p(input_df))
    cluster_id = int(kmeans.predict(scaled_vec)[0])
    label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
    
    # Distance to centroid for anomaly detection
    centroid = kmeans.cluster_centers_[cluster_id]
    dist = float(np.linalg.norm(scaled_vec - centroid))

    st.write("---")
    st.subheader("Classification Outcome")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Spend", f"£{spend:.2f}")
    with col_b:
        st.metric("Assigned Tier", f"Cluster #{cluster_id}")
    with col_c:
        st.metric("Centroid Distance", f"{dist:.2f}")

    st.success(f"📦 **Segment Profile:** {label}")
    
    if dist > 2.5:
        st.warning("⚠️ **Outlier Notice:** This order deviates significantly from standard density clusters (wholesale bulk or high-value outlier).")
    else:
        st.info("ℹ️ Standard retail density verified.")
