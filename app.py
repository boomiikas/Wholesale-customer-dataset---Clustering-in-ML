import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Clustering Analysis Dashboard")

# 1️⃣ Upload dataset
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select features to scale & cluster
    features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    X = df[features]

    # 2️⃣ Scale features using RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.subheader("Scaled Features (first 5 rows)")
    st.dataframe(pd.DataFrame(X_scaled, columns=features).head())

    # 3️⃣ Sidebar - algorithm & parameters
    st.sidebar.header("Clustering Options")
    algo = st.sidebar.selectbox("Select Algorithm", ["KMeans", "Hierarchical", "DBSCAN"])
    
    if algo in ["KMeans", "Hierarchical"]:
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    if algo == "DBSCAN":
        eps = st.sidebar.slider("eps", 0.1, 10.0, 0.5)
        min_samples = st.sidebar.slider("min_samples", 1, 20, 5)

    if st.button("Run Clustering"):
        # 4️⃣ Fit selected clustering algorithm
        if algo == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif algo == "Hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            model = DBSCAN(eps=eps, min_samples=min_samples)
        
        labels = model.fit_predict(X_scaled)

        # 5️⃣ Compute silhouette score
        if len(set(labels)) > 1 and len(set(labels)) != 1:
            sil = silhouette_score(X_scaled, labels)
        else:
            sil = -1  # silhouette not defined for single cluster
        st.write(f"**Silhouette Score:** {sil:.3f}")

        # 6️⃣ Show cluster sizes & outliers
        cluster_sizes = pd.Series(labels).value_counts().to_dict()
        st.write("**Cluster Sizes:**", cluster_sizes)

        if algo == "DBSCAN":
            n_outliers = sum(labels == -1)
            st.write(f"**Number of Outliers:** {n_outliers}")

        # 7️⃣ Add labels to dataset & allow download
        df['Cluster'] = labels
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV with Cluster Labels", csv, "clustered_data.csv", "text/csv")

        # 8️⃣ Visualize clusters using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette="tab10")
        plt.title(f"{algo} Clustering (PCA 2D Projection)")
        st.pyplot(plt)
