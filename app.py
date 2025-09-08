import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -------------------
# Load dataset
# -------------------
df = pd.read_csv("Wholesale customers data.csv")

# Select features
features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
X = df[features]

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)

# Add PCA + cluster labels back to df
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Cluster"] = labels

# -------------------
# Give each cluster a meaningful name
# -------------------
cluster_meanings = {
    0: "Cluster 0 - Fresh & Grocery Buyers",
    1: "Cluster 1 - Milk & Detergents Buyers",
    2: "Cluster 2 - Frozen & Delicassen Buyers"
}

# -------------------
# Prediction + Plot function
# -------------------
def predict_cluster(Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen):
    # Scale input
    input_data = np.array([[Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen]])
    input_scaled = scaler.transform(input_data)
    
    # PCA transform
    input_pca = pca.transform(input_scaled)
    
    # Predict cluster
    cluster = kmeans.predict(input_pca)[0]
    cluster_name = cluster_meanings.get(cluster, f"Cluster {cluster}")
    
    # Plot clusters + input point
    plt.figure(figsize=(7,5))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=df_pca, palette="Set2", s=60)
    plt.scatter(input_pca[0,0], input_pca[0,1], color="red", s=120, edgecolor="black", marker="X", label="Your Input")
    plt.legend()
    plt.title(cluster_name)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    # Save plot
    plt.tight_layout()
    plt.savefig("cluster_plot.png")
    plt.close()
    
    return cluster_name, "cluster_plot.png"

# -------------------
# Login check
# -------------------
USERNAME = "admin"
PASSWORD = "1234"

def login(username, password):
    if username == USERNAME and password == PASSWORD:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True, value="❌ Invalid login. Try again.")

# -------------------
# Gradio Blocks Layout
# -------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔐 Wholesale Customers Clustering Login")
    
    with gr.Row():
        user = gr.Textbox(label="Username")
        pwd = gr.Textbox(label="Password", type="password")
    
    login_btn = gr.Button("Login")
    error_msg = gr.Textbox(label="Message", visible=False)
    
    # Hidden clustering interface
    with gr.Group(visible=False) as clustering_ui:
        gr.Markdown("### 🛒 Wholesale Customers Clustering (KMeans + PCA)")
        inputs = [
            gr.Number(label="Fresh"),
            gr.Number(label="Milk"),
            gr.Number(label="Grocery"),
            gr.Number(label="Frozen"),
            gr.Number(label="Detergents_Paper"),
            gr.Number(label="Delicassen")
        ]
        outputs = [
            gr.Textbox(label="Predicted Cluster"),
            gr.Image(type="filepath", label="Cluster Visualization")
        ]
        gr.Interface(fn=predict_cluster, inputs=inputs, outputs=outputs).render()
    
    login_btn.click(fn=login, inputs=[user, pwd], outputs=[clustering_ui, error_msg])

if __name__ == "__main__":
    demo.launch()
