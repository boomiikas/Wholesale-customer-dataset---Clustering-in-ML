# Wholesale Customers Dataset Analysis  

This project explores the **Wholesale Customers Dataset** to understand customer purchasing behavior and patterns across different product categories. The analysis is performed in a **Jupyter Notebook** and an interactive **Gradio (Hugging Face)** app.  

🔗 **Live App:** [Wholesale Customer Clustering](https://huggingface.co/spaces/boomiikas/Wholesale-customer-clustering)  

---

## App Screenshot
<img width="1370" height="880" alt="image" src="https://github.com/user-attachments/assets/c7052f76-6a48-400d-97da-a89a24bce880" />

---
## Dataset  

The dataset contains annual spending by clients of a wholesale distributor across various product categories.  

**Columns:**  
- `Channel` → Type of distribution channel (e.g., Horeca, Retail)  
- `Region` → Customer’s geographical region  
- `Fresh` → Spending on fresh products  
- `Milk` → Spending on milk products  
- `Grocery` → Spending on grocery items  
- `Frozen` → Spending on frozen products  
- `Detergents_Paper` → Spending on detergents and paper products  
- `Delicassen` → Spending on delicatessen items  

---

## Key Steps in Analysis  

- Data loading and basic exploration with **Pandas**  
- Checking unique values in `Channel` and `Region`  
- Summary statistics of spending patterns  
- Visualization of distributions using **Matplotlib** and **Seaborn**  
- Correlation analysis between product categories  
- Insights into customer segmentation and behavior  

---

## Tech Stack  

- **Python 3**  
- **Pandas** & **NumPy** → Data manipulation  
- **Matplotlib** & **Seaborn** → Data visualization  
- **Streamlit** → Interactive web application  
- **Jupyter Notebook** → EDA and documentation  

---

## How to Run Locally  

1. Clone this repository
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>

   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```

---

# Clustering Analysis  

To identify customer segments, three clustering algorithms were applied on the **Wholesale Customers Dataset** after scaling the features:  

- **KMeans Clustering**  
- **Hierarchical Clustering (Agglomerative)**  
- **DBSCAN**  

The performance was evaluated using the **Silhouette Score**.  

---

## Results  

| Algorithm    | Silhouette | Clusters | Outliers | Cluster Sizes              |
|--------------|------------|----------|----------|----------------------------|
| KMeans       | 0.244      | 3        | 0        | {0: 209, 1: 149, 2: 82}    |
| Hierarchical | 0.243      | 3        | 0        | {0: 265, 1: 126, 2: 49}    |
| DBSCAN       | -1.000     | 0        | 18       | {0: 422}                   |

---

## Visualizations  

- **KMeans Clustering**  
<img width="546" height="433" alt="image" src="https://github.com/user-attachments/assets/5d6af0a8-a7c3-44ab-8505-71baaf420778" />


- **Hierarchical Clustering**
<img width="546" height="433" alt="image" src="https://github.com/user-attachments/assets/4e6bb093-fb61-4007-9b93-2bdf51700204" />


- **DBSCAN Clustering**
<img width="546" height="433" alt="image" src="https://github.com/user-attachments/assets/b1ea73b2-a2e6-4201-9617-36e4bf224a61" />


---

