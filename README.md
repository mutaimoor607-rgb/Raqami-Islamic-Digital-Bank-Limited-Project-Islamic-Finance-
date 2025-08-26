# Raqami-Islamic-Digital-Bank-Limited-Project-Islamic-Finance-
Survey on Retailers Preferences in Raqami Islamic Digital Bank Limited (Islamic Finance)
<br>
Author - Muhammad Taimoor
<br>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Sample simulated data (you should replace this with actual data from your Excel)
data = pd.DataFrame({
    'SmartphoneUse': [1 if i < 103 else 0 for i in range(115)],
    'Uses_DFS': [1 if i < 81 else 0 for i in range(115)],
    'Has_Savings': [1 if i < 23 else 0 for i in range(115)],
    'BankFinancingNeed': [2 if i < 45 else 1 if i < 97 else 0 for i in range(115)],
    'Uses_IslamicBank': [1 if i < 31 else 0 for i in range(115)],
    'Accepts_Digital': [2 if i < 42 else 1 if i < 103 else 0 for i in range(115)],
    'Uses_BankingApp': [1 if i < 64 else 0 for i in range(115)],
    'RecordKeeping': [1 if i < 71 else 0 for i in range(115)]
})

# Step 1: Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 2: Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Step 3: Add clusters back to the data
data['Cluster'] = clusters

# Step 4: Dimensionality reduction using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Step 5: Plot the labeled cluster graph
plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', s=60)

# Plot cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.6, marker='X', label='Cluster Centers')

# Label cluster centers
for i, (x, y) in enumerate(centers_pca):
    plt.text(x, y, f'Cluster {i}', fontsize=12, weight='bold', ha='center', va='center', color='black', backgroundcolor='white')

plt.title('Labeled K-Means Clustering of Kiryana Store Behavior')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
