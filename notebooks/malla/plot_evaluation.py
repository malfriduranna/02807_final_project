import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# File paths
input_csv = "../../data/clustered_reviews.csv"
output_plot_pca = "pca_cluster_visualization.png"
output_plot_distribution = "cluster_distribution.png"

# Load the data
df = pd.read_csv(input_csv)
print("Data loaded successfully.")

# Check available columns
print("Available columns:", df.columns)

# Specify the correct cluster column
best_cluster_col = 'cluster_2'  # Replace with 'cluster_2' if that column exists after clustering

# Ensure the cluster column exists
if best_cluster_col not in df.columns:
    raise ValueError(f"Column '{best_cluster_col}' not found in the dataset. Check your clustering script output.")

# Select only numerical features for PCA
features = df.select_dtypes(include=['float64', 'int64']).drop(columns=[best_cluster_col, 'label_binary'], errors='ignore')

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Plot PCA visualization
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='PCA1', 
    y='PCA2', 
    hue=best_cluster_col, 
    palette='tab10', 
    data=df, 
    s=50, 
    alpha=0.7
)
plt.title(f'PCA Visualization of Clusters ({best_cluster_col})', fontsize=14)
plt.xlabel('PCA1', fontsize=12)
plt.ylabel('PCA2', fontsize=12)
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig(output_plot_pca)
plt.show()

# Plot cluster distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=best_cluster_col, hue='label', data=df, palette='Set2')
plt.title(f'Cluster Distribution for {best_cluster_col}', fontsize=14)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Label')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(output_plot_distribution)
plt.show()

print("Plots saved successfully.")
