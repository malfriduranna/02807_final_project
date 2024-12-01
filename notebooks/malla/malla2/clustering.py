# clustering.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Set up logging
logfile = f"clustering_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log(message):
    print(message)
    logging.info(message)

def reduce_dimensions(df, method='pca'):
    log(f"Reducing dimensions using {method.upper()}...")
    features = df.select_dtypes(include=[np.number]).drop(columns=['label_binary'], errors='ignore')
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    if method == 'pca':
        pca = PCA(n_components=0.95, random_state=42)
        reduced_features = pca.fit_transform(features_scaled)
        log(f"PCA reduced dimensions to {reduced_features.shape[1]} components.")
    elif method == 'tsne':
        tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
        reduced_features = tsne.fit_transform(features_scaled)
        log("t-SNE dimensionality reduction complete.")
    else:
        log(f"Unknown dimensionality reduction method: {method}")
        return None
    return reduced_features

def perform_kmeans_clustering(df, cluster_range):
    log("Starting KMeans clustering process...")
    features = df.select_dtypes(include=[np.number]).drop(columns=['label_binary'], errors='ignore')
    for n_clusters in cluster_range:
        log(f"Clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        df[f'kmeans_{n_clusters}'] = cluster_labels
        log(f"KMeans clustering with {n_clusters} clusters completed.")
    log("KMeans clustering process complete.")
    return df

def perform_dbscan_clustering(df):
    log("Performing DBSCAN clustering...")
    reduced_features = reduce_dimensions(df, method='pca')
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(reduced_features)
    df['dbscan_cluster'] = cluster_labels
    log("DBSCAN clustering complete.")
    return df

def main():
    input_csv = "../../../data/processed_reviews.csv"
    output_csv = "../../../data/clustered_reviews.csv"
    log(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    log("Data loaded successfully.")

    # Perform KMeans clustering
    cluster_range = range(2, 11)  # Clusters from 2 to 10
    df = perform_kmeans_clustering(df, cluster_range)

    # Optionally perform DBSCAN clustering
    df = perform_dbscan_clustering(df)

    log(f"Saving clustered data to {output_csv}...")
    df.to_csv(output_csv, index=False)
    log(f"Clustered data saved to {output_csv}.")
    log("Clustering process completed successfully.")

if __name__ == "__main__":
    main()
