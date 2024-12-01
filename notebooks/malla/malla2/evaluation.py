# evaluation.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)
from sklearn.preprocessing import StandardScaler

# Set up logging
logfile = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log(message):
    print(message)
    logging.info(message)

def evaluate_clustering(df, cluster_columns):
    log("Starting clustering evaluation...")
    results = []
    features = df.select_dtypes(include=[np.number]).drop(columns=['label_binary'], errors='ignore')
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    for cluster_col in cluster_columns:
        try:
            log(f"Evaluating clustering for {cluster_col}...")
            labels = df[cluster_col]
            # Calculate evaluation metrics
            silhouette = silhouette_score(features_scaled, labels)
            log(f"Silhouette Score for {cluster_col}: {silhouette:.4f}")
            calinski_harabasz = calinski_harabasz_score(features_scaled, labels)
            log(f"Calinski-Harabasz Index for {cluster_col}: {calinski_harabasz:.4f}")
            davies_bouldin = davies_bouldin_score(features_scaled, labels)
            log(f"Davies-Bouldin Index for {cluster_col}: {davies_bouldin:.4f}")
            ari = adjusted_rand_score(df['label_binary'], labels)
            log(f"Adjusted Rand Index for {cluster_col}: {ari:.4f}")
            ami = adjusted_mutual_info_score(df['label_binary'], labels)
            log(f"Adjusted Mutual Information for {cluster_col}: {ami:.4f}")
            
            
            
            
            # Count CG and OR in each cluster
            cluster_counts = df.groupby(cluster_col)['label'].value_counts().unstack(fill_value=0)
            log(f"Cluster counts for {cluster_col}:\n{cluster_counts}")
            # Add summary to results
            results.append({
                "Cluster_Column": cluster_col,
                "Number_of_Clusters": len(labels.unique()),
                "Silhouette_Score": silhouette,
                "Calinski_Harabasz_Index": calinski_harabasz,
                "Davies_Bouldin_Index": davies_bouldin,
                "Adjusted_Rand_Index": ari,
                "Adjusted_Mutual_Info": ami,
                "CG_Counts": int(cluster_counts.get('CG', 0).sum()),
                "OR_Counts": int(cluster_counts.get('OR', 0).sum()),
                "Cluster_Breakdown": cluster_counts.to_dict()
            })
        except Exception as e:
            log(f"Error evaluating clustering for {cluster_col}: {e}")
    log("Clustering evaluation complete.")
    return pd.DataFrame(results)

def main():
    input_csv = "../../../data/clustered_reviews.csv"
    output_csv = "../../../data/clustering_evaluation_results.csv"
    log(f"Reading clustered data from {input_csv}...")
    df = pd.read_csv(input_csv)
    log("Clustered data loaded successfully.")

    # Identify cluster columns
    cluster_columns = [col for col in df.columns if col.startswith("kmeans_") or col == 'dbscan_cluster']
    if not cluster_columns:
        log("No clustering columns found in the dataset. Exiting.")
        return

    # Evaluate clustering performance
    evaluation_results = evaluate_clustering(df, cluster_columns)

    # Save results to a new CSV
    log(f"Saving evaluation results to {output_csv}...")
    evaluation_results['Cluster_Breakdown'] = evaluation_results['Cluster_Breakdown'].apply(str)
    evaluation_results.to_csv(output_csv, index=False)
    log(f"Evaluation results saved to {output_csv}.")
    log("Evaluation process completed successfully.")

if __name__ == "__main__":
    main()
