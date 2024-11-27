import pandas as pd
from sklearn.cluster import KMeans
import logging
from datetime import datetime

# Set up logging
logfile = f"clustering_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log(message):
    print(message)  # Print to console for real-time feedback
    logging.info(message)  # Write to the log file

def perform_clustering(df, cluster_range):
    """
    Perform clustering on the given DataFrame and return the results.
    """
    log("Starting clustering process...")
    results = []

    for n_clusters in cluster_range:
        log(f"Clustering with {n_clusters} clusters...")
        features = df.drop(columns=["label_binary", "category", "label", "text", "SENTIMENT_CATEGORY"], errors="ignore")
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            df[f'cluster_{n_clusters}'] = cluster_labels

            # Summarize cluster label counts
            label_summary = df[f'cluster_{n_clusters}'].value_counts().to_frame(name='count')
            label_summary.index.name = 'cluster_label'
            log(f"Clustering with {n_clusters} clusters completed successfully.")
            results.append((n_clusters, label_summary))
        except Exception as e:
            logging.error(f"Error clustering with {n_clusters} clusters: {e}")

    log("Clustering process complete.")
    return df, results

# Script Entry Point
if __name__ == "__main__":
    input_csv = "../data/processed_reviews.csv"
    output_csv = "../data/clustered_reviews.csv"
    log(f"Reading data from {input_csv}...")

    try:
        df = pd.read_csv(input_csv)
        log(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        log(f"Failed to read input file: {e}")
        exit(1)

    # Define cluster range
    cluster_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Perform clustering
    clustered_df, clustering_results = perform_clustering(df, cluster_range)

    # Save the clustered DataFrame to CSV
    log(f"Saving clustered data to {output_csv}...")
    try:
        clustered_df.to_csv(output_csv, index=False)
        log(f"Clustered data saved to {output_csv}.")
    except Exception as e:
        log(f"Failed to save clustered data: {e}")
        exit(1)

    log("Script completed successfully.")
