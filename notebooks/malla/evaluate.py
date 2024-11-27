import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import logging
from datetime import datetime
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# Set up logging
logfile = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log(message):
    print(message)  # Print to console for real-time feedback
    logging.info(message)  # Write to the log file

def evaluate_clustering(df, cluster_columns):
    """
    Evaluate clustering performance for each k, including counts of CG and OR in each cluster.
    """
    log("Starting clustering evaluation...")
    results = []

    features = df.drop(columns=["label_binary", "category", "label", "text", "SENTIMENT_CATEGORY"] + cluster_columns, errors="ignore")

    for cluster_col in cluster_columns:
        try:
            log(f"Evaluating clustering for {cluster_col}...")
            labels = df[cluster_col]

            # Calculate Silhouette Score
            silhouette = silhouette_score(features, labels)
            log(f"Silhouette Score for {cluster_col}: {silhouette:.4f}")

            # Calculate Calinski-Harabasz Index
            calinski_harabasz = calinski_harabasz_score(features, labels)
            log(f"Calinski-Harabasz Index for {cluster_col}: {calinski_harabasz:.4f}")

            davies_bouldin = davies_bouldin_score(features, labels)
            log(f"Davies-Bouldin Index for {cluster_col}: {davies_bouldin:.4f}")


            # Count CG and OR in each cluster
            cluster_counts = df.groupby(cluster_col)['label'].value_counts().unstack(fill_value=0)
            cg_counts = cluster_counts.get('CG', pd.Series(0, index=cluster_counts.index))
            or_counts = cluster_counts.get('OR', pd.Series(0, index=cluster_counts.index))

            log(f"Cluster counts for {cluster_col}:")
            log(f"\n{cluster_counts}")

            # Add summary to results
            results.append({
                "Cluster_Column": cluster_col,
                "Number_of_Clusters": len(labels.unique()),
                "Silhouette_Score": silhouette,
                "Calinski_Harabasz_Index": calinski_harabasz,
                "Davies_Bouldin_Index": davies_bouldin,
                "CG_Counts": cg_counts.sum(),
                "OR_Counts": or_counts.sum(),
                "Cluster_Breakdown": cluster_counts.to_dict()
            })
        except Exception as e:
            logging.error(f"Error evaluating clustering for {cluster_col}: {e}")

    log("Clustering evaluation complete.")
    return pd.DataFrame(results)

# Script Entry Point
if __name__ == "__main__":
    input_csv = "../data/clustered_reviews.csv"
    output_csv = "../data/clustering_evaluation_results_with_counts.csv"
    log(f"Reading clustered data from {input_csv}...")

    try:
        df = pd.read_csv(input_csv)
        log(f"Clustered data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        log(f"Failed to read input file: {e}")
        exit(1)

    # Identify cluster columns
    cluster_columns = [col for col in df.columns if col.startswith("cluster_")]
    if not cluster_columns:
        log("No clustering columns found in the dataset. Exiting.")
        exit(1)

    # Evaluate clustering performance
    evaluation_results = evaluate_clustering(df, cluster_columns)

    # Save results to a new CSV
    log(f"Saving evaluation results to {output_csv}...")
    try:
        # Flattening cluster breakdown for easier reading in the CSV
        evaluation_results['Cluster_Breakdown'] = evaluation_results['Cluster_Breakdown'].apply(lambda x: str(x))
        evaluation_results.to_csv(output_csv, index=False)
        log(f"Evaluation results saved to {output_csv}.")
    except Exception as e:
        log(f"Failed to save evaluation results: {e}")
        exit(1)

    log("Script completed successfully.")
