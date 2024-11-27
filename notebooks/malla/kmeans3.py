import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from umap import UMAP
import logging
from datetime import datetime

# Set up logging
logfile = f"clustering_pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log(message):
    print(message)
    logging.info(message)

# === STEP 1: Load and Preprocess Data ===
def load_and_preprocess_data(file_path):
    log("Loading data...")
    df = pd.read_csv(file_path)
    log("Data loaded successfully.")

    log("Preprocessing data...")
    df.rename(columns={'text_': 'text'}, inplace=True)
    df['text'] = df['text'].fillna("").astype(str)
    df['label_binary'] = df['label'].replace({'CG': 1, 'OR': 0})
    df.drop_duplicates(inplace=True)
    log("Data preprocessing complete.")
    return df

# === STEP 2: Feature Engineering ===
def add_tfidf_features(df):
    log("Adding TF-IDF features...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.index = df.index
    df = pd.concat([df, tfidf_df], axis=1)
    log("TF-IDF features added.")
    return df

def add_custom_features(df):
    log("Adding custom features...")
    from textstat import flesch_reading_ease
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    sid = SentimentIntensityAnalyzer()

    df['READABILITY_FRE'] = df['text'].apply(flesch_reading_ease)
    df['SENTIMENT_SCORE'] = df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['NUM_NOUNS'] = df['text'].apply(lambda x: sum(1 for _, tag in pos_tag(word_tokenize(x)) if tag.startswith('NN')))
    df['NUM_VERBS'] = df['text'].apply(lambda x: sum(1 for _, tag in pos_tag(word_tokenize(x)) if tag.startswith('VB')))
    df['REVIEW_LENGTH'] = df['text'].apply(len)
    log("Custom features added.")
    return df

# === STEP 3: Feature Importance Analysis ===
def analyze_feature_importance(df):
    log("Analyzing feature importance...")
    features = df.select_dtypes(include=[np.number]).drop(columns=['label_binary'], errors='ignore')
    target = df['label_binary']

    model = RandomForestClassifier(random_state=42)
    model.fit(features, target)
    feature_importance = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    log("Feature importance analysis complete.")
    return feature_importance

def select_top_features(df, n_top_features=20):
    feature_importance = analyze_feature_importance(df)
    top_features = feature_importance['Feature'].head(n_top_features).tolist()
    log(f"Selected top {n_top_features} features: {top_features}")
    return df[top_features + ['label_binary']]

# === STEP 4: Dimensionality Reduction ===
def reduce_features_with_pca(df):
    log("Reducing features with PCA...")
    features = df.drop(columns=['label_binary'], errors='ignore')
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=0.95)  # Retain 95% variance
    reduced_features = pca.fit_transform(features_scaled)
    log(f"PCA reduced dimensions to {reduced_features.shape[1]}")
    return reduced_features

# === STEP 5: Clustering ===
def perform_kmeans_clustering(features, cluster_range, df):
    log("Performing KMeans clustering...")
    results = []
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(features)
        df[f'cluster_{k}'] = labels
        silhouette = silhouette_score(features, labels)
        log(f"k={k}, Silhouette Score={silhouette:.4f}")
        results.append((k, labels, silhouette))
        
        # Save CG/OR counts per cluster
        cluster_counts = df.groupby(f'cluster_{k}')['label_binary'].value_counts().unstack(fill_value=0)
        cluster_counts['Total'] = cluster_counts.sum(axis=1)
        cluster_counts['CG_Percentage'] = cluster_counts[1] / cluster_counts['Total']
        cluster_counts['OR_Percentage'] = cluster_counts[0] / cluster_counts['Total']
        cluster_counts.to_csv(f"cluster_{k}_distribution.csv")
        log(f"Cluster distribution for k={k} saved.")
        
    return results, df

# === STEP 6: Main Pipeline ===
def main():
    # Load data
    file_path = "../../data/akereviewsdataset.csv"
    df = load_and_preprocess_data(file_path)

    # Add features
    df = add_tfidf_features(df)
    df = add_custom_features(df)

    # Select important features
    feature_importance = analyze_feature_importance(df)
    top_features_df = select_top_features(df)

    # Reduce features with PCA
    reduced_features = reduce_features_with_pca(top_features_df)

    # Perform clustering
    cluster_range = range(2, 11)
    clustering_results, df_with_clusters = perform_kmeans_clustering(reduced_features, cluster_range, df)

    # Save full dataset with clusters
    df_with_clusters.to_csv("clustering_results.csv", index=False)
    log("Clustering results saved to clustering_results.csv.")

    # Choose best k based on silhouette score
    best_k, best_labels, best_score = max(clustering_results, key=lambda x: x[2])
    log(f"Best k={best_k}, Silhouette Score={best_score:.4f}")

if __name__ == "__main__":
    main()
