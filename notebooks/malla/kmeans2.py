import pandas as pd
from textblob import TextBlob
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tree import Tree
from textstat import flesch_reading_ease
import string
import warnings
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Set up logging
# Logging setup
logfile = f"named_entities_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Script started")

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function for logging both to console and file
def log(message):
    print(message)
    logging.info(message)

log("Loading data...")
file_path = "../data/fake reviews dataset.csv"
df = pd.read_csv(file_path)
log("Data loaded successfully.")

log("Preparing data...")
df.rename(columns={'text_': 'text'}, inplace=True)
df['category'] = df['category'].str.replace('_5', '', regex=False)
df.drop_duplicates(inplace=True)
df['label_binary'] = df['label'].replace({'CG': 1, 'OR': 0})
df['text'] = df['text'].fillna("").astype(str)

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    words = [word for word in words if word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return ' '.join(words)

log("Preprocessing text...")
df['text'] = df['text'].apply(preprocess_text)
log("Text preprocessing complete.")

def add_tfidf_features(df):
    log("Adding TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.index = df.index  # Ensure indices align
    df = pd.concat([df, tfidf_df], axis=1)
    log("TF-IDF features added.")
    return df

from sklearn.preprocessing import StandardScaler

def scale_features(df):
    log("Scaling features...")
    # Identify features to scale
    features_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target and non-feature columns
    features_to_scale = [col for col in features_to_scale if col not in ['label_binary']]
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    log("Feature scaling complete.")
    return df


def add_features(df):
    log("Adding readability score...")
    df['READABILITY_FRE'] = df['text'].apply(flesch_reading_ease)
    log("Readability score added.")

    log("Adding sentiment score...")
    sid = SentimentIntensityAnalyzer()
    df['SENTIMENT_SCORE'] = df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['SENTIMENT_CATEGORY'] = df['SENTIMENT_SCORE'].apply(lambda x: 'positive' if x > 0 else 'negative')
    log("Sentiment score and category added.")

    log("Adding POS tags...")
    df['NUM_NOUNS'] = df['text'].apply(lambda x: sum(1 for _, tag in pos_tag(word_tokenize(x)) if tag.startswith('NN')))
    df['NUM_VERBS'] = df['text'].apply(lambda x: sum(1 for _, tag in pos_tag(word_tokenize(x)) if tag.startswith('VB')))
    df['NUM_ADJECTIVES'] = df['text'].apply(lambda x: sum(1 for _, tag in pos_tag(word_tokenize(x)) if tag.startswith('JJ')))
    df['NUM_ADVERBS'] = df['text'].apply(lambda x: sum(1 for _, tag in pos_tag(word_tokenize(x)) if tag.startswith('RB')))
    log("POS tags added.")

    log("Calculating review length...")
    df['REVIEW_LENGTH'] = df['text'].apply(len)
    log("Review length calculated.")

    log("Calculating average word length...")
    df['AVG_WORD_LENGTH'] = df['text'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if len(x.split()) > 0 else 0)
    log("Average word length calculated.")

log("Adding features to the dataset...")
add_features(df)
log("Feature addition complete.")

def perform_clustering(df, cluster_range):
    log("Starting clustering process...")
    results = []
    for n_clusters in cluster_range:
        log(f"Clustering with {n_clusters} clusters...")
        features = df.drop(columns=["label_binary", "category", "label", "text", "SENTIMENT_CATEGORY"])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        df['cluster_label'] = cluster_labels
        cluster_summary = df.groupby('cluster_label')['label_binary'].value_counts(normalize=True).unstack()
        log(f"Clustering with {n_clusters} clusters completed.")
        results.append((n_clusters, cluster_summary))
    log("Clustering process complete.")
    return results

def identify_distinctive_words(df):
    log("Identifying distinctive words between CG and OR reviews...")

    # Separate CG and OR reviews
    cg_reviews = df[df['label'] == 'CG']['text']
    or_reviews = df[df['label'] == 'OR']['text']

    # Vectorize the text data
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    cg_matrix = vectorizer.fit_transform(cg_reviews)
    or_matrix = vectorizer.transform(or_reviews)
    vocab = vectorizer.get_feature_names_out()

    # Sum word occurrences
    cg_word_counts = cg_matrix.sum(axis=0).A1
    or_word_counts = or_matrix.sum(axis=0).A1

    # Create DataFrames
    cg_freq = pd.DataFrame({'word': vocab, 'cg_count': cg_word_counts})
    or_freq = pd.DataFrame({'word': vocab, 'or_count': or_word_counts})

    # Merge DataFrames
    word_freq = cg_freq.merge(or_freq, on='word')
    word_freq['total'] = word_freq['cg_count'] + word_freq['or_count']

    # Calculate ratios
    word_freq['cg_ratio'] = word_freq['cg_count'] / word_freq['total']
    word_freq['or_ratio'] = word_freq['or_count'] / word_freq['total']

    # Identify distinctive words
    significant_words = word_freq[(word_freq['cg_ratio'] > 0.6) | (word_freq['or_ratio'] > 0.6)]
    log("Distinctive words identified.")
    return significant_words.sort_values(by='total', ascending=False)
# def visualize_word_usage(word_freq):
#     # Top words in CG reviews
#     cg_top_words = word_freq[word_freq['cg_ratio'] > 0.6].head(20)
#     or_top_words = word_freq[word_freq['or_ratio'] > 0.6].head(20)

#     # Word clouds

#     cg_wordcloud = WordCloud(width=800, height=400).generate(' '.join(cg_top_words['word']))
#     or_wordcloud = WordCloud(width=800, height=400).generate(' '.join(or_top_words['word']))

#     # Plotting
#     plt.figure(figsize=(15, 7))
#     plt.subplot(1, 2, 1)
#     plt.imshow(cg_wordcloud, interpolation='bilinear')
#     plt.title('Top Words in CG Reviews')
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.imshow(or_wordcloud, interpolation='bilinear')
#     plt.title('Top Words in OR Reviews')
#     plt.axis('off')

#     plt.show()
# def add_distinctive_word_features(df, significant_words):
#     log("Adding features based on distinctive words...")
#     top_cg_words = significant_words[significant_words['cg_ratio'] > 0.6]['word'].tolist()
#     top_or_words = significant_words[significant_words['or_ratio'] > 0.6]['word'].tolist()

#     for word in top_cg_words:
#         df[f'word_cg_{word}'] = df['text'].apply(lambda x: 1 if word in x else 0)

#     for word in top_or_words:
#         df[f'word_or_{word}'] = df['text'].apply(lambda x: 1 if word in x else 0)

#     log("Distinctive word features added.")
#     return df
def add_distinctive_word_features(df, significant_words):
    log("Adding features based on distinctive words...")
    top_cg_words = significant_words[significant_words['cg_ratio'] > 0.6]['word'].tolist()
    top_or_words = significant_words[significant_words['or_ratio'] > 0.6]['word'].tolist()

    # Initialize a dictionary to store new columns
    new_columns = {}

    # Create features for CG words
    for word in top_cg_words:
        new_columns[f'word_cg_{word}'] = df['text'].apply(lambda x: 1 if word in x else 0)

    # Create features for OR words
    for word in top_or_words:
        new_columns[f'word_or_{word}'] = df['text'].apply(lambda x: 1 if word in x else 0)

    # Concatenate new columns with the original DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    log("Distinctive word features added.")
    return df



# Add TF-IDF features
df = add_tfidf_features(df)
significant_words = identify_distinctive_words(df)
# visualize_word_usage(significant_words)
df = add_distinctive_word_features(df, significant_words)
# Scale the features
df = scale_features(df)

cluster_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
log("Performing clustering...")
cluster_results = perform_clustering(df, cluster_range)

log("Saving results to CSV...")
output_path = "../data/processed_reviews.csv"
df.to_csv(output_path, index=False)
log(f"Results saved to {output_path}.")
log("Script completed successfully.")
