# data_preprocessing.py

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions
from textstat import flesch_reading_ease
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Ensure necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

# Set up logging
logfile = f"data_preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log(message):
    print(message)
    logging.info(message)

def preprocess_text(text):
    # Expand contractions
    text = contractions.fix(text)
    # Lowercase the text
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def add_features(df):
    log("Adding readability score...")
    df['READABILITY_FRE'] = df['text'].apply(
        lambda x: flesch_reading_ease(x) if x.strip() != '' else 0
    )
    log("Readability score added.")

    log("Adding sentiment score...")
    sid = SentimentIntensityAnalyzer()
    df['SENTIMENT_SCORE'] = df['text'].apply(
        lambda x: sid.polarity_scores(x)['compound'] if x.strip() != '' else 0
    )
    df['SENTIMENT_CATEGORY'] = df['SENTIMENT_SCORE'].apply(
        lambda x: 'positive' if x > 0 else 'negative'
    )
    log("Sentiment score and category added.")

    log("Calculating POS tags...")
    df['NUM_NOUNS'] = df['text'].apply(
        lambda x: sum(1 for word, tag in nltk.pos_tag(word_tokenize(x)) if tag.startswith('NN')) if x.strip() != '' else 0
    )
    df['NUM_VERBS'] = df['text'].apply(
        lambda x: sum(1 for word, tag in nltk.pos_tag(word_tokenize(x)) if tag.startswith('VB')) if x.strip() != '' else 0
    )
    df['NUM_ADJECTIVES'] = df['text'].apply(
        lambda x: sum(1 for word, tag in nltk.pos_tag(word_tokenize(x)) if tag.startswith('JJ')) if x.strip() != '' else 0
    )
    df['NUM_ADVERBS'] = df['text'].apply(
        lambda x: sum(1 for word, tag in nltk.pos_tag(word_tokenize(x)) if tag.startswith('RB')) if x.strip() != '' else 0
    )
    log("POS tags calculated.")

    log("Calculating review length and average word length...")
    df['REVIEW_LENGTH'] = df['text'].apply(len)
    df['AVG_WORD_LENGTH'] = df['text'].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x.strip() != '' else 0
    )
    log("Review length and average word length calculated.")

def add_tfidf_features(df):
    log("Adding TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, max_df=0.95, min_df=5)
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=df.index
    )
    df = pd.concat([df, tfidf_df], axis=1)
    log("TF-IDF features added.")
    return df

def scale_features(df):
    log("Scaling features...")
    # Exclude non-feature columns
    non_feature_cols = ['label_binary', 'category', 'label', 'text', 'SENTIMENT_CATEGORY']
    # Get all columns except non-feature columns
    features_to_scale = df.columns.difference(non_feature_cols)
    # Ensure we only include numeric columns
    features_to_scale = df[features_to_scale].select_dtypes(include=[np.number]).columns.tolist()
    log(f"Features to scale: {features_to_scale}")
    
    # Check for non-numeric data in features_to_scale
    for col in features_to_scale.copy():  # Use copy to avoid modifying the list during iteration
        if not pd.api.types.is_numeric_dtype(df[col]):
            log(f"Column '{col}' is not numeric and will be excluded from scaling.")
            features_to_scale.remove(col)
    
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    log("Feature scaling complete.")
    return df

def main():
    log("Loading data...")
    file_path = "../../../data/akereviewsdataset.csv"
    df = pd.read_csv(file_path)
    log("Data loaded successfully.")

    log("Preparing data...")
    df.rename(columns={'text_': 'text'}, inplace=True)
    df['category'] = df['category'].str.replace('_5', '', regex=False)
    df.drop_duplicates(inplace=True)
    df['label_binary'] = df['label'].replace({'CG': 1, 'OR': 0})
    df['text'] = df['text'].fillna("").astype(str)
    # Ensure 'category' is of type string
    df['category'] = df['category'].astype(str)
    log("Data preparation complete.")

    log("Preprocessing text...")
    df['text'] = df['text'].apply(preprocess_text)
    log("Text preprocessing complete.")

    log("Adding features...")
    add_features(df)
    log("Feature addition complete.")

    log("Adding TF-IDF features...")
    df = add_tfidf_features(df)
    log("TF-IDF features added.")

    log("Data types before scaling:")
    log(df.dtypes)

    log("Scaling features...")
    df = scale_features(df)
    log("Feature scaling complete.")

    log("Saving processed data...")
    output_path = "../../../data/processed_reviews.csv"
    df.to_csv(output_path, index=False)
    log(f"Processed data saved to {output_path}.")
    log("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()
