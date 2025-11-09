import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

nltk.download("stopwords")
nltk.download("wordnet")

def preprocess_data(input_path="data/raw/reddit_raw.csv", output_path="data/processed/reddit_preprocessed.csv"):
    df = pd.read_csv(input_path)
    print(f"📥 Loaded data: {df.shape}")

    # Drop missing
    df.dropna(subset=["clean_comment"], inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Remove empty or whitespace-only comments
    df = df[df["clean_comment"].str.strip().astype(bool)]

    # Convert to lowercase
    df["clean_comment"] = df["clean_comment"].str.lower()

    # Remove unwanted characters
    df["clean_comment"] = df["clean_comment"].str.replace(r"[^A-Za-z0-9\s!?.,]", "", regex=True)

    # Remove stopwords (except sentiment-related)
    stop_words = set(stopwords.words("english")) - {"not", "but", "no", "however", "yet"}
    df["clean_comment"] = df["clean_comment"].apply(
        lambda x: " ".join([word for word in x.split() if word not in stop_words])
    )

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    df["clean_comment"] = df["clean_comment"].apply(
        lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
    )

    # Drop final empties
    df = df[df["clean_comment"].str.strip().astype(bool)]

    # Save processed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}, shape = {df.shape}")

    return df

if __name__ == "__main__":
    preprocess_data()
