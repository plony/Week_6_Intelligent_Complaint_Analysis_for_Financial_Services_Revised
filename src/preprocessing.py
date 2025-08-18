# src/preprocessing.py

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")


def clean_text(text: str) -> str:
    """
    Cleans a single text string by lowercasing, removing special
    characters, and stopwords.
    """
    if not isinstance(text, str):
        return ""

    stop_words = set(stopwords.words("english"))

    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove common boilerplate phrases (example)
    text = re.sub(r"i am writing to file a complaint", "", text, flags=re.I)
    # Tokenize and remove stopwords
    tokens = text.split()
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)


def filter_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame for specified products and cleans the consumer
    narratives.
    """
    if df is None:
        return None

    print("Starting data filtering and cleaning...")

    # Define the list of target products
    target_products = [
        "Credit card",
        "Personal loan",
        "Buy Now, Pay Later (BNPL)",
        "Savings account",
        "Money transfers",
    ]

    # Filter for target products
    filtered_df = df[df["Product"].isin(target_products)].copy()
    print(f"Dataset filtered to {len(filtered_df)} "
          f"records for target products.")

    # Remove records with empty narratives
    filtered_df = filtered_df.dropna(subset=["Consumer complaint narrative"]).copy()
    print(f"Dataset after removing empty narratives: "
          f"{len(filtered_df)} records.")

    # Apply text cleaning to the narrative column
    filtered_df["cleaned_narrative"] = filtered_df[
        "Consumer complaint narrative"
    ].apply(clean_text)

    print("Data cleaning complete.")
    return filtered_df