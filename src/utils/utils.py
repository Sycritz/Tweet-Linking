"""Utility functions for entity linking.

Contains data loading, training data generation, and helper functions.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from src.core import Candidate, InvertedIndex, PageContext
from src.candidate_generation.candidate_generator import generate_candidates
from src.features.features_extractor import extract_features, FEATURE_NAMES


def compute_commonness(score: int, total_scores: int) -> float:
    if total_scores == 0:
        return 0.0
    return score / total_scores


def string_similarity(s1: str, s2: str) -> float:
    s1, s2 = s1.lower(), s2.lower()
    if not s1 or not s2:
        return 0.0
    set1, set2 = set(s1.split()), set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def load_meij_dataset(
    tweets_path: str,
    annotations_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tweets_df = pd.read_csv(
        tweets_path,
        sep='\t',
        header=None,
        names=['tweet_id', 'text'],
        quoting=3
    )
    annotations_df = pd.read_csv(
        annotations_path,
        sep='\t',
        header=None,
        names=['tweet_id', 'page_id', 'mention', 'page_title'],
        quoting=3
    )
    return tweets_df, annotations_df


def create_training_data(
    tweets_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    index: InvertedIndex,
    context: PageContext,
    max_tweets: int | None = None,
    use_full_context: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
  
    gold_set = set()
    for _, row in annotations_df.iterrows():
        gold_set.add((str(row['tweet_id']), str(row['page_id'])))

    all_features = []
    all_labels = []

    tweets_to_process = tweets_df if max_tweets is None else tweets_df.head(max_tweets)

    for idx, row in tweets_to_process.iterrows():
        tweet_id = str(row['tweet_id'])
        tweet_text = str(row['text']) if pd.notna(row['text']) else ""
        if not tweet_text:
            continue

        candidates = generate_candidates(
            tweet_text, index, context, 
            top_k=3, use_full_context=use_full_context
        )
        if not candidates:
            continue

        total_score = sum(c.anchor_score for c in candidates)

        for c in candidates:
            features = extract_features(c, total_score, tweet_text)
            label = 1.0 if (tweet_id, c.page_id) in gold_set else 0.0
            all_features.append(features)
            all_labels.append(label)

    return np.array(all_features), np.array(all_labels)
