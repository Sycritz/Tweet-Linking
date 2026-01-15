from src.candidate_generation import generate_candidates
from typing import Optional
from dataclasses import dataclass
import lmdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys,os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.append(os.path.join(os.path.dirname(__file__), 'Provided-Resources'))
import SerializedListNew_pb2
import DictionaryWithTitle_pb2

@dataclass
class Candidate:
    ngram: str
    page_id: str
    anchor_score: int
    anchor_type: int
    page_title: str
    page_rank: float
    page_views: float
    num_categories: int
    num_anchors: int



class InvertedIndex:
    def __init__(self, path: str):
        self.env = lmdb.open(path, readonly=True, lock=False)

    def get(self, ngram: str) -> list[tuple[str, int, int]]:
        results = []
        with self.env.begin() as txn:
            val = txn.get(ngram.encode('utf-8'))
            if val:
                posting = SerializedListNew_pb2.SerializedListNew()
                posting.ParseFromString(val)
                for el in posting.Elements:
                    results.append((el.docId, el.score, el.typ))
        return results

    def close(self):
        self.env.close()


class PageContext:
    def __init__(self, path: str):
        self.env = lmdb.open(path, readonly=True, lock=False)

    def get(self, page_id: str) -> Optional[tuple[str, float, float, int, int]]:
        with self.env.begin() as txn:
            val = txn.get(page_id.encode('utf-8'))
            if val:
                dico = DictionaryWithTitle_pb2.Dico()
                dico.ParseFromString(val)
                return (
                    dico.PageTitle,
                    dico.PageRank,
                    dico.PageViews,
                    len(dico.Categories),
                    dico.length_anchors
                )
        return None

    def close(self):
        self.env.close()

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    max_tweets: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    gold_set = set()
    for _, row in annotations_df.iterrows():
        gold_set.add((str(row['tweet_id']), str(row['page_id'])))

    all_features = []
    all_labels = []

    for idx, row in tweets_df.head(max_tweets).iterrows():
        tweet_id = str(row['tweet_id'])
        tweet_text = str(row['text']) if pd.notna(row['text']) else ""
        if not tweet_text:
            continue

        candidates = generate_candidates(tweet_text, index, context, top_k=3)
        if not candidates:
            continue

        total_score = sum(c.anchor_score for c in candidates)

        for c in candidates:
            features = extract_features(c, total_score)
            label = 1.0 if (tweet_id, c.page_id) in gold_set else 0.0
            all_features.append(features)
            all_labels.append(label)

    return np.array(all_features), np.array(all_labels)

