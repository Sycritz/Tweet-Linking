import os
import sys
import re
import argparse
from dataclasses import dataclass
from typing import Optional

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import lmdb
import numpy as np
import torch
import torch.nn as nn

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


def clean_tweet(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def get_ngrams(text: str, min_n: int = 1, max_n: int = 6) -> list[str]:
    if not text:
        return []
    words = text.split()
    ngrams = []
    for n in range(min_n, max_n + 1):
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i + n]))
    return ngrams


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


def generate_candidates(
    tweet: str,
    index: InvertedIndex,
    context: PageContext,
    top_k: int = 10
) -> list[Candidate]:
    cleaned = clean_tweet(tweet)
    ngrams = get_ngrams(cleaned)
    candidates = []

    for ngram in ngrams:
        postings = index.get(ngram)
        for page_id, score, typ in postings[:top_k]:
            page_info = context.get(page_id)
            if page_info:
                title, rank, views, n_cats, n_anchors = page_info
                candidates.append(Candidate(
                    ngram=ngram,
                    page_id=page_id,
                    anchor_score=score,
                    anchor_type=typ,
                    page_title=title,
                    page_rank=rank,
                    page_views=views,
                    num_categories=n_cats,
                    num_anchors=n_anchors
                ))
    return candidates


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


def extract_features(candidate: Candidate, total_anchor_score: int) -> np.ndarray:
    commonness = compute_commonness(candidate.anchor_score, total_anchor_score)
    sim = string_similarity(candidate.ngram, candidate.page_title)
    ngram_len = len(candidate.ngram.split())
    is_redirect = 1.0 if candidate.anchor_type == 0 else 0.0
    is_anchor = 1.0 if candidate.anchor_type == 1 else 0.0
    is_both = 1.0 if candidate.anchor_type == 2 else 0.0
    log_rank = np.log1p(candidate.page_rank * 1e9)
    log_views = np.log1p(candidate.page_views)
    log_cats = np.log1p(candidate.num_categories)
    log_anchors = np.log1p(candidate.num_anchors)

    return np.array([
        commonness,
        sim,
        ngram_len,
        is_redirect,
        is_anchor,
        is_both,
        log_rank,
        log_views,
        log_cats,
        log_anchors
    ], dtype=np.float32)


class EntityLinkingDNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_inference(
    tweet: str,
    model: nn.Module,
    index: InvertedIndex,
    context: PageContext,
    threshold: float = 0.5
) -> list[dict]:
    candidates = generate_candidates(tweet, index, context, top_k=10)
    if not candidates:
        return []

    total_score = sum(c.anchor_score for c in candidates)
    features = np.array([extract_features(c, total_score) for c in candidates])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32).to(device)
        probs = model(X).cpu().numpy()

    results = []
    for c, prob in zip(candidates, probs):
        if prob >= threshold:
            results.append({
                'mention': c.ngram,
                'page_id': c.page_id,
                'page_title': c.page_title,
                'confidence': float(prob)
            })

    results.sort(key=lambda x: x['confidence'], reverse=True)
    seen_mentions = set()
    unique_results = []
    for r in results:
        if r['mention'] not in seen_mentions:
            seen_mentions.add(r['mention'])
            unique_results.append(r)

    return unique_results


def main():
    parser = argparse.ArgumentParser(description='TELS Inference - Link entities in tweets')
    parser.add_argument('--tweet', type=str, required=True, help='Tweet text to analyze')
    parser.add_argument('--model-path', type=str, default='models/dnn_entity_linker.pt')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()

    base_path = os.path.dirname(__file__)
    index_path = os.path.join(base_path, 'Provided-Resources', 'PostingsLast')
    context_path = os.path.join(base_path, 'Provided-Resources', 'PageIdToContexte2')

    print("Loading model and databases...")
    index = InvertedIndex(index_path)
    context = PageContext(context_path)

    model = EntityLinkingDNN(input_dim=10, hidden_dim=64)
    model_path = os.path.join(base_path, args.model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    print(f"\nTweet: {args.tweet}")
    print("-" * 50)

    results = run_inference(args.tweet, model, index, context, args.threshold)

    if results:
        print("Linked Entities:")
        for r in results:
            print(f"  '{r['mention']}' -> {r['page_title']} (confidence: {r['confidence']:.3f})")
    else:
        print("No entities linked above threshold.")

    index.close()
    context.close()


if __name__ == "__main__":
    main()
