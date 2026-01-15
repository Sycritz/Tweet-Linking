import os
import sys
import re
from dataclasses import dataclass
from typing import Optional

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
    top_k: int = 5
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


class EntityLinkingDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3
) -> nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                val_preds.extend((outputs.cpu() > 0.5).int().tolist())
                val_labels.extend(y_batch.int().tolist())

        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

    return model


def evaluate_model(model: nn.Module, loader: DataLoader) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds.extend((outputs.cpu() > 0.5).int().tolist())
            labels.extend(y_batch.int().tolist())

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0)
    }


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
    import argparse
    parser = argparse.ArgumentParser(description='TELS - Tweet Entity Linking System')
    parser.add_argument('--mode', choices=['train', 'infer'], default='train')
    parser.add_argument('--tweet', type=str, help='Tweet text for inference')
    parser.add_argument('--model-path', type=str, default='models/dnn_entity_linker.pt')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    base_path = os.path.dirname(__file__)
    index_path = os.path.join(base_path, 'Provided-Resources', 'PostingsLast')
    context_path = os.path.join(base_path, 'Provided-Resources', 'PageIdToContexte2')

    if args.mode == 'infer':
        if not args.tweet:
            print("Error: --tweet is required for inference mode")
            return

        print("Loading model and databases...")
        index = InvertedIndex(index_path)
        context = PageContext(context_path)

        model = EntityLinkingDNN(input_dim=10, hidden_dim=64)
        model_path = os.path.join(base_path, args.model_path)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        print(f"\nTweet: {args.tweet}")
        print("-" * 40)

        results = run_inference(args.tweet, model, index, context, args.threshold)

        if results:
            print("Linked Entities:")
            for r in results:
                print(f"  '{r['mention']}' -> {r['page_title']} (conf: {r['confidence']:.3f})")
        else:
            print("No entities linked.")

        index.close()
        context.close()
        return

    print("=" * 60)
    print("TELS MVP - Tweet Entity Linking System")
    print("=" * 60)

    tweets_path = os.path.join(base_path, 'Provided-Resources', 'Datasets', 'MeijRevisedAugmented', 'MeijTweets.tsv')
    annotations_path = os.path.join(base_path, 'Provided-Resources', 'Datasets', 'MeijRevisedAugmented', 'MeijAnnotations.tsv')

    print("\n[1/5] Loading databases...")
    index = InvertedIndex(index_path)
    context = PageContext(context_path)

    print("[2/5] Loading Meij dataset...")
    tweets_df, annotations_df = load_meij_dataset(tweets_path, annotations_path)
    print(f"      Loaded {len(tweets_df)} tweets, {len(annotations_df)} annotations")

    print("[3/5] Creating training data...")
    X, y = create_training_data(tweets_df, annotations_df, index, context, max_tweets=200)
    print(f"      Generated {len(X)} samples | Positive: {int(y.sum())} | Negative: {int(len(y) - y.sum())}")

    if len(X) == 0:
        print("No training data generated. Exiting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 1 else None)

    train_dataset = EntityLinkingDataset(X_train, y_train)
    test_dataset = EntityLinkingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print("\n[4/5] Training DNN model...")
    model = EntityLinkingDNN(input_dim=X.shape[1], hidden_dim=64)
    model = train_model(model, train_loader, test_loader, epochs=20, lr=1e-3)

    print("\n[5/5] Evaluating model...")
    metrics = evaluate_model(model, test_loader)
    print("\n" + "=" * 40)
    print("FINAL RESULTS")
    print("=" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("=" * 40)

    model_path = os.path.join(base_path, 'models', 'dnn_entity_linker.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    index.close()
    context.close()

    print("\nMVP Complete!")


if __name__ == "__main__":
    main()
