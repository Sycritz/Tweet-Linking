#!/usr/bin/env python3
"""Entity Linking Inference Script.

Link entities in tweets to Wikipedia pages using trained models.

Usage:
    python infer.py --tweet "Obama speaks at the White House" --model xgboost
"""

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np

from src.core import InvertedIndex, PageContext, Candidate
from src.candidate_generation.candidate_generator import generate_candidates
from src.features.features_extractor import extract_features
from src.models.svm_model import SVMEntityLinker
from src.models.xgboost_model import XGBoostEntityLinker
from src.models.dnn_model import DNNEntityLinker


PROJECT_ROOT = Path(__file__).parent


def load_model(model_type: str, models_dir: Path):
    if model_type == "svm":
        return SVMEntityLinker.load(models_dir / "svm_entity_linker.pkl")
    elif model_type == "xgboost":
        return XGBoostEntityLinker.load(models_dir / "xgboost_entity_linker.pkl")
    elif model_type == "dnn":
        return DNNEntityLinker.load(models_dir / "dnn_entity_linker.pt")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def link_entities(
    tweet: str,
    model,
    index: InvertedIndex,
    context: PageContext,
    threshold: float = 0.5,
    top_k: int = 10
) -> List[Dict]:
    candidates = generate_candidates(tweet, index, context, top_k=top_k)
    if not candidates:
        return []

    total_score = sum(c.anchor_score for c in candidates)
    features = np.array([extract_features(c, total_score, tweet) for c in candidates])

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(features)
    else:
        probs = model.predict(features).astype(float)

    results = []
    for c, prob in zip(candidates, probs):
        if prob >= threshold:
            results.append({
                'mention': c.ngram,
                'page_id': c.page_id,
                'page_title': c.page_title,
                'confidence': float(prob)
            })

    results.sort(key=lambda x: (-len(x['mention'].split()), -x['confidence']))
    
    seen_mentions = set()
    unique_results = []
    for r in results:
        mention_lower = r['mention'].lower()
        is_substring = any(mention_lower in seen for seen in seen_mentions)
        if not is_substring:
            seen_mentions.add(mention_lower)
            unique_results.append(r)

    return unique_results


def main():
    parser = argparse.ArgumentParser(description='Link entities in tweets to Wikipedia')
    parser.add_argument('--tweet', type=str, required=True, help='Tweet text to analyze')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['svm', 'xgboost', 'dnn'],
                        help='Model to use for prediction')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for entity links')
    args = parser.parse_args()

    index_path = PROJECT_ROOT / "Provided-Resources/PostingsLast"
    context_path = PROJECT_ROOT / "Provided-Resources/PageIdToContexte2"

    print("Loading resources...")
    index = InvertedIndex(str(index_path))
    context = PageContext(str(context_path))
    model = load_model(args.model, Path(args.models_dir))
    print(f"Using {args.model.upper()} model\n")

    print(f"Tweet: {args.tweet}")
    print("-" * 60)

    results = link_entities(args.tweet, model, index, context, args.threshold)

    if results:
        print("Linked Entities:")
        for r in results:
            print(f"  '{r['mention']}' → {r['page_title']} "
                  f"(confidence: {r['confidence']:.3f})")
    else:
        print("No entities linked above threshold.")

    index.close()
    context.close()


if __name__ == "__main__":
    main()
