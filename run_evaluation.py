"""Comprehensive evaluation script for entity linking models.

Evaluates all trained models (DNN, SVM, XGBoost) on multiple datasets,
compares with TagMe API, generates performance graphs and feature importance analysis.

Usage:
    python run_evaluation.py --output-dir results
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.inspection import permutation_importance

from src.core import InvertedIndex, PageContext
from src.utils.utils import create_training_data
from src.features.features_extractor import FEATURE_NAMES
from src.models.svm_model import SVMEntityLinker
from src.models.xgboost_model import XGBoostEntityLinker
from src.models.dnn_model import DNNEntityLinker


PROJECT_ROOT = Path(__file__).parent

def load_env_file(path: Path) -> dict:
    env_vars = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

_env_vars = load_env_file(PROJECT_ROOT / ".env")


@dataclass
class EvaluationResult:
    model_name: str
    dataset_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    num_samples: int


DATASET_CONFIGS = {
    "NEEL2016-Dev": {
        "tweets": "Provided-Resources/Datasets/NEEL2016/NEEL2016-UTF-Tweets-dev.tsv",
        "annotations": "Provided-Resources/Datasets/NEEL2016/DevAnnotationsCleaned.tsv",
        "format": "neel",
    },
    "NEEL2016-Train": {
        "tweets": "Provided-Resources/Datasets/NEEL2016/NEEL2016-UTF-Tweets-training.tsv",
        "annotations": "Provided-Resources/Datasets/NEEL2016/TrainAnnotationsCleaned.tsv",
        "format": "neel",
    },
    "Mena": {
        "tweets": "Provided-Resources/Datasets/Mena/MenaTweets.tsv",
        "annotations": "Provided-Resources/Datasets/Mena/MenaAnnotationsCleanedRefDel.tsv",
        "format": "mena",
    },
    "Meij": {
        "tweets": "Provided-Resources/Datasets/MeijRevisedAugmented/MeijTweets.tsv",
        "annotations": "Provided-Resources/Datasets/MeijRevisedAugmented/MeijAnnotations.tsv",
        "format": "meij",
    },
}


def load_dataset(
    tweets_path: str, 
    annotations_path: str, 
    fmt: str = "meij"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tweets_df = pd.read_csv(
        tweets_path, sep='\t', header=None,
        names=['tweet_id', 'text'], quoting=3
    )
    ann_df = pd.read_csv(annotations_path, sep='\t', header=None, quoting=3)
    
    if fmt == "meij":
        ann_df.columns = ['tweet_id', 'page_id', 'mention', 'page_title']
    elif fmt == "neel":
        ann_df.columns = ['tweet_id', 'mention', 'page_id', 'entity_type']
        ann_df['page_title'] = ann_df['mention']
    elif fmt == "mena":
        ann_df.columns = ['tweet_id', 'mention', 'col3', 'page_id']
        ann_df['page_title'] = ann_df['mention']
    else:
        ann_df.columns = ['tweet_id', 'page_id', 'mention', 'page_title']
    
    return tweets_df, ann_df


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
    }


def evaluate_tagme(
    tweets_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    api_token: str,
    max_tweets: Optional[int] = None
) -> Tuple[Optional[Dict[str, float]], int]:
    import requests
    import time
    
    TAGME_ENDPOINT = "https://tagme.d4science.org/tagme/tag"
    
    gold_by_tweet = annotations_df.groupby('tweet_id')['page_id'].apply(
        lambda x: set(str(v) for v in x)
    ).to_dict()
    
    y_true, y_pred = [], []
    eval_df = tweets_df if max_tweets is None else tweets_df.head(max_tweets)
    
    for _, row in eval_df.iterrows():
        tweet_id = row['tweet_id']
        text = str(row['text']) if pd.notna(row['text']) else ""
        if not text:
            continue
            
        try:
            response = requests.post(
                TAGME_ENDPOINT,
                data={'text': text, 'gcube-token': api_token, 'lang': 'en'},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                predicted_ids = {str(ann['id']) for ann in data.get('annotations', []) 
                                if ann.get('rho', 0) > 0.1}
                gold_ids = gold_by_tweet.get(tweet_id, set())
                
                has_gold = len(gold_ids) > 0
                has_correct = len(predicted_ids & gold_ids) > 0 if gold_ids else False
                
                y_true.append(1 if has_gold else 0)
                y_pred.append(1 if has_correct or (not has_gold and len(predicted_ids) == 0) else 0)
                
            time.sleep(0.1)
        except Exception as e:
            print(f"TagMe API error for tweet {tweet_id}: {e}")
            continue
    
    if not y_true:
        return None, 0
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }, len(y_true)


def evaluate_dbpedia_spotlight(
    tweets_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    max_tweets: Optional[int] = None
) -> Tuple[Optional[Dict[str, float]], int]:
    import requests
    import time
    
    SPOTLIGHT_ENDPOINT = "https://api.dbpedia-spotlight.org/en/annotate"
    
    def normalize(s: str) -> str:
        return s.lower().replace('_', ' ').replace('-', ' ').strip()
    
    gold_mentions_by_tweet = annotations_df.groupby('tweet_id', group_keys=False)[['mention', 'page_title']].apply(
        lambda df: set(normalize(str(m)) for m in df['mention']) | 
                   set(normalize(str(t)) for t in df.get('page_title', []) if pd.notna(t))
    ).to_dict()
    
    y_true, y_pred = [], []
    eval_df = tweets_df if max_tweets is None else tweets_df.head(max_tweets)
    
    for _, row in eval_df.iterrows():
        tweet_id = row['tweet_id']
        text = str(row['text']) if pd.notna(row['text']) else ""
        if not text:
            continue
            
        try:
            response = requests.get(
                SPOTLIGHT_ENDPOINT,
                params={'text': text, 'confidence': 0.35},
                headers={'Accept': 'application/json'},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                predicted_titles = set()
                for resource in data.get('Resources', []):
                    uri = resource.get('@URI', '')
                    surface = resource.get('@surfaceForm', '')
                    if uri:
                        title = uri.split('/')[-1].replace('_', ' ')
                        predicted_titles.add(normalize(title))
                    if surface:
                        predicted_titles.add(normalize(surface))
                
                gold_mentions = gold_mentions_by_tweet.get(tweet_id, set())
                
                has_gold = len(gold_mentions) > 0
                has_correct = len(predicted_titles & gold_mentions) > 0 if gold_mentions else False
                
                y_true.append(1 if has_gold else 0)
                y_pred.append(1 if has_correct or (not has_gold and len(predicted_titles) == 0) else 0)
                
            time.sleep(0.15)
        except Exception as e:
            print(f"DBpedia Spotlight API error for tweet {tweet_id}: {e}")
            continue
    
    if not y_true:
        return None, 0
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }, len(y_true)


def get_feature_importance_xgboost(model: XGBoostEntityLinker) -> Dict[str, float]:
    importance = model.get_feature_importance()
    return dict(sorted(zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True))


def compute_permutation_importance(
    model, X: np.ndarray, y: np.ndarray, n_repeats: int = 10
) -> Dict[str, float]:
    from sklearn.metrics import make_scorer
    scorer = make_scorer(f1_score, zero_division=0)
    result = permutation_importance(
        model, X, y, scoring=scorer, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    importance = dict(zip(FEATURE_NAMES, result.importances_mean))
    return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))


def plot_model_comparison(
    results: List[EvaluationResult], 
    save_path: str,
    title: str = "Model Performance Comparison"
):
    df = pd.DataFrame([{
        'Model': r.model_name,
        'Dataset': r.dataset_name,
        'Precision': r.precision,
        'Recall': r.recall,
        'F1': r.f1
    } for r in results])
    
    df_melted = df.melt(
        id_vars=['Model', 'Dataset'], 
        value_vars=['Precision', 'Recall', 'F1'],
        var_name='Metric', value_name='Score'
    )
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df_melted, x='Dataset', y='Score', hue='Model', ax=ax)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(
    importance: Dict[str, float],
    save_path: str,
    title: str = "Feature Importance",
    top_k: int = 15
):
    sorted_items = list(importance.items())[:top_k]
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_heatmap(
    results: List[EvaluationResult],
    save_path: str,
    metric: str = 'f1'
):
    models = list(set(r.model_name for r in results))
    datasets = list(set(r.dataset_name for r in results))
    
    matrix = np.zeros((len(models), len(datasets)))
    for r in results:
        i = models.index(r.model_name)
        j = datasets.index(r.dataset_name)
        matrix[i, j] = getattr(r, metric)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=datasets, yticklabels=models, ax=ax)
    ax.set_title(f'{metric.upper()} Score Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_vs_api_comparison(
    results: List[EvaluationResult],
    save_path: str,
    metric: str = 'f1'
):
    # Filter for datasets that have API results
    api_results = [r for r in results if r.model_name.endswith('-API') or 'Spotlight' in r.model_name]
    if not api_results:
        return
        
    datasets_with_api = set(r.dataset_name for r in api_results)
    filtered_results = [r for r in results if r.dataset_name in datasets_with_api]
    
    if not filtered_results:
        return

    df = pd.DataFrame([{
        'Model': r.model_name,
        'Dataset': r.dataset_name,
        'Score': getattr(r, metric)
    } for r in filtered_results])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='Dataset', y='Score', hue='Model', ax=ax)
    ax.set_title(f'Model vs API Performance ({metric.upper()})')
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_results_table(results: List[EvaluationResult]):
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'Dataset':<18} {'Prec':<10} {'Recall':<10} {'F1':<10} {'Acc':<10}")
    print("=" * 80)
    for r in results:
        print(f"{r.model_name:<20} {r.dataset_name:<18} "
              f"{r.precision:.4f}     {r.recall:.4f}     {r.f1:.4f}     {r.accuracy:.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate entity linking models")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results and plots")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory containing trained models")
    parser.add_argument("--max-tweets", type=int, default=None,
                        help="Max tweets per dataset for evaluation")
    parser.add_argument("--skip-tagme", action="store_true",
                        help="Skip TagMe API evaluation")
    parser.add_argument("--skip-spotlight", action="store_true",
                        help="Skip DBpedia Spotlight API evaluation")
    parser.add_argument("--max-api-tweets", type=int, default=None,
                        help="Max tweets for API evaluation (default: all)")
    parser.add_argument("--datasets", nargs="+", 
                        default=["NEEL2016-Dev", "NEEL2016-Train", "Mena", "Meij"],
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Datasets to evaluate on")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    
    print("Loading databases...")
    index = InvertedIndex(str(PROJECT_ROOT / "Provided-Resources/PostingsLast"))
    context = PageContext(str(PROJECT_ROOT / "Provided-Resources/PageIdToContexte2"))
    
    print("Loading trained models...")
    models = {}
    
    svm_path = models_dir / "svm_entity_linker.pkl"
    if svm_path.exists():
        models["SVM"] = SVMEntityLinker.load(svm_path)
        print("  Loaded SVM model")
    
    xgb_path = models_dir / "xgboost_entity_linker.pkl"
    if xgb_path.exists():
        models["XGBoost"] = XGBoostEntityLinker.load(xgb_path)
        print("  Loaded XGBoost model")
    
    dnn_path = models_dir / "dnn_entity_linker.pt"
    if dnn_path.exists():
        models["DNN"] = DNNEntityLinker.load(dnn_path)
        print("  Loaded DNN model")
    
    if not models:
        print("ERROR: No models found. Run training first.")
        return
    
    print(f"\nEvaluating on datasets: {args.datasets}")
    all_results = []
    dataset_features = {}
    
    for dataset_name in args.datasets:
        config = DATASET_CONFIGS[dataset_name]
        tweets_path = PROJECT_ROOT / config["tweets"]
        annotations_path = PROJECT_ROOT / config["annotations"]
        
        if not tweets_path.exists() or not annotations_path.exists():
            print(f"  Skipping {dataset_name}: files not found")
            continue
        
        print(f"\n  Processing {dataset_name}...")
        fmt = config.get("format", "meij")
        tweets_df, annotations_df = load_dataset(str(tweets_path), str(annotations_path), fmt)
        
        X, y = create_training_data(
            tweets_df, annotations_df, index, context,
            max_tweets=args.max_tweets
        )
        
        if len(X) == 0:
            print(f"    No samples generated for {dataset_name}")
            continue
            
        print(f"    Generated {len(X)} samples ({sum(y)} positive)")
        dataset_features[dataset_name] = (X, y)
        
        for model_name, model in models.items():
            metrics = evaluate_model(model, X, y)
            result = EvaluationResult(
                model_name=model_name,
                dataset_name=dataset_name,
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1=metrics['f1'],
                num_samples=len(y)
            )
            all_results.append(result)
            print(f"    {model_name}: F1={metrics['f1']:.4f}")
    
    tagme_token = _env_vars.get("TAGME_TOKEN") or os.getenv("TAGME_TOKEN")
    if tagme_token and not args.skip_tagme:
        print("\nEvaluating TagMe API...")
        for dataset_name in args.datasets[:1]:
            config = DATASET_CONFIGS[dataset_name]
            tweets_path = PROJECT_ROOT / config["tweets"]
            annotations_path = PROJECT_ROOT / config["annotations"]
            
            if tweets_path.exists() and annotations_path.exists():
                fmt = config.get("format", "meij")
                tweets_df, annotations_df = load_dataset(str(tweets_path), str(annotations_path), fmt)
                tagme_metrics, num_samples = evaluate_tagme(
                    tweets_df, annotations_df, tagme_token, args.max_api_tweets
                )
                
                if tagme_metrics:
                    result = EvaluationResult(
                        model_name="TagMe-API",
                        dataset_name=dataset_name,
                        accuracy=tagme_metrics['accuracy'],
                        precision=tagme_metrics['precision'],
                        recall=tagme_metrics['recall'],
                        f1=tagme_metrics['f1'],
                        num_samples=num_samples
                    )
                    all_results.append(result)
                    print(f"  TagMe on {dataset_name}: F1={tagme_metrics['f1']:.4f} ({num_samples} tweets)")
    
    if not args.skip_spotlight:
        print("\nEvaluating DBpedia Spotlight API...")
        for dataset_name in args.datasets[:1]:
            config = DATASET_CONFIGS[dataset_name]
            tweets_path = PROJECT_ROOT / config["tweets"]
            annotations_path = PROJECT_ROOT / config["annotations"]
            
            if tweets_path.exists() and annotations_path.exists():
                fmt = config.get("format", "meij")
                tweets_df, annotations_df = load_dataset(str(tweets_path), str(annotations_path), fmt)
                spotlight_metrics, num_samples = evaluate_dbpedia_spotlight(
                    tweets_df, annotations_df, args.max_api_tweets
                )
                
                if spotlight_metrics:
                    result = EvaluationResult(
                        model_name="DBpedia-Spotlight",
                        dataset_name=dataset_name,
                        accuracy=spotlight_metrics['accuracy'],
                        precision=spotlight_metrics['precision'],
                        recall=spotlight_metrics['recall'],
                        f1=spotlight_metrics['f1'],
                        num_samples=num_samples
                    )
                    all_results.append(result)
                    print(f"  Spotlight on {dataset_name}: F1={spotlight_metrics['f1']:.4f} ({num_samples} tweets)")
    
    print_results_table(all_results)
    
    print("\nGenerating plots...")
    
    plot_model_comparison(
        all_results, 
        str(output_dir / "model_comparison_per_dataset.png"),
        "Model Performance Comparison by Dataset"
    )
    
    if len(all_results) > 0:
        plot_heatmap(all_results, str(output_dir / "performance_heatmap.png"))
        
        # Plot Model vs API comparison for datasets where APIs were evaluated
        plot_model_vs_api_comparison(
            all_results,
            str(output_dir / "model_vs_api_comparison.png")
        )
    
    if "XGBoost" in models:
        xgb_importance = get_feature_importance_xgboost(models["XGBoost"])
        plot_feature_importance(
            xgb_importance,
            str(output_dir / "feature_importance_xgboost.png"),
            "XGBoost Feature Importance"
        )
        
        print("\nTop 10 XGBoost Features:")
        for i, (name, score) in enumerate(list(xgb_importance.items())[:10], 1):
            print(f"  {i}. {name}: {score:.4f}")
    
    if dataset_features and "XGBoost" in models:
        first_dataset = list(dataset_features.keys())[0]
        X, y = dataset_features[first_dataset]
        perm_importance = compute_permutation_importance(models["XGBoost"], X, y)
        plot_feature_importance(
            perm_importance,
            str(output_dir / "feature_importance_permutation.png"),
            f"Permutation Importance ({first_dataset})"
        )
    
    results_df = pd.DataFrame([{
        'Model': r.model_name,
        'Dataset': r.dataset_name,
        'Accuracy': r.accuracy,
        'Precision': r.precision,
        'Recall': r.recall,
        'F1': r.f1,
        'Samples': r.num_samples
    } for r in all_results])
    results_df.to_csv(output_dir / "evaluation_results.csv", index=False)
    print(f"\nSaved results to {output_dir / 'evaluation_results.csv'}")
    
    index.close()
    context.close()
    
    print("\n" + "="*50)
    print("Evaluation complete! Results saved to:", output_dir)


if __name__ == "__main__":
    main()
