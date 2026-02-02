"""Comprehensive evaluation module for entity linking models.

Features:
- Multi-dataset evaluation (NEEL2016, Mena, Meij)
- Metrics: F1, Precision, Recall
- Feature importance analysis
- TagMe/AIDA API comparison (optional)
- Extensible architecture for embedding models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
import warnings


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    model_name: str
    dataset_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    num_samples: int
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None


@dataclass
class ModelRegistry:
    """Registry for pluggable model evaluation."""
    models: Dict[str, Callable] = field(default_factory=dict)
    
    def register(self, name: str, predict_fn: Callable) -> None:
        """Register a model's prediction function."""
        self.models[name] = predict_fn
    
    def get(self, name: str) -> Callable:
        """Get registered model by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())


# Global registry for extensibility
MODEL_REGISTRY = ModelRegistry()


def load_dataset(tweets_path: str, annotations_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load tweets and annotations from TSV files.
    
    Args:
        tweets_path: Path to tweets TSV file
        annotations_path: Path to annotations TSV file
    
    Returns:
        Tuple of (tweets_df, annotations_df)
    """
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


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    dataset_name: str = "dataset"
) -> EvaluationResult:
    """Compute evaluation metrics for predictions."""
    return EvaluationResult(
        model_name=model_name,
        dataset_name=dataset_name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        num_samples=len(y_true),
        predictions=y_pred
    )


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "model",
    dataset_name: str = "dataset",
    threshold: float = 0.5
) -> EvaluationResult:
    """Evaluate a trained model on dataset."""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        preds = (probs >= threshold).astype(int)
    else:
        preds = model.predict(X)
        probs = None
    
    result = evaluate_predictions(y, preds, model_name, dataset_name)
    result.probabilities = probs
    return result


def analyze_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10
) -> Dict[str, float]:
    """Analyze feature importance using permutation importance.
    
    Works with any sklearn-compatible model.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
        )
    
    importance_dict = {}
    for i, name in enumerate(feature_names):
        importance_dict[name] = result.importances_mean[i]
    
    return dict(sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True))


def get_xgboost_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """Get native feature importance from XGBoost model."""
    if hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
        importances = model.model.feature_importances_
    else:
        return {}
    
    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: x[1], reverse=True
    ))


# ============== TagMe API Integration ==============

def evaluate_tagme(
    tweets: List[str],
    gold_annotations: Dict[str, set],
    api_key: Optional[str] = None
) -> Optional[EvaluationResult]:
    """Evaluate TagMe API on tweets (optional - requires API key).
    
    Args:
        tweets: List of tweet texts
        gold_annotations: Dict mapping tweet_id to set of gold page_ids
        api_key: TagMe API key (optional, skips if not provided)
    
    Returns:
        EvaluationResult or None if API unavailable
    """
    if not api_key:
        print("TagMe API key not provided, skipping TagMe evaluation")
        return None
    
    try:
        import requests
    except ImportError:
        print("requests library not available for TagMe API")
        return None
    
    TAGME_ENDPOINT = "https://tagme.d4science.org/tagme/tag"
    
    y_true_list = []
    y_pred_list = []
    
    for tweet_id, text in enumerate(tweets):
        try:
            response = requests.post(
                TAGME_ENDPOINT,
                data={'text': text, 'gcube-token': api_key},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                predicted_ids = {str(ann['id']) for ann in data.get('annotations', [])}
                gold_ids = gold_annotations.get(str(tweet_id), set())
                
                # Convert to binary: 1 if any correct link, 0 otherwise
                y_true_list.append(1 if gold_ids else 0)
                y_pred_list.append(1 if predicted_ids & gold_ids else 0)
        except Exception as e:
            print(f"TagMe API error: {e}")
            continue
    
    if not y_true_list:
        return None
    
    return evaluate_predictions(
        np.array(y_true_list),
        np.array(y_pred_list),
        model_name="TagMe",
        dataset_name="API"
    )


def evaluate_aida(
    tweets: List[str],
    gold_annotations: Dict[str, set],
    api_url: Optional[str] = None
) -> Optional[EvaluationResult]:
    """Evaluate AIDA API on tweets (optional).
    
    Note: AIDA requires local installation or custom endpoint.
    """
    if not api_url:
        print("AIDA API URL not provided, skipping AIDA evaluation")
        return None
    
    # AIDA integration placeholder - requires local AIDA setup
    print("AIDA evaluation requires local AIDA setup")
    return None


# ============== Visualization ==============

def plot_comparison(results: List[EvaluationResult], save_path: Optional[str] = None) -> None:
    """Plot bar chart comparing model performance."""
    if not results:
        return
    
    models = [r.model_name for r in results]
    metrics = ['Precision', 'Recall', 'F1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [getattr(r, metric.lower()) for r in results]
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(
    importance_dict: Dict[str, float],
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    top_k: int = 15
) -> None:
    """Plot feature importance bar chart."""
    sorted_items = list(importance_dict.items())[:top_k]
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def print_results_table(results: List[EvaluationResult]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Dataset':<15} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Acc':<8}")
    print("=" * 70)
    for r in results:
        print(f"{r.model_name:<20} {r.dataset_name:<15} "
              f"{r.precision:.4f}   {r.recall:.4f}   {r.f1:.4f}   {r.accuracy:.4f}")
    print("=" * 70)


# ============== Main Evaluation Runner ==============

def run_full_evaluation(
    models: Dict[str, Any],
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    feature_names: List[str],
    output_dir: str = "results",
    tagme_key: Optional[str] = None,
    aida_url: Optional[str] = None
) -> List[EvaluationResult]:
    """Run comprehensive evaluation across all models and datasets.
    
    Args:
        models: Dict mapping model names to trained model objects
        datasets: Dict mapping dataset names to (X, y) tuples
        feature_names: List of feature names for importance analysis
        output_dir: Directory to save plots and results
        tagme_key: Optional TagMe API key
        aida_url: Optional AIDA API URL
    
    Returns:
        List of all evaluation results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Evaluate each model on each dataset
    for model_name, model in models.items():
        for dataset_name, (X, y) in datasets.items():
            result = evaluate_model(model, X, y, model_name, dataset_name)
            all_results.append(result)
    
    # Print results table
    print_results_table(all_results)
    
    # Generate comparison plot
    plot_comparison(all_results, save_path=output_path / "model_comparison.png")
    
    # Feature importance for XGBoost
    for model_name, model in models.items():
        if 'xgboost' in model_name.lower():
            importance = get_xgboost_importance(model, feature_names)
            if importance:
                plot_feature_importance(
                    importance,
                    title=f"{model_name} Feature Importance",
                    save_path=output_path / f"{model_name}_importance.png"
                )
                print(f"\n{model_name} Top 5 Features:")
                for name, score in list(importance.items())[:5]:
                    print(f"  {name}: {score:.4f}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate entity linking models")
    parser.add_argument("--model", type=str, help="Path to model pickle file")
    parser.add_argument("--dataset", type=str, default="NEEL2016", help="Dataset to evaluate on")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Evaluation module loaded. Use run_full_evaluation() for comprehensive analysis.")
