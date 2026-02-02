"""Main training script for entity linking models.

Trains DNN, SVM, and XGBoost models on Meij dataset and saves as pkl files.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import load_meij_dataset, create_training_data
from src.core import InvertedIndex, PageContext
from src.candidate_generation.candidate_generator import generate_candidates
from src.features.features_extractor import extract_features, FEATURE_NAMES
from src.models.svm_model import train_svm
from src.models.xgboost_model import train_xgboost



def train_dnn_model(X_train, y_train, X_val, y_val, epochs: int = 20):
    import torch
    from torch.utils.data import DataLoader
    from src.models.dnn_model import EntityLinkingDataset, EntityLinkingDNN, train_model
    
    train_dataset = EntityLinkingDataset(X_train, y_train)
    val_dataset = EntityLinkingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    input_dim = X_train.shape[1]
    model = EntityLinkingDNN(input_dim=input_dim, hidden_dim=128)
    
    trained_model = train_model(model, train_loader, val_loader, epochs=epochs)
    return trained_model


def main():
    parser = argparse.ArgumentParser(description="Train entity linking models")
    parser.add_argument("--max-tweets", type=int, default=None, 
                        help="Max tweets to use (for testing)")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=20,
                        help="DNN training epochs")
    parser.add_argument("--xgb-grid-search", action="store_true",
                        help="Run XGBoost grid search")
    args = parser.parse_args()
    
    base_path = PROJECT_ROOT
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading inverted index and page context...")
    index_path = base_path / "Provided-Resources/PostingsLast"
    context_path = base_path / "Provided-Resources/PageIdToContexte2"
    
    index = InvertedIndex(str(index_path))
    context = PageContext(str(context_path))
    
    tweets_path = base_path / "Provided-Resources/Datasets/MeijRevisedAugmented/MeijTweets.tsv"
    annotations_path = base_path / "Provided-Resources/Datasets/MeijRevisedAugmented/MeijAnnotations.tsv"

    print("Loading Meij dataset...")
    tweets_df, annotations_df = load_meij_dataset( tweets_path, annotations_path)
    print(f"Loaded {len(tweets_df)} tweets, {len(annotations_df)} annotations")
    
    print("Generating features...")
    X, y = create_training_data(
        tweets_df, annotations_df, index, context,
        
    )
    print(f"Created {len(X)} samples, {sum(y)} positive ({100*sum(y)/len(y):.1f}%)")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    print("\n" + "="*50)
    print("Training SVM (poly kernel, degree=2, C=1)...")
    svm_model, svm_metrics = train_svm(X_train, y_train, X_val, y_val)
    print(f"SVM Val F1: {svm_metrics.get('val_f1', 0):.4f}")
    svm_model.save(output_dir / "svm_entity_linker.pkl")
    print(f"Saved SVM model to {output_dir / 'svm_entity_linker.pkl'}")
    
    print("\n" + "="*50)
    print("Training XGBoost...")
    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_val, y_val,
        grid_search=args.xgb_grid_search
    )
    print(f"XGBoost Val F1: {xgb_metrics.get('val_f1', 0):.4f}")
    xgb_model.save(output_dir / "xgboost_entity_linker.pkl")
    print(f"Saved XGBoost model to {output_dir / 'xgboost_entity_linker.pkl'}")
    
    print("\n" + "="*50)
    print("Training DNN...")
    import torch
    dnn_model = train_dnn_model(X_train, y_train, X_val, y_val, epochs=args.epochs)
    torch.save(dnn_model.state_dict(), output_dir / "dnn_entity_linker.pt")
    print(f"Saved DNN model to {output_dir / 'dnn_entity_linker.pt'}")
    
    index.close()
    context.close()
    
    print("\n" + "="*50)
    print("Training complete! Models saved to:", output_dir)
    print(f"  - svm_entity_linker.pkl")
    print(f"  - xgboost_entity_linker.pkl")
    print(f"  - dnn_entity_linker.pt")


if __name__ == "__main__":
    main()
