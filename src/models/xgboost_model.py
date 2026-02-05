"""XGBoost model for entity linking with grid search capability."""

import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Optional, Dict, Any
import joblib
from pathlib import Path


class XGBoostEntityLinker:
    """XGBoost classifier for entity linking."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        **kwargs
    ):
        self.scaler = StandardScaler()
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric='logloss',
            use_label_encoder=False,
            **kwargs
        )
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostEntityLinker':
        """Train the XGBoost model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores from trained model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model.feature_importances_
    
    def save(self, path: str | Path) -> None:
        """Save model and scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    @classmethod
    def load(cls, path: str | Path) -> 'XGBoostEntityLinker':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance._is_fitted = True
        return instance


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    grid_search: bool = False,
    **kwargs
) -> Tuple[XGBoostEntityLinker, dict]:
    """Train XGBoost model with optional grid search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        grid_search: Whether to perform grid search for hyperparameters
        **kwargs: Additional XGBoost parameters
    
    Returns:
        Tuple of (trained model, metrics dict)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    if grid_search:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
        }
        base_model = XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            subsample=0.8,
            colsample_bytree=0.8
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )
        search.fit(X_scaled, y_train)
        
        best_params = search.best_params_
        print(f"Best XGBoost params: {best_params}")
        
        model = XGBoostEntityLinker(**best_params, **kwargs)
        model.scaler = scaler
        model.model = search.best_estimator_
        model._is_fitted = True
    else:
        model = XGBoostEntityLinker(**kwargs)
        model.fit(X_train, y_train)
    
    metrics = {}
    train_preds = model.predict(X_train)
    metrics['train_accuracy'] = accuracy_score(y_train, train_preds)
    metrics['train_f1'] = f1_score(y_train, train_preds, zero_division=0)
    
    if X_val is not None and y_val is not None:
        val_preds = model.predict(X_val)
        metrics['val_accuracy'] = accuracy_score(y_val, val_preds)
        metrics['val_precision'] = precision_score(y_val, val_preds, zero_division=0)
        metrics['val_recall'] = recall_score(y_val, val_preds, zero_division=0)
        metrics['val_f1'] = f1_score(y_val, val_preds, zero_division=0)
    
    return model, metrics
