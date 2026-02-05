"""SVM model for entity linking using a polynomial kernel."""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import joblib
from pathlib import Path


class SVMEntityLinker:
    """SVM classifier for entity linking with polynomial kernel."""
    
    def __init__(self, kernel: str = 'poly', degree: int = 2, C: float = 1.0):
        self.scaler = StandardScaler()
        self.model = SVC(
            kernel=kernel,
            degree=degree,
            C=C,
            probability=True,
            class_weight='balanced'
        )
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMEntityLinker':
        """Train the SVM model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels of shape (n_samples,)
        
        Returns:
            Self for method chaining
        """
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
    
    def save(self, path: str | Path) -> None:
        """Save model and scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    @classmethod
    def load(cls, path: str | Path) -> 'SVMEntityLinker':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance._is_fitted = True
        return instance


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    kernel: str = 'poly',
    degree: int = 2,
    C: float = 1.0
) -> Tuple[SVMEntityLinker, dict]:
    """Train SVM model and return metrics.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        kernel: SVM kernel type
        degree: Polynomial degree
        C: Regularization parameter
    
    Returns:
        Tuple of (trained model, metrics dict)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    model = SVMEntityLinker(kernel=kernel, degree=degree, C=C)
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
