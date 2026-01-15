# svm_model.py
"""
SVM Model for Tweet Entity Linking System (TELS)

This module implements an SVM classifier for binary classification:
Given a (n-gram, PageID) pair, predict whether it's a valid entity link.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import json
from datetime import datetime
import os


class SVMEntityLinker:
    """
    SVM-based entity linking classifier.
    Predicts whether a (n-gram, PageID) candidate pair is a valid link.
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'):
        """
        Initialize SVM classifier.
        
        Args:
            kernel (str): SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C (float): Regularization parameter
            gamma (str/float): Kernel coefficient
            class_weight (str/dict): Weights for imbalanced classes
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.metrics = {}
        
    def prepare_features(self, X, feature_names=None):
        """
        Prepare and validate features.
        
        Args:
            X (array-like): Feature matrix
            feature_names (list): Names of features (optional)
            
        Returns:
            np.ndarray: Prepared feature matrix
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        
        X = np.array(X)
        
        # Handle missing values
        if np.any(np.isnan(X)):
            print("Warning: NaN values detected. Replacing with 0.")
            X = np.nan_to_num(X, nan=0.0)
        
        # Handle infinite values
        if np.any(np.isinf(X)):
            print("Warning: Infinite values detected. Replacing with max/min.")
            X = np.nan_to_num(X, posinf=np.finfo(np.float64).max, 
                             neginf=np.finfo(np.float64).min)
        
        if feature_names is not None and self.feature_names is None:
            self.feature_names = feature_names
        
        return X
    
    def fit(self, X_train, y_train, feature_names=None):
        """
        Train the SVM model.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels (0 or 1)
            feature_names (list): Names of features
            
        Returns:
            self
        """
        print("Training SVM model...")
        
        # Prepare features
        X_train = self.prepare_features(X_train, feature_names)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train_scaled)
        self.metrics['train'] = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, zero_division=0),
            'recall': recall_score(y_train, y_pred_train, zero_division=0),
            'f1': f1_score(y_train, y_pred_train, zero_division=0)
        }
        
        print(f"Training completed!")
        print(f"Train Accuracy: {self.metrics['train']['accuracy']:.4f}")
        print(f"Train F1-Score: {self.metrics['train']['f1']:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict labels for candidate pairs.
        
        Args:
            X (array-like): Feature matrix
            
        Returns:
            np.ndarray: Predicted labels (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        X = self.prepare_features(X)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict probability estimates for candidate pairs.
        
        Args:
            X (array-like): Feature matrix
            
        Returns:
            np.ndarray: Probability estimates (shape: [n_samples, 2])
                        Column 0: P(class=0), Column 1: P(class=1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        X = self.prepare_features(X)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X_test, y_test, set_name='test'):
        """
        Evaluate model performance.
        
        Args:
            X_test (array-like): Test features
            y_test (array-like): Test labels
            set_name (str): Name of the dataset (for logging)
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation!")
        
        X_test = self.prepare_features(X_test)
        y_test = np.array(y_test)
        
        # Make predictions
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.metrics[set_name] = metrics
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Evaluation Results ({set_name})")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TN: {metrics['confusion_matrix'][0][0]}, FP: {metrics['confusion_matrix'][0][1]}")
        print(f"FN: {metrics['confusion_matrix'][1][0]}, TP: {metrics['confusion_matrix'][1][1]}")
        print(f"{'='*60}\n")
        
        # Detailed classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Link', 'Valid Link']))
        
        return metrics
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            param_grid (dict): Parameter grid for search
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters found
        """
        print("Starting hyperparameter tuning...")
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        
        # Prepare features
        X_train = self.prepare_features(X_train)
        y_train = np.array(y_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Grid search
        grid_search = GridSearchCV(
            SVC(class_weight='balanced', probability=True, random_state=42),
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        print(f"\nBest parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best F1-Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def save_model(self, filepath='svm_entity_linker.pkl'):
        """
        Save the trained model and scaler.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving!")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath='svm_entity_linker.pkl'):
        """
        Load a trained model and scaler.
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data.get('metrics', {})
        self.is_fitted = True
        
        print(f"Model loaded from: {filepath}")
        print(f"Saved on: {model_data.get('timestamp', 'Unknown')}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("SVM Entity Linker - Example Usage\n")
    
    # Example: Create synthetic training data
    # In practice, you'll load this from your Meij dataset
    np.random.seed(42)
    
    # Simulate 1000 candidate pairs with 10 features
    n_samples = 1000
    n_features = 10
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (20% positive, 80% negative - typical for EL)
    y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    # Feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Negative class: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train SVM
    svm = SVMEntityLinker(kernel='rbf', C=10, gamma='scale')
    svm.fit(X_train, y_train, feature_names=feature_names)
    
    # Evaluate on test set
    svm.evaluate(X_test, y_test, set_name='test')
    
    # Example: Hyperparameter tuning (commented out for speed)
    # best_params = svm.hyperparameter_tuning(X_train, y_train, cv=3)
    
    # Make predictions
    sample_X = X_test[:5]
    predictions = svm.predict(sample_X)
    probabilities = svm.predict_proba(sample_X)
    
    print("\nExample Predictions:")
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"  Predicted: {predictions[i]} (True: {y_test[i]})")
        print(f"  Probability: {probabilities[i][1]:.4f}")
    
    # Save model
    svm.save_model('src/models/svm_entity_linker.pkl')
    
    print("\nSVM model training completed!")