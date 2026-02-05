"""Enhanced DNN model for entity linking.

Deep neural network with batch normalization, residual connections,
early stopping, and learning rate scheduling for improved performance.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from typing import Tuple, Optional
from pathlib import Path


class EntityLinkingDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.linear(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + identity


class EntityLinkingDNN(nn.Module):
    def __init__(self, input_dim: int = 17, hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return self.sigmoid(x).squeeze(-1)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_state = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10
) -> nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        
        if early_stopping(val_f1, model):
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_state)
            break

    return model


def evaluate_model(model: nn.Module, loader: DataLoader) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
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


class DNNEntityLinker:
    def __init__(self, input_dim: int = 17):
        self.model = EntityLinkingDNN(input_dim=input_dim)
        self.input_dim = input_dim
        self._is_fitted = False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            probs = self.model(X_tensor).cpu().numpy()
        return (probs >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            return self.model(X_tensor).cpu().numpy()
    
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'input_dim': self.input_dim
        }, path)
    
    @classmethod
    def load(cls, path: str | Path) -> 'DNNEntityLinker':
        data = torch.load(path, weights_only=False)
        instance = cls(input_dim=data['input_dim'])
        instance.model.load_state_dict(data['model_state'])
        instance._is_fitted = True
        return instance
