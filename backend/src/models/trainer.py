"""
Model training module for AMSE.
Trains and saves ML models for signal prediction.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier

from src.features import compute_features, get_feature_columns, prepare_training_data

logger = logging.getLogger(__name__)

# Model save directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'trained_models')


def ensure_models_dir():
    """Create models directory if it doesn't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def train_logistic_regression(X: np.ndarray, y: np.ndarray, 
                               test_size: float = 0.2) -> Tuple[LogisticRegression, StandardScaler, dict]:
    """
    Train a Logistic Regression model.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data for testing
    
    Returns:
        Tuple of (model, scaler, metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'positive_rate': y.mean(),
        'model_type': 'LogisticRegression'
    }
    
    logger.info(f"Logistic Regression - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return model, scaler, metrics


def train_xgboost(X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2) -> Tuple[XGBClassifier, StandardScaler, dict]:
    """
    Train an XGBoost model.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data for testing
    
    Returns:
        Tuple of (model, scaler, metrics)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # Scale features (XGBoost can work without, but helps consistency)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'positive_rate': y.mean(),
        'model_type': 'XGBoost'
    }
    
    logger.info(f"XGBoost - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return model, scaler, metrics


def save_model(model, scaler, feature_cols: list, metrics: dict, 
               model_name: str = 'model') -> str:
    """
    Save model, scaler, and metadata.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_cols: List of feature column names
        metrics: Training metrics
        model_name: Name for the saved model
    
    Returns:
        Path to saved model
    """
    ensure_models_dir()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{timestamp}.joblib"
    filepath = os.path.join(MODELS_DIR, filename)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'metrics': metrics,
        'created_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, filepath)
    logger.info(f"Model saved to {filepath}")
    
    # Also save as latest
    latest_path = os.path.join(MODELS_DIR, f"{model_name}_latest.joblib")
    joblib.dump(model_data, latest_path)
    
    return filepath


def load_model(model_name: str = 'model_latest') -> Optional[dict]:
    """
    Load a saved model.
    
    Args:
        model_name: Name of the model file (without .joblib)
    
    Returns:
        Dict with model, scaler, feature_columns, metrics
    """
    filepath = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    
    if not os.path.exists(filepath):
        logger.error(f"Model not found: {filepath}")
        return None
    
    model_data = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    
    return model_data


def list_models() -> list:
    """List all saved models."""
    ensure_models_dir()
    models = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.joblib'):
            models.append(f.replace('.joblib', ''))

def save_torch_model(model, metrics: dict, scaler=None, model_name: str = 'brain_a') -> str:
    """Save PyTorch model."""
    ensure_models_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{timestamp}.pth"
    filepath = os.path.join(MODELS_DIR, filename)
    
    # Save Model State and Metadata
    state = {
        'state_dict': model.state_dict(),
        'metrics': metrics,
        'scaler': scaler,
        'input_dim': model.lstm.input_size,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'created_at': datetime.now().isoformat()
    }
    
    torch.save(state, filepath)
    logger.info(f"PyTorch model saved to {filepath}")
    
    latest_path = os.path.join(MODELS_DIR, f"{model_name}_latest.pth")
    torch.save(state, latest_path)
    
    return filepath

def load_torch_model(model_class, model_name: str = 'brain_a_latest') -> Optional[Any]:
    """Load PyTorch model."""
    filepath = os.path.join(MODELS_DIR, f"{model_name}.pth")
    if not os.path.exists(filepath):
        logger.error(f"Model not found: {filepath}")
        return None
        
    state = torch.load(filepath, weights_only=False)
    
    model = model_class(
        input_dim=state['input_dim'],
        hidden_dim=state['hidden_dim'],
        num_layers=state['num_layers']
    )
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    return model, state['metrics']

