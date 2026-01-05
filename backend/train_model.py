#!/usr/bin/env python3
"""
Model Training Script for AMSE.
Trains ML models on historical market data.

Usage:
    python train_model.py --model xgboost --symbol RELIANCE.NS
    python train_model.py --model logistic --data data/historical.csv
"""

import argparse
import sys
import logging
import pandas as pd

# Add src to path
sys.path.insert(0, 'src')

from src.fetcher import fetch_historical_candles
from src.features import prepare_training_data, get_feature_columns
from src.models.trainer import (
    train_logistic_regression, 
    train_xgboost, 
    save_model,
    save_torch_model
)
from src.models.brain_a import train_brain_a
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data_from_csv(filepath: str) -> pd.DataFrame:
    """Load historical data from CSV file."""
    df = pd.read_csv(filepath)
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Parse date/timestamp
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
        
    return df


def fetch_data_for_training(symbol: str, period: str = '1mo') -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance."""
    logger.info(f"Fetching {period} of data for {symbol}...")
    candles = fetch_historical_candles(symbol, period, '5m')
    
    if not candles:
        raise ValueError(f"No data fetched for {symbol}")
    
    df = pd.DataFrame(candles)
    logger.info(f"Fetched {len(df)} candles")
    return df


def train_model(model_type: str, df: pd.DataFrame, threshold: float = 0.005) -> dict:
    """Train a model on the data."""
    logger.info(f"Preparing training data with threshold {threshold*100:.1f}%...")
    
    X, y, feature_cols = prepare_training_data(df, threshold)
    
    logger.info(f"Training samples: {len(X)}")
    logger.info(f"Positive class rate: {y.mean()*100:.1f}%")
    logger.info(f"Features: {feature_cols}")
    
    if model_type == 'logistic':
        model, scaler, metrics = train_logistic_regression(X, y)
        model_name = 'logistic_regression'
    elif model_type == 'xgboost':
        model, scaler, metrics = train_xgboost(X, y)
        model_name = 'xgboost'
    elif model_type == 'brain_a':
        # Brain A setup
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Brain A
        model, metrics = train_brain_a(X_scaled, y, input_dim=X.shape[1])
        model_name = 'brain_a'
        
        # Save specialized torch model
        filepath = save_torch_model(model, metrics, scaler, model_name)
        metrics['model_path'] = filepath
        return metrics
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save model
    filepath = save_model(model, scaler, feature_cols, metrics, model_name)
    metrics['model_path'] = filepath
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train ML models for AMSE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_model.py --model xgboost --symbol RELIANCE.NS
    python train_model.py --model logistic --symbol TCS.NS --period 3mo
    python train_model.py --model xgboost --data historical.csv
        """
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=['logistic', 'xgboost', 'brain_a'],
        help='Model type to train'
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default=None,
        help='Stock symbol to fetch data for'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to CSV file with historical data'
    )
    parser.add_argument(
        '--period', '-p',
        type=str,
        default='1mo',
        help='Data period for fetching (e.g., 1mo, 3mo, 6mo)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.005,
        help='Signal threshold for positive class (default: 0.005 = 0.5%%)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.symbol and not args.data:
        parser.error("Must specify either --symbol or --data")
    
    print(f"""
============================================================
            AMSE - Model Training                          
============================================================
  Model:     {args.model}
  Source:    {args.symbol or args.data}
  Threshold: {args.threshold*100:.1f}%
============================================================
    """)
    
    try:
        # Load data
        if args.data:
            df = load_data_from_csv(args.data)
        else:
            df = fetch_data_for_training(args.symbol, args.period)
        
        # Train model
        metrics = train_model(args.model, df, args.threshold)
        
        # Display results
        print(f"""
============================================================
                 TRAINING COMPLETE                         
============================================================
  Model Type:   {metrics['model_type']}
  Accuracy:     {metrics['accuracy']*100:.2f}%
  F1 Score:     {metrics['f1_score']*100:.2f}%
  Train Size:   {metrics['train_samples']}
  Test Size:    {metrics['test_samples']}
  Saved To:     {metrics['model_path']}
============================================================
        """)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
