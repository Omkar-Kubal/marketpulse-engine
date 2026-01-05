"""
Feature engineering module for AMSE.
Computes features for ML model training and inference.
Includes advanced technical indicators and market regime detection.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import ta

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from OHLCV data using `ta` library and custom logic.
    
    Features:
    - Returns: Log returns (1, 3, 5 periods)
    - Volatility: ATR, Rolling Std Dev
    - Momentum: RSI, MACD, ROC
    - Trend: ADX, SMA Crossovers
    - Market Regime: Volatility Regime, Trend Strength
    
    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume
    
    Returns:
        DataFrame with original data plus computed features
    """
    df = df.copy()
    
    # Ensure proper column names
    if hasattr(df, 'columns'):
        df.columns = [c.lower() for c in df.columns]

    # Validate required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        raise ValueError(f"DataFrame missing required columns: {required}")

    # --- 1. Returns (Log scale for stationarity) ---
    df['log_ret_1'] = np.log(df['close'] / df['close'].shift(1))
    df['log_ret_5'] = np.log(df['close'] / df['close'].shift(5))

    # --- 2. Volatility ---
    # ATR (Average True Range) - Measure of volatility
    indicator_atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = indicator_atr.average_true_range()
    df['atr_norm'] = df['atr'] / df['close']  # Normalized ATR

    # Rolling Volatility (Standard Deviation of returns)
    df['volatility_20'] = df['log_ret_1'].rolling(window=20).std()

    # Bollinger Bands Width (Bandwidth)
    indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_width'] = indicator_bb.bollinger_wband()
    df['bb_pband'] = indicator_bb.bollinger_pband() # %B (position within bands)

    # --- 3. Momentum ---
    # RSI (Relative Strength Index)
    df['rsi'] = ta.momentum.rsi(close=df['close'], window=14)

    # MACD (Moving Average Convergence Divergence)
    indicator_macd = ta.trend.MACD(close=df['close'])
    df['macd'] = indicator_macd.macd()
    df['macd_signal'] = indicator_macd.macd_signal()
    df['macd_diff'] = indicator_macd.macd_diff() # Histogram

    # ROC (Rate of Change)
    df['roc_10'] = ta.momentum.roc(close=df['close'], window=10)

    # --- 4. Trend & Strength ---
    # ADX (Average Directional Index) - Trend strength
    indicator_adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = indicator_adx.adx()
    df['adx_pos'] = indicator_adx.adx_pos()
    df['adx_neg'] = indicator_adx.adx_neg()

    # SMA Crossovers
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['sma_cross'] = (df['sma_20'] - df['sma_50']) / df['close']

    # --- 5. Market Regime (Categorical/One-Hot) ---
    # Simple logic: High ADX + Positive SMA Cross = Bull Trend
    #               High ADX + Overall Negative Return = Bear Trend
    #               Low ADX = Ranging/Choppy
    
    # We create a continuous 'regime_score' (-1 to 1) for ML models better than categories
    # +1 = Strong Bull, -1 = Strong Bear, 0 = Chop
    
    df['trend_strength'] = df['adx'] / 100.0 # 0 to 1
    
    # Direction based on EMA alignment or MACD
    df['trend_dir'] = np.where(df['close'] > df['sma_50'], 1, -1)
    
    df['regime'] = df['trend_strength'] * df['trend_dir']
    
    # Fill NaN values (resulting from rolling windows)
    # Forward fill then Backward fill (or just drop in training)
    df = df.bfill().ffill()
    
    return df

def get_feature_columns() -> List[str]:
    """Get list of feature column names for model training."""
    return [
        'log_ret_1', 'log_ret_5',
        'atr_norm', 'volatility_20',
        'bb_width', 'bb_pband',
        'rsi', 
        'macd', 'macd_diff', 
        'roc_10',
        'adx', 'sma_cross',
        'regime'
    ]

def prepare_training_data(df: pd.DataFrame, target_threshold: float = 0.005) -> tuple:
    """
    Prepare data for model training.
    
    Args:
        df: DataFrame with OHLCV data
        target_threshold: Threshold for positive class labels (feature engineering only)
    
    Returns:
        X (features), y (target), feature_cols (list of names)
    """
    # Compute features
    df = compute_features(df)
    
    # Create target: 1 if next period return > threshold, else 0 (Simple Classification)
    # Note: For RL, we might not use this specific target, but good for Supervised Brain A
    
    # Target: Log return of NEXT candle
    df['target_return'] = np.log(df['close'].shift(-1) / df['close'])
    
    # Binary Target for Classifier
    df['target_class'] = (df['target_return'] > target_threshold).astype(int)
    
    feature_cols = get_feature_columns()
    
    # Drop NaNs created by lagging/shifting
    df_clean = df.dropna(subset=feature_cols + ['target_return'])
    
    X = df_clean[feature_cols].values
    y = df_clean['target_class'].values # Return binary class by default, can change to regression
    
    return X, y, feature_cols

def prepare_inference_data(candles: list) -> Optional[np.ndarray]:
    """
    Prepare data for inference from candle list.
    
    Args:
        candles: List of candle dicts or ORM objects
    
    Returns:
        Feature array for the latest candle (1, n_features)
    """
    # Need enough candles for max window (ADX/MACD/SMA50 require ~50+)
    # Providing 100 is safe
    if len(candles) < 50:
        return None
    
    # Convert to DataFrame
    if hasattr(candles[0], 'close'):
        # ORM objects
        data = [{
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in candles]
    else:
        data = candles
    
    df = pd.DataFrame(data)
    df = compute_features(df)
    
    feature_cols = get_feature_columns()
    
    # Get last row
    last_row = df[feature_cols].iloc[-1]
    
    if last_row.isna().any():
        return None
        
    return last_row.values.reshape(1, -1)
