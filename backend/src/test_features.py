
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import compute_features, get_feature_columns

def test_features():
    print("Testing features module...")
    
    # Create dummy data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 105,
        'low': np.random.randn(100) + 95,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(100, 1000, 100)
    })
    
    # Ensure high >= low etc
    df['high'] = df[['open', 'close']].max(axis=1) + 1
    df['low'] = df[['open', 'close']].min(axis=1) - 1
    
    try:
        df_feat = compute_features(df)
        print(f"Features computed. Shape: {df_feat.shape}")
        
        cols = get_feature_columns()
        print(f"Feature columns: {cols}")
        
        # Check if columns exist
        missing = [c for c in cols if c not in df_feat.columns]
        if missing:
            print(f"FAILED: Missing columns: {missing}")
        else:
            print("SUCCESS: All feature columns present.")
            
        # Check RSI
        print(f"RSI Sample: {df_feat['rsi'].tail().values}")
        
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_features()
