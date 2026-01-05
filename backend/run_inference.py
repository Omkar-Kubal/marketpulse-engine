
#!/usr/bin/env python3
"""
ML Inference Script for AMSE (Brain A + Brain B).
Runs trained models on live data to generate trading signals.

Usage:
    python run_inference.py --symbol RELIANCE.NS
    python run_inference.py --check-pnl
"""

import argparse
import sys
import logging
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, 'src')

from src.database import init_db, get_latest_candles, insert_candle, candle_exists
from src.fetcher import fetch_latest_candle, fetch_historical_candles
from src.features import compute_features, get_feature_columns
from src.models.trainer import load_torch_model
from src.models.brain_a import BrainA
from src.rl.env import TradingEnv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')

def get_latest_brain_b_path():
    """Find latest PPO model."""
    files = [f for f in os.listdir(MODELS_DIR) if f.startswith('brain_b_ppo') and f.endswith('.zip')]
    if not files:
        return None
    # Sort by timestamp (filename)
    files.sort(reverse=True)
    return os.path.join(MODELS_DIR, files[0])

def run_inference_pipeline(symbol: str, brain_a_name: str = 'brain_a_latest') -> dict:
    
    # 1. Ensure Data Availability
    # We need at least 50 candles for features + Sequence
    init_db()
    
    # Check DB
    candles_db = get_latest_candles(symbol, limit=100)
    
    if len(candles_db) < 60:
        logger.info("Insufficient historical data in DB. Fetching history...")
        # Fetch 5 days history
        history = fetch_historical_candles(symbol, period='5d', interval='5m')
        # Insert into DB
        for c in history:
             if not candle_exists(symbol, c['timestamp']):
                insert_candle(
                    symbol=c['symbol'],
                    timestamp=c['timestamp'],
                    open_price=c['open'],
                    high=c['high'],
                    low=c['low'],
                    close=c['close'],
                    volume=c['volume']
                )
        candles_db = get_latest_candles(symbol, limit=100)
    else:
        # Fetch just latest 
        latest = fetch_latest_candle(symbol, '5m')
        if latest and not candle_exists(symbol, latest['timestamp']):
             insert_candle(
                symbol=latest['symbol'],
                timestamp=latest['timestamp'],
                open_price=latest['open'],
                high=latest['high'],
                low=latest['low'],
                close=latest['close'],
                volume=latest['volume']
            )
             candles_db = get_latest_candles(symbol, limit=100)
    
    if len(candles_db) < 50:
         return {'error': f"Not enough data even after fetch. Count: {len(candles_db)}"}
         
    # 2. Prepare Feature Sequence
    # Convert to DF
    data = [{
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in candles_db]
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp')
    
    # Compute Features
    df_feat = compute_features(df)
    feature_cols = get_feature_columns()
    
    # We need the LAST sequence of length 30
    # Brain A expects (1, 30, n_features)
    # The sequence ends at the current candle.
    
    seq_len = 30
    if len(df_feat) < seq_len:
        return {'error': "Not enough data for sequence"}
        
    last_seq_df = df_feat.iloc[-seq_len:]
    
    # Check for NaNs
    if last_seq_df[feature_cols].isna().any().any():
        # Handle NaNs (ffill or fail)
        last_seq_df = last_seq_df.ffill().bfill()
        if last_seq_df[feature_cols].isna().any().any():
             return {'error': "Features contain NaNs"}

    X_seq = last_seq_df[feature_cols].values
    
    # Scale? 
    # Brain A was trained on SCALED data.
    # We need to load the scaler used during Brain A training.
    # load_torch_model returns metrics which SHOULD contain scaler if I saved it?
    # I modified save_torch_model to save 'scaler' in STATE dict.
    # load_torch_model returns (model, metrics).
    # Does it return scaler?
    # My `load_torch_model` returns `model, state['metrics']`.
    # It DISCARDS the scaler! 
    # I need to fix `load_torch_model` to return scaler if needed, or put scaler in metrics.
    # OR simpler: Brain A metrics dict usually has metadata.
    # Wait, I saved 'scaler': scaler IN THE STATE DICT.
    # But `load_torch_model` only extracts `state['metrics']`.
    
    # I MUST FIX `load_torch_model` to return scaler. 
    # OR, I can just load the state dict manually here.
    
    model_path = os.path.join(MODELS_DIR, f"{brain_a_name}.pth")
    if not os.path.exists(model_path):
         return {'error': f"Model {brain_a_name} not found"}
    
    state = torch.load(model_path, weights_only=False)
    scaler = state.get('scaler')
    if not scaler:
         logger.warning("No scaler found in model state. Using raw features (Prediction may be bad).")
    else:
         X_seq = scaler.transform(X_seq)
         
    # 3. Brain A Inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrainA(input_dim=X_seq.shape[1], 
                   hidden_dim=state['hidden_dim'], 
                   num_layers=state['num_layers'])
    model.load_state_dict(state['state_dict'])
    model.to(device)
    model.eval()
    
    x_input = torch.FloatTensor(X_seq).unsqueeze(0).to(device) # (1, 30, feat)
    
    with torch.no_grad():
        prob, embedding = model(x_input)
        
    embedding_np = embedding.cpu().numpy().flatten()
    
    # 4. Brain B Inference
    brain_b_path = get_latest_brain_b_path()
    if not brain_b_path:
        return {'error': "No Brain B (PPO) model found"}
        
    ppo_agent = PPO.load(brain_b_path)
    
    # Construct Observation for Brain B
    # Env _get_observation: [embedding, balance_pct, position_pct, unreal_pnl, drawdown]
    # For signal generation, assume neutral/start state
    account_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    obs = np.concatenate((embedding_np, account_state))
    
    action, _ = ppo_agent.predict(obs, deterministic=True)
    
    # Action Map: 0=HOLD, 1=BUY, 2=SELL
    actions_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    signal = actions_map.get(int(action), "UNKNOWN")
    
    return {
        'symbol': symbol,
        'signal': signal,
        'brain_a_prob': float(prob.item()), # Direction probability
        'close': float(last_seq_df.iloc[-1]['close']),
        'timestamp': datetime.now()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='RELIANCE.NS')
    parser.add_argument('--model-a', default='brain_a_latest')
    args = parser.parse_args()
    
    print(f"\nRunning Inference for {args.symbol}...")
    try:
        result = run_inference_pipeline(args.symbol, args.model_a)
        
        if result.get('error'):
            print(f"[!] Error: {result['error']}")
        else:
            print(f"""
============================================================
             MARKETPULSE SIGNAL
============================================================
  Symbol:      {result['symbol']}
  Start Time:  {result['timestamp']}
  Price:       {result['close']:.2f}
------------------------------------------------------------
  Brain A (Direction Prob): {result['brain_a_prob']:.4f}
  Brain B (RL Action):      {result['signal']}
============================================================
            """)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
