
import argparse
import sys
import logging
import os
import torch
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Add src to path
sys.path.insert(0, 'src')

from src.fetcher import fetch_historical_candles
from src.features import prepare_training_data
from src.models.trainer import load_torch_model
from src.models.brain_a import BrainA
from src.rl.env import TradingEnv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')

def train_rl_agent(symbol: str, 
                   brain_a_name: str = 'brain_a_latest',
                   data_path: str = None,
                   timesteps: int = 100000) -> str:
    
    # 1. Load Data
    if data_path:
        logger.info(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        logger.info(f"Fetching data for {symbol}...")
        candles = fetch_historical_candles(symbol, period='1mo', interval='5m')
        df = pd.DataFrame(candles)
    
    # 2. Prepare Features
    logger.info("Preparing features...")
    # Using 0.0 as threshold since we just want X, y structure, y is ignored for RL env (rewards are calculated dynamically)
    X, y, _ = prepare_training_data(df, target_threshold=0.0)
    
    # Align df with X (dropping first N rows used for feature calculation)
    # feature calc drops rows with NaN. prepare_training_data returns X, y.
    # We need the corresponding price data (df_clean) for the environment.
    # Re-running compute_features to get the clean DataFrame
    from src.features import compute_features, get_feature_columns
    df_feat = compute_features(df)
    df_feat['target_return'] = np.log(df_feat['close'].shift(-1) / df_feat['close'])
    features = get_feature_columns() + ['target_return']
    df_clean = df_feat.dropna(subset=features)
    
    assert len(df_clean) == len(X), f"Length mismatch: DF={len(df_clean)}, X={len(X)}"
    
    # 3. Load Brain A & Generate Embeddings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading Brain A: {brain_a_name}")
    
    brain_a, metrics = load_torch_model(BrainA, brain_a_name)
    if not brain_a:
        raise ValueError("Failed to load Brain A model")
        
    brain_a.to(device)
    brain_a.eval() # Set to eval mode
    
    # Create sequences for Brain A
    # Brain A expects (batch, seq_len, features)
    # But our BrainADataset handles slicing.
    # To get embeddings for ALL steps, we need to process step-by-step or batch processing.
    # Simplest: use a DataLoader with the same Dataset logic
    
    from src.models.brain_a import BrainADataset
    from torch.utils.data import DataLoader
    
    # Note: BrainADataset trims the first `seq_len` data points as it needs history.
    # So `embeddings` will be smaller than `X`.
    # We must trim `df_clean` accordingly.
    
    seq_len = 30 # Must match training seq_len of Brain A
    dataset = BrainADataset(X, y, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    logger.info("Generating embeddings from Brain A...")
    embeddings_list = []
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            _, embedding = brain_a(batch_x)
            embeddings_list.append(embedding.cpu().numpy())
            
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    # Trim Dataframe to match embeddings
    # BrainADataset starts at index 0 (which corresponds to X[0:seq_len]).
    # The output label corresponds to X[seq_len-1].
    # So the embedding corresponds to time `seq_len-1`.
    # We need to drop the first `seq_len` rows of df_clean to match (or `seq_len-1`?)
    # Let's align exactly:
    # Dataset length is len(X) - seq_len.
    # Index i in embeddings corresponds to X[i : i+seq_len].
    # The "current state" at end of sequence is X[i+seq_len-1].
    # So we should use df_clean starting from index `seq_len-1`.
    # However, Dataset `__len__` is X-seq_len.
    # If X has 100 items, seq=30. len=70.
    # Indicies 0 to 69.
    # Index 0: X[0:30]. Ends at 29.
    # Index 69: X[69:99]. Ends at 99.
    # So we slice df_clean from index `seq_len-1` to end?
    # df_clean row 29 matches embedding 0.
    # df_clean row 99 matches embedding 69.
    
    df_env = df_clean.iloc[seq_len:] # Drop first seq_len rows?
    # Wait, if seq_len=30, index 0..29 are used for first embedding.
    # The "current time" for that embedding is t=29.
    # So we need df_clean at t=29.
    # So df_env = df_clean.iloc[seq_len-1 : ] ?
    # Let's verify lengths.
    # len(embeddings) = N - seq_len.
    # len(df_env) must be N - seq_len.
    # len(df_clean) = N.
    # df_clean.iloc[seq_len:] has length N - seq_len.
    # Correct.
    
    df_env = df_clean.iloc[seq_len:].reset_index(drop=True)
    
    logger.info(f"Environment Data: {len(df_env)} steps")
    
    # 4. Initialize Env
    env = TradingEnv(df_env, embeddings, initial_balance=100000)
    env = DummyVecEnv([lambda: env])
    
    # 5. Train PPO
    logger.info("Training PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    model.learn(total_timesteps=timesteps)
    
    # 6. Save
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(MODELS_DIR, f"brain_b_ppo_{timestamp}")
    model.save(save_path)
    logger.info(f"Agent saved to {save_path}")
    
    return save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='RELIANCE.NS')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV data file')
    parser.add_argument('--brain_a', type=str, default='brain_a_latest')
    parser.add_argument('--timesteps', type=int, default=10000)
    
    args = parser.parse_args()
    
    train_rl_agent(args.symbol, args.brain_a, args.data, args.timesteps)
