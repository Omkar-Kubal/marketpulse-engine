
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    Trading Environment for AMSE.
    
    State Space:
    - Market State: Brain A Embeddings (64 dim) + Current Market Features (optional)
    - Account State: [Balance_Pct, Position_Pct, Unrealized_PnL_Pct, Drawdown_Pct]
    
    Action Space:
    - Discrete(3): 0=HOLD, 1=BUY, 2=SELL/CLOSE
    - Discrete(5) MultiDiscrete could be added for TP/SL later.
    """
    
    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray, 
                 initial_balance: float = 10000.0,
                 max_steps: int = 0,
                 commission: float = 0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.embeddings = embeddings
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Data validation
        assert len(df) == len(embeddings), "Data and embeddings must match length"
        
        self.n_steps = len(df)
        self.max_steps = max_steps if max_steps > 0 else self.n_steps - 1
        
        # Action Space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation Space
        # Embedding (e.g. 64) + Account State (4) = 68
        self.embedding_dim = embeddings.shape[1]
        self.account_dim = 4
        total_obs_dim = self.embedding_dim + self.account_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0 # 0 or 1 (Fixed size for now) or units
        self.entry_price = 0.0
        self.peak_balance = initial_balance
        self.trades = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        
        # Random start implementation can be added here
        # For now, start from index 0 or random index
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.peak_balance = self.initial_balance
        self.trades = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # 1. Market State (Embedding)
        market_state = self.embeddings[self.current_step]
        
        # 2. Account State
        # Normalize relative to initial balance or current bounds
        balance_pct = self.balance / self.initial_balance
        
        current_price = self.df.iloc[self.current_step]['close']
        
        if self.position > 0:
            position_val = self.position * current_price
            position_pct = position_val / (self.balance + position_val)
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            position_pct = 0.0
            unrealized_pnl = 0.0
            
        # Drawdown from peak
        total_equity = self.balance + (self.position * current_price)
        self.peak_balance = max(self.peak_balance, total_equity)
        drawdown = (total_equity - self.peak_balance) / self.peak_balance
        
        account_state = np.array([balance_pct, position_pct, unrealized_pnl, drawdown], dtype=np.float32)
        
        # Concatenate
        return np.concatenate((market_state, account_state))
    
    def step(self, action: int):
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        terminated = False
        truncated = False
        
        # Execute Action
        if action == 1: # BUY
            if self.position == 0:
                # Enter Long (Use 95% of cash)
                units = (self.balance * 0.95) / current_price
                cost = units * current_price * self.commission
                self.balance -= (units * current_price + cost)
                self.position = units
                self.entry_price = current_price
                # Penalty for transaction? Maybe implicit in cost
        
        elif action == 2: # SELL
            if self.position > 0:
                # Close Position
                revenue = self.position * current_price
                cost = revenue * self.commission
                self.balance += (revenue - cost)
                
                # Calculate PnL for reward
                # Reward = Realized PnL %
                pnl_pct = (current_price - self.entry_price) / self.entry_price
                reward = pnl_pct * 100 # Scale up
                
                self.position = 0
                self.entry_price = 0
        
        # HOLD (action=0) -> No change in position
        
        # Step Forward
        self.current_step += 1
        
        # Check termination
        if self.current_step >= self.max_steps:
            terminated = True
        
        if self.balance < self.initial_balance * 0.5: # Bankruptcy condition
            terminated = True
            reward -= 10 # Big Penalty
            
        # Shaping Reward: Change in Equity
        # Usually better for training than sparse realized PnL
        new_price = self.df.iloc[self.current_step]['close'] if not terminated else current_price
        prev_equity = self.balance + (self.position * current_price)
        new_equity = self.balance + (self.position * new_price)
        
        step_reward = (new_equity - prev_equity) / self.initial_balance * 100
        
        reward += step_reward
        
        return self._get_observation(), reward, terminated, truncated, {}
