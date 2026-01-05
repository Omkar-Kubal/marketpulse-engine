
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    type: str # 'LONG' or 'SHORT'
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    duration: int = 0 # candles

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: pd.Series

class BacktestEngine:
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission # 0.1% per side
        
    def run(self, df: pd.DataFrame, signals: pd.Series, 
            sl_pct: float = 0.02, tp_pct: float = 0.04) -> BacktestResult:
        """
        Run backtest simulation.
        
        Args:
            df: OHLCV DataFrame
            signals: Series of 1 (Buy), -1 (Sell), 0 (Hold)
            sl_pct: Stop Less percentage (fixed or dynamic support later)
            tp_pct: Take Profit percentage
        """
        capital = self.initial_capital
        position = 0 # Currently held quantity
        entry_price = 0.0
        trades = []
        equity_curve = []
        
        # Ensure df is sorted
        df = df.sort_index() if isinstance(df.index, pd.DatetimeIndex) else df
        
        # We iterate through the dataframe
        # Using itertuples for speed
        
        # Align signals with df
        # If signals is a Series, assume same index
        
        # Simple Loop
        for i, row in enumerate(df.itertuples()):
            timestamp = row.timestamp if hasattr(row, 'timestamp') else df.index[i]
            close = row.close
            high = row.high
            low = row.low
            
            # Record equity (Cash + Unrealized PnL)
            unrealized = 0
            if position > 0:
                unrealized = (close - entry_price) * position
            current_equity = capital + unrealized
            equity_curve.append(current_equity)
            
            # Check Exit (SL/TP)
            if position > 0:
                # Check Low for SL
                if low <= entry_price * (1 - sl_pct):
                    # Trigger SL
                    exit_price = entry_price * (1 - sl_pct)
                    pnl = (exit_price - entry_price) * position
                    cost = exit_price * position * self.commission
                    capital += (exit_price * position) - cost
                    
                    trades.append(Trade(
                        entry_time=trade_entry_time,
                        entry_price=entry_price,
                        type='LONG',
                        size=position,
                        exit_time=timestamp,
                        exit_price=exit_price,
                        pnl=pnl - cost - entry_cost,
                        pnl_pct=(exit_price - entry_price) / entry_price,
                        exit_reason="SL"
                    ))
                    position = 0
                    continue
                
                # Check High for TP
                elif high >= entry_price * (1 + tp_pct):
                    # Trigger TP
                    exit_price = entry_price * (1 + tp_pct)
                    pnl = (exit_price - entry_price) * position
                    cost = exit_price * position * self.commission
                    capital += (exit_price * position) - cost
                    
                    trades.append(Trade(
                        entry_time=trade_entry_time,
                        entry_price=entry_price,
                        type='LONG',
                        size=position,
                        exit_time=timestamp,
                        exit_price=exit_price,
                        pnl=pnl - cost - entry_cost,
                        pnl_pct=(exit_price - entry_price) / entry_price,
                        exit_reason="TP"
                    ))
                    position = 0
                    continue
            
            # Check Signal
            # 1 = Buy, 0 = Hold, -1 = Sell/Close
            # NOTE: Signals usually come from PREVIOUS candle close. 
            # So if row i has signal, we act on Open of i? Or Close of i?
            # Standard: Signal generated at Close[i], Trade at Open[i+1] or Close[i].
            # For simplicity: Trade at Close[i] (Instant Execution simulation)
            
            signal = signals.iloc[i] if i < len(signals) else 0
            
            if position == 0 and signal == 1:
                # Buy
                # Calculate size (e.g., 95% of capital)
                size = (capital * 0.95) / close
                cost = close * size * self.commission
                
                position = size
                entry_price = close
                trade_entry_time = timestamp
                entry_cost = cost
                
                capital -= (close * size) + cost
                
            elif position > 0 and signal == -1:
                # Sell
                exit_price = close
                pnl = (exit_price - entry_price) * position
                cost = exit_price * position * self.commission
                
                capital += (exit_price * position) - cost
                
                trades.append(Trade(
                    entry_time=trade_entry_time,
                    entry_price=entry_price,
                    type='LONG',
                    size=position,
                    exit_time=timestamp,
                    exit_price=exit_price,
                    pnl=pnl - cost - entry_cost,
                    pnl_pct=(exit_price - entry_price) / entry_price,
                    exit_reason="SIGNAL"
                ))
                position = 0

        # Calculate Stats
        df_trades = pd.DataFrame([t.__dict__ for t in trades])
        if not df_trades.empty:
            total_trades = len(df_trades)
            win_rate = len(df_trades[df_trades['pnl'] > 0]) / total_trades
            total_pnl = df_trades['pnl'].sum()
            total_pnl_pct = (capital - self.initial_capital) / self.initial_capital
        else:
            total_trades = 0
            win_rate = 0.0
            total_pnl = 0.0
            total_pnl_pct = 0.0
            
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=len(df_trades[df_trades['pnl'] > 0]) if not df_trades.empty else 0,
            losing_trades=len(df_trades[df_trades['pnl'] <= 0]) if not df_trades.empty else 0,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=0.0, # TODO: Calc from equity curve
            sharpe_ratio=0.0, # TODO: Calc
            trades=trades,
            equity_curve=pd.Series(equity_curve)
        )
