"""
Signal generation module for AMSE.
Generates BUY/NO_BUY signals based on momentum.
"""

import logging
from datetime import datetime
from typing import Optional
from src.database import get_session, Signal, get_latest_candles

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_LOOKBACK = 3  # Number of candles for momentum calculation
DEFAULT_THRESHOLD = 0.005  # 0.5% momentum threshold for BUY signal


def calculate_momentum(candles: list, lookback: int = DEFAULT_LOOKBACK) -> Optional[float]:
    """
    Calculate momentum as percentage change over lookback period.
    
    Args:
        candles: List of candle dicts (in chronological order)
        lookback: Number of periods to look back
    
    Returns:
        Momentum as decimal (0.01 = 1%) or None if insufficient data
    """
    if len(candles) < lookback + 1:
        logger.warning(f"Insufficient candles for momentum calculation. Need {lookback + 1}, got {len(candles)}")
        return None
    
    # Get close prices
    close_now = candles[-1].close if hasattr(candles[-1], 'close') else candles[-1]['close']
    close_ago = candles[-(lookback + 1)].close if hasattr(candles[-(lookback + 1)], 'close') else candles[-(lookback + 1)]['close']
    
    if close_ago == 0:
        return None
    
    momentum = (close_now - close_ago) / close_ago
    return momentum


def generate_signal(momentum: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """
    Generate trading signal based on momentum.
    
    Args:
        momentum: Momentum value as decimal
        threshold: Threshold for BUY signal
    
    Returns:
        "BUY" or "NO_BUY"
    """
    if momentum > threshold:
        return "BUY"
    else:
        return "NO_BUY"


def get_signal_for_symbol(symbol: str, lookback: int = DEFAULT_LOOKBACK, 
                          threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Get the current signal for a symbol.
    
    Args:
        symbol: Stock symbol
        lookback: Momentum lookback period
        threshold: Signal threshold
    
    Returns:
        Dict with signal details
    """
    # Get recent candles
    candles = get_latest_candles(symbol, limit=lookback + 5)
    
    if not candles:
        return {
            'symbol': symbol,
            'signal': None,
            'momentum': None,
            'error': 'No candle data available',
            'timestamp': datetime.now()
        }
    
    # Calculate momentum
    momentum = calculate_momentum(candles, lookback)
    
    if momentum is None:
        return {
            'symbol': symbol,
            'signal': None,
            'momentum': None,
            'error': 'Insufficient data for momentum calculation',
            'timestamp': datetime.now()
        }
    
    # Generate signal
    signal = generate_signal(momentum, threshold)
    
    # Get latest candle info
    latest_candle = candles[-1]
    latest_close = latest_candle.close if hasattr(latest_candle, 'close') else latest_candle['close']
    latest_timestamp = latest_candle.timestamp if hasattr(latest_candle, 'timestamp') else latest_candle['timestamp']
    
    return {
        'symbol': symbol,
        'signal': signal,
        'momentum': momentum,
        'momentum_pct': f"{momentum * 100:.2f}%",
        'close': latest_close,
        'candle_timestamp': latest_timestamp,
        'timestamp': datetime.now(),
        'threshold': threshold,
        'lookback': lookback
    }


def save_signal(symbol: str, signal: str, momentum: float) -> Signal:
    """Save a signal to the database."""
    session = get_session()
    try:
        signal_record = Signal(
            timestamp=datetime.now(),
            symbol=symbol,
            signal=signal,
            momentum_value=momentum
        )
        session.add(signal_record)
        session.commit()
        return signal_record
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_latest_signals(symbol: str, limit: int = 10) -> list:
    """Get the most recent signals for a symbol."""
    session = get_session()
    try:
        signals = session.query(Signal)\
            .filter(Signal.symbol == symbol)\
            .order_by(Signal.timestamp.desc())\
            .limit(limit)\
            .all()
        return list(reversed(signals))
    finally:
        session.close()
