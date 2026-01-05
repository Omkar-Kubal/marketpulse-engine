"""
Data fetcher module for AMSE.
Handles fetching OHLCV data from Yahoo Finance.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
import pytz

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IST = pytz.timezone('Asia/Kolkata')

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds


def fetch_latest_candle(symbol: str, interval: str = '5m') -> Optional[dict]:
    """
    Fetch the latest candle for a symbol using yfinance.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS')
        interval: Candle interval (e.g., '5m', '15m', '1h')
    
    Returns:
        Dict with OHLCV data or None if fetch fails
    """
    retries = 0
    backoff = INITIAL_BACKOFF
    
    while retries < MAX_RETRIES:
        try:
            # Fetch data for the last 1 day to get recent candles
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1d', interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Get the most recent complete candle
            latest = df.iloc[-1]
            timestamp = df.index[-1]
            
            # Convert to IST if needed
            if timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone(IST)
            
            return {
                'timestamp': timestamp.to_pydatetime().replace(tzinfo=None),
                'symbol': symbol,
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'close': float(latest['Close']),
                'volume': int(latest['Volume'])
            }
            
        except Exception as e:
            retries += 1
            logger.warning(f"Fetch attempt {retries} failed for {symbol}: {e}")
            
            if retries < MAX_RETRIES:
                logger.info(f"Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
            else:
                logger.error(f"All retries exhausted for {symbol}")
                return None
    
    return None


def fetch_historical_candles(symbol: str, period: str = '1mo', 
                             interval: str = '5m') -> list[dict]:
    """
    Fetch historical candles for training/backtesting.
    
    Args:
        symbol: Stock symbol
        period: Data period (e.g., '1mo', '3mo', '1y')
        interval: Candle interval
    
    Returns:
        List of OHLCV dicts
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No historical data for {symbol}")
            return []
        
        candles = []
        for idx, row in df.iterrows():
            timestamp = idx
            if timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone(IST)
            
            candles.append({
                'timestamp': timestamp.to_pydatetime().replace(tzinfo=None),
                'symbol': symbol,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        logger.info(f"Fetched {len(candles)} historical candles for {symbol}")
        return candles
        
    except Exception as e:
        logger.error(f"Failed to fetch historical data for {symbol}: {e}")
        return []


if __name__ == '__main__':
    # Quick test
    print("Testing fetcher with RELIANCE.NS...")
    candle = fetch_latest_candle('RELIANCE.NS', '5m')
    if candle:
        print(f"Latest candle:")
        print(f"  Time: {candle['timestamp']}")
        print(f"  Open: {candle['open']:.2f}")
        print(f"  High: {candle['high']:.2f}")
        print(f"  Low: {candle['low']:.2f}")
        print(f"  Close: {candle['close']:.2f}")
        print(f"  Volume: {candle['volume']:,}")
    else:
        print("Failed to fetch candle (market may be closed)")
