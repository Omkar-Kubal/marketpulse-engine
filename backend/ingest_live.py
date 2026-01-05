#!/usr/bin/env python3
"""
Live Market Data Ingestion Script for AMSE.
Continuously fetches and stores OHLCV data during market hours.

Usage:
    python ingest_live.py --symbol RELIANCE.NS --interval 5m
"""

import argparse
import signal
import sys
import time
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from src.database import init_db, insert_candle, candle_exists, get_last_candle_timestamp
from src.fetcher import fetch_latest_candle
from src.market_hours import is_market_open, get_market_status, get_seconds_until_market_open

# Ensure data directory exists
import os
os.makedirs('data', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info("Shutdown signal received. Stopping gracefully...")
    running = False


def parse_interval(interval: str) -> int:
    """Parse interval string to seconds."""
    if interval.endswith('m'):
        return int(interval[:-1]) * 60
    elif interval.endswith('h'):
        return int(interval[:-1]) * 3600
    else:
        return 300  # Default 5 minutes


def run_ingestion(symbol: str, interval: str):
    """Main ingestion loop."""
    global running
    
    # Initialize database
    init_db()
    logger.info(f"Database initialized")
    
    poll_seconds = parse_interval(interval)
    logger.info(f"Starting ingestion for {symbol} with {interval} interval ({poll_seconds}s)")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    candles_ingested = 0
    
    while running:
        try:
            # Check market status
            if not is_market_open():
                status = get_market_status()
                wait_seconds = get_seconds_until_market_open()
                
                if wait_seconds > 0:
                    logger.info(f"Market closed. Next open: {status['next_open']}")
                    logger.info(f"Waiting {wait_seconds/3600:.1f} hours...")
                    
                    # Sleep in chunks to allow graceful shutdown
                    sleep_chunk = min(300, wait_seconds)  # Max 5 min chunks
                    while wait_seconds > 0 and running:
                        time.sleep(min(sleep_chunk, wait_seconds))
                        wait_seconds -= sleep_chunk
                    continue
            
            # Fetch latest candle
            candle = fetch_latest_candle(symbol, interval)
            
            if candle is None:
                logger.warning(f"Failed to fetch candle for {symbol}")
                time.sleep(30)  # Wait before retry
                continue
            
            # Check if candle already exists
            if candle_exists(symbol, candle['timestamp']):
                logger.debug(f"Candle {candle['timestamp']} already exists, skipping")
            else:
                # Insert new candle
                insert_candle(
                    symbol=candle['symbol'],
                    timestamp=candle['timestamp'],
                    open_price=candle['open'],
                    high=candle['high'],
                    low=candle['low'],
                    close=candle['close'],
                    volume=candle['volume']
                )
                candles_ingested += 1
                logger.info(
                    f"[{candles_ingested}] Ingested: {symbol} @ {candle['timestamp']} | "
                    f"O:{candle['open']:.2f} H:{candle['high']:.2f} "
                    f"L:{candle['low']:.2f} C:{candle['close']:.2f} V:{candle['volume']:,}"
                )
            
            # Wait for next poll
            logger.debug(f"Waiting {poll_seconds}s for next poll...")
            
            # Sleep in small chunks for graceful shutdown
            for _ in range(poll_seconds):
                if not running:
                    break
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            running = False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(30)  # Wait before retry
    
    logger.info(f"Ingestion stopped. Total candles ingested: {candles_ingested}")


def main():
    parser = argparse.ArgumentParser(
        description='Live market data ingestion for AMSE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ingest_live.py --symbol RELIANCE.NS --interval 5m
    python ingest_live.py --symbol TCS.NS --interval 15m
        """
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='RELIANCE.NS',
        help='Stock symbol (default: RELIANCE.NS)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=str,
        default='5m',
        choices=['1m', '5m', '15m', '30m', '1h'],
        help='Candle interval (default: 5m)'
    )
    
    args = parser.parse_args()
    
    print(f"""
============================================================
            AMSE - Live Market Data Ingestion              
============================================================
  Symbol:   {args.symbol}
  Interval: {args.interval}
============================================================
    """)
    
    run_ingestion(args.symbol, args.interval)


if __name__ == '__main__':
    main()
