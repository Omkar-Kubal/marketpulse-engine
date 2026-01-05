#!/usr/bin/env python3
"""
Signal Generation Script for AMSE.
Generates BUY/NO_BUY signals based on momentum.

Usage:
    python generate_signal.py --symbol RELIANCE.NS
    python generate_signal.py --symbol RELIANCE.NS --watch
"""

import argparse
import sys
import time
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from src.database import init_db
from src.signals import get_signal_for_symbol, save_signal
from src.market_hours import is_market_open, get_market_status
from src.fetcher import fetch_latest_candle
from src.database import insert_candle, candle_exists

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def display_signal(result: dict):
    """Display signal result in a formatted way."""
    if result.get('error'):
        print(f"\n[!] Error: {result['error']}")
        return
    
    signal = result['signal']
    momentum_pct = result['momentum_pct']
    close = result['close']
    symbol = result['symbol']
    
    # Signal indicator
    if signal == "BUY":
        indicator = "[+] BUY"
    else:
        indicator = "[-] NO_BUY"
    
    print(f"""
============================================================
                     SIGNAL UPDATE                         
============================================================
  Symbol:     {symbol}
  Signal:     {indicator}
  Momentum:   {momentum_pct}
  Close:      Rs.{close:.2f}
  Threshold:  {result['threshold']*100:.1f}% (lookback: {result['lookback']} candles)
  Updated:    {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
============================================================
    """)


def fetch_and_store_candle(symbol: str, interval: str = '5m') -> bool:
    """Fetch latest candle and store if new."""
    candle = fetch_latest_candle(symbol, interval)
    if candle and not candle_exists(symbol, candle['timestamp']):
        insert_candle(
            symbol=candle['symbol'],
            timestamp=candle['timestamp'],
            open_price=candle['open'],
            high=candle['high'],
            low=candle['low'],
            close=candle['close'],
            volume=candle['volume']
        )
        logger.info(f"Stored new candle: {candle['timestamp']}")
        return True
    return False


def run_signal_generation(symbol: str, lookback: int, threshold: float, 
                          watch: bool = False, interval: str = '5m'):
    """Generate and display signal."""
    init_db()
    
    # First, ensure we have recent data
    fetch_and_store_candle(symbol, interval)
    
    if not watch:
        # Single run
        result = get_signal_for_symbol(symbol, lookback, threshold)
        display_signal(result)
        
        # Save to database if valid
        if result['signal']:
            save_signal(symbol, result['signal'], result['momentum'])
            logger.info(f"Signal saved: {result['signal']}")
    else:
        # Watch mode - continuous updates
        print(f"\n[*] Watching {symbol} for signal changes (Ctrl+C to stop)...\n")
        last_signal = None
        
        try:
            while True:
                # Fetch new data if market is open
                if is_market_open():
                    fetch_and_store_candle(symbol, interval)
                
                result = get_signal_for_symbol(symbol, lookback, threshold)
                
                # Only display if signal changes or every minute
                current_signal = result.get('signal')
                if current_signal != last_signal or current_signal:
                    display_signal(result)
                    
                    if result['signal']:
                        save_signal(symbol, result['signal'], result['momentum'])
                    
                    last_signal = current_signal
                
                # Wait before next check
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\n\nWatch mode stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate trading signals for AMSE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_signal.py --symbol RELIANCE.NS
    python generate_signal.py --symbol TCS.NS --lookback 5 --threshold 0.01
    python generate_signal.py --symbol RELIANCE.NS --watch
        """
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='RELIANCE.NS',
        help='Stock symbol (default: RELIANCE.NS)'
    )
    parser.add_argument(
        '--lookback', '-l',
        type=int,
        default=3,
        help='Momentum lookback period in candles (default: 3)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.005,
        help='Signal threshold as decimal, e.g., 0.005 for 0.5%% (default: 0.005)'
    )
    parser.add_argument(
        '--watch', '-w',
        action='store_true',
        help='Watch mode - continuously update signal'
    )
    parser.add_argument(
        '--interval', '-i',
        type=str,
        default='5m',
        choices=['1m', '5m', '15m', '30m', '1h'],
        help='Candle interval for fetching new data (default: 5m)'
    )
    
    args = parser.parse_args()
    
    print(f"""
============================================================
            AMSE - Signal Generation                       
============================================================
  Symbol:    {args.symbol}
  Lookback:  {args.lookback}
  Threshold: {args.threshold*100:.1f}%
============================================================
    """)
    
    run_signal_generation(
        symbol=args.symbol,
        lookback=args.lookback,
        threshold=args.threshold,
        watch=args.watch,
        interval=args.interval
    )


if __name__ == '__main__':
    main()
