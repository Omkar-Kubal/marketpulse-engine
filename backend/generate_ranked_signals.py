#!/usr/bin/env python3
"""
Ranked Signal Generation Script for AMSE.
Generates and ranks signals for multiple stocks.

Usage:
    python generate_ranked_signals.py
    python generate_ranked_signals.py --symbols stock_list.txt
"""

import argparse
import sys
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Add src to path
sys.path.insert(0, 'src')

from src.database import init_db
from src.fetcher import fetch_latest_candle
from src.database import insert_candle, candle_exists, get_latest_candles
from src.signals import get_signal_for_symbol
from src.features import prepare_inference_data
from src.models.trainer import load_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default stock list
DEFAULT_STOCKS = [
    'RELIANCE.NS',
    'TCS.NS',
    'HDFCBANK.NS',
    'INFY.NS',
    'ICICIBANK.NS',
    'HINDUNILVR.NS',
    'SBIN.NS',
    'BHARTIARTL.NS',
    'ITC.NS',
    'KOTAKBANK.NS'
]


def fetch_and_store_candle(symbol: str) -> Optional[dict]:
    """Fetch and store latest candle for a symbol."""
    try:
        candle = fetch_latest_candle(symbol, '5m')
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
        return candle
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return None


def get_ml_score(symbol: str, model_data: dict) -> Optional[float]:
    """Get ML buy score for a symbol."""
    try:
        candles = get_latest_candles(symbol, limit=20)
        if len(candles) < 15:
            return None
        
        X = prepare_inference_data(candles)
        if X is None:
            return None
        
        X_scaled = model_data['scaler'].transform(X)
        proba = model_data['model'].predict_proba(X_scaled)[0]
        return float(proba[1])  # Probability of BUY class
    except Exception as e:
        logger.error(f"ML inference failed for {symbol}: {e}")
        return None


def calculate_market_condition_score(result: dict) -> float:
    """
    Calculate market condition score based on momentum.
    Returns a score between 0 and 1.
    """
    momentum = result.get('momentum', 0)
    if momentum is None:
        return 0.5  # Neutral
    
    # Normalize momentum to 0-1 range
    # Typical momentum range: -0.05 to +0.05
    normalized = (momentum + 0.05) / 0.1
    return max(0, min(1, normalized))


def calculate_final_score(ml_score: Optional[float], 
                          market_score: float,
                          ml_weight: float = 0.6) -> float:
    """
    Calculate final ranked score.
    Formula: 0.6 * ML_score + 0.4 * market_condition_score
    """
    if ml_score is None:
        # Fall back to market score only
        return market_score
    
    market_weight = 1 - ml_weight
    return ml_weight * ml_score + market_weight * market_score


def generate_ranked_signals(symbols: list, use_ml: bool = True) -> list:
    """
    Generate and rank signals for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        use_ml: Whether to use ML model for scoring
    
    Returns:
        List of ranked signal results
    """
    init_db()
    
    # Load ML model if available
    model_data = None
    if use_ml:
        model_data = load_model('xgboost_latest')
        if model_data is None:
            logger.warning("No ML model found, using momentum-based scoring only")
    
    results = []
    
    # Fetch data in parallel
    logger.info(f"Fetching data for {len(symbols)} symbols...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {
            executor.submit(fetch_and_store_candle, symbol): symbol 
            for symbol in symbols
        }
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
    
    # Generate signals
    for symbol in symbols:
        try:
            # Get momentum-based signal
            result = get_signal_for_symbol(symbol, lookback=3, threshold=0.005)
            
            if result.get('error'):
                continue
            
            # Get ML score
            ml_score = None
            if model_data:
                ml_score = get_ml_score(symbol, model_data)
            
            # Calculate scores
            market_score = calculate_market_condition_score(result)
            final_score = calculate_final_score(ml_score, market_score)
            
            results.append({
                'symbol': symbol,
                'signal': result['signal'],
                'final_score': final_score,
                'ml_score': ml_score,
                'market_score': market_score,
                'momentum': result.get('momentum'),
                'momentum_pct': result.get('momentum_pct'),
                'close': result.get('close'),
                'timestamp': result['timestamp']
            })
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Sort by final score (descending)
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return results


def load_symbols_from_file(filepath: str) -> list:
    """Load symbol list from file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def display_ranked_results(results: list):
    """Display ranked results in a formatted table."""
    print(f"""
============================================================
            RANKED SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
============================================================
{'Rank':<5} {'Symbol':<15} {'Signal':<10} {'Score':<8} {'ML':<8} {'Momentum':<10}
------------------------------------------------------------""")
    
    for i, r in enumerate(results, 1):
        ml_str = f"{r['ml_score']*100:.1f}%" if r['ml_score'] is not None else "N/A"
        mom_str = r.get('momentum_pct', 'N/A')
        signal_str = "[+]" if r['signal'] == 'BUY' else "[-]"
        
        print(f"{i:<5} {r['symbol']:<15} {signal_str:<10} {r['final_score']*100:.1f}%    {ml_str:<8} {mom_str:<10}")
    
    print("============================================================")


def main():
    parser = argparse.ArgumentParser(
        description='Generate ranked signals for multiple stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_ranked_signals.py
    python generate_ranked_signals.py --symbols stock_list.txt
    python generate_ranked_signals.py --no-ml
        """
    )
    parser.add_argument(
        '--symbols', '-s',
        type=str,
        default=None,
        help='Path to file with symbol list (one per line)'
    )
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Do not use ML model, use only momentum-based scoring'
    )
    parser.add_argument(
        '--top', '-t',
        type=int,
        default=10,
        help='Number of top signals to show (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Load symbols
    if args.symbols:
        symbols = load_symbols_from_file(args.symbols)
    else:
        symbols = DEFAULT_STOCKS
    
    print(f"""
============================================================
            AMSE - Ranked Signal Generation                
============================================================
  Symbols: {len(symbols)} stocks
  ML Model: {'Enabled' if not args.no_ml else 'Disabled'}
============================================================
    """)
    
    # Generate ranked signals
    results = generate_ranked_signals(symbols, use_ml=not args.no_ml)
    
    if not results:
        print("\n[!] No signals generated. Check that data is available.")
        return
    
    # Display top results
    display_ranked_results(results[:args.top])
    
    # Show buy recommendations
    buy_signals = [r for r in results if r['signal'] == 'BUY']
    if buy_signals:
        print(f"\n[+] BUY Recommendations: {', '.join(r['symbol'] for r in buy_signals[:5])}")
    else:
        print("\n[-] No BUY signals at current thresholds")


if __name__ == '__main__':
    main()
