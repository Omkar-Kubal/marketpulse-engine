"""
Market hours module for AMSE.
Handles IST market hours detection for NSE.
"""

from datetime import datetime, time, timedelta
import pytz

# Indian Standard Time
IST = pytz.timezone('Asia/Kolkata')

# NSE Market hours
MARKET_OPEN = time(9, 15)   # 9:15 AM
MARKET_CLOSE = time(15, 30)  # 3:30 PM


def get_current_ist_time() -> datetime:
    """Get current time in IST."""
    return datetime.now(IST)


def is_weekday(dt: datetime = None) -> bool:
    """Check if given datetime is a weekday (Mon-Fri)."""
    if dt is None:
        dt = get_current_ist_time()
    return dt.weekday() < 5  # 0=Monday, 4=Friday


def is_market_hours(dt: datetime = None) -> bool:
    """Check if given datetime is within market hours."""
    if dt is None:
        dt = get_current_ist_time()
    
    current_time = dt.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def is_market_open(dt: datetime = None) -> bool:
    """
    Check if market is currently open.
    Returns True if it's a weekday and within market hours.
    """
    if dt is None:
        dt = get_current_ist_time()
    
    return is_weekday(dt) and is_market_hours(dt)


def get_next_market_open() -> datetime:
    """
    Get the datetime of the next market open.
    If market is currently open, returns today's open time.
    """
    now = get_current_ist_time()
    
    # Start with today's market open
    next_open = now.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, 
                            second=0, microsecond=0)
    
    # If we're past today's open time, move to tomorrow
    if now.time() > MARKET_OPEN:
        next_open += timedelta(days=1)
    
    # Skip weekends
    while next_open.weekday() >= 5:  # Saturday=5, Sunday=6
        next_open += timedelta(days=1)
    
    return next_open


def get_seconds_until_market_open() -> float:
    """Get seconds until next market open."""
    now = get_current_ist_time()
    next_open = get_next_market_open()
    return (next_open - now).total_seconds()


def get_market_status() -> dict:
    """Get current market status as a dict."""
    now = get_current_ist_time()
    is_open = is_market_open(now)
    
    return {
        'is_open': is_open,
        'current_time': now.isoformat(),
        'market_open': MARKET_OPEN.isoformat(),
        'market_close': MARKET_CLOSE.isoformat(),
        'next_open': get_next_market_open().isoformat() if not is_open else None
    }


if __name__ == '__main__':
    # Quick test
    status = get_market_status()
    print(f"Market Status: {'OPEN' if status['is_open'] else 'CLOSED'}")
    print(f"Current IST: {status['current_time']}")
    if not status['is_open']:
        print(f"Next Open: {status['next_open']}")
