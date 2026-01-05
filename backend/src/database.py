"""
Database module for AMSE.
Handles SQLite connection and OHLCV data storage.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Index
from sqlalchemy.orm import sessionmaker, declarative_base

# Database path
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DB_PATH = os.path.join(DATA_DIR, 'market.db')

Base = declarative_base()


class Candle(Base):
    """OHLCV candle data model."""
    __tablename__ = 'candles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(50), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )


class Signal(Base):
    """Signal data model (for Phase 2)."""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(50), nullable=False)
    signal = Column(String(20), nullable=False)  # BUY or NO_BUY
    momentum_value = Column(Float, nullable=True)
    
    __table_args__ = (
        Index('idx_signal_symbol_timestamp', 'symbol', 'timestamp'),
    )


# Global engine and session
_engine = None
_Session = None


def init_db():
    """Initialize database and create tables."""
    global _engine, _Session
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    _engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine)
    
    return _engine


def get_session():
    """Get a new database session."""
    global _Session
    if _Session is None:
        init_db()
    return _Session()


def insert_candle(symbol: str, timestamp: datetime, open_price: float, 
                  high: float, low: float, close: float, volume: int) -> Candle:
    """Insert a new candle into the database."""
    session = get_session()
    try:
        candle = Candle(
            timestamp=timestamp,
            symbol=symbol,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume
        )
        session.add(candle)
        session.commit()
        return candle
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_latest_candles(symbol: str, limit: int = 10) -> list:
    """Get the most recent candles for a symbol."""
    session = get_session()
    try:
        candles = session.query(Candle)\
            .filter(Candle.symbol == symbol)\
            .order_by(Candle.timestamp.desc())\
            .limit(limit)\
            .all()
        return list(reversed(candles))  # Return in chronological order
    finally:
        session.close()


def get_last_candle_timestamp(symbol: str) -> datetime | None:
    """Get the timestamp of the most recent candle."""
    session = get_session()
    try:
        candle = session.query(Candle)\
            .filter(Candle.symbol == symbol)\
            .order_by(Candle.timestamp.desc())\
            .first()
        return candle.timestamp if candle else None
    finally:
        session.close()


def candle_exists(symbol: str, timestamp: datetime) -> bool:
    """Check if a candle with given timestamp already exists."""
    session = get_session()
    try:
        exists = session.query(Candle)\
            .filter(Candle.symbol == symbol, Candle.timestamp == timestamp)\
            .first() is not None
        return exists
    finally:
        session.close()
