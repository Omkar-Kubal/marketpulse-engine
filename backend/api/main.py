"""
FastAPI Backend for AMSE.
Provides REST endpoints for signals and price data.
"""

import sys
import os
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.database import init_db, get_latest_candles, get_last_candle_timestamp
from src.signals import get_signal_for_symbol, get_latest_signals
from src.market_hours import get_market_status
from src.fetcher import fetch_latest_candle

# Initialize database
init_db()

# Create FastAPI app
app = FastAPI(
    title="AMSE API",
    description="Automated Market Signal Engine API",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class SignalResponse(BaseModel):
    symbol: str
    signal: Optional[str]
    momentum: Optional[float]
    momentum_pct: Optional[str]
    close: Optional[float]
    timestamp: datetime
    candle_timestamp: Optional[datetime]
    error: Optional[str] = None


class PriceResponse(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketStatusResponse(BaseModel):
    is_open: bool
    current_time: str
    market_open: str
    market_close: str
    next_open: Optional[str]


class HealthResponse(BaseModel):
    status: str
    last_update: Optional[datetime]
    market_status: dict


# Active WebSocket connections
active_connections: list[WebSocket] = []


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "AMSE API",
        "version": "1.0.0",
        "endpoints": {
            "signal": "/api/signal/{symbol}",
            "price": "/api/price/{symbol}",
            "status": "/api/status",
            "market": "/api/market"
        }
    }


@app.get("/api/signal/{symbol}", response_model=SignalResponse)
async def get_signal(symbol: str, lookback: int = 3, threshold: float = 0.005):
    """
    Get current signal for a symbol.
    
    - **symbol**: Stock symbol (e.g., RELIANCE.NS)
    - **lookback**: Number of candles for momentum calculation
    - **threshold**: Signal threshold (0.005 = 0.5%)
    """
    try:
        result = get_signal_for_symbol(symbol, lookback, threshold)
        return SignalResponse(
            symbol=result['symbol'],
            signal=result.get('signal'),
            momentum=result.get('momentum'),
            momentum_pct=result.get('momentum_pct'),
            close=result.get('close'),
            timestamp=result['timestamp'],
            candle_timestamp=result.get('candle_timestamp'),
            error=result.get('error')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/price/{symbol}")
async def get_price(symbol: str, limit: int = 10):
    """
    Get latest price data for a symbol.
    
    - **symbol**: Stock symbol
    - **limit**: Number of candles to return
    """
    try:
        candles = get_latest_candles(symbol, limit)
        if not candles:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        return {
            "symbol": symbol,
            "count": len(candles),
            "candles": [
                {
                    "timestamp": c.timestamp.isoformat(),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume
                }
                for c in candles
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/price/{symbol}/latest")
async def get_latest_price(symbol: str, fetch_new: bool = False):
    """
    Get the most recent price for a symbol.
    
    - **symbol**: Stock symbol
    - **fetch_new**: If True, fetch fresh data from Yahoo Finance
    """
    try:
        if fetch_new:
            candle = fetch_latest_candle(symbol)
            if candle:
                return {
                    "symbol": symbol,
                    "timestamp": candle['timestamp'].isoformat(),
                    "open": candle['open'],
                    "high": candle['high'],
                    "low": candle['low'],
                    "close": candle['close'],
                    "volume": candle['volume'],
                    "source": "live"
                }
        
        candles = get_latest_candles(symbol, limit=1)
        if not candles:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        c = candles[0]
        return {
            "symbol": symbol,
            "timestamp": c.timestamp.isoformat(),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "source": "database"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market", response_model=MarketStatusResponse)
async def get_market():
    """Get current market status."""
    status = get_market_status()
    return MarketStatusResponse(**status)


@app.get("/api/status", response_model=HealthResponse)
async def get_status(symbol: str = "RELIANCE.NS"):
    """
    Get system health and status.
    
    - **symbol**: Symbol to check for last update
    """
    last_update = get_last_candle_timestamp(symbol)
    market_status = get_market_status()
    
    return HealthResponse(
        status="healthy",
        last_update=last_update,
        market_status=market_status
    )


@app.get("/api/signals/{symbol}/history")
async def get_signal_history(symbol: str, limit: int = 20):
    """
    Get signal history for a symbol.
    
    - **symbol**: Stock symbol
    - **limit**: Number of signals to return
    """
    try:
        signals = get_latest_signals(symbol, limit)
        return {
            "symbol": symbol,
            "count": len(signals),
            "signals": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "signal": s.signal,
                    "momentum": s.momentum_value
                }
                for s in signals
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket for real-time updates
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time signal updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Wait for client messages (keep-alive)
            data = await websocket.receive_text()
            
            # If client sends a symbol, respond with current signal
            if data:
                result = get_signal_for_symbol(data)
                await websocket.send_json({
                    "type": "signal",
                    "data": {
                        "symbol": result['symbol'],
                        "signal": result.get('signal'),
                        "momentum_pct": result.get('momentum_pct'),
                        "close": result.get('close'),
                        "timestamp": result['timestamp'].isoformat()
                    }
                })
    except WebSocketDisconnect:
        active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
