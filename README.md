# AMSE - Automated Market Signal Engine

An automated market signal system that consumes near real-time market data and generates buy signals.

## Quick Start

```bash
cd backend
pip install -r requirements.txt

# Start live ingestion (during market hours)
python ingest_live.py --symbol RELIANCE.NS --interval 5m
```

## Project Structure

```
backend/
├── src/           # Core modules
├── api/           # FastAPI endpoints
├── data/          # SQLite database
└── trained_models/# ML model artifacts
```

## Phases

1. **Phase 1** - Live Market Data Ingestion ✓
2. **Phase 2** - Basic Signal Generation
3. **Phase 3** - Backend API
4. **Phase 4** - ML Model Integration
5. **Phase 5** - Multi-Stock Scaling
