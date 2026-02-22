"""
ChainWatch Configuration Module
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Ethereum RPC (supports Alchemy, Infura, or public nodes)
ETH_RPC_URL = os.getenv("ETH_RPC_URL", "https://ethereum-rpc.publicnode.com")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'chainwatch.db'}")

# Polling interval (seconds) – Ethereum avg block time is ~12s
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "12"))

# ML settings
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "5"))

# Risk weights
RISK_WEIGHTS = {
    "ml_anomaly": 0.35,
    "graph_suspicion": 0.25,
    "flash_loan": 0.20,
    "wash_trade": 0.20,
}

# CORS
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:8000"
).split(",")

# Server port (for local deployment)
PORT = int(os.getenv("PORT", "8000"))
