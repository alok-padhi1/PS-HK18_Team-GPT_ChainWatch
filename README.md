# ChainWatch — AI-Powered Blockchain Monitoring Platform


THE PPTX link:https://www.canva.com/design/DAHB693E9jI/MVEiSukOJX6tMBBIGvC0SA/edit?utm_content=DAHB693E9jI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

> Real-time Ethereum blockchain monitoring with ML-based anomaly detection, graph analysis, flash-loan detection, and wash-trading identification.

![ChainWatch](https://img.shields.io/badge/ChainWatch-AI%20Monitor-6366f1?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)

---

## Architecture

```
ChainWatch
├── backend/              # FastAPI + Python ML pipeline
│   ├── app/
│   │   ├── main.py           # API endpoints & background polling
│   │   ├── config.py         # Environment configuration
│   │   ├── database.py       # SQLAlchemy models & session
│   │   └── services/
│   │       ├── blockchain.py     # Ethereum Web3 integration
│   │       ├── ml_engine.py      # Isolation Forest + KMeans
│   │       ├── graph_analysis.py # NetworkX graph analysis
│   │       ├── flash_loan.py     # Flash-loan detection
│   │       └── risk_engine.py    # Composite risk scoring
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/             # React 18 + Vite + TailwindCSS
│   ├── src/
│   │   ├── App.jsx           # Layout & routing
│   │   ├── api.js            # API client
│   │   ├── index.css         # Tailwind + glassmorphism styles
│   │   └── pages/
│   │       ├── Dashboard.jsx     # Main dashboard
│   │       ├── WalletDetail.jsx  # Wallet profiles & risk
│   │       ├── GraphView.jsx     # Force-directed graph
│   │       └── TimelineView.jsx  # Timeline & flash-loans
│   ├── Dockerfile
│   └── nginx.conf
└── docker-compose.yml
```

## Features

- **Live Ethereum Monitoring** — Polls Ethereum mainnet every 12s for new blocks
- **ML Anomaly Detection** — Isolation Forest identifies suspicious transaction patterns
- **Wallet Clustering** — KMeans groups wallets by behavioural similarity
- **Flash-Loan Detection** — Identifies same-block high-value round-trip patterns
- **Wash-Trade Detection** — Graph cycle analysis finds bidirectional value transfers
- **Composite Risk Scoring** — Weighted combination of 4 detection engines (0-100 scale)
- **Explainable Alerts** — Each alert includes human-readable explanation of triggers
- **Network Graph Visualization** — Interactive force-directed wallet interaction graph
- **Dark Theme Dashboard** — Glassmorphism UI with animated charts

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + Web3 connection status |
| GET | `/api/live-transactions` | Recent transactions from DB |
| POST | `/api/run-detection` | Trigger full ML/graph detection pipeline |
| GET | `/api/wallet-profiles` | All wallet profiles sorted by risk |
| GET | `/api/alerts` | Generated alerts (filterable) |
| GET | `/api/graph-data` | Wallet interaction graph for visualization |
| GET | `/api/timeline-data` | Time-series transaction/alert volume |
| GET | `/api/risk-score/{address}` | Composite risk breakdown for a wallet |
| POST | `/api/ingest` | Manually ingest latest N blocks |
| GET | `/api/flash-loans` | Detected flash-loan events |
| GET | `/api/wash-trades` | Detected wash-trading pairs |

---

## Quick Start (Local Development)

### Prerequisites

- Python 3.11+
- Node.js 18+
- An Ethereum RPC URL ([Alchemy](https://alchemy.com) or [Infura](https://infura.io) free tier)

### 1. Backend

```bash
cd backend

# Create virtualenv
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and set your ETH_RPC_URL

# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (proxies /api to backend on port 8000)
npm run dev
```

Open `http://localhost:5173` to access the dashboard.

---

## Docker Deployment

```bash
# Copy and configure env
copy backend\.env.example backend\.env
# Edit backend/.env with your ETH_RPC_URL

# Build and run
docker-compose up --build -d

# Access:
# Frontend: http://localhost
# Backend:  http://localhost:8000/api/health
```

---

## Deploy to Cloud

### Backend → Render

1. Create a new **Web Service** on [Render](https://render.com)
2. Connect your GitHub repo, set root directory to `backend/`
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `ETH_RPC_URL` = your Alchemy/Infura URL
   - `DATABASE_URL` = `sqlite:///data/chainwatch.db` (or a PostgreSQL URL)

### Frontend → Vercel

1. Create a new project on [Vercel](https://vercel.com)
2. Set root directory to `frontend/`
3. Set **Build Command**: `npm run build`
4. Set **Output Directory**: `dist`
5. Add a rewrite rule in `vercel.json`:
```json
{
  "rewrites": [
    { "source": "/api/:path*", "destination": "https://your-render-backend.onrender.com/api/:path*" }
  ]
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ETH_RPC_URL` | Alchemy demo | Ethereum JSON-RPC endpoint |
| `DATABASE_URL` | SQLite (local) | Database connection string |
| `POLL_INTERVAL` | `12` | Block polling interval (seconds) |
| `ANOMALY_CONTAMINATION` | `0.05` | Isolation Forest contamination rate |
| `N_CLUSTERS` | `5` | KMeans cluster count |
| `CORS_ORIGINS` | `localhost` | Allowed CORS origins (comma-separated) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.11, SQLAlchemy, Web3.py |
| ML | scikit-learn (Isolation Forest, KMeans) |
| Graph | NetworkX |
| Frontend | React 18, Vite, TailwindCSS 3 |
| Charts | Recharts, React Force Graph 2D |
| Animations | Framer Motion |
| Icons | Lucide React |
| Database | SQLite (dev) / PostgreSQL (prod) |
| Deployment | Docker, Render, Vercel |

---

## License

MIT
