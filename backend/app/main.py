"""
ChainWatch FastAPI Application
Main entry point with all REST API endpoints, background polling,
and automatic analysis pipeline triggers.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import CORS_ORIGINS
from app.database import init_db, get_db, Transaction, WalletProfile, Alert, RiskScore
from app.services.blockchain import blockchain_service
from app.services.ml_engine import ml_engine
from app.services.graph_analysis import graph_analyzer
from app.services.flash_loan import flash_loan_detector
from app.services.risk_engine import risk_engine

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("chainwatch")


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB, bulk-ingest blocks, start background polling with auto-analysis."""
    init_db()
    logger.info("ChainWatch starting up...")

    # ── Initial bulk ingest: fetch 20 blocks for massive starting dataset ──
    async def initial_ingest():
        """Ingest 20 recent blocks on startup for a rich initial dataset."""
        try:
            logger.info("▶ Starting bulk ingest of 20 blocks for initial data...")
            result = await asyncio.to_thread(
                blockchain_service.ingest_latest_blocks, 20
            )
            logger.info(
                f"✅ Bulk ingest complete: {result['transactions_fetched']} txs from "
                f"{result['blocks_processed']} blocks ({result['new_transactions_stored']} stored)"
            )

            # Run full detection pipeline on initial data
            def run_initial_pipeline():
                from app.database import SessionLocal
                db = SessionLocal()
                try:
                    train_result = ml_engine.train(db)
                    logger.info(f"ML training: {train_result.get('status', 'unknown')}")

                    predictions = ml_engine.predict(db)
                    if predictions:
                        ml_engine.update_wallet_profiles(db, predictions)
                        logger.info(f"Profiled {len(predictions)} wallets")

                    graph_result = graph_analyzer.build_graph(db)
                    logger.info(
                        f"Graph: {graph_result.get('nodes', 0)} nodes, "
                        f"{graph_result.get('edges', 0)} edges, "
                        f"{graph_result.get('communities', 0)} communities"
                    )

                    result = risk_engine.run_full_detection(db)
                    logger.info(
                        f"Initial detection: {result.get('alerts_generated', 0)} alerts, "
                        f"{result.get('wallets_profiled', 0)} wallets profiled"
                    )
                    blockchain_service.last_analysis_at = datetime.utcnow()
                except Exception as e:
                    logger.error(f"Initial pipeline error: {e}", exc_info=True)
                finally:
                    db.close()

            await asyncio.to_thread(run_initial_pipeline)
            logger.info("✅ Initial analysis pipeline complete. Dashboard ready.")

        except Exception as e:
            logger.error(f"Bulk ingest error: {e}", exc_info=True)

    await initial_ingest()

    # ── Background polling for new blocks ──
    async def on_new_block(block_number: int):
        """Callback after each polled block batch – run full detection pipeline."""
        logger.info(f"Block {block_number} ingested. Running detection pipeline...")

        def run_full_pipeline():
            from app.database import SessionLocal
            db = SessionLocal()
            try:
                train_result = ml_engine.train(db)
                predictions = ml_engine.predict(db)
                if predictions:
                    ml_engine.update_wallet_profiles(db, predictions)

                graph_analyzer.build_graph(db)

                result = risk_engine.run_full_detection(db)
                logger.info(
                    f"Detection: {result.get('alerts_generated', 0)} alerts, "
                    f"{result.get('wallets_profiled', 0)} wallets"
                )

                blockchain_service.last_analysis_at = datetime.utcnow()
            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)
            finally:
                db.close()

        try:
            await asyncio.to_thread(run_full_pipeline)
        except Exception as e:
            logger.error(f"Post-block detection thread error: {e}")

    polling_task = asyncio.create_task(
        blockchain_service.start_polling(callback=on_new_block)
    )

    yield

    blockchain_service.stop_polling()
    polling_task.cancel()
    logger.info("ChainWatch shut down.")


app = FastAPI(
    title="ChainWatch API",
    description="AI-powered Ethereum blockchain monitoring and anomaly detection platform",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS – allow all common origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "web3_connected": blockchain_service.is_connected,
        "mode": blockchain_service.mode,
        "last_processed_block": blockchain_service.last_processed_block,
    }


# ─── Status (detailed monitoring) ─────────────────────────────────────────────

@app.get("/api/status")
def get_status(db: Session = Depends(get_db)):
    """
    Detailed system status: RPC connection, last block, total transactions,
    last analysis timestamp, and mode (live / simulation).
    """
    total_tx = db.query(func.count(Transaction.id)).scalar() or 0
    total_wallets = db.query(func.count(WalletProfile.id)).scalar() or 0
    total_alerts = db.query(func.count(Alert.id)).scalar() or 0

    return {
        "rpc_connected": blockchain_service.is_connected,
        "mode": blockchain_service.mode,
        "rpc_url": blockchain_service._rpc_url[:50] + "..." if len(blockchain_service._rpc_url) > 50 else blockchain_service._rpc_url,
        "last_processed_block": blockchain_service.last_processed_block,
        "total_blocks_processed": blockchain_service.total_blocks_processed,
        "total_transactions": total_tx,
        "total_wallets_profiled": total_wallets,
        "total_alerts": total_alerts,
        "last_analysis_at": blockchain_service.last_analysis_at.isoformat() if blockchain_service.last_analysis_at else None,
        "polling_active": blockchain_service._polling,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─── Live Transactions ────────────────────────────────────────────────────────

@app.get("/api/live-transactions")
def get_live_transactions(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get the most recent transactions from the database."""
    txns = (
        db.query(Transaction)
        .order_by(Transaction.block_number.desc(), Transaction.id.desc())
        .limit(limit)
        .all()
    )
    return {
        "count": len(txns),
        "transactions": [
            {
                "tx_hash": tx.tx_hash,
                "block_number": tx.block_number,
                "from_address": tx.from_address,
                "to_address": tx.to_address,
                "value_eth": tx.value_eth,
                "gas_price_gwei": tx.gas_price_gwei,
                "timestamp": tx.timestamp.isoformat() if tx.timestamp else None,
                "is_contract_call": tx.is_contract_call,
            }
            for tx in txns
        ],
    }


# ─── Run Detection Pipeline ───────────────────────────────────────────────────

@app.post("/api/run-detection")
def run_detection(db: Session = Depends(get_db)):
    """
    Manually trigger the full detection pipeline:
    ML training, graph analysis, flash-loan detection, wash-trade detection,
    risk scoring, and alert generation.
    """
    try:
        result = risk_engine.run_full_detection(db)
        blockchain_service.last_analysis_at = datetime.utcnow()
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Detection pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Wallet Profiles ──────────────────────────────────────────────────────────

@app.get("/api/wallet-profiles")
def get_wallet_profiles(
    limit: int = Query(100, ge=1, le=1000),
    sort_by: str = Query("risk_score", pattern="^(risk_score|tx_count|total_value_sent)$"),
    db: Session = Depends(get_db),
):
    """Get wallet profiles sorted by risk score or other metrics."""
    order_col = getattr(WalletProfile, sort_by, WalletProfile.risk_score)
    profiles = (
        db.query(WalletProfile)
        .order_by(order_col.desc())
        .limit(limit)
        .all()
    )
    return {
        "count": len(profiles),
        "profiles": [
            {
                "address": p.address,
                "tx_count": p.tx_count,
                "total_value_sent": round(p.total_value_sent, 6),
                "total_value_received": round(p.total_value_received, 6),
                "avg_value": round(p.avg_value, 6),
                "unique_counterparties": p.unique_counterparties,
                "inflow_outflow_ratio": round(p.inflow_outflow_ratio, 4),
                "tx_frequency": round(p.tx_frequency, 4),
                "burst_score": p.burst_score,
                "cluster_label": p.cluster_label,
                "risk_score": round(p.risk_score, 2),
                "last_active": p.last_active.isoformat() if p.last_active else None,
            }
            for p in profiles
        ],
    }


# ─── Alerts ───────────────────────────────────────────────────────────────────

@app.get("/api/alerts")
def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    alert_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Get generated alerts, optionally filtered by type or severity."""
    query = db.query(Alert).order_by(Alert.created_at.desc())

    if alert_type:
        query = query.filter(Alert.alert_type == alert_type)
    if severity:
        query = query.filter(Alert.severity == severity)

    alerts = query.limit(limit).all()
    return {
        "count": len(alerts),
        "alerts": [
            {
                "id": a.id,
                "wallet_address": a.wallet_address,
                "alert_type": a.alert_type,
                "severity": a.severity,
                "risk_score": round(a.risk_score, 2),
                "explanation": a.explanation,
                "tx_hash": a.tx_hash,
                "block_number": a.block_number,
                "is_resolved": a.is_resolved,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in alerts
        ],
    }


# ─── Graph Data ───────────────────────────────────────────────────────────────

@app.get("/api/graph-data")
def get_graph_data(db: Session = Depends(get_db)):
    """
    Get wallet interaction graph data for visualization.
    Returns nodes (wallets) and links (transactions) for force-graph rendering.
    """
    if graph_analyzer.graph.number_of_nodes() == 0:
        graph_analyzer.build_graph(db)

    data = graph_analyzer.get_graph_data()
    return data


# ─── Timeline Data ────────────────────────────────────────────────────────────

@app.get("/api/timeline-data")
def get_timeline_data(
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db),
):
    """
    Get time-series data for transaction volume and alert count.
    Grouped by hour for the specified window.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    # Transaction volume per hour
    tx_timeline = (
        db.query(
            func.strftime("%Y-%m-%d %H:00", Transaction.timestamp).label("hour"),
            func.count(Transaction.id).label("tx_count"),
            func.sum(Transaction.value_eth).label("total_value"),
        )
        .filter(Transaction.timestamp >= cutoff)
        .group_by("hour")
        .order_by("hour")
        .all()
    )

    # Alert count per hour
    alert_timeline = (
        db.query(
            func.strftime("%Y-%m-%d %H:00", Alert.created_at).label("hour"),
            func.count(Alert.id).label("alert_count"),
        )
        .filter(Alert.created_at >= cutoff)
        .group_by("hour")
        .order_by("hour")
        .all()
    )

    alert_map = {a.hour: a.alert_count for a in alert_timeline}

    return {
        "hours": hours,
        "timeline": [
            {
                "hour": row.hour,
                "tx_count": row.tx_count,
                "total_value": round(row.total_value or 0, 4),
                "alert_count": alert_map.get(row.hour, 0),
            }
            for row in tx_timeline
        ],
    }


# ─── Risk Score ───────────────────────────────────────────────────────────────

@app.get("/api/risk-score/{wallet_address}")
def get_risk_score(wallet_address: str, db: Session = Depends(get_db)):
    """
    Get composite risk score for a specific wallet.
    Computes score if missing, otherwise returns cached version.
    """
    wallet_address = wallet_address.lower()

    # Check if wallet has transactions
    tx_count = db.query(Transaction).filter(
        (Transaction.from_address == wallet_address) |
        (Transaction.to_address == wallet_address)
    ).count()

    if tx_count == 0:
        return {
            "wallet_address": wallet_address,
            "composite_score": 0,
            "severity": "unknown",
            "explanation": "No transactions found for this wallet.",
            "tx_count": 0,
        }

    try:
        risk = risk_engine.compute_risk(db, wallet_address)
        risk["tx_count"] = tx_count
        return risk
    except Exception as e:
        logger.error(f"Risk computation error for {wallet_address}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Ingest Endpoint (manual trigger) ─────────────────────────────────────────

@app.post("/api/ingest")
def ingest_blocks(n_blocks: int = Query(3, ge=1, le=1000)):
    """Manually trigger ingestion of the latest N blocks."""
    try:
        result = blockchain_service.ingest_latest_blocks(n_blocks=n_blocks)
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Flash Loan Events ────────────────────────────────────────────────────────

@app.get("/api/flash-loans")
def get_flash_loans(db: Session = Depends(get_db)):
    """Get detected flash-loan-like events."""
    events = flash_loan_detector.detect(db)
    return {"count": len(events), "events": events}


# ─── Wash Trading Pairs ───────────────────────────────────────────────────────

@app.get("/api/wash-trades")
def get_wash_trades(db: Session = Depends(get_db)):
    """Get detected wash-trading pairs."""
    if graph_analyzer.graph.number_of_nodes() == 0:
        graph_analyzer.build_graph(db)
    pairs = graph_analyzer.detect_wash_trading(db)
    return {"count": len(pairs), "pairs": pairs}
