"""
Ethereum Blockchain Service
Handles Web3 connection, block polling, transaction extraction.
Includes retry logic, connection validation, and simulation fallback.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable

from web3 import Web3
from web3.exceptions import Web3RPCError
from sqlalchemy.orm import Session

from app.config import ETH_RPC_URL, POLL_INTERVAL
from app.database import SessionLocal, Transaction, init_db

logger = logging.getLogger("chainwatch.blockchain")

# ── Fallback RPC endpoints (tried in order if primary fails) ──────────────────
FALLBACK_RPCS = [
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.ankr.com/eth",
    "https://eth.llamarpc.com",
    "https://1rpc.io/eth",
]

MAX_CONSECUTIVE_ERRORS = 10  # Number of consecutive poll errors before simulation fallback


class BlockchainService:
    """Manages Ethereum blockchain connection and data ingestion."""

    def __init__(self):
        self._rpc_url = ETH_RPC_URL
        self.w3: Optional[Web3] = None
        self.last_processed_block: Optional[int] = None
        self.last_analysis_at: Optional[datetime] = None
        self.total_blocks_processed: int = 0
        self._polling = False
        self._simulation_mode = False
        self._connect()
        # Log connection status on startup
        if self.is_connected:
            logger.info(f"✅ Live mode active — connected to {self._rpc_url}")
        else:
            logger.warning(f"⚠️ Could not connect to any RPC. Mode: {self.mode}")

    # ── Connection management ─────────────────────────────────────────────────

    def _connect(self):
        """Try to connect to primary RPC, then fallbacks."""
        urls_to_try = [self._rpc_url] + FALLBACK_RPCS

        for url in urls_to_try:
            try:
                provider = Web3.HTTPProvider(url, request_kwargs={"timeout": 15})
                w3 = Web3(provider)
                if w3.is_connected():
                    self.w3 = w3
                    self._rpc_url = url
                    self._simulation_mode = False
                    logger.info(f"Web3 connected via {url}")
                    return
                else:
                    logger.warning(f"Web3 not connected via {url}")
            except Exception as e:
                logger.warning(f"Failed to connect to {url}: {e}")

        # All RPCs failed → simulation mode
        logger.error("All RPC endpoints failed. Entering simulation mode.")
        self._simulation_mode = True
        self.w3 = Web3()  # disconnected instance

    def reconnect(self):
        """Attempt to reconnect after failures."""
        logger.info("Attempting reconnection...")
        self._connect()

    @property
    def is_connected(self) -> bool:
        if self._simulation_mode:
            return False
        try:
            return self.w3.is_connected() if self.w3 else False
        except Exception:
            return False

    @property
    def mode(self) -> str:
        return "simulation" if self._simulation_mode else "live"

    # ── Block / Transaction fetching ──────────────────────────────────────────

    def get_latest_block_number(self) -> int:
        """Return the latest block number on chain."""
        if self._simulation_mode:
            # Return a simulated block number (advances with time)
            base = 22_100_000
            offset = int((datetime.utcnow() - datetime(2026, 1, 1)).total_seconds() / 12)
            return base + offset
        return self.w3.eth.block_number

    def fetch_block_transactions(self, block_number: int) -> List[Dict[str, Any]]:
        """
        Fetch all transactions from a specific block and return parsed dicts.
        Falls back to simulation data if RPC fails.
        """
        if self._simulation_mode:
            return self._generate_simulated_transactions(block_number)

        for attempt in range(3):
            try:
                block = self.w3.eth.get_block(block_number, full_transactions=True)
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/3 fetching block {block_number}: {e}")
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                else:
                    logger.error(f"All retries failed for block {block_number}, generating simulation data")
                    return self._generate_simulated_transactions(block_number)

        timestamp = datetime.utcfromtimestamp(block["timestamp"])
        parsed: List[Dict[str, Any]] = []

        for tx in block.transactions:
            try:
                value_eth = float(self.w3.from_wei(tx["value"], "ether"))
                gas_price_gwei = float(self.w3.from_wei(tx.get("gasPrice", 0), "gwei"))
                input_data = tx.get("input", "0x")
                is_contract = len(input_data) > 2 if input_data else False

                parsed.append({
                    "tx_hash": tx["hash"].hex() if isinstance(tx["hash"], bytes) else str(tx["hash"]),
                    "block_number": block_number,
                    "from_address": tx["from"].lower() if tx["from"] else "",
                    "to_address": tx["to"].lower() if tx["to"] else None,
                    "value_eth": value_eth,
                    "gas_price_gwei": gas_price_gwei,
                    "gas_used": tx.get("gas", 0),
                    "timestamp": timestamp,
                    "input_data_length": len(input_data) if input_data else 0,
                    "is_contract_call": is_contract,
                })
            except Exception as e:
                logger.warning(f"Skipping tx {tx.get('hash', '?')}: {e}")

        return parsed

    def _generate_simulated_transactions(self, block_number: int) -> List[Dict[str, Any]]:
        """Generate realistic-looking simulated transactions for demo/fallback."""
        rng = random.Random(block_number)
        n_txs = rng.randint(5, 25)
        timestamp = datetime.utcnow() - timedelta(seconds=rng.randint(0, 120))
        wallets = [f"0x{rng.randbytes(20).hex()}" for _ in range(15)]
        parsed = []

        for i in range(n_txs):
            sender = rng.choice(wallets)
            receiver = rng.choice([w for w in wallets if w != sender])
            value_eth = round(rng.uniform(0.001, 50.0), 6)
            gas_price = round(rng.uniform(5.0, 80.0), 2)
            input_len = rng.choice([0, 0, 0, 68, 136, 200, 580])
            parsed.append({
                "tx_hash": f"0x{rng.randbytes(32).hex()}",
                "block_number": block_number,
                "from_address": sender,
                "to_address": receiver if rng.random() > 0.05 else None,
                "value_eth": value_eth,
                "gas_price_gwei": gas_price,
                "gas_used": rng.randint(21000, 500000),
                "timestamp": timestamp,
                "input_data_length": input_len,
                "is_contract_call": input_len > 2,
            })

        return parsed

    # ── Storage ───────────────────────────────────────────────────────────────

    def store_transactions(self, db: Session, tx_list: List[Dict[str, Any]]) -> int:
        """Store parsed transactions in the database. Returns count of new rows."""
        stored = 0
        for tx_data in tx_list:
            try:
                existing = db.query(Transaction).filter_by(tx_hash=tx_data["tx_hash"]).first()
                if existing:
                    continue
                db.add(Transaction(**tx_data))
                stored += 1
            except Exception as e:
                logger.warning(f"DB insert error for tx {tx_data.get('tx_hash', '?')}: {e}")
                db.rollback()

        if stored > 0:
            try:
                db.commit()
            except Exception as e:
                logger.error(f"DB commit error: {e}")
                db.rollback()
                stored = 0
        return stored

    # ── Manual bulk ingest ────────────────────────────────────────────────────

    def ingest_latest_blocks(self, n_blocks: int = 3) -> Dict[str, Any]:
        """Fetch the latest n blocks and store their transactions."""
        init_db()
        latest = self.get_latest_block_number()
        start_block = latest - n_blocks + 1

        total_txs = 0
        total_stored = 0
        db = SessionLocal()
        try:
            for block_num in range(start_block, latest + 1):
                txs = self.fetch_block_transactions(block_num)
                stored = self.store_transactions(db, txs)
                total_txs += len(txs)
                total_stored += stored
                logger.info(f"Block {block_num}: {len(txs)} txs fetched, {stored} new stored")
                time.sleep(0.15)  # Rate limit protection

            self.last_processed_block = latest
            self.total_blocks_processed += n_blocks
        finally:
            db.close()

        return {
            "latest_block": latest,
            "blocks_processed": n_blocks,
            "transactions_fetched": total_txs,
            "new_transactions_stored": total_stored,
            "mode": self.mode,
        }

    # ── Background polling ────────────────────────────────────────────────────

    async def start_polling(self, callback: Optional[Callable] = None):
        """
        Continuously poll for new blocks in the background.
        Calls optional callback(block_number) after each new block batch.
        Retries connection every 5 seconds on failure.
        """
        self._polling = True
        init_db()
        logger.info(f"Starting block polling loop (interval={POLL_INTERVAL}s, mode={self.mode})...")
        consecutive_errors = 0

        while self._polling:
            try:
                if not self.is_connected and not self._simulation_mode:
                    logger.warning("RPC disconnected. Retrying in 5s...")
                    await asyncio.sleep(5)
                    self.reconnect()
                    continue

                latest = self.get_latest_block_number()
                if self.last_processed_block is None:
                    self.last_processed_block = latest - 1

                if latest > self.last_processed_block:
                    start = self.last_processed_block + 1
                    end = latest

                    def process_blocks():
                        db = SessionLocal()
                        total_stored = 0
                        try:
                            for block_num in range(start, end + 1):
                                txs = self.fetch_block_transactions(block_num)
                                stored = self.store_transactions(db, txs)
                                total_stored += stored
                                logger.info(f"Polled block {block_num}: {len(txs)} txs, {stored} new stored")
                        finally:
                            db.close()
                        return total_stored

                    total = await asyncio.to_thread(process_blocks)
                    self.last_processed_block = latest
                    self.total_blocks_processed += (end - start + 1)
                    consecutive_errors = 0

                    if callback and total > 0:
                        try:
                            await callback(latest)
                        except Exception as cb_err:
                            logger.error(f"Post-block callback error: {cb_err}")

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Polling error (#{consecutive_errors}): {e}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS and not self._simulation_mode:
                    logger.warning(f"Too many consecutive errors ({consecutive_errors}). Switching to simulation mode.")
                    self._simulation_mode = True
                    consecutive_errors = 0
                elif consecutive_errors > 0 and consecutive_errors % 3 == 0 and not self._simulation_mode:
                    # Attempt reconnection every 3 errors
                    logger.info("Attempting RPC reconnection after errors...")
                    self.reconnect()

            await asyncio.sleep(POLL_INTERVAL)

    def stop_polling(self):
        """Stop the background polling loop."""
        self._polling = False
        logger.info("Polling stopped.")


# Singleton instance
blockchain_service = BlockchainService()
