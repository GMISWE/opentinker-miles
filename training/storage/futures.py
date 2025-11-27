"""
Futures storage module for managing async operations.

This module provides a storage layer for tracking asynchronous operations
in the training API, using SQLite for persistence and in-memory caching
for performance.
"""
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List
from threading import Lock

logger = logging.getLogger(__name__)


def _serialize_result(result: Any) -> Optional[str]:
    """
    Serialize result to JSON, handling Pydantic models.

    Args:
        result: Result object to serialize (dict, Pydantic model, or None)

    Returns:
        JSON string or None
    """
    if result is None:
        return None

    # Handle Pydantic models (have model_dump or dict method)
    if hasattr(result, 'model_dump'):
        # Pydantic v2
        return json.dumps(result.model_dump())
    elif hasattr(result, 'dict'):
        # Pydantic v1
        return json.dumps(result.dict())
    else:
        # Plain dict or JSON-serializable object
        return json.dumps(result)


class FuturesStorage:
    """
    Manages futures for async operations with SQLite persistence.

    This class provides thread-safe storage for tracking async operations,
    combining in-memory caching with SQLite persistence for reliability.
    """

    def __init__(self, db_path: Path):
        """
        Initialize futures storage with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS futures (
                    request_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    operation TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_futures_status
                ON futures(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_futures_created
                ON futures(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_futures_model
                ON futures(model_id)
            """)

            conn.commit()
            logger.info(f"Initialized futures database at {self.db_path}")
        finally:
            conn.close()

    def save_future(
        self,
        request_id: str,
        operation: str,
        payload: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> None:
        """
        Save a new future for an async operation.

        Args:
            request_id: Unique identifier for the request
            operation: Type of operation (e.g., "create_model", "forward_backward")
            payload: Operation payload/parameters
            model_id: Optional model identifier
        """
        future_data = {
            "request_id": request_id,
            "model_id": model_id,
            "operation": operation,
            "payload": payload,
            "status": "pending",
            "result": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # Save to memory with thread safety
        with self._lock:
            self._memory_store[request_id] = future_data

        # Persist to SQLite
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO futures
                (request_id, model_id, operation, payload, status, result, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id,
                model_id,
                operation,
                json.dumps(payload),
                future_data["status"],
                None,
                future_data["created_at"],
                future_data["updated_at"]
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to save future {request_id}: {e}")
            raise
        finally:
            conn.close()

    def update_status(
        self,
        request_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update the status of an existing future.

        Args:
            request_id: Future identifier
            status: New status ("pending", "completed", "failed")
            result: Optional result data

        Returns:
            True if update successful, False if future not found
        """
        # Update in memory
        with self._lock:
            if request_id not in self._memory_store:
                # Try to load from database
                future = self._load_from_db(request_id)
                if not future:
                    logger.warning(f"Future {request_id} not found for update")
                    return False
                self._memory_store[request_id] = future

            self._memory_store[request_id]["status"] = status
            self._memory_store[request_id]["updated_at"] = datetime.utcnow().isoformat()
            if result is not None:
                # Convert Pydantic models to dict before storing in memory
                if hasattr(result, 'model_dump'):
                    self._memory_store[request_id]["result"] = result.model_dump()
                elif hasattr(result, 'dict'):
                    self._memory_store[request_id]["result"] = result.dict()
                else:
                    self._memory_store[request_id]["result"] = result

            updated_at = self._memory_store[request_id]["updated_at"]

        # Persist to SQLite
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE futures
                SET status = ?, result = ?, updated_at = ?
                WHERE request_id = ?
            """, (
                status,
                _serialize_result(result),
                updated_at,
                request_id
            ))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update future {request_id}: {e}")
            raise
        finally:
            conn.close()

    def get_future(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a future by request ID.

        Args:
            request_id: Future identifier

        Returns:
            Future data dict or None if not found
        """
        # Check memory first
        with self._lock:
            if request_id in self._memory_store:
                return self._memory_store[request_id].copy()

        # Fallback to database
        future = self._load_from_db(request_id)
        if future:
            # Cache in memory for future access
            with self._lock:
                self._memory_store[request_id] = future
        return future

    def _load_from_db(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Load a future from the database."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT request_id, model_id, operation, payload,
                       status, result, created_at, updated_at
                FROM futures
                WHERE request_id = ?
            """, (request_id,))
            row = cursor.fetchone()

            if row:
                return {
                    "request_id": row[0],
                    "model_id": row[1],
                    "operation": row[2],
                    "payload": json.loads(row[3]),
                    "status": row[4],
                    "result": json.loads(row[5]) if row[5] else None,
                    "created_at": row[6],
                    "updated_at": row[7]
                }
            return None
        finally:
            conn.close()

    def list_futures(
        self,
        model_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List futures with optional filters.

        Args:
            model_id: Filter by model ID
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of future data dicts
        """
        query = "SELECT * FROM futures WHERE 1=1"
        params = []

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)

            futures = []
            for row in cursor.fetchall():
                futures.append({
                    "request_id": row[0],
                    "model_id": row[1],
                    "operation": row[2],
                    "payload": json.loads(row[3]),
                    "status": row[4],
                    "result": json.loads(row[5]) if row[5] else None,
                    "created_at": row[6],
                    "updated_at": row[7]
                })
            return futures
        finally:
            conn.close()

    def cleanup_old_futures(self, max_age_hours: int = 24) -> int:
        """
        Remove futures older than specified age.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of futures removed
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cutoff_str = cutoff_time.isoformat()

        # Clean memory store
        with self._lock:
            to_remove = [
                request_id
                for request_id, future in self._memory_store.items()
                if future["created_at"] < cutoff_str
            ]
            for request_id in to_remove:
                del self._memory_store[request_id]
            memory_removed = len(to_remove)

        # Clean database
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM futures WHERE created_at < ?",
                (cutoff_str,)
            )
            db_removed = cursor.rowcount
            conn.commit()
        finally:
            conn.close()

        total_removed = memory_removed + db_removed
        if total_removed > 0:
            logger.info(
                f"Cleaned up {memory_removed} futures from memory "
                f"and {db_removed} from database (older than {max_age_hours}h)"
            )

        return total_removed

    def get_stats(self) -> Dict[str, int]:
        """
        Get storage statistics.

        Returns:
            Dict with counts by status and totals
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()

            # Count by status
            cursor.execute("""
                SELECT status, COUNT(*)
                FROM futures
                GROUP BY status
            """)
            stats = dict(cursor.fetchall())

            # Total count
            cursor.execute("SELECT COUNT(*) FROM futures")
            stats["total"] = cursor.fetchone()[0]

            # Memory store size
            with self._lock:
                stats["in_memory"] = len(self._memory_store)

            return stats
        finally:
            conn.close()