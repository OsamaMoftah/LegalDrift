"""Drift history persistence layer.

Save, load, and query drift detection results over time.
Supports JSON files and SQLite backends.
"""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from legaldrift.core.detector import DriftResult

logger = logging.getLogger(__name__)


@dataclass
class DriftRecord:
    """A single drift detection record with metadata."""

    timestamp: str
    baseline_id: str
    current_id: str
    result: Dict[str, Any]
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "baseline_id": self.baseline_id,
            "current_id": self.current_id,
            "result": self.result,
            "notes": self.notes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftRecord":
        return cls(
            timestamp=data["timestamp"],
            baseline_id=data["baseline_id"],
            current_id=data["current_id"],
            result=data["result"],
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
        )


class DriftHistory:
    """Persistent storage for drift detection history.

    Supports two backends:
      - JSON file (simple, human-readable)
      - SQLite database (queryable, scalable)
    """

    def __init__(self, path: Optional[Path] = None, backend: str = "json"):
        """Initialize drift history.

        Args:
            path: Path to storage file (auto-created if missing).
            backend: "json" or "sqlite".
        """
        self.backend = backend
        if path is None:
            path = Path("drift_history.json") if backend == "json" else Path("drift_history.db")
        self.path = Path(path)
        self._records: List[DriftRecord] = []

        if self.backend == "sqlite":
            self._init_sqlite()
        elif self.path.exists():
            self._load_json()

    def _init_sqlite(self):
        """Create SQLite schema if not exists."""
        conn = sqlite3.connect(str(self.path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                baseline_id TEXT NOT NULL,
                current_id TEXT NOT NULL,
                drift_detected INTEGER NOT NULL,
                p_value REAL NOT NULL,
                severity REAL NOT NULL,
                result_json TEXT NOT NULL,
                notes TEXT,
                tags TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON drift_records(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_baseline ON drift_records(baseline_id)"
        )
        conn.commit()
        conn.close()

    def _load_json(self):
        """Load records from JSON file."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._records = [DriftRecord.from_dict(r) for r in data]
            logger.info("Loaded %d drift records from %s", len(self._records), self.path)
        except Exception as e:
            logger.warning("Failed to load drift history: %s", e)
            self._records = []

    def save(
        self,
        baseline_id: str,
        current_id: str,
        result: DriftResult,
        notes: str = "",
        tags: Optional[List[str]] = None,
    ) -> DriftRecord:
        """Save a drift detection result.

        Args:
            baseline_id: ID of the baseline document.
            current_id: ID of the current document.
            result: DriftResult to store.
            notes: Optional human-readable notes.
            tags: Optional tags for categorization.

        Returns:
            The saved DriftRecord.
        """
        record = DriftRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            baseline_id=baseline_id,
            current_id=current_id,
            result=result.to_dict(),
            notes=notes,
            tags=tags or [],
        )

        if self.backend == "sqlite":
            self._save_sqlite(record)
        else:
            self._records.append(record)
            self._save_json()

        logger.info(
            "Saved drift record: %s vs %s (drift=%s)",
            baseline_id, current_id, result.drift_detected
        )
        return record

    def _save_json(self):
        """Write records to JSON file."""
        data = [r.to_dict() for r in self._records]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _save_sqlite(self, record: DriftRecord):
        """Insert record into SQLite database."""
        conn = sqlite3.connect(str(self.path))
        conn.execute(
            """
            INSERT INTO drift_records
            (timestamp, baseline_id, current_id, drift_detected, p_value, severity, result_json, notes, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp,
                record.baseline_id,
                record.current_id,
                int(record.result["drift_detected"]),
                record.result["p_value"],
                record.result["severity"],
                json.dumps(record.result),
                record.notes,
                json.dumps(record.tags),
            ),
        )
        conn.commit()
        conn.close()

    def query(
        self,
        baseline_id: Optional[str] = None,
        current_id: Optional[str] = None,
        drift_detected: Optional[bool] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> List[DriftRecord]:
        """Query historical drift records.

        Args:
            baseline_id: Filter by baseline document ID.
            current_id: Filter by current document ID.
            drift_detected: Filter by drift detection status.
            since: ISO timestamp to filter records after.
            limit: Maximum records to return.

        Returns:
            List of matching DriftRecords.
        """
        if self.backend == "sqlite":
            return self._query_sqlite(baseline_id, current_id, drift_detected, since, limit)
        return self._query_json(baseline_id, current_id, drift_detected, since, limit)

    def _query_json(
        self,
        baseline_id: Optional[str],
        current_id: Optional[str],
        drift_detected: Optional[bool],
        since: Optional[str],
        limit: int,
    ) -> List[DriftRecord]:
        records = self._records
        if baseline_id:
            records = [r for r in records if r.baseline_id == baseline_id]
        if current_id:
            records = [r for r in records if r.current_id == current_id]
        if drift_detected is not None:
            records = [r for r in records if r.result["drift_detected"] == drift_detected]
        if since:
            records = [r for r in records if r.timestamp >= since]
        return records[-limit:][::-1]

    def _query_sqlite(
        self,
        baseline_id: Optional[str],
        current_id: Optional[str],
        drift_detected: Optional[bool],
        since: Optional[str],
        limit: int,
    ) -> List[DriftRecord]:
        conn = sqlite3.connect(str(self.path))
        sql = "SELECT timestamp, baseline_id, current_id, result_json, notes, tags FROM drift_records WHERE 1=1"
        params: List[Any] = []
        if baseline_id:
            sql += " AND baseline_id = ?"
            params.append(baseline_id)
        if current_id:
            sql += " AND current_id = ?"
            params.append(current_id)
        if drift_detected is not None:
            sql += " AND drift_detected = ?"
            params.append(int(drift_detected))
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append(
                DriftRecord(
                    timestamp=row[0],
                    baseline_id=row[1],
                    current_id=row[2],
                    result=json.loads(row[3]),
                    notes=row[4] or "",
                    tags=json.loads(row[5]) if row[5] else [],
                )
            )
        return records

    def get_latest(self, baseline_id: Optional[str] = None) -> Optional[DriftRecord]:
        """Get the most recent drift record.

        Args:
            baseline_id: Optional baseline ID to filter by.

        Returns:
            Latest DriftRecord or None.
        """
        results = self.query(baseline_id=baseline_id, limit=1)
        return results[0] if results else None

    def clear(self):
        """Delete all stored records."""
        if self.backend == "sqlite":
            conn = sqlite3.connect(str(self.path))
            conn.execute("DELETE FROM drift_records")
            conn.commit()
            conn.close()
        else:
            self._records = []
            if self.path.exists():
                self.path.unlink()
        logger.info("Cleared all drift history from %s", self.path)
