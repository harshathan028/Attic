"""
Run History Store - Persistent storage for pipeline execution history.

Stores detailed records of each pipeline run for analysis and learning.
Uses SQLite for lightweight, file-based persistence.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Complete record of a pipeline execution."""
    run_id: str
    timestamp: str
    topic: str
    agents_executed: List[str]
    prompts_used: Dict[str, str]
    tool_usage: Dict[str, int]
    output_length: int
    evaluation_scores: Dict[str, float]
    retry_counts: Dict[str, int]
    execution_time: float
    final_quality_score: float
    live_data_enabled: bool = False
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        """Create from dictionary."""
        return cls(**data)


class RunHistoryStore:
    """
    SQLite-based run history storage.
    
    Provides persistent storage for pipeline execution records,
    enabling cross-run analysis and learning.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the run history store.

        Args:
            db_path: Path to SQLite database file.
        """
        if db_path is None:
            db_dir = Path(__file__).parent.parent / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "run_history.db")
        
        self.db_path = db_path
        self._init_db()
        logger.info(f"Initialized RunHistoryStore: {db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS run_history (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    agents_executed TEXT NOT NULL,
                    prompts_used TEXT NOT NULL,
                    tool_usage TEXT NOT NULL,
                    output_length INTEGER NOT NULL,
                    evaluation_scores TEXT NOT NULL,
                    retry_counts TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    final_quality_score REAL NOT NULL,
                    live_data_enabled INTEGER DEFAULT 0,
                    strategy_id TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic ON run_history(topic)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON run_history(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality ON run_history(final_quality_score)
            """)
            conn.commit()

    def store(self, record: RunRecord) -> None:
        """
        Store a run record.

        Args:
            record: RunRecord to store.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO run_history (
                    run_id, timestamp, topic, agents_executed, prompts_used,
                    tool_usage, output_length, evaluation_scores, retry_counts,
                    execution_time, final_quality_score, live_data_enabled,
                    strategy_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.run_id,
                record.timestamp,
                record.topic,
                json.dumps(record.agents_executed),
                json.dumps(record.prompts_used),
                json.dumps(record.tool_usage),
                record.output_length,
                json.dumps(record.evaluation_scores),
                json.dumps(record.retry_counts),
                record.execution_time,
                record.final_quality_score,
                1 if record.live_data_enabled else 0,
                record.strategy_id,
                json.dumps(record.metadata),
            ))
            conn.commit()
        
        logger.info(f"Stored run record: {record.run_id}")

    def get(self, run_id: str) -> Optional[RunRecord]:
        """
        Get a specific run record.

        Args:
            run_id: Run ID to retrieve.

        Returns:
            RunRecord if found, None otherwise.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM run_history WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            
        return self._row_to_record(row) if row else None

    def get_by_topic(
        self,
        topic: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[RunRecord]:
        """
        Get run records for a topic.

        Args:
            topic: Topic to search for.
            limit: Maximum records to return.
            min_score: Minimum quality score filter.

        Returns:
            List of matching RunRecords.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM run_history 
                WHERE topic LIKE ? AND final_quality_score >= ?
                ORDER BY final_quality_score DESC
                LIMIT ?
            """, (f"%{topic}%", min_score, limit))
            rows = cursor.fetchall()
        
        return [self._row_to_record(row) for row in rows]

    def get_best_runs(
        self,
        limit: int = 10,
        min_score: float = 70.0,
    ) -> List[RunRecord]:
        """
        Get highest-scoring run records.

        Args:
            limit: Maximum records to return.
            min_score: Minimum quality score.

        Returns:
            List of top-scoring RunRecords.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM run_history 
                WHERE final_quality_score >= ?
                ORDER BY final_quality_score DESC
                LIMIT ?
            """, (min_score, limit))
            rows = cursor.fetchall()
        
        return [self._row_to_record(row) for row in rows]

    def get_recent(self, limit: int = 20) -> List[RunRecord]:
        """
        Get most recent run records.

        Args:
            limit: Maximum records to return.

        Returns:
            List of recent RunRecords.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM run_history 
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
        
        return [self._row_to_record(row) for row in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all runs.

        Returns:
            Statistics dictionary.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_runs,
                    AVG(final_quality_score) as avg_score,
                    MAX(final_quality_score) as max_score,
                    MIN(final_quality_score) as min_score,
                    AVG(execution_time) as avg_time,
                    SUM(output_length) as total_output
                FROM run_history
            """)
            row = cursor.fetchone()
            
            # Get unique topics
            cursor = conn.execute("SELECT DISTINCT topic FROM run_history")
            topics = [r[0] for r in cursor.fetchall()]
        
        return {
            "total_runs": row[0] or 0,
            "avg_score": round(row[1] or 0, 2),
            "max_score": round(row[2] or 0, 2),
            "min_score": round(row[3] or 0, 2),
            "avg_execution_time": round(row[4] or 0, 2),
            "total_output_chars": row[5] or 0,
            "unique_topics": len(topics),
        }

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        """Convert database row to RunRecord."""
        return RunRecord(
            run_id=row["run_id"],
            timestamp=row["timestamp"],
            topic=row["topic"],
            agents_executed=json.loads(row["agents_executed"]),
            prompts_used=json.loads(row["prompts_used"]),
            tool_usage=json.loads(row["tool_usage"]),
            output_length=row["output_length"],
            evaluation_scores=json.loads(row["evaluation_scores"]),
            retry_counts=json.loads(row["retry_counts"]),
            execution_time=row["execution_time"],
            final_quality_score=row["final_quality_score"],
            live_data_enabled=bool(row["live_data_enabled"]),
            strategy_id=row["strategy_id"],
            metadata=json.loads(row["metadata"]),
        )

    def delete_old_runs(self, days: int = 90) -> int:
        """
        Delete runs older than specified days.

        Args:
            days: Age threshold in days.

        Returns:
            Number of deleted records.
        """
        cutoff = datetime.now().isoformat()[:10]  # Simple date comparison
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM run_history 
                WHERE date(timestamp) < date(?, '-' || ? || ' days')
            """, (cutoff, days))
            deleted = cursor.rowcount
            conn.commit()
        
        logger.info(f"Deleted {deleted} old run records")
        return deleted
