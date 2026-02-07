"""
Performance Tracker - Metrics collection and analysis.

Tracks agent performance, output quality, and identifies
patterns for optimization.
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
class PerformanceMetrics:
    """Performance metrics for an agent or run."""
    agent_name: str
    run_id: str
    timestamp: str
    output_score: float
    readability_score: float
    length_score: float
    keyword_density_score: float
    structure_score: float
    citation_count: int
    execution_time: float
    retry_count: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create from dictionary."""
        return cls(**data)


class PerformanceTracker:
    """
    Performance metrics storage and analysis.
    
    Tracks detailed performance data per agent and run,
    enabling performance analysis and optimization.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize performance tracker.

        Args:
            db_path: Path to SQLite database file.
        """
        if db_path is None:
            db_dir = Path(__file__).parent.parent / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "performance.db")
        
        self.db_path = db_path
        self._init_db()
        logger.info(f"Initialized PerformanceTracker: {db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    output_score REAL NOT NULL,
                    readability_score REAL DEFAULT 0,
                    length_score REAL DEFAULT 0,
                    keyword_density_score REAL DEFAULT 0,
                    structure_score REAL DEFAULT 0,
                    citation_count INTEGER DEFAULT 0,
                    execution_time REAL NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    success INTEGER NOT NULL,
                    error_message TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent 
                ON performance_metrics(agent_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_run 
                ON performance_metrics(run_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON performance_metrics(timestamp)
            """)
            
            # Failure tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failure_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    context TEXT NOT NULL
                )
            """)
            
            conn.commit()

    def record(self, metrics: PerformanceMetrics) -> None:
        """
        Record performance metrics.

        Args:
            metrics: PerformanceMetrics to store.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_metrics (
                    agent_name, run_id, timestamp, output_score,
                    readability_score, length_score, keyword_density_score,
                    structure_score, citation_count, execution_time,
                    retry_count, success, error_message, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.agent_name,
                metrics.run_id,
                metrics.timestamp,
                metrics.output_score,
                metrics.readability_score,
                metrics.length_score,
                metrics.keyword_density_score,
                metrics.structure_score,
                metrics.citation_count,
                metrics.execution_time,
                metrics.retry_count,
                1 if metrics.success else 0,
                metrics.error_message,
                json.dumps(metrics.metadata),
            ))
            conn.commit()
        
        logger.debug(f"Recorded metrics for {metrics.agent_name}")

    def record_failure(
        self,
        agent_name: str,
        run_id: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
    ) -> None:
        """
        Record a failure case.

        Args:
            agent_name: Name of the failing agent.
            run_id: Run identifier.
            error_type: Type of error.
            error_message: Error message.
            context: Execution context.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO failure_cases (
                    agent_name, run_id, timestamp, error_type,
                    error_message, context
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                agent_name,
                run_id,
                datetime.now().isoformat(),
                error_type,
                error_message,
                json.dumps(context),
            ))
            conn.commit()
        
        logger.info(f"Recorded failure: {agent_name} - {error_type}")

    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """
        Get statistics for an agent.

        Args:
            agent_name: Agent name to analyze.

        Returns:
            Statistics dictionary.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_runs,
                    SUM(success) as successful_runs,
                    AVG(output_score) as avg_score,
                    AVG(readability_score) as avg_readability,
                    AVG(execution_time) as avg_time,
                    SUM(retry_count) as total_retries,
                    MAX(output_score) as best_score,
                    MIN(output_score) as worst_score
                FROM performance_metrics
                WHERE agent_name = ?
            """, (agent_name,))
            row = cursor.fetchone()
            
            # Get recent trend (last 10 vs previous 10)
            cursor = conn.execute("""
                SELECT output_score FROM performance_metrics
                WHERE agent_name = ?
                ORDER BY timestamp DESC
                LIMIT 20
            """, (agent_name,))
            scores = [r[0] for r in cursor.fetchall()]
        
        recent = scores[:10] if len(scores) >= 10 else scores
        previous = scores[10:20] if len(scores) >= 20 else []
        
        trend = "stable"
        if recent and previous:
            recent_avg = sum(recent) / len(recent)
            prev_avg = sum(previous) / len(previous)
            if recent_avg > prev_avg * 1.1:
                trend = "improving"
            elif recent_avg < prev_avg * 0.9:
                trend = "declining"
        
        return {
            "agent_name": agent_name,
            "total_runs": row[0] or 0,
            "success_rate": (row[1] / row[0] * 100) if row[0] else 0,
            "avg_score": round(row[2] or 0, 2),
            "avg_readability": round(row[3] or 0, 2),
            "avg_execution_time": round(row[4] or 0, 2),
            "total_retries": row[5] or 0,
            "best_score": round(row[6] or 0, 2),
            "worst_score": round(row[7] or 0, 2),
            "trend": trend,
        }

    def get_run_metrics(self, run_id: str) -> List[PerformanceMetrics]:
        """
        Get all metrics for a run.

        Args:
            run_id: Run identifier.

        Returns:
            List of PerformanceMetrics.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM performance_metrics
                WHERE run_id = ?
                ORDER BY timestamp
            """, (run_id,))
            rows = cursor.fetchall()
        
        return [self._row_to_metrics(row) for row in rows]

    def get_failure_patterns(self, limit: int = 10) -> Dict[str, Any]:
        """
        Analyze failure patterns.

        Args:
            limit: Maximum failure types to return.

        Returns:
            Failure analysis dictionary.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Group by error type
            cursor = conn.execute("""
                SELECT error_type, COUNT(*) as count, agent_name
                FROM failure_cases
                GROUP BY error_type, agent_name
                ORDER BY count DESC
                LIMIT ?
            """, (limit,))
            error_types = [
                {"type": row[0], "count": row[1], "agent": row[2]}
                for row in cursor.fetchall()
            ]
            
            # Recent failures
            cursor = conn.execute("""
                SELECT agent_name, error_type, error_message, timestamp
                FROM failure_cases
                ORDER BY timestamp DESC
                LIMIT 5
            """)
            recent = [
                {
                    "agent": row[0],
                    "type": row[1],
                    "message": row[2],
                    "timestamp": row[3],
                }
                for row in cursor.fetchall()
            ]
        
        return {
            "error_distribution": error_types,
            "recent_failures": recent,
        }

    def get_performance_trends(self, days: int = 30) -> Dict[str, List[Dict]]:
        """
        Get performance trends over time.

        Args:
            days: Number of days to analyze.

        Returns:
            Trends by agent.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    agent_name,
                    date(timestamp) as date,
                    AVG(output_score) as avg_score,
                    COUNT(*) as runs
                FROM performance_metrics
                WHERE timestamp >= date('now', '-' || ? || ' days')
                GROUP BY agent_name, date(timestamp)
                ORDER BY date
            """, (days,))
            
            trends = {}
            for row in cursor.fetchall():
                agent = row[0]
                if agent not in trends:
                    trends[agent] = []
                trends[agent].append({
                    "date": row[1],
                    "avg_score": round(row[2], 2),
                    "runs": row[3],
                })
        
        return trends

    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall performance statistics.

        Returns:
            Aggregate statistics.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_measurements,
                    COUNT(DISTINCT run_id) as total_runs,
                    COUNT(DISTINCT agent_name) as agents_tracked,
                    AVG(output_score) as overall_avg_score,
                    SUM(success) * 100.0 / COUNT(*) as success_rate,
                    AVG(execution_time) as avg_execution_time
                FROM performance_metrics
            """)
            row = cursor.fetchone()
            
            cursor = conn.execute(
                "SELECT COUNT(*) FROM failure_cases"
            )
            failures = cursor.fetchone()[0]
        
        return {
            "total_measurements": row[0] or 0,
            "total_runs": row[1] or 0,
            "agents_tracked": row[2] or 0,
            "overall_avg_score": round(row[3] or 0, 2),
            "success_rate": round(row[4] or 0, 2),
            "avg_execution_time": round(row[5] or 0, 2),
            "total_failures": failures,
        }

    def _row_to_metrics(self, row: sqlite3.Row) -> PerformanceMetrics:
        """Convert database row to PerformanceMetrics."""
        return PerformanceMetrics(
            agent_name=row["agent_name"],
            run_id=row["run_id"],
            timestamp=row["timestamp"],
            output_score=row["output_score"],
            readability_score=row["readability_score"],
            length_score=row["length_score"],
            keyword_density_score=row["keyword_density_score"],
            structure_score=row["structure_score"],
            citation_count=row["citation_count"],
            execution_time=row["execution_time"],
            retry_count=row["retry_count"],
            success=bool(row["success"]),
            error_message=row["error_message"],
            metadata=json.loads(row["metadata"]),
        )
