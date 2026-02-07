"""
Strategy Memory - Learning from successful execution strategies.

Stores and retrieves successful strategies for topic categories,
enabling adaptive pipeline behavior based on past performance.
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
class Strategy:
    """A successful execution strategy."""
    strategy_id: str
    topic_category: str
    topic_keywords: List[str]
    agent_order: List[str]
    prompt_overrides: Dict[str, str]
    tool_preferences: Dict[str, List[str]]
    live_data_sources: List[str]
    average_score: float
    use_count: int
    last_used: str
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Strategy":
        """Create from dictionary."""
        return cls(**data)


class StrategyMemory:
    """
    Strategy storage and retrieval system.
    
    Learns from successful pipeline runs and provides
    optimized strategies for similar topics.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize strategy memory.

        Args:
            db_path: Path to SQLite database file.
        """
        if db_path is None:
            db_dir = Path(__file__).parent.parent / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "strategy_memory.db")
        
        self.db_path = db_path
        self._init_db()
        logger.info(f"Initialized StrategyMemory: {db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    topic_category TEXT NOT NULL,
                    topic_keywords TEXT NOT NULL,
                    agent_order TEXT NOT NULL,
                    prompt_overrides TEXT NOT NULL,
                    tool_preferences TEXT NOT NULL,
                    live_data_sources TEXT NOT NULL,
                    average_score REAL NOT NULL,
                    use_count INTEGER DEFAULT 1,
                    last_used TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Create keyword search table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_keywords (
                    keyword TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category 
                ON strategies(topic_category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_score 
                ON strategies(average_score)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_keywords 
                ON strategy_keywords(keyword)
            """)
            conn.commit()

    def store_strategy(
        self,
        topic: str,
        strategy: Strategy,
        score: float,
    ) -> None:
        """
        Store or update a strategy.

        Args:
            topic: Topic that was processed.
            strategy: Strategy configuration used.
            score: Quality score achieved.
        """
        # Generate strategy ID if not set
        if not strategy.strategy_id:
            strategy.strategy_id = f"strat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract keywords from topic
        keywords = self._extract_keywords(topic)
        strategy.topic_keywords = keywords
        
        with sqlite3.connect(self.db_path) as conn:
            # First, check if this exact strategy_id already exists
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM strategies WHERE strategy_id = ?",
                (strategy.strategy_id,)
            )
            existing_by_id = cursor.fetchone()
            
            if existing_by_id:
                # Update the existing strategy by ID
                existing = self._row_to_strategy(existing_by_id)
                new_avg = (existing.average_score * existing.use_count + score) / (existing.use_count + 1)
                conn.execute("""
                    UPDATE strategies SET
                        average_score = ?,
                        use_count = use_count + 1,
                        last_used = ?,
                        prompt_overrides = ?,
                        tool_preferences = ?
                    WHERE strategy_id = ?
                """, (
                    new_avg,
                    datetime.now().isoformat(),
                    json.dumps(strategy.prompt_overrides),
                    json.dumps(strategy.tool_preferences),
                    existing.strategy_id,
                ))
                logger.info(f"Updated strategy {existing.strategy_id} by ID (avg: {new_avg:.1f})")
            else:
                # Check if similar strategy exists by keywords
                existing = self._find_similar_strategy(conn, keywords)
                
                if existing:
                    # Update existing strategy
                    new_avg = (existing.average_score * existing.use_count + score) / (existing.use_count + 1)
                    conn.execute("""
                        UPDATE strategies SET
                            average_score = ?,
                            use_count = use_count + 1,
                            last_used = ?,
                            prompt_overrides = ?,
                            tool_preferences = ?
                        WHERE strategy_id = ?
                    """, (
                        new_avg,
                        datetime.now().isoformat(),
                        json.dumps(strategy.prompt_overrides),
                        json.dumps(strategy.tool_preferences),
                        existing.strategy_id,
                    ))
                    logger.info(f"Updated strategy {existing.strategy_id} by keywords (avg: {new_avg:.1f})")
                else:
                    # Generate a new unique ID to avoid conflicts
                    import uuid
                    new_id = f"strat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                    
                    # Insert new strategy
                    strategy.average_score = score
                    strategy.last_used = datetime.now().isoformat()
                    strategy.created_at = datetime.now().isoformat()
                    
                    conn.execute("""
                        INSERT INTO strategies (
                            strategy_id, topic_category, topic_keywords, agent_order,
                            prompt_overrides, tool_preferences, live_data_sources,
                            average_score, use_count, last_used, created_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        new_id,
                        strategy.topic_category,
                        json.dumps(strategy.topic_keywords),
                        json.dumps(strategy.agent_order),
                        json.dumps(strategy.prompt_overrides),
                        json.dumps(strategy.tool_preferences),
                        json.dumps(strategy.live_data_sources),
                        strategy.average_score,
                        1,
                        strategy.last_used,
                        strategy.created_at,
                        json.dumps(strategy.metadata),
                    ))
                    
                    # Store keywords for search
                    for kw in keywords:
                        conn.execute(
                            "INSERT INTO strategy_keywords (keyword, strategy_id) VALUES (?, ?)",
                            (kw.lower(), new_id)
                        )
                    
                    logger.info(f"Stored new strategy {new_id}")
            
            conn.commit()

    def get_best_strategy(self, topic: str) -> Optional[Strategy]:
        """
        Get the best strategy for a topic.

        Args:
            topic: Topic to find strategy for.

        Returns:
            Best matching Strategy or None.
        """
        keywords = self._extract_keywords(topic)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Find strategies with matching keywords
            placeholders = ",".join("?" * len(keywords))
            cursor = conn.execute(f"""
                SELECT s.*, COUNT(sk.keyword) as match_count
                FROM strategies s
                LEFT JOIN strategy_keywords sk ON s.strategy_id = sk.strategy_id
                WHERE sk.keyword IN ({placeholders})
                GROUP BY s.strategy_id
                ORDER BY match_count DESC, s.average_score DESC
                LIMIT 1
            """, [kw.lower() for kw in keywords])
            
            row = cursor.fetchone()
            
            if row:
                return self._row_to_strategy(row)
            
            # Fallback: get highest scoring strategy
            cursor = conn.execute("""
                SELECT * FROM strategies
                ORDER BY average_score DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            
        return self._row_to_strategy(row) if row else None

    def get_strategies_for_category(
        self,
        category: str,
        limit: int = 5,
    ) -> List[Strategy]:
        """
        Get strategies for a topic category.

        Args:
            category: Topic category.
            limit: Maximum strategies to return.

        Returns:
            List of matching strategies.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM strategies
                WHERE topic_category = ?
                ORDER BY average_score DESC
                LIMIT ?
            """, (category, limit))
            rows = cursor.fetchall()
        
        return [self._row_to_strategy(row) for row in rows]

    def get_all_strategies(
        self,
        min_score: float = 0.0,
        limit: int = 50,
    ) -> List[Strategy]:
        """
        Get all strategies above minimum score.

        Args:
            min_score: Minimum average score.
            limit: Maximum strategies to return.

        Returns:
            List of strategies.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM strategies
                WHERE average_score >= ?
                ORDER BY average_score DESC
                LIMIT ?
            """, (min_score, limit))
            rows = cursor.fetchall()
        
        return [self._row_to_strategy(row) for row in rows]

    def _extract_keywords(self, topic: str) -> List[str]:
        """Extract keywords from topic string."""
        # Simple keyword extraction
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', topic.lower())
        
        # Filter common words
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this",
            "are", "was", "were", "been", "have", "has", "had",
            "will", "would", "could", "should", "may", "might",
            "about", "into", "through", "during", "before", "after",
        }
        
        return [w for w in words if w not in stopwords][:10]

    def _find_similar_strategy(
        self,
        conn: sqlite3.Connection,
        keywords: List[str],
    ) -> Optional[Strategy]:
        """Find existing strategy with similar keywords."""
        if not keywords:
            return None
        
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(keywords))
        
        cursor = conn.execute(f"""
            SELECT s.*, COUNT(sk.keyword) as match_count
            FROM strategies s
            JOIN strategy_keywords sk ON s.strategy_id = sk.strategy_id
            WHERE sk.keyword IN ({placeholders})
            GROUP BY s.strategy_id
            HAVING match_count >= ?
            ORDER BY match_count DESC
            LIMIT 1
        """, [kw.lower() for kw in keywords] + [len(keywords) // 2 + 1])
        
        row = cursor.fetchone()
        return self._row_to_strategy(row) if row else None

    def _row_to_strategy(self, row: sqlite3.Row) -> Strategy:
        """Convert database row to Strategy."""
        return Strategy(
            strategy_id=row["strategy_id"],
            topic_category=row["topic_category"],
            topic_keywords=json.loads(row["topic_keywords"]),
            agent_order=json.loads(row["agent_order"]),
            prompt_overrides=json.loads(row["prompt_overrides"]),
            tool_preferences=json.loads(row["tool_preferences"]),
            live_data_sources=json.loads(row["live_data_sources"]),
            average_score=row["average_score"],
            use_count=row["use_count"],
            last_used=row["last_used"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]),
        )

    def delete_low_performers(self, min_score: float = 50.0) -> int:
        """
        Delete strategies below minimum score.

        Args:
            min_score: Minimum score threshold.

        Returns:
            Number of deleted strategies.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get IDs to delete
            cursor = conn.execute(
                "SELECT strategy_id FROM strategies WHERE average_score < ?",
                (min_score,)
            )
            ids = [row[0] for row in cursor.fetchall()]
            
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM strategy_keywords WHERE strategy_id IN ({placeholders})",
                    ids
                )
                conn.execute(
                    f"DELETE FROM strategies WHERE strategy_id IN ({placeholders})",
                    ids
                )
            
            conn.commit()
        
        logger.info(f"Deleted {len(ids)} low-performing strategies")
        return len(ids)
