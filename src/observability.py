import os
import json
import datetime
from src import config

class QueryAuditLogger:
    """Local JSONL audit logger and performance tracer."""

    def __init__(self, log_path: str = None):
        self.log_path = log_path or os.path.join(config.DATA_DIR, "audit_log.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

    def log_query_execution(
        self,
        query: str,
        category: str,
        latency_ms: float,
        chunk_count: int,
        edge_count: int,
        confidence: int,
        sources_used: list[str],
        warnings: list[str] = None
    ):
        """Appends structured audit record to JSONL."""
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "category": category,
            "latency_ms": round(latency_ms, 2),
            "chunk_count": chunk_count,
            "edge_count": edge_count,
            "confidence": confidence,
            "sources_used": sources_used or [],
            "warnings": warnings or []
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"Audit log write failed: {e}")

    def read_recent_records(self, limit: int = 50) -> list[dict]:
        """Reads recent audit log records."""
        if not os.path.exists(self.log_path):
            return []
        records = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records[-limit:]
        except Exception as e:
            print(f"Error reading audit log: {e}")
            return []
