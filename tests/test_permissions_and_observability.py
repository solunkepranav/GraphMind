import os
import tempfile
import pytest
from src.permissions import AccessControlManager
from src.observability import QueryAuditLogger

def test_access_control_intersection_rule():
    acl = AccessControlManager({
        "admin": ["*"],
        "finance": ["financials.pdf", "tax_2024.pdf"],
        "engineer": ["system.md", "api.py"]
    })

    all_docs = ["financials.pdf", "tax_2024.pdf", "system.md", "api.py"]

    # Admin user selected everything -> gets everything
    assert set(acl.get_effective_sources("admin", all_docs, all_docs)) == set(all_docs)

    # Finance user tries to select system.md -> pruned out by intersection
    user_attempt = ["financials.pdf", "system.md"]
    effective = acl.get_effective_sources("finance", all_docs, user_attempt)
    assert effective == ["financials.pdf"]
    assert "system.md" not in effective

def test_query_audit_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_audit.jsonl")
        logger = QueryAuditLogger(log_path=log_file)

        logger.log_query_execution(
            query="What is the revenue?",
            category="SIMPLE",
            latency_ms=124.5,
            chunk_count=3,
            edge_count=0,
            confidence=92,
            sources_used=["financials.pdf"],
            warnings=[]
        )

        records = logger.read_recent_records(10)
        assert len(records) == 1
        r = records[0]
        assert r["query"] == "What is the revenue?"
        assert r["latency_ms"] == 124.5
        assert r["confidence"] == 92
        assert r["sources_used"] == ["financials.pdf"]
