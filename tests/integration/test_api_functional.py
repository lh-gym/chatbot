from pathlib import Path

from fastapi.testclient import TestClient


def test_api_ingest_query_trace_metrics() -> None:
    # Import after environment setup to use default deterministic planner.
    from rag_agent.api.main import app

    client = TestClient(app)

    doc = Path("/tmp/rag_agent_policy.txt")
    doc.write_text(
        "Company policy states employees must encrypt customer data at rest.\n" * 800,
        encoding="utf-8",
    )

    ingest_resp = client.post(
        "/ingest",
        json={"path": str(doc), "doc_id": "policy-doc", "metadata": {"source": "policy"}},
    )
    assert ingest_resp.status_code == 200
    assert ingest_resp.json()["chunks_created"] >= 1

    query_resp = client.post(
        "/query",
        json={"question": "What does policy require for customer data?", "chat_history": []},
    )
    assert query_resp.status_code == 200
    query_payload = query_resp.json()
    assert query_payload["citations"]
    assert query_payload["groundedness"] >= 0.95

    trace_resp = client.get(f"/traces/{query_payload['trace_id']}")
    assert trace_resp.status_code == 200
    assert trace_resp.json()["tool_traces"]

    source_resp = client.post(
        "/sources/search",
        json={"query": "encrypt customer data", "top_k": 3},
    )
    assert source_resp.status_code == 200
    assert source_resp.json()["items"]

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert metrics_resp.json()["total_requests"] >= 1
