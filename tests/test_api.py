"""
End-to-end integration tests for the API itself — the full session lifecycle
a real frontend (or study participant) would drive: create session, send
messages, force an intervention, select a branch, fetch history, fetch
evaluation. These force the mock LLM + mock embeddings regardless of your
local .env, so the suite stays free and fast to run at any time, including
right before a demo, without touching your Gemini quota.
"""
import os

# Force mock backends for this test module, regardless of .env — must happen
# before app.config (and anything that imports it) is first imported.
os.environ["FIXATION_LLM_BACKEND"] = "mock"
os.environ["FIXATION_EMBEDDING_BACKEND"] = "mock"
os.environ["FIXATION_DATABASE_URL"] = "sqlite:///./test_api.db"

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    # Dispose the engine first so SQLAlchemy releases its file handle on the
    # test DB — without this, deleting the file fails on Windows (though not
    # on Linux/Mac, which don't lock open files the same way).
    from app.database import engine

    engine.dispose()
    if os.path.exists("test_api.db"):
        try:
            os.remove("test_api.db")
        except OSError:
            pass  # best-effort cleanup; leftover test DB file is harmless


def test_create_session_with_seed_prompt(client):
    resp = client.post(
        "/sessions",
        json={"task_type": "story_generation", "strategy": "baseline", "seed_prompt": "A story about a lighthouse."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"]
    assert data["assistant_message"] is not None
    assert "lighthouse" in data["assistant_message"] or "A story about a lighthouse." in data["assistant_message"]


def test_send_message_returns_metrics(client):
    create_resp = client.post(
        "/sessions", json={"task_type": "product_brainstorming", "strategy": "baseline", "seed_prompt": "Ideas for a new app."}
    )
    session_id = create_resp.json()["session_id"]

    resp = client.post(f"/sessions/{session_id}/messages", json={"content": "Tell me more."})
    assert resp.status_code == 200
    data = resp.json()
    assert "avg_similarity" in data["metrics"]
    assert "fixation_score" in data["metrics"]
    assert data["intervened"] is False  # baseline never intervenes


def test_force_intervene_returns_branches(client):
    create_resp = client.post(
        "/sessions", json={"task_type": "interface_design", "strategy": "user_triggered", "seed_prompt": "Design a dashboard."}
    )
    session_id = create_resp.json()["session_id"]

    resp = client.post(
        f"/sessions/{session_id}/messages", json={"content": "give me alternatives", "force_intervene": True}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intervened"] is True
    assert set(data["branches"].keys()) == {
        "abstract_reframing",
        "contradictory_perspective",
        "adjacent_domain",
        "unconventional_alternative",
        "speculative_future",
    }


def test_select_branch_continues_conversation(client):
    create_resp = client.post(
        "/sessions", json={"task_type": "story_generation", "strategy": "user_triggered", "seed_prompt": "A story about the sea."}
    )
    session_id = create_resp.json()["session_id"]
    client.post(f"/sessions/{session_id}/messages", json={"content": "more", "force_intervene": True})

    resp = client.post(f"/sessions/{session_id}/branches/select", json={"branch_category": "adjacent_domain"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["assistant_message"] is not None
    assert data["intervened"] is False


def test_select_branch_without_pending_tree_fails_cleanly(client):
    create_resp = client.post(
        "/sessions", json={"task_type": "story_generation", "strategy": "baseline", "seed_prompt": "A story."}
    )
    session_id = create_resp.json()["session_id"]

    resp = client.post(f"/sessions/{session_id}/branches/select", json={"branch_category": "adjacent_domain"})
    assert resp.status_code == 400


def test_get_session_history_is_correctly_ordered(client):
    create_resp = client.post(
        "/sessions", json={"task_type": "story_generation", "strategy": "baseline", "seed_prompt": "Turn zero."}
    )
    session_id = create_resp.json()["session_id"]
    client.post(f"/sessions/{session_id}/messages", json={"content": "Turn two content."})

    resp = client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    turn_indices = [t["turn_index"] for t in data["turns"]]
    assert turn_indices == sorted(turn_indices)
    assert turn_indices == list(range(len(turn_indices)))  # no gaps — this is the bug we fixed earlier
    assert len(data["metrics_timeline"]) == len(data["turns"])


def test_get_evaluation_returns_expected_fields(client):
    create_resp = client.post(
        "/sessions", json={"task_type": "story_generation", "strategy": "baseline", "seed_prompt": "Eval test."}
    )
    session_id = create_resp.json()["session_id"]
    client.post(f"/sessions/{session_id}/messages", json={"content": "One more turn."})

    resp = client.get(f"/sessions/{session_id}/evaluation")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {
        "semantic_diversity",
        "embedding_dispersion",
        "lexical_diversity",
        "novelty_score",
        "num_turns",
        "num_interventions",
    }


def test_nonexistent_session_returns_404(client):
    resp = client.get("/sessions/does-not-exist")
    assert resp.status_code == 404


def test_health_endpoint_reports_backend_config(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_backend"] == "mock"
    assert data["embedding_backend"] == "mock"
