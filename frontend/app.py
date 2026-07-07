"""
Minimal Streamlit frontend for the Conversational Fixation Detection API.

This is a thin client — all the real logic (fixation detection, strategies,
LLM calls) lives in the FastAPI backend. This UI just makes it usable for a
human participant instead of clicking through /docs.

Run the backend first (`uvicorn app.main:app --reload` from the project
root), then run this with:

    streamlit run frontend/app.py

Works identically whether the backend is running mock or real (Gemini +
sentence-transformers) — the frontend doesn't know or care which.
"""
import requests
import streamlit as st

API_BASE = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000")

st.set_page_config(page_title="Fixation Detection Demo", layout="wide")
st.title("Conversational Fixation Detection & Mitigation")

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": ..., "content": ...}
if "metrics_log" not in st.session_state:
    st.session_state.metrics_log = []  # list of dicts, one per assistant turn
if "pending_branches" not in st.session_state:
    st.session_state.pending_branches = None


def reset_session_state():
    st.session_state.session_id = None
    st.session_state.history = []
    st.session_state.metrics_log = []
    st.session_state.pending_branches = None


# ---------------------------------------------------------------------------
# Sidebar: session setup + live metrics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Session Setup")

    task_type = st.selectbox(
        "Task type",
        ["story_generation", "product_brainstorming", "interface_design", "other"],
        disabled=st.session_state.session_id is not None,
    )
    strategy = st.selectbox(
        "Intervention strategy",
        ["baseline", "static", "fixed_interval", "user_triggered", "adaptive"],
        index=4,
        disabled=st.session_state.session_id is not None,
    )
    seed_prompt = st.text_area(
        "Opening prompt",
        placeholder="e.g. A detective finds a locked door in an old mansion.",
        disabled=st.session_state.session_id is not None,
    )

    if st.session_state.session_id is None:
        if st.button("Start Session", type="primary", use_container_width=True):
            if not seed_prompt.strip():
                st.error("Enter an opening prompt first.")
            else:
                try:
                    resp = requests.post(
                        f"{API_BASE}/sessions",
                        json={"task_type": task_type, "strategy": strategy, "seed_prompt": seed_prompt},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.session_id = data["session_id"]
                    st.session_state.history.append({"role": "user", "content": seed_prompt})
                    if data.get("initial_branches"):
                        st.session_state.pending_branches = data["initial_branches"]
                    elif data.get("assistant_message"):
                        st.session_state.history.append(
                            {"role": "assistant", "content": data["assistant_message"]}
                        )
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Could not reach backend: {e}")
    else:
        st.success(f"Session active\n\n`{st.session_state.session_id[:8]}...`")
        st.caption(f"Strategy: **{strategy}** | Task: **{task_type}**")
        if st.button("End & Start New Session", use_container_width=True):
            reset_session_state()
            st.rerun()

    st.divider()
    st.header("Fixation Metrics")
    if st.session_state.metrics_log:
        latest = st.session_state.metrics_log[-1]
        col1, col2 = st.columns(2)
        col1.metric("Avg Similarity", f"{latest['avg_similarity']:.3f}")
        col2.metric("Dispersion", f"{latest['dispersion']:.3f}")
        col1.metric("Trajectory Mvmt", f"{latest['trajectory_movement']:.3f}")
        col2.metric("Fixation Score", f"{latest['fixation_score']:.3f}")
        if latest["is_fixated"]:
            st.warning("⚠️ Conversation flagged as fixated")
        else:
            st.info("Conversation is exploring freely")

        with st.expander("Fixation score over time"):
            st.line_chart(
                {"fixation_score": [m["fixation_score"] for m in st.session_state.metrics_log]}
            )
    else:
        st.caption("Metrics will appear here once the conversation starts.")

    st.divider()
    if st.session_state.session_id and st.button("View Evaluation Summary", use_container_width=True):
        try:
            resp = requests.get(f"{API_BASE}/sessions/{st.session_state.session_id}/evaluation", timeout=30)
            resp.raise_for_status()
            st.json(resp.json())
        except requests.RequestException as e:
            st.error(f"Could not fetch evaluation: {e}")


# ---------------------------------------------------------------------------
# Main panel: conversation + branch selection
# ---------------------------------------------------------------------------
for turn in st.session_state.history:
    with st.chat_message(turn["role"] if turn["role"] in ("user", "assistant") else "user"):
        st.write(turn["content"])

if st.session_state.pending_branches:
    st.subheader("🌳 Explore a different direction")
    st.caption("The conversation may be converging — pick a direction to continue from:")
    cols = st.columns(len(st.session_state.pending_branches))
    for col, (category, node) in zip(cols, st.session_state.pending_branches.items()):
        with col:
            st.markdown(f"**{node['title']}**")
            st.caption(node["prompt"])
            if st.button("Choose this", key=f"branch_{category}"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/sessions/{st.session_state.session_id}/branches/select",
                        json={"branch_category": category},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.history.append({"role": "user", "content": node["prompt"]})
                    st.session_state.history.append(
                        {"role": "assistant", "content": data["assistant_message"]}
                    )
                    st.session_state.metrics_log.append(data["metrics"])
                    st.session_state.pending_branches = None
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Could not reach backend: {e}")

elif st.session_state.session_id:
    user_input = st.chat_input("Continue the conversation...")
    force_col1, force_col2 = st.columns([4, 1])
    force_intervene = force_col2.button("💡 Give me alternatives")

    if user_input or force_intervene:
        content = user_input if user_input else "Show me some alternative directions."
        try:
            resp = requests.post(
                f"{API_BASE}/sessions/{st.session_state.session_id}/messages",
                json={"content": content, "force_intervene": force_intervene},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state.history.append({"role": "user", "content": content})
            st.session_state.metrics_log.append(data["metrics"])

            if data["intervened"]:
                st.session_state.pending_branches = data["branches"]
            else:
                st.session_state.history.append(
                    {"role": "assistant", "content": data["assistant_message"]}
                )
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Could not reach backend: {e}")
else:
    st.info("👈 Start a session in the sidebar to begin.")
