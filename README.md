# Adaptive Detection and Mitigation of Conversational Fixation

A stateful FastAPI backend that detects **semantic fixation** during long-form
human-AI co-creation and delivers structured **exploration-tree** interventions
to push the user toward unexplored regions of the idea space.

This is the backend skeleton for the project described in the proposal: it
implements the fixation-detection pipeline, all four intervention-timing
strategies, the exploration-tree generation mechanism, and the evaluation
metrics needed for the experimental comparison.

## Why it runs offline by default

Everything defaults to **mock** LLM + **mock** embedding backends, so you can
develop, test, and demo the whole pipeline with zero API keys and zero
network access. Flip two env vars to go live with Gemini + real sentence
embeddings once you're ready (see `.env.example`).

> Note: this matters if you're running inside a sandboxed environment (like
> the one this was built in) — `sentence-transformers` needs to download model
> weights from huggingface.co, and the Gemini client needs to reach Google's
> API, neither of which may be reachable from a locked-down network. The mock
> backends have zero external dependencies for exactly this reason.

## Architecture

```
app/
  config.py         Settings (env-var driven), all thresholds live here
  database.py        SQLAlchemy models: ConversationSession, Turn, InterventionEvent
  schemas.py         Pydantic request/response models
  embeddings.py       EmbeddingBackend: MockEmbeddingBackend | SentenceTransformerBackend
  llm_client.py       LLMClient: MockLLMClient | GeminiLLMClient
  fixation.py         FixationAnalyzer — the core detection math (see below)
  strategies.py       The 4 intervention-timing strategies + baseline
  intervention.py     ExplorationTreeGenerator — the intervention content
  evaluation.py       Post-hoc metrics: semantic diversity, dispersion, lexical
                       diversity, novelty
  routers/sessions.py FastAPI endpoints tying it all together
  main.py             App entrypoint
tests/                Unit tests for fixation math + strategy logic
scripts/run_experiment.py   Offline 5-way experimental comparison runner
```

### Fixation detection (`fixation.py`)

For a sliding window of the most recent N turn embeddings, three signals are
computed:

1. **avg_similarity** — mean pairwise cosine similarity within the window.
   High similarity ⇒ turns are saying near-identical things.
2. **dispersion** — mean distance of each embedding to the window centroid
   (how spread out the point cloud is). Low dispersion ⇒ convergence.
3. **trajectory_movement** — distance between the centroid of the first half
   and second half of the window. Low movement ⇒ the conversation isn't
   drifting anywhere new even turn-to-turn.

These combine into a single `fixation_score ∈ [0, 1]`; when it crosses
`FIXATION_FIXATION_SCORE_THRESHOLD` (default 0.65), the conversation is
flagged as fixated. All thresholds are configurable in `.env`.

### Intervention strategies (`strategies.py`)

All four share one interface — `decide(ctx) -> StrategyDecision` — so the
experimental comparison is just "run the same conversation loop with a
different strategy object":

| Strategy | When it fires |
|---|---|
| `baseline` | Never (control condition) |
| `static` | Once, before the very first assistant turn |
| `fixed_interval` | Every N turns (default 5), regardless of content |
| `user_triggered` | Only when the user explicitly asks (`force_intervene=true`) |
| `adaptive` | When `FixationAnalyzer` detects convergence, with a cooldown so it doesn't refire every turn |

A manual override (`force_intervene: true` in `/sessions/{id}/messages`) will
trigger an intervention **regardless of strategy** — handy for demos/debugging,
but you should disable that override path (or just not expose the button) when
running the actual controlled experiment, since it defeats the point of
comparing timing strategies.

### Exploration tree (`intervention.py`)

Five branch categories, per the proposal: `abstract_reframing`,
`contradictory_perspective`, `adjacent_domain`, `unconventional_alternative`,
`speculative_future`. Each is `{title, prompt}`. The user picks one via
`POST /sessions/{id}/branches/select`; the branch's prompt becomes the seed for
the next turn, redirecting the conversation without discarding history.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # defaults already run fully offline
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

### Going live with Gemini + real embeddings

```bash
pip install sentence-transformers google-genai
```

> **Important:** use `google-genai` (the current, actively-maintained SDK —
> `from google import genai`). The older `google-generativeai` package is
> deprecated as of Nov 2025 and won't work with this client. Also, model
> availability changes over time — `gemini-2.5-flash` is the default here as
> of mid-2026, but double-check https://ai.google.dev/gemini-api/docs/models
> for the current lineup before you run experiments, since older model IDs
> (e.g. `gemini-2.0-flash`) do eventually get shut down.

In `.env`:
```
FIXATION_LLM_BACKEND=gemini
FIXATION_GEMINI_API_KEY=your-key-here
FIXATION_EMBEDDING_BACKEND=sentence-transformers
```

## API walkthrough

```bash
# 1. Start a session
curl -X POST localhost:8000/sessions -H "Content-Type: application/json" -d '{
  "task_type": "product_brainstorming",
  "strategy": "adaptive",
  "seed_prompt": "I want to brainstorm a product for reducing food waste."
}'
# -> {"session_id": "...", "assistant_message": "...", ...}

# 2. Keep chatting
curl -X POST localhost:8000/sessions/{id}/messages -H "Content-Type: application/json" \
  -d '{"content": "Can you refine that idea further?"}'
# -> {"assistant_message": "...", "metrics": {...}, "intervened": false}

# When fixation is detected (or the strategy's timing rule fires), "intervened"
# becomes true and "branches" is populated instead of "assistant_message".

# 3. Force an intervention manually (debugging / user_triggered UI button)
curl -X POST localhost:8000/sessions/{id}/messages -H "Content-Type: application/json" \
  -d '{"content": "show me other directions", "force_intervene": true}'

# 4. Pick a branch to continue from
curl -X POST localhost:8000/sessions/{id}/branches/select -H "Content-Type: application/json" \
  -d '{"branch_category": "adjacent_domain"}'

# 5. Inspect full history + fixation metrics timeline
curl localhost:8000/sessions/{id}

# 6. Post-hoc evaluation metrics for this conversation
curl localhost:8000/sessions/{id}/evaluation
```

## Running the experimental comparison

```bash
python -m scripts.run_experiment --turns 10
```

Runs baseline + all 4 strategies across the core task types (story
generation, product brainstorming), fully offline, and prints per-run
and averaged metrics (semantic diversity, embedding dispersion, lexical
diversity, novelty score, intervention count). This is your starting point
for the quantitative half of the evaluation section — swap in real
Gemini/sentence-transformers backends and larger `--turns` once you're ready
to generate results for the writeup.

**Scope note:** the core matrix is 2 task types × 2 seeds × 5 strategies (20
trials, 4 blocks for the Friedman test), reduced from an original 3-task
design to fit the remaining quota/timeline. A third task type
(`interface_design`) was piloted early on; those trials are kept in
`experiment_results_supplementary.jsonl` as exploratory/illustrative data
rather than part of the primary statistical comparison, since it doesn't
have a complete, matched set of trials across all 5 strategies and both
seeds. This decision was made before the branch-selection fix (see below)
was validated against any adaptive-strategy numbers, so it isn't
result-contingent.

## Frontend

A minimal Streamlit UI is included in `frontend/app.py` — this is what your
1-2 study participants should actually use, instead of `/docs`.

```bash
pip install streamlit requests
```

With the backend already running (`uvicorn app.main:app --reload` in one
terminal), run the frontend in a second terminal:

```bash
streamlit run frontend/app.py
```

It opens in your browser automatically. Features:
- Sidebar: pick task type + strategy, enter an opening prompt, start a session
- Chat interface for the conversation itself
- Live fixation metrics (avg similarity, dispersion, fixation score) updating
  every turn, plus a running chart
- When an intervention fires, the five exploration branches appear as cards
  you can click to continue from
- A manual "Give me alternatives" button (maps to `force_intervene`) for
  demoing the mechanism on demand, independent of the strategy being tested
- "View Evaluation Summary" button to see the post-hoc metrics for the
  current session

The frontend is a thin client — it only talks to the FastAPI backend over
HTTP, so it works identically whether the backend is running mock or real
(Gemini + sentence-transformers) backends. Point it at a different backend
URL via the sidebar field if needed.

## Running tests

```bash
pytest tests/ -v
```

Covers the fixation-detection math (fixated vs. exploratory synthetic
conversations) and all four strategies' triggering logic (including cooldown
behavior for the adaptive strategy).

## What's deliberately left for you to build next

This skeleton covers the full proposal architecture end to end, but a few
things are intentionally left as extension points rather than fully built out,
since they depend on choices you'll want to make for the course deliverable:

- **Real embedding/LLM wiring**: works today with mocks; swap backends per
  the instructions above once you have API access and want real numbers.
- **Better lexical diversity metric**: currently a simple type-token ratio;
  swap in `lexicalrichness` (MTLD/MATTR) if the writeup calls for it.
- **User study harness**: `run_experiment.py` automates the *system-vs-system*
  comparison; the human user study (perceived creativity support, fixation
  reduction, satisfaction) needs a small survey instrument and a UI on top of
  this API — the API is designed so a simple frontend can be a thin client
  over `/sessions/*`.
- **Frontend**: none included. The API is REST/JSON and CORS-enabled so any
  frontend (a simple React chat UI, or even Streamlit for a fast prototype)
  can sit on top of it.
- **Statistical testing**: `run_experiment.py` prints averages; for the
  writeup you'll want to run many more trials per condition and run proper
  significance tests (e.g. paired t-tests or Wilcoxon across matched seeds)
  on the metrics it logs.
