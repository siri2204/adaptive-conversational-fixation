# Adaptive Detection and Mitigation of Conversational Fixation

A stateful FastAPI backend that detects **semantic fixation** during long-form
human-AI co-creation and delivers structured **exploration-tree** interventions
to push the user toward unexplored regions of the idea space.

This implements the full pipeline described in the proposal: the
fixation-detection math, all four intervention-timing strategies (plus a
baseline control), the exploration-tree generation mechanism, and the
evaluation metrics used for the experimental comparison — validated end-to-end
against real Gemini + sentence-transformer embeddings, with a completed,
statistically significant experimental comparison across all 5 strategies.

## Results

The core hypothesis — that **adaptive**, fixation-triggered intervention
timing outperforms fixed-schedule or manual timing — is supported by the
data. Full matrix: 3 task types × 2 seeds × 5 strategies = 30 trials (6
blocks), run against `gemini-3.5-flash-lite` + `sentence-transformers`
(`all-MiniLM-L6-v2`) embeddings.

**Friedman test (6 blocks × 5 strategies):**

| metric | χ² | p-value | significant (α=.05)? |
|---|---|---|---|
| semantic diversity | 13.73 | 0.0082 | yes |
| embedding dispersion | 15.07 | 0.0046 | yes |
| novelty score | 14.40 | 0.0061 | yes |
| lexical diversity | 8.40 | 0.0780 | no |

**Average rank per strategy (1 = best), across all 6 blocks:**

| strategy | avg. rank |
|---|---|
| adaptive | 1.33 |
| static | 2.67 |
| fixed_interval | 3.17 |
| user_triggered | 3.17 |
| baseline | 4.67 |

`adaptive` ranked 1st or 2nd in every single block. `baseline` (no
intervention) was reliably worst, as expected. Lexical diversity — a simple
type-token ratio — didn't reach significance; see "What's deliberately left"
below for a likely fix.

**A confound worth disclosing:** an earlier pass at this comparison showed
`adaptive` performing *worse* than every other strategy, despite intervening
more often. Root cause: the experiment runner passed a single fixed branch
category to every intervention across the whole matrix, so `adaptive`
(intervening ~3x per conversation) kept re-offering the *same* exploration
direction instead of a different one each time, compounding rather than
broadening. Fixed by rotating through the five branch categories by
intervention count (`scripts/run_experiment.py`). The numbers above are from
the corrected runner; raw per-trial data is in `experiment_results.jsonl`.

## Why it can also run fully offline

Everything defaults to **mock** LLM + **mock** embedding backends, so the
whole pipeline is developable/testable with zero API keys and zero network
access. Flip two env vars to go live with Gemini + real sentence embeddings
(see `.env.example`).

> Sandboxed/locked-down environments: `sentence-transformers` needs to
> download model weights from huggingface.co, and the Gemini client needs to
> reach Google's API — neither may be reachable on a restricted network. The
> mock backends have zero external dependencies for exactly this reason.

## Architecture

```
app/
  config.py         Settings (env-var driven), all thresholds live here
  database.py       SQLAlchemy models: ConversationSession, Turn, InterventionEvent
  schemas.py        Pydantic request/response models
  embeddings.py     EmbeddingBackend: MockEmbeddingBackend | SentenceTransformerBackend
  llm_client.py     LLMClient: MockLLMClient | GeminiLLMClient (with retry/backoff
                    for both rate limits and transient server overload)
  fixation.py       FixationAnalyzer — the core detection math (see below)
  strategies.py     The 4 intervention-timing strategies + baseline
  intervention.py   ExplorationTreeGenerator — the intervention content
  evaluation.py     Post-hoc metrics: semantic diversity, dispersion, lexical
                     diversity, novelty
  routers/sessions.py FastAPI endpoints tying it all together
  main.py           App entrypoint
tests/              Unit tests for fixation math + strategy logic (30 tests)
scripts/
  run_experiment.py    Resumable, budget-aware experimental comparison runner
  analyze_results.py   Friedman test + descriptive stats over experiment_results.jsonl
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
`FIXATION_FIXATION_SCORE_THRESHOLD`, the conversation is flagged as fixated.
**Default thresholds are tuned for mock embeddings.** Real
`sentence-transformers` output has a very different scale — the values
actually used for the experiment (in `.env`) were empirically recalibrated:
`FIXATION_FIXATION_SIMILARITY_THRESHOLD=0.27`,
`FIXATION_FIXATION_DISPERSION_THRESHOLD=0.39`,
`FIXATION_FIXATION_SCORE_THRESHOLD=0.17`. If you change embedding models,
recalibrate against a few real conversations before trusting the defaults.

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
but disable that override path (or just don't expose the button) when running
the actual controlled experiment, since it defeats the point of comparing
timing strategies.

### Exploration tree (`intervention.py`)

Five branch categories, per the proposal: `abstract_reframing`,
`contradictory_perspective`, `adjacent_domain`, `unconventional_alternative`,
`speculative_future`. Each is `{title, prompt}`. In interactive use, the user
picks one via `POST /sessions/{id}/branches/select`; in the automated
experiment runner, each successive intervention within a conversation rotates
to the next category (see "Results" above for why this matters).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
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
> deprecated and won't work with this client.
>
> **Model availability changes fast.** This project's default model has
> already changed twice over the course of development, purely due to
> Google retiring/restricting models: it started on `gemini-2.5-flash`,
> which became unreachable for newly-created API keys mid-project (a known,
> reported issue — see [Google's model list](https://ai.google.dev/gemini-api/docs/models)
> for the current lineup), and is currently pinned to
> `gemini-3.5-flash-lite` for stability under free-tier load. If you're
> reproducing this later, check current model availability before assuming
> the configured default still works — `llm_client.py`'s retry logic handles
> transient 503 overload gracefully, but a 404 "no longer available to new
> users" needs a model swap in `.env`, not a retry.

In `.env`:
```
FIXATION_LLM_BACKEND=gemini
FIXATION_GEMINI_API_KEY=your-key-here
FIXATION_GEMINI_MODEL=gemini-3.5-flash-lite
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
python -m scripts.run_experiment --plan                     # show what's pending, no calls spent
python -m scripts.run_experiment --turns 6 --max-calls 12    # run until today's call budget is used
python -m scripts.run_experiment --only-strategy adaptive    # prioritize one strategy's remaining trials
python -m scripts.run_experiment --summarize                 # aggregate stats + CSV export
```

Resumable and budget-aware: it persists each completed trial to
`experiment_results.jsonl` and skips anything already done, so it's safe to
run once a day against a free-tier daily quota and pick up where it left off.
`--only-strategy` is useful for prioritizing whichever strategy your
hypothesis depends on most if you're worried about running out of time
before finishing the full matrix.

The full matrix here is 3 task types (story generation, product
brainstorming, interface design) × 2 seed prompts × 5 strategies = 30 trials,
6 blocks for the Friedman test. `experiment_results.jsonl` in this repo
contains the complete, final dataset.

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

**Don't run the frontend while `run_experiment.py` is mid-run** — both draw
on the same daily API quota.

## Running tests

```bash
pytest tests/ -v
```

30 tests covering the fixation-detection math (fixated vs. exploratory
synthetic conversations), all four strategies' triggering logic (including
cooldown behavior for the adaptive strategy), and the evaluation metrics.

## What's deliberately left for you to build next

- **Better lexical diversity metric**: currently a simple type-token ratio,
  and the one metric that didn't reach significance in the Friedman test —
  swap in `lexicalrichness` (MTLD/MATTR) if the writeup calls for it, since
  raw TTR is highly sensitive to conversation length rather than genuine
  lexical variety.
- **User study harness**: `run_experiment.py` automates the *system-vs-system*
  comparison; the human user study (perceived creativity support, fixation
  reduction, satisfaction) needs a small survey instrument on top of the
  Streamlit frontend above.
- **Post-hoc pairwise tests**: the Friedman test establishes an overall
  difference across strategies; a Nemenyi or Wilcoxon post-hoc test would
  pin down exactly which pairwise differences (e.g. adaptive vs. static)
  are individually significant.
