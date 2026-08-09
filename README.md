# Adaptive Detection and Mitigation of Conversational Fixation

Seminar project — Universität des Saarlandes.

A stateful FastAPI backend that detects **semantic fixation** during long-form
human–AI co-creation and delivers structured **exploration-tree** interventions
intended to push the conversation toward unexplored regions of the idea space.

This repository contains the full pipeline described in the report: the
fixation-detection math, four intervention-timing strategies plus a baseline
control, the exploration-tree generation mechanism, the evaluation metrics, the
automated experimental comparison, and the materials and analysis scripts used
for the human study.

---

## Results

Full matrix: 3 task types × 2 seed prompts × 5 strategies = 30 conversations
(6 blocks), run against `gemini-3.5-flash-lite` with `sentence-transformers`
(`all-MiniLM-L6-v2`) embeddings. Raw per-trial data is in
`experiment_results.jsonl`.

**Friedman test (6 blocks × 5 strategies, df = 4):**

| metric | χ² | p-value | significant (α = .05)? |
|---|---|---|---|
| semantic diversity | 13.73 | 0.008 | yes |
| embedding dispersion | 15.07 | 0.005 | yes |
| novelty score | 14.40 | 0.006 | yes |
| lexical diversity | 8.40 | 0.078 | no |

**Average rank per strategy (lower is better):**

| strategy | avg. rank |
|---|---|
| adaptive | 1.375 |
| user_triggered | 2.833 |
| static | 3.208 |
| fixed_interval | 3.333 |
| baseline | 4.250 |

The adaptive condition achieved the best overall ranking across all four
metrics. Two caveats are discussed in the report and repeated here:

- **Intervention frequency is confounded with timing.** Because conversations
  are twelve turns long, the static, fixed-interval and user-triggered
  strategies each perform a single intervention, while the adaptive strategy
  averaged three. The advantage therefore cannot be attributed to *when* it
  intervened rather than *how often*. A dose-matched control condition would
  be needed to separate the two.
- **Metrics are computed over complete conversations, including user turns.**
  Since intervention turns replace the fixed user prompt with exploration-tree
  branch text, lexical diversity in particular is partly influenced by the
  injected prompts. It was also the only metric not to reach significance.

Reproduce both tables with:

```bash
python -m scripts.friedman_test    # Friedman tests, average ranks, strategy means
python -m scripts.analyze_results  # paired comparisons of each strategy vs. baseline
```

`friedman_test.py` reproduces both tables above exactly and writes
`friedman_results.json`, `average_ranks.csv` and `strategy_means.csv`. It also
prints the mean intervention count per strategy, which is the basis for the
frequency caveat noted above. `analyze_results.py` performs a separate
analysis — paired t-tests and Wilcoxon signed-rank tests against the baseline
condition — and writes `stats_report.csv`.

### A bug found and fixed during development

An earlier pass at this comparison showed `adaptive` performing *worse* than
every other strategy despite intervening more often. The experiment runner was
passing a single fixed branch category to every intervention across the whole
matrix, so `adaptive` kept re-offering the *same* exploration direction rather
than a different one each time — compounding convergence instead of broadening
it. Fixed by rotating through the five branch categories by intervention count
in `scripts/run_experiment.py`. The results above come from the corrected
runner.

---

## Architecture

```
app/
  config.py            Settings (env-var driven); all thresholds live here
  database.py          SQLAlchemy models: ConversationSession, Turn, InterventionEvent
  schemas.py           Pydantic request/response models
  embeddings.py        MockEmbeddingBackend | SentenceTransformerBackend
  llm_client.py        MockLLMClient | GeminiLLMClient (retry/backoff for rate
                       limits and transient server overload)
  fixation.py          FixationAnalyzer — the core detection math
  strategies.py        The four intervention-timing strategies + baseline
  intervention.py      ExplorationTreeGenerator — the intervention content
  evaluation.py        Post-hoc metrics: semantic diversity, dispersion,
                       lexical diversity, novelty
  routers/sessions.py  FastAPI endpoints
  main.py              App entrypoint

frontend/app.py        Streamlit interface used by the study participants

scripts/
  run_experiment.py         Resumable, budget-aware experiment runner
  friedman_test.py          Friedman tests + average ranks (reproduces the
                            tables above)
  analyze_results.py        Paired comparisons of each strategy vs. baseline
  calibrate_thresholds.py   Threshold calibration against real embeddings
  analyze_session.py        Per-session record for the human study: turns,
                            interventions, and evaluation metrics
  reconstruct_trajectory.py Recomputes the per-turn fixation trajectory from
                            stored embeddings using the live FixationAnalyzer

study/
  participant_materials.pdf Task brief, questionnaire, and open questions
  build_materials.py        Generates the above
  data/                     Per-participant session records and responses

tests/                 Unit tests for the fixation math, strategies, evaluation
                       metrics, and API
```

### Fixation detection (`fixation.py`)

Over a sliding window of the six most recent turn embeddings, three signals are
computed:

1. **avg_similarity** — mean pairwise cosine similarity within the window.
   High similarity means turns are saying near-identical things.
2. **dispersion** — mean distance of each embedding to the window centroid.
   Low dispersion means convergence.
3. **trajectory_movement** — distance between the centroids of the first and
   second halves of the window. Low movement means the conversation isn't
   drifting anywhere new.

These combine into a `fixation_score ∈ [0, 1]`, weighted 0.5 / 0.3 / 0.2. A
window is flagged as fixated when the composite score crosses its threshold, or
when the similarity–dispersion criterion is met.

**The defaults in `config.py` are tuned for mock embeddings and should not be
used with real ones.** Real `sentence-transformers` output sits on a very
different scale. The values used for all reported experiments were recalibrated
empirically and live in `.env`:

```
FIXATION_FIXATION_SIMILARITY_THRESHOLD=0.27
FIXATION_FIXATION_DISPERSION_THRESHOLD=0.39
FIXATION_FIXATION_SCORE_THRESHOLD=0.17
```

If you change embedding models, recalibrate against real conversations
(`scripts/calibrate_thresholds.py`) before trusting any defaults.

### Intervention strategies (`strategies.py`)

All strategies share one interface — `decide(ctx) -> StrategyDecision` — so the
comparison is simply the same conversation loop with a different strategy
object:

| Strategy | When it fires |
|---|---|
| `baseline` | Never (control condition) |
| `static` | Once, before the first assistant turn |
| `fixed_interval` | Every N turns (default 5), regardless of content |
| `user_triggered` | Only on explicit request (`force_intervene=true`) |
| `adaptive` | When `FixationAnalyzer` detects convergence, subject to a four-turn cooldown |

A manual override (`force_intervene: true` on `/sessions/{id}/messages`)
triggers an intervention regardless of strategy. Events written by this path are
recorded with `user_forced = 1` in `intervention_events`, so manually requested
interventions can be distinguished from automatically detected ones after the
fact.

### Exploration tree (`intervention.py`)

Five branch categories: `abstract_reframing`, `contradictory_perspective`,
`adjacent_domain`, `unconventional_alternative`, `speculative_future`. Each is
`{title, prompt}`. In interactive use the user picks one via
`POST /sessions/{id}/branches/select`; in the automated runner, successive
interventions rotate through the categories.

The generator normalises the language model's response before use. Gemini does
not return a consistent JSON structure between calls — it may return a dict
keyed by category, a bare list of branch objects, or a wrapper object — and
`_coerce_tree()` handles all three. An unhandled case caused a crash during a
live session that thirty automated trials had not triggered.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env           # defaults run fully offline
uvicorn app.main:app
```

Interactive API docs at `http://localhost:8000/docs`.

Everything defaults to **mock** LLM and embedding backends, so the pipeline is
developable and testable with no API key and no network access.

### Running with Gemini and real embeddings

```bash
pip install sentence-transformers google-genai
```

Use `google-genai` (`from google import genai`), not the deprecated
`google-generativeai`.

**Model availability changes quickly.** This project's default model changed
twice during development as Google retired or restricted models: it began on
`gemini-2.5-flash`, which became unreachable for newly created API keys
mid-project, and is currently pinned to `gemini-3.5-flash-lite` for stability
under free-tier load. Check
[the current model list](https://ai.google.dev/gemini-api/docs/models) before
assuming the configured default still works. `llm_client.py` retries transient
503 overload, but a 404 needs a model swap in `.env`, not a retry.

In `.env`:

```
FIXATION_LLM_BACKEND=gemini
FIXATION_GEMINI_API_KEY=your-key-here
FIXATION_GEMINI_MODEL=gemini-3.5-flash-lite
FIXATION_EMBEDDING_BACKEND=sentence-transformers
```

---

## API walkthrough

```bash
# 1. Start a session
curl -X POST localhost:8000/sessions -H "Content-Type: application/json" -d '{
  "task_type": "product_brainstorming",
  "strategy": "adaptive",
  "seed_prompt": "I want to brainstorm a product for reducing food waste."
}'

# 2. Continue the conversation
curl -X POST localhost:8000/sessions/{id}/messages -H "Content-Type: application/json" \
  -d '{"content": "Can you refine that idea further?"}'
# When an intervention fires, "intervened" becomes true and "branches" is
# populated instead of "assistant_message".

# 3. Force an intervention manually
curl -X POST localhost:8000/sessions/{id}/messages -H "Content-Type: application/json" \
  -d '{"content": "show me other directions", "force_intervene": true}'

# 4. Continue from a chosen branch
curl -X POST localhost:8000/sessions/{id}/branches/select -H "Content-Type: application/json" \
  -d '{"branch_category": "adjacent_domain"}'

# 5. Full history and fixation metrics timeline
curl localhost:8000/sessions/{id}

# 6. Post-hoc evaluation metrics
curl localhost:8000/sessions/{id}/evaluation
```

---

## Reproducing the experiment

```bash
python -m scripts.run_experiment --plan                    # pending trials, no calls spent
python -m scripts.run_experiment --turns 6 --max-calls 12  # run within a call budget
python -m scripts.run_experiment --only-strategy adaptive  # prioritise one strategy
python -m scripts.run_experiment --summarize               # aggregate stats + CSV export
```

The runner is resumable and budget-aware: each completed trial is appended to
`experiment_results.jsonl` and already-completed combinations are skipped, so it
can be run across several days against a free-tier daily quota.
`experiment_results.jsonl` in this repository is the complete final dataset used
for all reported results.

---

## Human study

A small qualitative study with two participants assessed whether the
exploration-tree intervention is perceived as useful during collaborative
creative work. Both completed a 15-minute story-generation session through the
Streamlit interface with the adaptive strategy enabled, followed by a
fourteen-item Likert questionnaire and a short interview.

- `study/participant_materials.pdf` — task brief, questionnaire, open questions
- `study/build_materials.py` — regenerates the above
- `study/data/P0*.json`, `P0*.md` — per-session records from `analyze_session.py`
- `study/data/P0*_traj.json` — per-turn fixation trajectories
- `study/data/P0*_responses.pdf` — transcribed questionnaire and interview responses

Participants are identified only by code; no names or contact details were
collected. Session analysis:

```bash
python scripts/analyze_session.py --session <SESSION_ID> --json out.json --md out.md
python scripts/reconstruct_trajectory.py --session <SESSION_ID> --offset 1
```

`reconstruct_trajectory.py` recomputes the fixation score at every turn from the
embeddings stored in `fixation.db`, using the project's own `FixationAnalyzer`.
This is necessary because the backend persists an `InterventionEvent` only when
an intervention fires, leaving no stored score for intervening turns. Run
`--validate` to replay all logged events and confirm the recomputed scores match
the stored ones exactly.

---

## Frontend

```bash
pip install streamlit requests
streamlit run frontend/app.py
```

Requires the backend running in another terminal. The interface provides task
and strategy selection, the chat itself, live fixation metrics with a running
chart, exploration branches rendered as selectable cards when an intervention
fires, a manual "give me alternatives" button, and a post-hoc evaluation
summary. It is a thin HTTP client, so it behaves identically against mock or
real backends.

Do not run the frontend while `run_experiment.py` is mid-run — both draw on the
same daily API quota.

---

## Tests

```bash
pytest tests/ -v
```

Covers the fixation-detection math against synthetic fixated and exploratory
conversations, all strategies' triggering logic including adaptive cooldown
behaviour, the evaluation metrics, and the API endpoints.

---

## Known limitations

- Lexical diversity uses a plain type-token ratio, which is sensitive to
  conversation length rather than genuine lexical variety. MTLD or MATTR
  (via `lexicalrichness`) would be a better measure.
- The Friedman test establishes an overall difference across strategies; a
  Nemenyi or Wilcoxon post-hoc test would identify which pairwise differences
  are individually significant.
- The fixation thresholds were calibrated against embedding statistics, not
  against human-labelled fixation events.
