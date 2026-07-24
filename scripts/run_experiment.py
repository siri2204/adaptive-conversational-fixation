"""
Runs the experimental comparison described in the proposal (section 6):
baseline vs. the four exploration-support strategies, across several task
types and seed prompts.

DESIGNED FOR A LIMITED DAILY API QUOTA. Every completed trial is appended to
`experiment_results.jsonl` immediately. Re-running this script:
  - skips any (task_type, seed_index, strategy) combination already completed
  - stops itself once it estimates the next trial would exceed --max-calls
    for this invocation, so you never blow through a day's quota mid-run
  - can be safely re-run tomorrow, and the day after, picking up where it
    left off, until the full experiment matrix is complete

This means you can run this once a day against your free-tier quota and it
will gradually complete the full comparison over several days without ever
repeating work or wasting a call.

Usage:
    # See what's left to run and how many calls it's estimated to need:
    python -m scripts.run_experiment --plan

    # Run today's batch, capped at ~18 real LLM calls (leaves a safety
    # margin under a 20/day free-tier cap):
    python -m scripts.run_experiment --max-calls 18

    # Print the aggregated summary + export a CSV for stats, once you have
    # enough completed trials:
    python -m scripts.run_experiment --summarize
"""
import argparse
import json
import statistics
import uuid
from pathlib import Path

from app.database import init_db, SessionLocal, ConversationSession
from app.embeddings import get_embedding_backend
from app.llm_client import get_llm_client
from app.intervention import ExplorationTreeGenerator, BRANCH_CATEGORIES
from app.fixation import FixationAnalyzer
from app.strategies import get_strategy, StrategyContext
from app.evaluation import evaluate_conversation
from app.config import settings

RESULTS_FILE = Path(__file__).parent.parent / "experiment_results.jsonl"

# One or more seed prompts per task type. Add more here later if you have
# quota to spare — each seed x strategy combination is one independent trial.
#
# NOTE: `interface_design` was intentionally dropped from the core matrix to
# fit the remaining quota/timeline (2 tasks x 2 seeds x 5 strategies = 20
# combos, 4 blocks for the Friedman test, vs. 30 combos / 6 blocks with it
# included). The 4 interface_design trials already collected are kept as
# supplementary/exploratory data in experiment_results_supplementary.jsonl —
# not part of the primary statistical comparison. This scope decision was
# made before any post-fix adaptive results existed, so it isn't
# result-contingent.
SEED_PROMPTS = {
    "story_generation": [
        "A detective finds a locked door in an old mansion.",
        "A lighthouse keeper receives a mysterious radio signal one stormy night.",
    ],
    "product_brainstorming": [
        "I want to brainstorm a new product for helping people reduce food waste at home.",
        "Brainstorm a product that helps remote teams feel more connected.",
    ],
    "interface_design": [
        "Help me design a mobile app interface for tracking personal habits.",
        "Design an interface for a smart home energy dashboard.",
    ],
}
# Reverted to the full original 3-task x 2-seed x 5-strategy matrix (30
# combos, 6 blocks) since there was time to complete it after all. The
# scope reductions considered earlier (dropping interface_design, then
# further to story_generation only) are no longer needed.

STRATEGIES = ["baseline", "static", "fixed_interval", "user_triggered", "adaptive"]


def all_combos() -> list[dict]:
    combos = []
    for task_type, seeds in SEED_PROMPTS.items():
        for seed_index, seed in enumerate(seeds):
            for strategy_name in STRATEGIES:
                combos.append(
                    {"task_type": task_type, "seed_index": seed_index, "seed": seed, "strategy": strategy_name}
                )
    return combos


def load_completed() -> set:
    if not RESULTS_FILE.exists():
        return set()
    completed = set()
    with open(RESULTS_FILE) as f:
        for line in f:
            row = json.loads(line)
            completed.add((row["task_type"], row["seed_index"], row["strategy"]))
    return completed


def append_result(row: dict):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


class CountingLLMClient:
    """Wraps the real LLM client just to count actual calls made, so the
    budget check reflects reality rather than a rough estimate."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return self.inner.generate(messages)

    def generate_json(self, messages):
        self.calls += 1
        return self.inner.generate_json(messages)


def run_one(db, combo: dict, num_turns: int, branch_pref: str, llm) -> dict:
    task_type, seed, strategy_name = combo["task_type"], combo["seed"], combo["strategy"]

    analyzer = FixationAnalyzer(
        window=settings.fixation_window,
        similarity_threshold=settings.fixation_similarity_threshold,
        dispersion_threshold=settings.fixation_dispersion_threshold,
        score_threshold=settings.fixation_score_threshold,
    )
    strategy = get_strategy(
        strategy_name, analyzer, interval=settings.fixed_interval_turns, cooldown=settings.adaptive_cooldown_turns
    )
    embedder = get_embedding_backend()
    tree_gen = ExplorationTreeGenerator(llm)

    sess = ConversationSession(id=str(uuid.uuid4()), task_type=task_type, strategy=strategy_name, strategy_params="{}")
    db.add(sess)
    db.flush()

    history: list[dict] = []
    embeddings: list[list[float]] = []
    turns_log: list[dict] = []
    last_intervention_turn = -1
    intervention_count = 0

    def add_turn(role, content):
        emb = embedder.embed(content)
        history.append({"role": role, "content": content})
        embeddings.append(emb)
        turns_log.append({"content": content, "embedding": emb})

    add_turn("user", seed)

    seed_ctx = StrategyContext(turn_index=0, embeddings_so_far=embeddings, last_intervention_turn=-1, user_forced=False)
    seed_decision = strategy.decide(seed_ctx)
    if seed_decision.intervene:
        branches = tree_gen.generate(history, task_type)
        last_intervention_turn = 0
        branch_key = BRANCH_CATEGORIES[intervention_count % len(BRANCH_CATEGORIES)]
        intervention_count += 1
        chosen = branches.get(branch_key, next(iter(branches.values())))
        add_turn("user", chosen["prompt"])
    reply = llm.generate(history)
    add_turn("assistant", reply)

    for i in range(num_turns - 1):
        ctx = StrategyContext(
            turn_index=len(history),
            embeddings_so_far=embeddings,
            last_intervention_turn=last_intervention_turn,
            user_forced=(strategy_name == "user_triggered" and i > 0 and i % 4 == 0),
        )
        decision = strategy.decide(ctx)

        if decision.intervene:
            branches = tree_gen.generate(history, task_type)
            last_intervention_turn = ctx.turn_index
            branch_key = BRANCH_CATEGORIES[intervention_count % len(BRANCH_CATEGORIES)]
            intervention_count += 1
            chosen = branches.get(branch_key, next(iter(branches.values())))
            add_turn("user", chosen["prompt"])
        else:
            add_turn("user", "Can you refine and expand on that a bit more?")

        reply = llm.generate(history)
        add_turn("assistant", reply)

    metrics = evaluate_conversation(turns_log)
    metrics.update(
        {
            "task_type": task_type,
            "seed_index": combo["seed_index"],
            "strategy": strategy_name,
            "intervention_count": intervention_count,
        }
    )
    return metrics


def estimate_calls_for_combo(combo: dict, num_turns: int) -> int:
    """Rough upper-bound estimate: 1 call per turn, plus up to 1 extra call
    per turn if that strategy could plausibly intervene every turn. Used only
    to decide whether to *start* a combo, not for anything scientific."""
    base = num_turns
    if combo["strategy"] == "baseline":
        return base
    if combo["strategy"] == "static":
        return base + 1
    return base + max(2, num_turns // 3)


def cmd_plan(num_turns: int):
    completed = load_completed()
    combos = all_combos()
    pending = [c for c in combos if (c["task_type"], c["seed_index"], c["strategy"]) not in completed]

    print(f"Total combinations in experiment matrix: {len(combos)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(pending)}\n")

    if not pending:
        print("Nothing left to run! Use --summarize to see results.")
        return

    total_est = sum(estimate_calls_for_combo(c, num_turns) for c in pending)
    print(f"Estimated calls to finish everything: ~{total_est}")
    print(f"At ~18 calls/day (free tier, safety margin), that's ~{-(-total_est // 18)} more days.\n")
    print("Next combos, in order:")
    for c in pending[:10]:
        est = estimate_calls_for_combo(c, num_turns)
        print(f"  {c['task_type']:22s} seed#{c['seed_index']} {c['strategy']:16s} (~{est} calls)")
    if len(pending) > 10:
        print(f"  ... and {len(pending) - 10} more")


def cmd_run(num_turns: int, branch_pref: str, max_calls: int, only_strategy: str | None = None):
    completed = load_completed()
    combos = all_combos()
    pending = [c for c in combos if (c["task_type"], c["seed_index"], c["strategy"]) not in completed]

    if only_strategy:
        pending = [c for c in pending if c["strategy"] == only_strategy]

    if not pending:
        if only_strategy:
            print(f"Nothing left to run for strategy='{only_strategy}'.")
        else:
            print("Nothing left to run — the full experiment matrix is already complete.")
        print("Use --summarize to see aggregated results.")
        return

    init_db()
    db = SessionLocal()
    raw_llm = get_llm_client()
    llm = CountingLLMClient(raw_llm)

    print(f"Backend: {settings.llm_backend} / {settings.embedding_backend}")
    print(f"{len(pending)} combos remaining. Budget for this run: {max_calls} calls.\n")

    try:
        for combo in pending:
            est = estimate_calls_for_combo(combo, num_turns)
            if llm.calls + est > max_calls:
                print(
                    f"Stopping here — next combo ({combo['task_type']}/{combo['strategy']}) "
                    f"estimated at ~{est} calls would exceed today's budget of {max_calls}. "
                    f"Used {llm.calls} calls this run."
                )
                break

            print(f"Running: {combo['task_type']} seed#{combo['seed_index']} strategy={combo['strategy']} ...")
            try:
                result = run_one(db, combo, num_turns, branch_pref, llm)
                db.commit()
                append_result(result)
                print(f"  -> done. {result}")
            except RuntimeError as e:
                print(f"\nStopped: {e}")
                break
    finally:
        db.close()

    print(f"\nTotal calls used this run: {llm.calls}")
    remaining = len(all_combos()) - len(load_completed())
    print(f"Combos still remaining: {remaining}")
    if remaining > 0:
        print("Re-run this script tomorrow (or after your quota resets) to continue.")
    else:
        print("Experiment matrix complete! Run with --summarize to see results.")


def cmd_summarize():
    if not RESULTS_FILE.exists():
        print("No results yet — run some trials first.")
        return

    rows = [json.loads(line) for line in open(RESULTS_FILE)]
    print(f"Total completed trials: {len(rows)}\n")

    print("=== Summary by strategy (mean across all completed trials) ===")
    for strategy_name in STRATEGIES:
        strategy_rows = [r for r in rows if r["strategy"] == strategy_name]
        if not strategy_rows:
            print(f"{strategy_name:16s} | (no trials yet)")
            continue

        def avg(key):
            return statistics.mean(r[key] for r in strategy_rows)

        def std(key):
            vals = [r[key] for r in strategy_rows]
            return statistics.stdev(vals) if len(vals) > 1 else 0.0

        print(
            f"{strategy_name:16s} | n={len(strategy_rows):2d} "
            f"semantic_diversity={avg('semantic_diversity'):.3f}±{std('semantic_diversity'):.3f} "
            f"dispersion={avg('embedding_dispersion'):.3f}±{std('embedding_dispersion'):.3f} "
            f"lexical_diversity={avg('lexical_diversity'):.3f}±{std('lexical_diversity'):.3f} "
            f"novelty={avg('novelty_score'):.3f}±{std('novelty_score'):.3f} "
            f"interventions={avg('intervention_count'):.1f}"
        )

    csv_path = RESULTS_FILE.with_suffix(".csv")
    if rows:
        keys = ["task_type", "seed_index", "strategy", "semantic_diversity", "embedding_dispersion",
                "lexical_diversity", "novelty_score", "num_turns", "intervention_count"]
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"\nExported {csv_path} for use in a stats package (R, pandas, SPSS, etc.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=6, help="User/assistant turn-pairs per trial")
    parser.add_argument("--branch", type=str, default="abstract_reframing", choices=BRANCH_CATEGORIES)
    parser.add_argument("--max-calls", type=int, default=12, help="Stop before exceeding this many real LLM calls")
    parser.add_argument("--plan", action="store_true", help="Show what's left to run and estimated cost, then exit")
    parser.add_argument("--summarize", action="store_true", help="Print aggregated results + export CSV, then exit")
    parser.add_argument(
        "--only-strategy",
        type=str,
        default=None,
        choices=STRATEGIES,
        help="Run only pending combos for this strategy (e.g. --only-strategy adaptive to prioritize it)",
    )
    args = parser.parse_args()

    if args.plan:
        cmd_plan(args.turns)
    elif args.summarize:
        cmd_summarize()
    else:
        cmd_run(args.turns, args.branch, args.max_calls, args.only_strategy)


if __name__ == "__main__":
    main()
