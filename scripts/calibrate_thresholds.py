"""
Calibrates fixation-detection thresholds against your REAL LLM + embedding
backend. The defaults in config.py were educated guesses made before any real
data existed. Real sentence-transformer embeddings (and real, verbose LLM
output) sit on a very different numeric scale than those guesses assumed —
running this once against your actual Gemini + sentence-transformers setup
gives you empirically-grounded thresholds instead of arbitrary constants.

Method: runs several "fixated" conversations (deliberately repetitive/
restating) and several "divergent" conversations (deliberately topic-jumping)
through your real backends, measures the three raw signals for each, and
recommends thresholds at the midpoint between the two clusters.

REQUIRES FIXATION_LLM_BACKEND=gemini and FIXATION_EMBEDDING_BACKEND=
sentence-transformers in your .env — calibrating against the mock backends
defeats the purpose (this will still run against mocks without erroring, but
prints a warning, since it's useful for smoke-testing the script itself).

Usage:
    python -m scripts.calibrate_thresholds --trials 3
"""
import argparse
import statistics
import time

from app.llm_client import get_llm_client
from app.embeddings import get_embedding_backend
from app.fixation import FixationAnalyzer
from app.config import settings

# Free-tier Gemini keys are commonly capped at 5 requests/minute. We pace calls
# proactively rather than relying purely on the retry-after-429 logic in
# GeminiLLMClient, since that logic works but wastes time hitting the limit
# repeatedly before settling into a sustainable cadence.
SECONDS_BETWEEN_CALLS = 13

SEED = "Let's write a short story about someone who discovers a hidden room in their house."

FIXATED_FOLLOWUPS = [
    "Just summarize what you've told me so far in 2-3 sentences, without adding any new details.",
    "Restate the same summary again, as concisely as possible, with no new information.",
    "Say the same thing one more time, just rephrased slightly.",
    "Repeat the core idea again in a single sentence.",
]

DIVERGENT_FOLLOWUPS = [
    "Actually, let's abandon that — instead, write about a chef competing in a cooking contest.",
    "Now switch again — tell me about an astronaut stranded on Mars.",
    "Now switch to a completely different topic: a medieval blacksmith's daily routine.",
    "Switch one more time: describe a jazz musician's first big performance.",
]


def run_trial(followups: list[str]) -> list[list[float]]:
    llm = get_llm_client()
    embedder = get_embedding_backend()
    history = [{"role": "user", "content": SEED}]
    embeddings = [embedder.embed(SEED)]

    reply = llm.generate(history)
    history.append({"role": "assistant", "content": reply})
    embeddings.append(embedder.embed(reply))

    for followup in followups:
        time.sleep(SECONDS_BETWEEN_CALLS)
        history.append({"role": "user", "content": followup})
        embeddings.append(embedder.embed(followup))
        reply = llm.generate(history)
        history.append({"role": "assistant", "content": reply})
        embeddings.append(embedder.embed(reply))

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2, help="Trials per condition (fixated/divergent). Each trial makes 5 LLM calls, paced ~13s apart to respect free-tier rate limits.")
    args = parser.parse_args()

    if settings.llm_backend != "gemini" or settings.embedding_backend != "sentence-transformers":
        print(
            "WARNING: calibrating against mock backends won't give meaningful thresholds. "
            "Set FIXATION_LLM_BACKEND=gemini and FIXATION_EMBEDDING_BACKEND=sentence-transformers "
            "in your .env first if you want real numbers.\n"
        )

    analyzer = FixationAnalyzer(window=6)  # only used to extract raw signals here; its own thresholds are irrelevant

    fixated_runs, divergent_runs = [], []
    for i in range(args.trials):
        print(f"Running fixated trial {i + 1}/{args.trials}...")
        fixated_runs.append(run_trial(FIXATED_FOLLOWUPS))
        time.sleep(SECONDS_BETWEEN_CALLS)
        print(f"Running divergent trial {i + 1}/{args.trials}...")
        divergent_runs.append(run_trial(DIVERGENT_FOLLOWUPS))
        if i < args.trials - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)

    def last_window_signals(embeddings):
        r = analyzer.analyze(embeddings)
        return r.avg_similarity, r.dispersion, r.trajectory_movement

    fixated_signals = [last_window_signals(e) for e in fixated_runs]
    divergent_signals = [last_window_signals(e) for e in divergent_runs]

    def col(signals, i):
        return [s[i] for s in signals]

    print("\n=== Fixated trials (avg_similarity, dispersion, trajectory_movement) ===")
    for s in fixated_signals:
        print(f"  {s[0]:.3f}  {s[1]:.3f}  {s[2]:.3f}")
    print("=== Divergent trials ===")
    for s in divergent_signals:
        print(f"  {s[0]:.3f}  {s[1]:.3f}  {s[2]:.3f}")

    sim_fix_mean = statistics.mean(col(fixated_signals, 0))
    sim_div_mean = statistics.mean(col(divergent_signals, 0))
    disp_fix_mean = statistics.mean(col(fixated_signals, 1))
    disp_div_mean = statistics.mean(col(divergent_signals, 1))

    recommended_similarity_threshold = (sim_fix_mean + sim_div_mean) / 2
    # dispersion/trajectory are "lower = more fixated"; fixation.py's contribution
    # formula uses dispersion_threshold*2 as its normalization scale, so we pick
    # dispersion_threshold such that *2 lands near the midpoint between clusters.
    disp_midpoint = (disp_fix_mean + disp_div_mean) / 2
    recommended_dispersion_threshold = disp_midpoint / 2

    print("\n=== Recommended .env values (based on this run) ===")
    print(f"FIXATION_FIXATION_SIMILARITY_THRESHOLD={recommended_similarity_threshold:.3f}")
    print(f"FIXATION_FIXATION_DISPERSION_THRESHOLD={recommended_dispersion_threshold:.3f}")

    calibrated_analyzer = FixationAnalyzer(
        window=6,
        similarity_threshold=recommended_similarity_threshold,
        dispersion_threshold=recommended_dispersion_threshold,
    )
    fixated_scores = [calibrated_analyzer.analyze(e).fixation_score for e in fixated_runs]
    divergent_scores = [calibrated_analyzer.analyze(e).fixation_score for e in divergent_runs]
    recommended_score_threshold = (statistics.mean(fixated_scores) + statistics.mean(divergent_scores)) / 2

    print(f"FIXATION_FIXATION_SCORE_THRESHOLD={recommended_score_threshold:.3f}")
    print(f"\nFixated fixation_scores:   {[round(s, 3) for s in fixated_scores]}")
    print(f"Divergent fixation_scores: {[round(s, 3) for s in divergent_scores]}")
    print(
        "\nIf these two score lists overlap a lot, the separation is weak — consider "
        "more trials, or accept that avg_similarity alone may need more weight than "
        "dispersion/trajectory for your specific LLM's writing style."
    )


if __name__ == "__main__":
    main()
