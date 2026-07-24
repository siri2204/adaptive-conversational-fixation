"""
Statistical significance testing for the experimental comparison
(proposal section 7 / README "Statistical testing" gap).

Reads experiment_results.jsonl (written incrementally by run_experiment.py)
and runs paired comparisons between the baseline strategy and each of the
four exploration-support strategies, matched on (task_type, seed_index) so
each pair being compared ran on the *same* seed prompt.

For each metric (semantic_diversity, embedding_dispersion, lexical_diversity,
novelty_score) and each non-baseline strategy, this reports:
  - mean difference (strategy - baseline)
  - paired t-test (parametric; assumes roughly normal differences)
  - Wilcoxon signed-rank test (non-parametric; safer with small n or skew)
  - Cohen's d for effect size

With n=2 seeds x 3 task_types = 6 pairs per strategy (the current
SEED_PROMPTS setup), these tests are underpowered for strong claims -- this
script is meant to give you the right numbers to report, with the right
caveats, not to manufacture significance that isn't there. Add more seeds in
run_experiment.py's SEED_PROMPTS if you want more statistical power.

Usage:
    python -m scripts.analyze_results
    python -m scripts.analyze_results --alpha 0.10
    python -m scripts.analyze_results --results-file path/to/other.jsonl
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

DEFAULT_RESULTS_FILE = Path(__file__).parent.parent / "experiment_results.jsonl"
METRICS = ["semantic_diversity", "embedding_dispersion", "lexical_diversity", "novelty_score"]
STRATEGIES = ["static", "fixed_interval", "user_triggered", "adaptive"]
BASELINE = "baseline"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m scripts.run_experiment` first to generate results."
        )
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_seed(rows: list[dict], strategy: str) -> dict[tuple, dict]:
    """Key: (task_type, seed_index) -> result row, for a given strategy."""
    return {
        (r["task_type"], r["seed_index"]): r
        for r in rows
        if r["strategy"] == strategy
    }


def paired_values(rows: list[dict], strategy: str, metric: str) -> tuple[list[float], list[float], list[str]]:
    """Returns (baseline_values, strategy_values, labels) for seeds present
    in BOTH baseline and the given strategy, in matched order."""
    baseline_idx = index_by_seed(rows, BASELINE)
    strategy_idx = index_by_seed(rows, strategy)
    common_keys = sorted(set(baseline_idx) & set(strategy_idx))

    baseline_vals = [baseline_idx[k][metric] for k in common_keys]
    strategy_vals = [strategy_idx[k][metric] for k in common_keys]
    labels = [f"{k[0]}/seed{k[1]}" for k in common_keys]
    return baseline_vals, strategy_vals, labels


def cohens_d_paired(diffs: list[float]) -> float:
    """Cohen's d for paired samples: mean difference / std of differences."""
    if len(diffs) < 2:
        return 0.0
    sd = statistics.stdev(diffs)
    if sd == 0:
        return 0.0
    return statistics.mean(diffs) / sd


def run_comparison(rows: list[dict], strategy: str, metric: str, alpha: float) -> dict | None:
    baseline_vals, strategy_vals, labels = paired_values(rows, strategy, metric)
    n = len(baseline_vals)
    if n < 2:
        return None

    diffs = [s - b for s, b in zip(strategy_vals, baseline_vals)]
    mean_diff = statistics.mean(diffs)
    d = cohens_d_paired(diffs)

    result = {
        "strategy": strategy,
        "metric": metric,
        "n_pairs": n,
        "baseline_mean": statistics.mean(baseline_vals),
        "strategy_mean": statistics.mean(strategy_vals),
        "mean_diff": mean_diff,
        "cohens_d": d,
        "t_stat": None,
        "t_pvalue": None,
        "wilcoxon_stat": None,
        "wilcoxon_pvalue": None,
        "significant_at_alpha": None,
    }

    if scipy_stats is None:
        return result

    if n >= 2 and any(d != 0 for d in diffs):
        t_stat, t_p = scipy_stats.ttest_rel(strategy_vals, baseline_vals)
        result["t_stat"], result["t_pvalue"] = float(t_stat), float(t_p)

    # Wilcoxon needs at least a few non-zero differences and n>=~6 to be
    # meaningful, but scipy will run with fewer; we still report it,
    # flagging low-n cases in the printed output.
    try:
        if any(diff != 0 for diff in diffs):
            w_stat, w_p = scipy_stats.wilcoxon(strategy_vals, baseline_vals)
            result["wilcoxon_stat"], result["wilcoxon_pvalue"] = float(w_stat), float(w_p)
    except ValueError:
        pass  # e.g. all differences are zero, or n too small for scipy's exact method

    if result["t_pvalue"] is not None:
        result["significant_at_alpha"] = result["t_pvalue"] < alpha

    return result


def print_report(all_results: list[dict], alpha: float):
    print(f"\n{'='*100}")
    print(f"Paired comparisons vs. baseline (alpha={alpha})")
    print(f"{'='*100}\n")

    if scipy_stats is None:
        print("NOTE: scipy not installed (`pip install scipy`) — showing means/effect size only,")
        print("      no p-values. Install scipy for the significance tests.\n")

    for metric in METRICS:
        print(f"--- {metric} ---")
        metric_results = [r for r in all_results if r["metric"] == metric]
        if not metric_results:
            print("  (no matched pairs yet for any strategy)\n")
            continue

        for r in metric_results:
            sig_marker = ""
            if r["significant_at_alpha"] is True:
                sig_marker = "  *** significant ***"
            elif r["significant_at_alpha"] is False:
                sig_marker = "  (not significant)"

            line = (
                f"  {r['strategy']:16s} n={r['n_pairs']:2d}  "
                f"baseline={r['baseline_mean']:.3f}  strategy={r['strategy_mean']:.3f}  "
                f"diff={r['mean_diff']:+.3f}  d={r['cohens_d']:+.2f}"
            )
            if r["t_pvalue"] is not None:
                line += f"  t_p={r['t_pvalue']:.3f}"
            if r["wilcoxon_pvalue"] is not None:
                line += f"  wilcoxon_p={r['wilcoxon_pvalue']:.3f}"
            line += sig_marker
            print(line)

            if r["n_pairs"] < 6:
                print(
                    f"    (n={r['n_pairs']} pairs — underpowered; treat p-values as indicative, "
                    f"not confirmatory. Add more seeds to SEED_PROMPTS in run_experiment.py for more power.)"
                )
        print()

    print(f"{'='*100}")
    print("Effect size guide (Cohen's d): |d|~0.2 small, ~0.5 medium, ~0.8+ large.")
    print("Positive diff/d means the strategy scored HIGHER than baseline on that metric")
    print("(for these four metrics, higher is generally the desired direction — more")
    print("diversity/dispersion/novelty is the goal of the intervention).")
    print(f"{'='*100}\n")


def export_csv(all_results: list[dict], path: Path):
    if not all_results:
        return
    keys = list(all_results[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in all_results:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"Exported {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=str, default=str(DEFAULT_RESULTS_FILE))
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for the t-test")
    args = parser.parse_args()

    rows = load_rows(Path(args.results_file))
    print(f"Loaded {len(rows)} trial results from {args.results_file}")

    all_results = []
    for strategy in STRATEGIES:
        for metric in METRICS:
            r = run_comparison(rows, strategy, metric, args.alpha)
            if r is not None:
                all_results.append(r)

    if not all_results:
        print(
            "\nNo matched baseline/strategy pairs found yet. You need at least one "
            "completed (task_type, seed_index) combo for BOTH 'baseline' and a given "
            "strategy before a paired comparison is possible. Run more of "
            "`python -m scripts.run_experiment` and try again."
        )
        return

    print_report(all_results, args.alpha)

    csv_path = Path(args.results_file).with_name("stats_report.csv")
    export_csv(all_results, csv_path)


if __name__ == "__main__":
    main()
