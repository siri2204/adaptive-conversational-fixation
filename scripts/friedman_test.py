#!/usr/bin/env python3
"""
friedman_test.py — reproduce the Friedman results reported in the paper.

    python -m scripts.friedman_test

Reads experiment_results.jsonl and, for each evaluation metric, runs a Friedman
test across the six experimental blocks (task_type x seed_index) with the five
intervention strategies as repeated measures. Also reports the average rank of
each strategy across all metrics and blocks.

This reproduces Tables 5 and 6 of the report. `analyze_results.py` performs a
different analysis (paired comparisons against baseline) and does not produce
these numbers.

Outputs friedman_results.json, average_ranks.csv and strategy_means.csv
alongside experiment_results.jsonl.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

try:
    from scipy.stats import friedmanchisquare, rankdata
except ImportError:
    sys.exit("error: scipy is required (pip install scipy)")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "experiment_results.jsonl")

METRICS = ["semantic_diversity", "embedding_dispersion",
           "lexical_diversity", "novelty_score"]
STRATEGIES = ["baseline", "static", "fixed_interval",
              "user_triggered", "adaptive"]


def load(path):
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load(RESULTS)
    print(f"loaded {len(rows)} trials from {os.path.basename(RESULTS)}\n")

    # index by (block, strategy); a block is one task_type x seed_index pair
    table = {}
    blocks = set()
    for r in rows:
        block = (r["task_type"], r["seed_index"])
        blocks.add(block)
        table[(block, r["strategy"])] = r
    blocks = sorted(blocks)

    missing = [(b, s) for b in blocks for s in STRATEGIES
               if (b, s) not in table]
    if missing:
        print(f"warning: {len(missing)} block/strategy combinations missing; "
              f"the design is incomplete and results may be unreliable")

    print(f"{len(blocks)} blocks x {len(STRATEGIES)} strategies\n")

    # ---- Friedman test per metric (Table 5) --------------------------------
    print("=" * 62)
    print("FRIEDMAN TESTS")
    print("=" * 62)
    print(f"  {'metric':<24}{'chi2':>9}{'p':>10}   significant (a=.05)")
    print("  " + "-" * 58)

    results = {}
    for m in METRICS:
        cols = []
        for s in STRATEGIES:
            col = [table[(b, s)][m] for b in blocks if (b, s) in table]
            cols.append(col)
        if min(len(c) for c in cols) < 2:
            print(f"  {m:<24}  too few blocks")
            continue
        stat, p = friedmanchisquare(*cols)
        results[m] = {"chi2": float(stat), "p": float(p),
                      "df": len(STRATEGIES) - 1, "n_blocks": len(blocks)}
        print(f"  {m:<24}{stat:>9.2f}{p:>10.4f}   "
              f"{'yes' if p < 0.05 else 'no'}")

    print(f"\n  df = {len(STRATEGIES) - 1} for all tests\n")

    # ---- Average ranks (Table 6) ------------------------------------------
    # Rank strategies within each (block, metric); higher value = better = rank 1
    rank_sums = defaultdict(list)
    for b in blocks:
        for m in METRICS:
            vals = [table[(b, s)][m] for s in STRATEGIES if (b, s) in table]
            present = [s for s in STRATEGIES if (b, s) in table]
            if len(vals) < 2:
                continue
            # negate so that the largest value receives rank 1
            ranks = rankdata([-v for v in vals])
            for s, rk in zip(present, ranks):
                rank_sums[s].append(float(rk))

    avg_ranks = {s: sum(v) / len(v) for s, v in rank_sums.items() if v}
    ordered = sorted(avg_ranks.items(), key=lambda kv: kv[1])

    print("=" * 62)
    print("AVERAGE RANKS ACROSS ALL METRICS AND BLOCKS (lower is better)")
    print("=" * 62)
    print(f"  {'strategy':<20}{'avg rank':>10}")
    print("  " + "-" * 30)
    for s, r in ordered:
        print(f"  {s:<20}{r:>10.3f}")

    # ---- Strategy means ---------------------------------------------------
    print("\n" + "=" * 62)
    print("MEAN METRIC VALUES PER STRATEGY")
    print("=" * 62)
    header = f"  {'strategy':<18}" + "".join(f"{m[:12]:>14}" for m in METRICS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    means = {}
    for s in STRATEGIES:
        vals = {m: [table[(b, s)][m] for b in blocks if (b, s) in table]
                for m in METRICS}
        means[s] = {m: sum(v) / len(v) for m, v in vals.items() if v}
        line = f"  {s:<18}"
        for m in METRICS:
            line += f"{means[s].get(m, float('nan')):>14.4f}"
        print(line)

    # also report intervention counts, since they differ across strategies
    print("\n  mean interventions per conversation:")
    for s in STRATEGIES:
        ic = [table[(b, s)].get("intervention_count")
              for b in blocks if (b, s) in table]
        ic = [x for x in ic if x is not None]
        if ic:
            spread = "" if len(set(ic)) == 1 else f"  (range {min(ic)}-{max(ic)})"
            print(f"    {s:<20}{sum(ic)/len(ic):>6.2f}{spread}")

    # ---- exports -----------------------------------------------------------
    with open(os.path.join(HERE, "friedman_results.json"), "w") as f:
        json.dump({"blocks": len(blocks), "tests": results,
                   "average_ranks": avg_ranks}, f, indent=2)
    with open(os.path.join(HERE, "average_ranks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "average_rank"])
        for s, r in ordered:
            w.writerow([s, f"{r:.3f}"])
    with open(os.path.join(HERE, "strategy_means.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy"] + METRICS)
        for s in STRATEGIES:
            w.writerow([s] + [f"{means[s].get(m, ''):.4f}" for m in METRICS])

    print("\nwrote friedman_results.json, average_ranks.csv, strategy_means.csv")


if __name__ == "__main__":
    main()
