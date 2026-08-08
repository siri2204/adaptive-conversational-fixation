#!/usr/bin/env python3
"""
reconstruct_trajectory.py — recompute the fixation score at every turn.

Your app only writes an InterventionEvent row when an intervention actually
fires, so the database holds no score for the turns in between. It does, however,
store `embedding_json` for every turn — and the fixation score is a deterministic
function of those embeddings. So the full trajectory can be rebuilt after the
fact, exactly, with no live logging and no code change to the write path.

Run from the project root (the folder containing app/) with the venv active:

    python scripts/reconstruct_trajectory.py --validate
    python scripts/reconstruct_trajectory.py --session <ID> --json p01_traj.json

--validate replays every InterventionEvent already in the database and checks
that the recomputed score matches the stored one. If it matches, the
reconstruction is provably faithful to what the live system computed.

This imports YOUR FixationAnalyzer and YOUR settings. It never reimplements the
formula, so it cannot drift from the real detector.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.getcwd())

try:
    from app.fixation import FixationAnalyzer
except ImportError as e:
    sys.exit(f"error: could not import app.fixation ({e}).\n"
             f"       run this from the project root with the venv active.")


def build_analyzer(args) -> FixationAnalyzer:
    """Construct the analyzer the same way the running app does."""
    kw = {}
    try:
        from app.config import settings
        mapping = {
            "window": ["fixation_window", "window", "fixation_window_size"],
            "similarity_threshold": ["fixation_similarity_threshold"],
            "dispersion_threshold": ["fixation_dispersion_threshold"],
            "score_threshold": ["fixation_score_threshold"],
            "similarity_weight": ["fixation_similarity_weight", "similarity_weight"],
            "dispersion_weight": ["fixation_dispersion_weight", "dispersion_weight"],
            "trajectory_weight": ["fixation_trajectory_weight", "trajectory_weight"],
        }
        for param, names in mapping.items():
            for n in names:
                if hasattr(settings, n):
                    kw[param] = getattr(settings, n)
                    break
    except Exception as e:
        print(f"warning: could not read app.config settings ({e}); "
              f"falling back to FixationAnalyzer defaults")

    for k in ("window", "similarity_threshold", "dispersion_threshold",
              "score_threshold"):
        v = getattr(args, k, None)
        if v is not None:
            kw[k] = v

    print("analyzer configuration:")
    for k, v in sorted(kw.items()):
        print(f"    {k:<22} = {v}")
    if not kw:
        print("    (all defaults — check your .env is being loaded)")
    return FixationAnalyzer(**kw)


def load_session_embeddings(con, session_id):
    """All turns in order, with parsed embeddings. Matches what the app feeds
    the analyzer: every turn, not just the participant's."""
    rows = con.execute(
        "SELECT turn_index, role, content, embedding_json FROM turns "
        "WHERE session_id = ? ORDER BY turn_index ASC", (session_id,)).fetchall()
    out = []
    for ti, role, content, ej in rows:
        emb = None
        if ej:
            try:
                emb = json.loads(ej)
            except Exception:
                emb = None
        out.append({"turn_index": ti, "role": role,
                    "content": content or "", "embedding": emb})
    return out


def trajectory(analyzer, turns, offset=0):
    """Score at each prefix. offset shifts which prefix maps to which turn_index."""
    embs = [t["embedding"] for t in turns]
    if any(e is None for e in embs):
        missing = [t["turn_index"] for t in turns if t["embedding"] is None]
        print(f"warning: {len(missing)} turns have no stored embedding "
              f"(turn_index {missing[:8]}). They are skipped, which changes the "
              f"window contents and therefore the scores.")
        keep = [(t, e) for t, e in zip(turns, embs) if e is not None]
        turns = [t for t, _ in keep]
        embs = [e for _, e in keep]

    pts = []
    for L in range(2, len(embs) + 1):
        r = analyzer.analyze(embs[:L])
        pts.append({
            "prefix_len": L,
            "turn_index": turns[L - 1]["turn_index"] + offset,
            "role": turns[L - 1]["role"],
            "fixation_score": r.fixation_score,
            "avg_similarity": r.avg_similarity,
            "dispersion": r.dispersion,
            "trajectory_movement": r.trajectory_movement,
            "is_fixated": bool(r.is_fixated),
        })
    return pts


def validate(con, analyzer, tol=1e-6):
    """Replay every stored InterventionEvent and check the recomputed score."""
    evs = con.execute(
        "SELECT session_id, turn_index, fixation_score, avg_similarity "
        "FROM intervention_events ORDER BY created_at").fetchall()
    if not evs:
        print("no intervention_events rows to validate against.")
        return

    print(f"\nvalidating {len(evs)} stored event(s)\n" + "-" * 68)
    offsets_found = []
    for sid, ti, stored_score, stored_sim in evs:
        turns = load_session_embeddings(con, sid)
        if not turns:
            print(f"  {sid[:8]} turn {ti}: no turns found — session was deleted")
            continue
        embs = [t["embedding"] for t in turns if t["embedding"] is not None]
        best = None
        for L in range(2, len(embs) + 1):
            r = analyzer.analyze(embs[:L])
            d = abs(r.fixation_score - stored_score)
            if best is None or d < best[1]:
                best = (L, d, r)
        L, d, r = best
        ok = d <= tol
        status = "MATCH" if ok else ("close" if d < 1e-3 else "NO MATCH")
        print(f"  {sid[:8]} turn_index={ti:<3} stored={stored_score:.6f}")
        print(f"           best recompute = {r.fixation_score:.6f} at prefix "
              f"len {L}  (diff {d:.2e})  -> {status}")
        if ok:
            offsets_found.append(ti - L)

    print("-" * 68)
    if offsets_found:
        uniq = sorted(set(offsets_found))
        print(f"exact matches: {len(offsets_found)}/{len(evs)}")
        print(f"turn_index - prefix_len offset(s): {uniq}")
        if len(uniq) == 1:
            print(f"\nconsistent offset {uniq[0]} — pass --offset {uniq[0]} "
                  f"when generating a trajectory so turn numbers line up with "
                  f"your observation log.")
        else:
            print("\ninconsistent offsets — inspect before trusting turn alignment.")
    else:
        print("no exact matches. The analyzer settings here differ from the ones "
              "that produced those rows (thresholds/weights/window changed since, "
              "or .env is not loading). The trajectory shape is still valid; only "
              "comparison against these historical rows is unreliable.")


def sparkline(vals, threshold=None):
    blocks = "▁▂▃▄▅▆▇█"
    if not vals:
        return "(none)"
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    line = "".join(blocks[min(7, int((v - lo) / span * 7.999))] for v in vals)
    tail = f"   min {lo:.4f}  max {hi:.4f}"
    if threshold is not None:
        tail += f"  |  >= {threshold:.2f} on {sum(1 for v in vals if v >= threshold)}/{len(vals)}"
    return line + tail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="fixation.db")
    ap.add_argument("--session")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--json")
    ap.add_argument("--window", type=int)
    ap.add_argument("--similarity-threshold", type=float, dest="similarity_threshold")
    ap.add_argument("--dispersion-threshold", type=float, dest="dispersion_threshold")
    ap.add_argument("--score-threshold", type=float, dest="score_threshold")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"error: {args.db} not found")
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    analyzer = build_analyzer(args)

    if args.validate:
        validate(con, analyzer)
        if not args.session:
            return

    if not args.session:
        print("\npass --session <ID> to reconstruct one session's trajectory.")
        rows = con.execute(
            "SELECT session_id, COUNT(*) n FROM turns GROUP BY session_id "
            "HAVING n >= 4 ORDER BY n DESC LIMIT 10").fetchall()
        if rows:
            print("\nsessions with >= 4 turns:")
            for sid, n in rows:
                print(f"    {sid}  ({n} turns)")
        return

    turns = load_session_embeddings(con, args.session)
    if not turns:
        sys.exit(f"no turns for session {args.session!r}")
    pts = trajectory(analyzer, turns, args.offset)

    fired = {r[0] for r in con.execute(
        "SELECT turn_index FROM intervention_events WHERE session_id = ?",
        (args.session,))}

    thr = getattr(analyzer, "score_threshold", None)
    print(f"\nsession {args.session}  —  {len(turns)} turns, "
          f"{len(pts)} scored points\n")
    print(f"  {'turn':>4} {'role':<10} {'score':>8} {'sim':>7} {'disp':>7} "
          f"{'traj':>7}  flags")
    print("  " + "-" * 62)
    for p in pts:
        flags = []
        if p["is_fixated"]:
            flags.append("FIXATED")
        if p["turn_index"] in fired:
            flags.append("<<< INTERVENTION LOGGED")
        print(f"  {p['turn_index']:>4} {p['role'][:10]:<10} "
              f"{p['fixation_score']:>8.4f} {p['avg_similarity']:>7.3f} "
              f"{p['dispersion']:>7.3f} {p['trajectory_movement']:>7.3f}  "
              f"{' '.join(flags)}")

    print("\n  trajectory: " + sparkline([p["fixation_score"] for p in pts], thr))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"session_id": args.session,
                       "offset": args.offset,
                       "intervention_turns": sorted(fired),
                       "trajectory": pts}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
