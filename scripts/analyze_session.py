#!/usr/bin/env python3
"""
analyze_session.py — pull the objective record for one human-study session.

Drop this in scripts/ and run it from the project root.

    python scripts/analyze_session.py --list
    python scripts/analyze_session.py --session <SESSION_ID>
    python scripts/analyze_session.py --session <SESSION_ID> --json p01.json --md p01.md

It reports, for a single session:
  * the turn-by-turn fixation trajectory, with the turns where interventions fired
  * whether the fixation score actually rose in the turns *before* each intervention
  * whether semantic diversity / dispersion / novelty rose in the turns *after*
  * the same four summary metrics used in the automated experiment

This does not call any LLM API. It reads the database and (optionally) computes
embeddings locally, so it costs no quota and can be re-run freely.

--- Schema handling -------------------------------------------------------
The script does NOT assume your column names. It introspects the SQLite file
and maps columns onto roles by name. If it guesses wrong, run --schema to see
what it found and then override with the flags shown in that output, e.g.

    python scripts/analyze_session.py --schema
    python scripts/analyze_session.py --session S1 --turns-table messages \
        --col-content body --col-fixation score
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# --------------------------------------------------------------------------
# Column-role candidates, best guess first. Matching is case-insensitive and
# exact; add your own names here if the auto-detection misses.
# --------------------------------------------------------------------------
ROLE_CANDIDATES: dict[str, list[str]] = {
    "session_fk": ["session_id", "sess_id", "conversation_id", "convo_id"],
    "turn_index": ["turn_index", "turn_number", "turn_no", "turn", "idx",
                   "index", "position", "seq", "sequence", "order", "n"],
    "speaker": ["role", "speaker", "author", "sender", "actor", "source"],
    "content": ["content", "text", "message", "body", "message_text", "utterance"],
    "fixation": ["fixation_score", "fixation", "score", "fix_score"],
    "similarity": ["similarity", "cosine_similarity", "mean_similarity",
                   "consecutive_similarity", "sim"],
    "dispersion": ["dispersion", "embedding_dispersion", "disp"],
    "trajectory": ["trajectory", "trajectory_movement", "movement", "traj"],
    "intervened": ["intervened", "intervention", "is_intervention",
                   "intervention_triggered", "did_intervene", "triggered"],
    "branch": ["branch", "branch_category", "selected_branch", "branch_name",
               "chosen_branch"],
    "created": ["created_at", "timestamp", "ts", "time", "created", "datetime"],
    "embedding": ["embedding_json", "embedding", "embeddings", "vector", "emb"],
    "triggered": ["triggered", "did_trigger", "fired", "intervened",
                  "intervention_triggered"],
}

SESSION_ID_CANDIDATES = ["id", "session_id", "sess_id", "uuid"]
STRATEGY_CANDIDATES = ["strategy", "strategy_name", "condition", "arm"]
TASK_CANDIDATES = ["task", "task_type", "task_name", "prompt_type"]

USER_SPEAKERS = {"user", "human", "participant", "p", "you"}


# ==========================================================================
# Database introspection
# ==========================================================================
def connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        sys.exit(f"error: database not found at {db_path!r}\n"
                 f"       pass --db /path/to/fixation.db")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def list_tables(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'").fetchall()
    return [r["name"] for r in rows]


def columns_of(con: sqlite3.Connection, table: str) -> list[str]:
    return [r["name"] for r in con.execute(f'PRAGMA table_info("{table}")')]


def match_role(cols: list[str], role: str) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in ROLE_CANDIDATES[role]:
        if cand in lower:
            return lower[cand]
    # fall back to substring match, longest candidate first
    for cand in sorted(ROLE_CANDIDATES[role], key=len, reverse=True):
        for lc, orig in lower.items():
            if cand in lc:
                return orig
    return None


def find_turns_table(con: sqlite3.Connection) -> Optional[str]:
    """The turns table is the one that has both a content column and a session FK."""
    best, best_score = None, 0
    for t in list_tables(con):
        cols = columns_of(con, t)
        score = 0
        if match_role(cols, "content"):
            score += 3
        if match_role(cols, "session_fk"):
            score += 3
        if match_role(cols, "speaker"):
            score += 2
        if match_role(cols, "fixation"):
            score += 2
        if match_role(cols, "turn_index"):
            score += 1
        # prefer plausible names
        if t.lower() in ("turns", "turn", "messages", "message"):
            score += 2
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= 5 else None


def find_sessions_table(con: sqlite3.Connection, turns_table: str) -> Optional[str]:
    best, best_score = None, 0
    for t in list_tables(con):
        if t == turns_table:
            continue
        cols = columns_of(con, t)
        lower = [c.lower() for c in cols]
        score = 0
        if any(c in lower for c in SESSION_ID_CANDIDATES):
            score += 2
        if any(c in lower for c in STRATEGY_CANDIDATES):
            score += 3
        if any(c in lower for c in TASK_CANDIDATES):
            score += 2
        if t.lower() in ("sessions", "session", "conversations"):
            score += 3
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= 3 else None


def find_events_table(con: sqlite3.Connection, turns_table: str) -> Optional[str]:
    """Some schemas keep fixation scores in a separate per-evaluation events table."""
    best, best_score = None, 0
    for t in list_tables(con):
        if t == turns_table:
            continue
        cols = columns_of(con, t)
        score = 0
        if match_role(cols, "fixation"):
            score += 3
        if match_role(cols, "session_fk"):
            score += 2
        if match_role(cols, "turn_index"):
            score += 2
        if match_role(cols, "triggered"):
            score += 2
        if match_role(cols, "branch"):
            score += 1
        if "event" in t.lower() or "intervention" in t.lower():
            score += 3
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= 6 else None


@dataclass
class Mapping:
    turns_table: str
    sessions_table: Optional[str] = None
    events_table: Optional[str] = None
    cols: dict[str, Optional[str]] = field(default_factory=dict)
    ecols: dict[str, Optional[str]] = field(default_factory=dict)
    session_pk: Optional[str] = None
    strategy_col: Optional[str] = None
    task_col: Optional[str] = None

    def has(self, role: str) -> bool:
        return bool(self.cols.get(role))


def build_mapping(con: sqlite3.Connection, args) -> Mapping:
    turns = args.turns_table or find_turns_table(con)
    if not turns:
        sys.exit("error: could not identify a turns/messages table.\n"
                 "       run with --schema and pass --turns-table explicitly.")
    tcols = columns_of(con, turns)
    cols = {role: match_role(tcols, role) for role in ROLE_CANDIDATES}

    # explicit CLI overrides
    for role, override in [("content", args.col_content),
                           ("fixation", args.col_fixation),
                           ("session_fk", args.col_session_fk),
                           ("speaker", args.col_speaker),
                           ("turn_index", args.col_turn_index),
                           ("intervened", args.col_intervened),
                           ("branch", args.col_branch)]:
        if override:
            cols[role] = override

    ev = args.events_table or find_events_table(con, turns)
    ecols: dict[str, Optional[str]] = {}
    if ev:
        evcols = columns_of(con, ev)
        for role in ("session_fk", "turn_index", "fixation", "similarity",
                     "dispersion", "trajectory", "triggered", "branch"):
            ecols[role] = match_role(evcols, role)
        if args.col_triggered:
            ecols["triggered"] = args.col_triggered
        if args.col_branch:
            ecols["branch"] = args.col_branch

    sess = args.sessions_table or find_sessions_table(con, turns)
    m = Mapping(turns_table=turns, sessions_table=sess, events_table=ev,
                cols=cols, ecols=ecols)
    if sess:
        scols = columns_of(con, sess)
        lower = {c.lower(): c for c in scols}
        for c in SESSION_ID_CANDIDATES:
            if c in lower:
                m.session_pk = lower[c]
                break
        for c in STRATEGY_CANDIDATES:
            if c in lower:
                m.strategy_col = lower[c]
                break
        for c in TASK_CANDIDATES:
            if c in lower:
                m.task_col = lower[c]
                break
    return m


def print_schema(con: sqlite3.Connection, m: Mapping) -> None:
    print("\n=== TABLES IN DATABASE ===")
    for t in list_tables(con):
        n = con.execute(f'SELECT COUNT(*) c FROM "{t}"').fetchone()["c"]
        print(f"  {t:<24} {n:>6} rows   [{', '.join(columns_of(con, t))}]")
    print("\n=== DETECTED MAPPING ===")
    print(f"  turns table    : {m.turns_table}")
    print(f"  sessions table : {m.sessions_table or '(none found)'}")
    print(f"  events table   : {m.events_table or '(none found)'}")
    print("\n  -- columns in turns table --")
    for role, col in m.cols.items():
        if role in ("similarity", "dispersion", "trajectory", "triggered"):
            continue
        mark = " " if col else "!"
        print(f" {mark}  {role:<14}: {col or '-- not present --'}")
    if m.events_table:
        print(f"\n  -- columns in {m.events_table} (joined on session + turn_index) --")
        for role, col in m.ecols.items():
            mark = " " if col else "!"
            print(f" {mark}  {role:<14}: {col or '-- not present --'}")
    print("\nIf any of the above is wrong, override it, e.g.:")
    print("  --turns-table messages --col-content body --col-fixation score\n")


# ==========================================================================
# Data loading
# ==========================================================================
@dataclass
class Turn:
    n: int
    speaker: str
    content: str
    fixation: Optional[float] = None
    similarity: Optional[float] = None
    dispersion: Optional[float] = None
    trajectory: Optional[float] = None
    intervened: bool = False
    evaluated: bool = False
    branch: Optional[str] = None
    created: Optional[str] = None
    embedding: Optional[list] = None

    @property
    def is_user(self) -> bool:
        return (self.speaker or "").strip().lower() in USER_SPEAKERS


def truthy(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y") or (s not in ("", "0", "false",
                                                            "none", "null", "no"))


def load_turns(con: sqlite3.Connection, m: Mapping, session_id: str) -> list[Turn]:
    c = m.cols
    sel, roles = [], []
    for role in ("turn_index", "speaker", "content", "fixation",
                 "intervened", "branch", "created", "embedding"):
        if c.get(role):
            sel.append(f'"{c[role]}"')
            roles.append(role)
    if not sel:
        sys.exit("error: no usable columns found in the turns table.")

    order = f'"{c["turn_index"]}"' if c.get("turn_index") else (
        f'"{c["created"]}"' if c.get("created") else "rowid")
    fk = c.get("session_fk")
    if not fk:
        sys.exit("error: no session foreign-key column found; pass --col-session-fk.")

    q = (f'SELECT {", ".join(sel)} FROM "{m.turns_table}" '
         f'WHERE "{fk}" = ? ORDER BY {order} ASC')
    rows = con.execute(q, (session_id,)).fetchall()

    turns = []
    for i, r in enumerate(rows, start=1):
        d = {roles[j]: r[j] for j in range(len(roles))}
        emb = None
        if d.get("embedding"):
            try:
                emb = json.loads(d["embedding"])
            except Exception:
                emb = None
        turns.append(Turn(
            n=int(d["turn_index"]) if d.get("turn_index") is not None else i,
            speaker=str(d.get("speaker") or "?"),
            content=str(d.get("content") or ""),
            fixation=(float(d["fixation"])
                      if d.get("fixation") is not None else None),
            intervened=truthy(d.get("intervened")),
            branch=(str(d["branch"]) if d.get("branch") else None),
            created=(str(d["created"]) if d.get("created") else None),
            embedding=emb,
        ))

    # --- merge the events table, if fixation data lives there ---------------
    if m.events_table and m.ecols.get("session_fk"):
        e = m.ecols
        want = [r for r in ("turn_index", "fixation", "similarity", "dispersion",
                            "trajectory", "triggered", "branch") if e.get(r)]
        cols_sql = ", ".join(f'"{e[r]}"' for r in want)
        erows = con.execute(
            f'SELECT {cols_sql} FROM "{m.events_table}" WHERE "{e["session_fk"]}" = ?',
            (session_id,)).fetchall()
        by_turn = {}
        for er in erows:
            d = {want[j]: er[j] for j in range(len(want))}
            if d.get("turn_index") is None:
                continue
            by_turn[int(d["turn_index"])] = d
        matched = 0
        for t in turns:
            d = by_turn.get(t.n)
            if not d:
                continue
            matched += 1
            t.evaluated = True
            for role in ("fixation", "similarity", "dispersion", "trajectory"):
                if d.get(role) is not None:
                    setattr(t, role, float(d[role]))
            if "triggered" in d:
                t.intervened = truthy(d.get("triggered"))
            if d.get("branch"):
                t.branch = str(d["branch"])
        orphans = set(by_turn) - {t.n for t in turns}
        if orphans:
            print(f"note: {len(orphans)} event rows reference turn indices with no "
                  f"matching turn: {sorted(orphans)[:10]}")
        if erows and matched == 0:
            print("warning: found event rows for this session but none matched a "
                  "turn_index. The two tables may index turns differently.")
    return turns


def list_sessions(con: sqlite3.Connection, m: Mapping) -> None:
    fk = m.cols.get("session_fk")
    counts = {}
    if fk:
        for r in con.execute(
                f'SELECT "{fk}" s, COUNT(*) c FROM "{m.turns_table}" '
                f'GROUP BY "{fk}"'):
            counts[str(r["s"])] = r["c"]

    print("\n=== SESSIONS ===")
    if m.sessions_table and m.session_pk:
        extra = [x for x in (m.strategy_col, m.task_col) if x]
        cols = ", ".join(f'"{x}"' for x in [m.session_pk] + extra)
        rows = con.execute(
            f'SELECT {cols} FROM "{m.sessions_table}"').fetchall()
        hdr = f'{"session id":<40} {"turns":>5}  ' + "  ".join(
            f"{e:<20}" for e in extra)
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            sid = str(r[m.session_pk])
            line = f"{sid:<40} {counts.get(sid, 0):>5}  "
            line += "  ".join(f"{str(r[e])[:20]:<20}" for e in extra)
            print(line)
    else:
        print(f'{"session id":<40} {"turns":>5}')
        for sid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"{sid:<40} {n:>5}")
    print()


# ==========================================================================
# Metrics — prefers YOUR implementations, falls back to local ones
# ==========================================================================
def try_import_project_metrics():
    """Look for the evaluation functions the automated experiment used."""
    sys.path.insert(0, os.getcwd())
    candidates = [
        "app.evaluation", "app.metrics", "app.eval_metrics",
        "app.services.evaluation", "app.services.metrics",
        "evaluation", "metrics", "app.core.evaluation",
    ]
    wanted = ["semantic_diversity", "embedding_dispersion", "novelty_score"]
    for modname in candidates:
        try:
            mod = __import__(modname, fromlist=["*"])
        except Exception:
            continue
        found = {w: getattr(mod, w) for w in wanted if hasattr(mod, w)}
        if len(found) == len(wanted):
            return modname, found
    return None, None


def cosine(a, b) -> float:
    import numpy as np
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def local_semantic_diversity(embs) -> float:
    """1 - mean pairwise cosine similarity."""
    n = len(embs)
    if n < 2:
        return float("nan")
    sims = [cosine(embs[i], embs[j]) for i in range(n) for j in range(i + 1, n)]
    return 1.0 - (sum(sims) / len(sims))


def local_embedding_dispersion(embs) -> float:
    """Mean Euclidean distance from the centroid."""
    import numpy as np
    if len(embs) < 2:
        return float("nan")
    arr = np.asarray(embs)
    centroid = arr.mean(axis=0)
    return float(np.mean(np.linalg.norm(arr - centroid, axis=1)))


def local_novelty_score(embs) -> float:
    """Mean over turns 2..n of (1 - max cosine similarity to any earlier turn)."""
    if len(embs) < 2:
        return float("nan")
    vals = []
    for i in range(1, len(embs)):
        vals.append(1.0 - max(cosine(embs[i], embs[j]) for j in range(i)))
    return sum(vals) / len(vals)


def local_consecutive_similarity(embs) -> float:
    if len(embs) < 2:
        return float("nan")
    return sum(cosine(embs[i - 1], embs[i])
               for i in range(1, len(embs))) / (len(embs) - 1)


def embed(texts: list[str], model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("error: sentence-transformers not installed in this environment.\n"
                 "       activate your project venv, or re-run with --no-metrics.")
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=False)


# ==========================================================================
# Reporting
# ==========================================================================
def sparkline(values: list[Optional[float]], threshold: Optional[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    vals = [v for v in values if v is not None]
    if not vals:
        return "(no fixation scores stored)"
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        if v is None:
            out.append(" ")
        else:
            out.append(blocks[min(7, int((v - lo) / span * 7.999))])
    line = "".join(out)
    tail = f"   min {lo:.3f}  max {hi:.3f}"
    if threshold is not None:
        crossings = sum(1 for v in vals if v >= threshold)
        tail += f"  |  >= threshold ({threshold:.2f}) on {crossings}/{len(vals)} turns"
    return line + tail


def report(turns: list[Turn], session_id: str, meta: dict,
           metrics: dict, threshold: Optional[float]) -> None:
    W = 78
    print("\n" + "=" * W)
    print(f" SESSION {session_id}")
    for k, v in meta.items():
        print(f"   {k:<18}: {v}")
    print(f"   {'turns':<18}: {len(turns)} "
          f"({sum(1 for t in turns if t.is_user)} from participant)")
    print("=" * W)

    # --- turn table -------------------------------------------------------
    has_comp = any(t.similarity is not None for t in turns)
    print("\n TURN-BY-TURN")
    hdr = f" {'#':>3} {'who':<9} {'fixation':>9}"
    if has_comp:
        hdr += f" {'sim':>6} {'disp':>6} {'traj':>6}"
    hdr += f" {'int':>4} {'branch':<16}"
    print(hdr)
    print(" " + "-" * min(W - 2, len(hdr)))
    for t in turns:
        fx = f"{t.fixation:.4f}" if t.fixation is not None else "   -"
        flag = "***" if t.intervened else ("ev" if t.evaluated else "")
        mark = "^" if (threshold is not None and t.fixation is not None
                       and t.fixation >= threshold) else " "
        line = f" {t.n:>3} {t.speaker[:9]:<9} {fx:>8}{mark}"
        if has_comp:
            for v in (t.similarity, t.dispersion, t.trajectory):
                line += f" {v:>6.3f}" if v is not None else f" {'-':>6}"
        line += f" {flag:>4} {(t.branch or '')[:16]:<16}"
        print(line)
    ev_n = sum(1 for t in turns if t.evaluated)
    if ev_n:
        print(f"\n   'ev' = fixation evaluated but no intervention fired. "
              f"{ev_n}/{len(turns)} turns evaluated.")

    # --- trajectory -------------------------------------------------------
    scored = [t for t in turns if t.fixation is not None]
    print(f"\n FIXATION TRAJECTORY  ({len(scored)} scored turns, "
          f"'!' marks an intervention)")
    print("  " + sparkline([t.fixation for t in scored], threshold))
    print("  " + "".join("!" if t.intervened else " " for t in scored))

    # --- intervention alignment -------------------------------------------
    iv = [t for t in turns if t.intervened]
    print(f"\n INTERVENTIONS: {len(iv)}")
    if not iv:
        print("   none recorded — check --col-intervened is mapped correctly,")
        print("   or the strategy genuinely never triggered.")
    for t in iv:
        idx = turns.index(t)
        # window over *scored* turns only — assistant turns carry no score
        before = [x.fixation for x in turns[:idx] if x.fixation is not None][-3:]
        after = [x.fixation for x in turns[idx + 1:]
                 if x.fixation is not None][:3]
        b = f"{sum(before)/len(before):.4f}" if before else "n/a"
        a = f"{sum(after)/len(after):.4f}" if after else "n/a"
        rose = ("rising" if len(before) >= 2 and before[-1] > before[0]
                else "not rising" if len(before) >= 2 else "too few scored turns")
        arrow = ""
        if before and after:
            d = (sum(after) / len(after)) - (sum(before) / len(before))
            arrow = f"  [{d:+.4f}]"
        print(f"   turn {t.n:>3}  branch={t.branch or '?':<18}")
        print(f"        fixation: mean of 3 scored turns before {b} "
              f"-> after {a}{arrow}")
        print(f"        pre-intervention trend: {rose}")

    # --- metrics ----------------------------------------------------------
    if metrics:
        print("\n CONVERSATION METRICS (participant turns only)")
        src = metrics.pop("_source", "local implementation")
        for k, v in metrics.items():
            if isinstance(v, dict):
                print(f"   {k}")
                for kk, vv in v.items():
                    sv = f"{vv:.4f}" if isinstance(vv, float) and not math.isnan(vv) else str(vv)
                    print(f"      {kk:<28}: {sv}")
            else:
                sv = f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v)
                print(f"   {k:<31}: {sv}")
        print(f"\n   metric source: {src}")
    print()


def write_markdown(path: str, session_id: str, meta: dict, turns: list[Turn],
                   metrics: dict) -> None:
    L = [f"# Session record — `{session_id}`", ""]
    for k, v in meta.items():
        L.append(f"- **{k}**: {v}")
    L.append(f"- **turns**: {len(turns)}")
    L += ["", "## Fixation trajectory", "",
          "| # | speaker | fixation | intervention | branch |",
          "|---|---------|----------|--------------|--------|"]
    for t in turns:
        fx = f"{t.fixation:.4f}" if t.fixation is not None else ""
        L.append(f"| {t.n} | {t.speaker} | {fx} | "
                 f"{'yes' if t.intervened else ''} | {t.branch or ''} |")
    if metrics:
        L += ["", "## Metrics", "", "| metric | value |", "|--------|-------|"]
        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    L.append(f"| {k}: {kk} | {vv:.4f} |"
                             if isinstance(vv, float) else f"| {k}: {kk} | {vv} |")
            elif isinstance(v, float):
                L.append(f"| {k} | {v:.4f} |")
            else:
                L.append(f"| {k} | {v} |")
    open(path, "w").write("\n".join(L) + "\n")
    print(f"wrote {path}")


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="fixation.db")
    ap.add_argument("--session", help="session id to analyse")
    ap.add_argument("--list", action="store_true", help="list sessions and exit")
    ap.add_argument("--schema", action="store_true",
                    help="dump tables/columns and the detected mapping, then exit")
    ap.add_argument("--threshold", type=float, default=0.17,
                    help="fixation score threshold, for the trajectory markers")
    ap.add_argument("--model", default=os.environ.get(
        "FIXATION_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    ap.add_argument("--no-metrics", action="store_true",
                    help="skip embedding computation (no sentence-transformers needed)")
    ap.add_argument("--json", help="write the full record to this JSON path")
    ap.add_argument("--md", help="write a markdown table to this path")
    # schema overrides
    ap.add_argument("--turns-table")
    ap.add_argument("--sessions-table")
    ap.add_argument("--events-table")
    ap.add_argument("--col-triggered")
    ap.add_argument("--recompute-embeddings", action="store_true",
                    help="ignore embeddings stored in the DB and re-encode")
    ap.add_argument("--col-content")
    ap.add_argument("--col-fixation")
    ap.add_argument("--col-session-fk")
    ap.add_argument("--col-speaker")
    ap.add_argument("--col-turn-index")
    ap.add_argument("--col-intervened")
    ap.add_argument("--col-branch")
    args = ap.parse_args()

    con = connect(args.db)
    m = build_mapping(con, args)

    if args.schema:
        print_schema(con, m)
        return
    if args.list or not args.session:
        list_sessions(con, m)
        if not args.session:
            print("pass --session <id> to analyse one of the above.\n")
        return

    turns = load_turns(con, m, args.session)
    if not turns:
        sys.exit(f"error: no turns found for session {args.session!r}. "
                 f"run --list to see valid ids.")

    meta = {}
    if m.sessions_table and m.session_pk:
        r = con.execute(f'SELECT * FROM "{m.sessions_table}" '
                        f'WHERE "{m.session_pk}" = ?', (args.session,)).fetchone()
        if r:
            meta = {k: r[k] for k in r.keys() if k != m.session_pk}

    metrics: dict = {}
    if not args.no_metrics:
        user_texts = [t.content for t in turns if t.is_user and t.content.strip()]
        if len(user_texts) < 2:
            print("note: fewer than 2 participant turns — skipping metrics.")
        else:
            modname, proj = try_import_project_metrics()
            user_turns = [t for t in turns if t.is_user and t.content.strip()]
            stored = [t.embedding for t in user_turns]
            if all(x for x in stored) and not args.recompute_embeddings:
                import numpy as np
                embs = np.asarray(stored, dtype=float)
                print(f"using {len(embs)} embeddings already stored in the database "
                      f"(pass --recompute-embeddings to re-encode instead)")
            else:
                embs = embed(user_texts, args.model)
            if proj:
                metrics["_source"] = f"your project module `{modname}` (preferred)"
                for name, fn in proj.items():
                    try:
                        metrics[name] = float(fn(embs))
                    except Exception:
                        try:
                            metrics[name] = float(fn(user_texts))
                        except Exception as e:
                            metrics[name] = f"failed: {e}"
            else:
                metrics["_source"] = ("local fallback — project metrics module not "
                                      "importable from cwd; verify definitions match")
                metrics["semantic_diversity"] = local_semantic_diversity(embs)
                metrics["embedding_dispersion"] = local_embedding_dispersion(embs)
                metrics["novelty_score"] = local_novelty_score(embs)
                metrics["consecutive_similarity"] = local_consecutive_similarity(embs)

            # split around the first intervention: the triangulation payload
            first = next((i for i, t in enumerate(turns) if t.intervened), None)
            if first is not None:
                pre_idx = [i for i, t in enumerate(turns)
                           if t.is_user and i < first and t.content.strip()]
                post_idx = [i for i, t in enumerate(turns)
                            if t.is_user and i > first and t.content.strip()]
                if len(pre_idx) >= 2 and len(post_idx) >= 2:
                    order = [i for i, t in enumerate(turns)
                             if t.is_user and t.content.strip()]
                    pos = {orig: k for k, orig in enumerate(order)}
                    pre = [embs[pos[i]] for i in pre_idx]
                    post = [embs[pos[i]] for i in post_idx]
                    metrics["before_vs_after_first_intervention"] = {
                        "diversity_before": local_semantic_diversity(pre),
                        "diversity_after": local_semantic_diversity(post),
                        "dispersion_before": local_embedding_dispersion(pre),
                        "dispersion_after": local_embedding_dispersion(post),
                        "novelty_before": local_novelty_score(pre),
                        "novelty_after": local_novelty_score(post),
                    }
                else:
                    print("note: too few participant turns either side of the first "
                          "intervention for a before/after split.")

    report(turns, args.session, meta, dict(metrics), args.threshold)

    if args.json:
        metrics.pop("_source", None)
        payload = {"session_id": args.session, "meta": meta,
                   "turns": [asdict(t) for t in turns], "metrics": metrics}
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"wrote {args.json}")
    if args.md:
        metrics.pop("_source", None)
        write_markdown(args.md, args.session, meta, turns, metrics)


if __name__ == "__main__":
    main()
