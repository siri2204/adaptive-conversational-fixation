"""
Post-hoc evaluation metrics used to compare strategies against each other
(proposal section 7). Operate on a full logged conversation: a list of
{"content": str, "embedding": list[float]} dicts in chronological order.
"""
from __future__ import annotations
import re

from app.fixation import cosine_similarity, centroid, euclidean_distance


def semantic_diversity(embeddings: list[list[float]]) -> float:
    """Mean pairwise cosine *distance* (1 - similarity) across the whole
    conversation. Higher = the conversation covered more distinct semantic
    territory overall."""
    if len(embeddings) < 2:
        return 0.0
    dists = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dists.append(1 - cosine_similarity(embeddings[i], embeddings[j]))
    return sum(dists) / len(dists)


def embedding_dispersion(embeddings: list[list[float]]) -> float:
    """Mean distance of every turn embedding to the global centroid. Higher =
    turns are spread more widely around the conversation's 'center of mass'."""
    if len(embeddings) < 2:
        return 0.0
    c = centroid(embeddings)
    dists = [euclidean_distance(v, c) for v in embeddings]
    return sum(dists) / len(dists)


def lexical_diversity(texts: list[str]) -> float:
    """Simple type-token ratio (TTR) across all turns combined. A cheap,
    dependency-free stand-in for MTLD; swap in `lexicalrichness` later if you
    want the real MTLD/MATTR/HD-D metrics for the writeup."""
    tokens = []
    for t in texts:
        tokens.extend(re.findall(r"[a-z0-9']+", t.lower()))
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def novelty_score(embeddings: list[list[float]]) -> float:
    """For each turn (after the first few), measure its distance to the
    centroid of *all prior* turns, then average. Higher = each new turn tends
    to introduce genuinely new semantic content rather than restating what
    came before."""
    if len(embeddings) < 3:
        return 0.0
    novelties = []
    for i in range(2, len(embeddings)):
        prior_centroid = centroid(embeddings[:i])
        novelties.append(euclidean_distance(embeddings[i], prior_centroid))
    return sum(novelties) / len(novelties)


def evaluate_conversation(turns: list[dict]) -> dict:
    """turns: list of {"content": str, "embedding": list[float]}"""
    embeddings = [t["embedding"] for t in turns]
    texts = [t["content"] for t in turns]
    return {
        "semantic_diversity": semantic_diversity(embeddings),
        "embedding_dispersion": embedding_dispersion(embeddings),
        "lexical_diversity": lexical_diversity(texts),
        "novelty_score": novelty_score(embeddings),
        "num_turns": len(turns),
    }
