"""
Tests for the post-hoc evaluation metrics (semantic diversity, dispersion,
lexical diversity, novelty). These directly back the numbers your experiment
results and report will rely on, so it's worth being confident they behave
correctly on known-shape inputs before trusting them on real data.
"""
from app.evaluation import (
    semantic_diversity,
    embedding_dispersion,
    lexical_diversity,
    novelty_score,
    evaluate_conversation,
)


def test_semantic_diversity_zero_for_identical_embeddings():
    embeddings = [[1.0, 0.0, 0.0]] * 5
    assert semantic_diversity(embeddings) == 0.0


def test_semantic_diversity_high_for_orthogonal_embeddings():
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    # Orthogonal vectors have cosine similarity 0, so distance (1 - sim) = 1
    assert semantic_diversity(embeddings) == 1.0


def test_semantic_diversity_needs_at_least_two_points():
    assert semantic_diversity([[1.0, 0.0]]) == 0.0
    assert semantic_diversity([]) == 0.0


def test_embedding_dispersion_zero_when_all_identical():
    embeddings = [[2.0, 3.0]] * 4
    assert embedding_dispersion(embeddings) == 0.0


def test_embedding_dispersion_positive_when_spread_out():
    embeddings = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]]
    assert embedding_dispersion(embeddings) > 0.0


def test_lexical_diversity_all_unique_words_is_one():
    texts = ["the quick brown fox jumps"]
    assert lexical_diversity(texts) == 1.0


def test_lexical_diversity_lower_with_repeated_words():
    texts = ["the cat sat on the mat the cat slept"]
    result = lexical_diversity(texts)
    assert 0.0 < result < 1.0


def test_lexical_diversity_empty_input():
    assert lexical_diversity([]) == 0.0
    assert lexical_diversity([""]) == 0.0


def test_novelty_score_needs_at_least_three_turns():
    assert novelty_score([[1.0, 0.0], [0.0, 1.0]]) == 0.0


def test_novelty_score_positive_for_diverging_conversation():
    # Each new turn moves further from the centroid of everything before it.
    embeddings = [[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [10.0, -10.0]]
    assert novelty_score(embeddings) > 0.0


def test_novelty_score_zero_for_static_conversation():
    embeddings = [[1.0, 1.0]] * 5
    assert novelty_score(embeddings) == 0.0


def test_evaluate_conversation_integration():
    turns = [
        {"content": "the quick brown fox", "embedding": [1.0, 0.0, 0.0]},
        {"content": "jumps over the lazy dog", "embedding": [0.0, 1.0, 0.0]},
        {"content": "a completely different sentence here", "embedding": [0.0, 0.0, 1.0]},
    ]
    result = evaluate_conversation(turns)
    assert set(result.keys()) == {
        "semantic_diversity",
        "embedding_dispersion",
        "lexical_diversity",
        "novelty_score",
        "num_turns",
    }
    assert result["num_turns"] == 3
    assert result["semantic_diversity"] > 0
    assert result["embedding_dispersion"] > 0
