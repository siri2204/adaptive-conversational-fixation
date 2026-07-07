"""
Sanity-check the fixation math with synthetic embeddings:
  - a "fixated" conversation: near-identical vectors (should score high)
  - an "exploratory" conversation: spread-out, orthogonal-ish vectors (should score low)
"""
import math
import random

from app.fixation import FixationAnalyzer


def _unit(vec):
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def test_fixated_conversation_scores_high():
    random.seed(0)
    base = _unit([1.0, 0.1, 0.0, 0.0])
    embeddings = []
    for _ in range(8):
        noisy = _unit([base[i] + random.uniform(-0.02, 0.02) for i in range(len(base))])
        embeddings.append(noisy)

    analyzer = FixationAnalyzer(window=6)
    result = analyzer.analyze(embeddings)

    assert result.avg_similarity > 0.9
    assert result.is_fixated is True


def test_exploratory_conversation_scores_low():
    random.seed(1)
    dim = 16
    embeddings = [_unit([random.uniform(-1, 1) for _ in range(dim)]) for _ in range(8)]

    analyzer = FixationAnalyzer(window=6)
    result = analyzer.analyze(embeddings)

    assert result.avg_similarity < 0.5
    assert result.is_fixated is False


def test_too_few_turns_is_never_fixated():
    analyzer = FixationAnalyzer()
    result = analyzer.analyze([[1.0, 0.0]])
    assert result.is_fixated is False
