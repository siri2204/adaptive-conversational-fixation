"""
Core fixation-detection logic: given a list of turn embeddings (chronological),
estimate whether the conversation has converged onto a narrow semantic region.

Three signals, each computed over a sliding window of the most recent N turns:

1. avg_similarity — mean pairwise cosine similarity within the window.
   High similarity => turns are saying near-identical things => fixation.

2. dispersion — mean distance of each embedding to the window's centroid
   (i.e. how spread out the point cloud is). Low dispersion => fixation.

3. trajectory_movement — how far the centroid of the *second half* of the
   window has moved from the centroid of the *first half*. Low movement
   means the conversation isn't drifting anywhere new even turn-to-turn.

These are combined into a single fixation_score in [0, 1] (higher = more
fixated), which is what the intervention strategies threshold against.
"""
from __future__ import annotations
import math
from dataclasses import dataclass


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


def centroid(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += x
    return [x / len(vectors) for x in out]


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class FixationResult:
    avg_similarity: float
    dispersion: float
    trajectory_movement: float
    fixation_score: float
    is_fixated: bool


class FixationAnalyzer:
    def __init__(
        self,
        window: int = 6,
        similarity_threshold: float = 0.82,
        dispersion_threshold: float = 0.15,
        score_threshold: float = 0.65,
        similarity_weight: float = 0.5,
        dispersion_weight: float = 0.3,
        trajectory_weight: float = 0.2,
    ):
        self.window = window
        self.similarity_threshold = similarity_threshold
        self.dispersion_threshold = dispersion_threshold
        self.score_threshold = score_threshold
        self.w_sim = similarity_weight
        self.w_disp = dispersion_weight
        self.w_traj = trajectory_weight

    def analyze(self, embeddings: list[list[float]]) -> FixationResult:
        if len(embeddings) < 2:
            return FixationResult(0.0, 1.0, 1.0, 0.0, False)

        window_embeds = embeddings[-self.window :]

        avg_similarity = self._avg_pairwise_similarity(window_embeds)
        dispersion = self._dispersion(window_embeds)
        trajectory_movement = self._trajectory_movement(window_embeds)

        # Normalize each raw signal into a "fixation contribution" in [0,1].
        sim_contrib = _clamp01(avg_similarity)  # already in [-1,1]-ish, clamp to [0,1]
        disp_contrib = _clamp01(1.0 - dispersion / max(self.dispersion_threshold * 2, 1e-6))
        traj_contrib = _clamp01(1.0 - trajectory_movement / max(self.dispersion_threshold * 2, 1e-6))

        fixation_score = (
            self.w_sim * sim_contrib + self.w_disp * disp_contrib + self.w_traj * traj_contrib
        )

        is_fixated = (
            fixation_score >= self.score_threshold
            or (avg_similarity >= self.similarity_threshold and dispersion <= self.dispersion_threshold)
        )

        return FixationResult(
            avg_similarity=avg_similarity,
            dispersion=dispersion,
            trajectory_movement=trajectory_movement,
            fixation_score=fixation_score,
            is_fixated=is_fixated,
        )

    def _avg_pairwise_similarity(self, embeds: list[list[float]]) -> float:
        if len(embeds) < 2:
            return 0.0
        sims = []
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                sims.append(cosine_similarity(embeds[i], embeds[j]))
        return sum(sims) / len(sims)

    def _dispersion(self, embeds: list[list[float]]) -> float:
        c = centroid(embeds)
        dists = [euclidean_distance(v, c) for v in embeds]
        return sum(dists) / len(dists)

    def _trajectory_movement(self, embeds: list[list[float]]) -> float:
        if len(embeds) < 4:
            return 1.0  # not enough data to call it stagnant; default to "moving"
        mid = len(embeds) // 2
        first_half_centroid = centroid(embeds[:mid])
        second_half_centroid = centroid(embeds[mid:])
        return euclidean_distance(first_half_centroid, second_half_centroid)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
