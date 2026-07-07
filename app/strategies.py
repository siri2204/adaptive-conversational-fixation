"""
Implements the four intervention-timing strategies from the proposal:

  (a) StaticBranchingStrategy   — offer branches once, at conversation start.
  (b) FixedIntervalStrategy     — offer branches every N turns, regardless of content.
  (c) UserTriggeredStrategy     — only offer branches when the user explicitly asks.
  (d) AdaptiveStrategy          — offer branches when FixationAnalyzer detects
                                  convergence, subject to a cooldown so it doesn't
                                  fire every single turn once triggered.

Each strategy exposes the same interface: `should_intervene(context) -> bool`,
where `context` carries everything a strategy might need (turn index, embeddings
so far, whether the user explicitly requested it, cooldown state). This uniform
interface is what makes the four-way experimental comparison in the proposal
possible with one code path.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.fixation import FixationAnalyzer, FixationResult


@dataclass
class StrategyContext:
    turn_index: int  # index of the assistant turn about to be produced
    embeddings_so_far: list[list[float]]
    last_intervention_turn: int  # -1 if never
    user_forced: bool = False


@dataclass
class StrategyDecision:
    intervene: bool
    fixation_result: FixationResult | None = None


class InterventionStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        ...


class BaselineStrategy(InterventionStrategy):
    """No intervention, ever. This is the control condition in the experiment."""

    name = "baseline"

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        return StrategyDecision(intervene=False)


class StaticBranchingStrategy(InterventionStrategy):
    """Offer the exploration tree exactly once, before the very first assistant turn."""

    name = "static"

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        return StrategyDecision(intervene=ctx.turn_index == 0 and ctx.last_intervention_turn == -1)


class FixedIntervalStrategy(InterventionStrategy):
    name = "fixed_interval"

    def __init__(self, interval: int = 5):
        self.interval = interval

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        due = ctx.turn_index > 0 and ctx.turn_index % self.interval == 0
        already_done_this_turn = ctx.last_intervention_turn == ctx.turn_index
        return StrategyDecision(intervene=due and not already_done_this_turn)


class UserTriggeredStrategy(InterventionStrategy):
    name = "user_triggered"

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        return StrategyDecision(intervene=ctx.user_forced)


class AdaptiveStrategy(InterventionStrategy):
    """Fire when the FixationAnalyzer says the conversation has converged,
    but not more often than `cooldown` turns apart."""

    name = "adaptive"

    def __init__(self, analyzer: FixationAnalyzer, cooldown: int = 4):
        self.analyzer = analyzer
        self.cooldown = cooldown

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        result = self.analyzer.analyze(ctx.embeddings_so_far)
        cooldown_ok = (ctx.last_intervention_turn == -1) or (
            ctx.turn_index - ctx.last_intervention_turn >= self.cooldown
        )
        return StrategyDecision(intervene=result.is_fixated and cooldown_ok, fixation_result=result)


def get_strategy(name: str, analyzer: FixationAnalyzer, **params) -> InterventionStrategy:
    if name == "baseline":
        return BaselineStrategy()
    if name == "static":
        return StaticBranchingStrategy()
    if name == "fixed_interval":
        return FixedIntervalStrategy(interval=params.get("interval", 5))
    if name == "user_triggered":
        return UserTriggeredStrategy()
    if name == "adaptive":
        return AdaptiveStrategy(analyzer=analyzer, cooldown=params.get("cooldown", 4))
    raise ValueError(f"Unknown strategy: {name}")
