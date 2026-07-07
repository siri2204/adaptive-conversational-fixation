from app.fixation import FixationAnalyzer
from app.strategies import (
    StrategyContext,
    BaselineStrategy,
    StaticBranchingStrategy,
    FixedIntervalStrategy,
    UserTriggeredStrategy,
    AdaptiveStrategy,
    get_strategy,
)


def _ctx(turn_index, last_intervention_turn=-1, user_forced=False, embeddings=None):
    return StrategyContext(
        turn_index=turn_index,
        embeddings_so_far=embeddings or [[1.0, 0.0]] * (turn_index + 1),
        last_intervention_turn=last_intervention_turn,
        user_forced=user_forced,
    )


def test_baseline_never_intervenes():
    s = BaselineStrategy()
    assert s.decide(_ctx(0)).intervene is False
    assert s.decide(_ctx(10)).intervene is False


def test_static_only_fires_once_at_start():
    s = StaticBranchingStrategy()
    assert s.decide(_ctx(0, last_intervention_turn=-1)).intervene is True
    assert s.decide(_ctx(0, last_intervention_turn=0)).intervene is False
    assert s.decide(_ctx(3, last_intervention_turn=-1)).intervene is False


def test_fixed_interval_fires_on_multiples():
    s = FixedIntervalStrategy(interval=5)
    assert s.decide(_ctx(5)).intervene is True
    assert s.decide(_ctx(10)).intervene is True
    assert s.decide(_ctx(6)).intervene is False
    assert s.decide(_ctx(0)).intervene is False  # turn 0 excluded


def test_user_triggered_only_on_force():
    s = UserTriggeredStrategy()
    assert s.decide(_ctx(3, user_forced=False)).intervene is False
    assert s.decide(_ctx(3, user_forced=True)).intervene is True


def test_adaptive_respects_cooldown():
    analyzer = FixationAnalyzer(window=6, score_threshold=0.0)  # force is_fixated True whenever enough turns
    s = AdaptiveStrategy(analyzer=analyzer, cooldown=4)

    fixated_embeddings = [[1.0, 0.0]] * 6

    # First check: never intervened before -> cooldown satisfied -> should fire
    d1 = s.decide(_ctx(6, last_intervention_turn=-1, embeddings=fixated_embeddings))
    assert d1.intervene is True

    # Immediately after (turn 7, last intervention at 6) -> within cooldown -> should NOT fire
    d2 = s.decide(_ctx(7, last_intervention_turn=6, embeddings=fixated_embeddings))
    assert d2.intervene is False

    # After cooldown elapses -> should fire again
    d3 = s.decide(_ctx(10, last_intervention_turn=6, embeddings=fixated_embeddings))
    assert d3.intervene is True


def test_get_strategy_factory():
    analyzer = FixationAnalyzer()
    assert get_strategy("baseline", analyzer).name == "baseline"
    assert get_strategy("static", analyzer).name == "static"
    assert get_strategy("fixed_interval", analyzer, interval=3).interval == 3
    assert get_strategy("user_triggered", analyzer).name == "user_triggered"
    assert get_strategy("adaptive", analyzer, cooldown=2).cooldown == 2
