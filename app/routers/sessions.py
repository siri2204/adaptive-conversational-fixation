import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from app.config import settings
from app.database import get_db, ConversationSession, Turn, InterventionEvent
from app.embeddings import get_embedding_backend
from app.llm_client import get_llm_client
from app.fixation import FixationAnalyzer
from app.strategies import get_strategy, StrategyContext
from app.intervention import ExplorationTreeGenerator
from app.evaluation import evaluate_conversation
from app.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    SendMessageRequest,
    SendMessageResponse,
    SelectBranchRequest,
    SessionHistoryResponse,
    TurnOut,
    FixationMetrics,
    EvaluationResponse,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])

analyzer = FixationAnalyzer(
    window=settings.fixation_window,
    similarity_threshold=settings.fixation_similarity_threshold,
    dispersion_threshold=settings.fixation_dispersion_threshold,
    score_threshold=settings.fixation_score_threshold,
)


def _get_session_or_404(db: DBSession, session_id: str) -> ConversationSession:
    sess = db.query(ConversationSession).filter(ConversationSession.id == session_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


def _get_turns(db: DBSession, session_id: str) -> list[Turn]:
    """Always query turns fresh, ordered by turn_index. We deliberately do NOT
    read through the `sess.turns` relationship here: SQLAlchemy lazy-loads and
    caches that collection on first access, and since new Turn rows are
    inserted via their session_id foreign key directly (not via the
    relationship), that cached collection goes stale for the rest of the
    request. Querying explicitly avoids that class of bug entirely."""
    return db.query(Turn).filter(Turn.session_id == session_id).order_by(Turn.turn_index).all()


def _history_as_messages(turns: list[Turn]) -> list[dict]:
    return [{"role": t.role if t.role in ("user", "assistant") else "user", "content": t.content} for t in turns]


def _append_turn(db: DBSession, sess: ConversationSession, role: str, content: str, next_index: int) -> Turn:
    embedding = get_embedding_backend().embed(content)
    turn = Turn(
        session_id=sess.id,
        turn_index=next_index,
        role=role,
        content=content,
        embedding_json=json.dumps(embedding),
    )
    db.add(turn)
    db.flush()
    return turn


def _run_strategy_and_maybe_intervene(
    db: DBSession, sess: ConversationSession, turns: list[Turn], user_forced: bool
) -> tuple[bool, dict | None, FixationMetrics]:
    strategy = get_strategy(sess.strategy, analyzer, **sess.get_strategy_params())
    embeddings = [t.embedding() for t in turns]
    ctx = StrategyContext(
        turn_index=len(turns),
        embeddings_so_far=embeddings,
        last_intervention_turn=sess.last_intervention_turn,
        user_forced=user_forced,
    )
    decision = strategy.decide(ctx)
    if user_forced and not decision.intervene:
        decision.intervene = True

    # Always compute fixation metrics for observability/evaluation, even for
    # strategies that don't use them to decide (baseline, static, fixed_interval).
    fixation_result = decision.fixation_result or analyzer.analyze(embeddings)
    metrics = FixationMetrics(
        avg_similarity=fixation_result.avg_similarity,
        dispersion=fixation_result.dispersion,
        trajectory_movement=fixation_result.trajectory_movement,
        fixation_score=fixation_result.fixation_score,
        is_fixated=fixation_result.is_fixated,
    )

    branches = None
    if decision.intervene:
        generator = ExplorationTreeGenerator(get_llm_client())
        branches = generator.generate(_history_as_messages(turns), sess.task_type)
        sess.last_intervention_turn = ctx.turn_index
        db.add(
            InterventionEvent(
                session_id=sess.id,
                turn_index=ctx.turn_index,
                strategy=sess.strategy,
                triggered=True,
                fixation_score=metrics.fixation_score,
                avg_similarity=metrics.avg_similarity,
                dispersion=metrics.dispersion,
                trajectory_movement=metrics.trajectory_movement,
                branches_json=json.dumps(branches),
            )
        )
        db.flush()

    return decision.intervene, branches, metrics


@router.post("", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest, db: DBSession = Depends(get_db)):
    session_id = str(uuid.uuid4())
    sess = ConversationSession(
        id=session_id,
        task_type=req.task_type,
        strategy=req.strategy,
        strategy_params=json.dumps(req.strategy_params),
    )
    db.add(sess)
    db.flush()

    initial_branches = None
    assistant_message = None

    if req.seed_prompt:
        _append_turn(db, sess, "user", req.seed_prompt, next_index=0)
        turns = _get_turns(db, session_id)
        intervened, branches, _ = _run_strategy_and_maybe_intervene(db, sess, turns, user_forced=False)
        if intervened:
            initial_branches = branches
        else:
            reply = get_llm_client().generate(_history_as_messages(turns))
            _append_turn(db, sess, "assistant", reply, next_index=len(turns))
            assistant_message = reply

    db.commit()
    return CreateSessionResponse(
        session_id=session_id,
        strategy=req.strategy,
        initial_branches=initial_branches,
        assistant_message=assistant_message,
    )


@router.post("/{session_id}/messages", response_model=SendMessageResponse)
def send_message(session_id: str, req: SendMessageRequest, db: DBSession = Depends(get_db)):
    sess = _get_session_or_404(db, session_id)
    existing_turns = _get_turns(db, session_id)
    _append_turn(db, sess, "user", req.content, next_index=len(existing_turns))

    turns = _get_turns(db, session_id)
    intervened, branches, metrics = _run_strategy_and_maybe_intervene(db, sess, turns, user_forced=req.force_intervene)

    assistant_message = None
    if not intervened:
        reply = get_llm_client().generate(_history_as_messages(turns))
        _append_turn(db, sess, "assistant", reply, next_index=len(turns))
        assistant_message = reply
        turns = _get_turns(db, session_id)

    db.commit()
    return SendMessageResponse(
        turn_index=len(turns) - 1,
        assistant_message=assistant_message,
        metrics=metrics,
        intervened=intervened,
        branches=branches,
    )


@router.post("/{session_id}/branches/select", response_model=SendMessageResponse)
def select_branch(session_id: str, req: SelectBranchRequest, db: DBSession = Depends(get_db)):
    sess = _get_session_or_404(db, session_id)

    last_event = (
        db.query(InterventionEvent)
        .filter(InterventionEvent.session_id == session_id)
        .order_by(InterventionEvent.id.desc())
        .first()
    )
    if not last_event or not last_event.branches_json:
        raise HTTPException(status_code=400, detail="No pending exploration tree for this session")

    branches = json.loads(last_event.branches_json)
    node = branches.get(req.branch_category)
    if not node:
        raise HTTPException(status_code=400, detail=f"Unknown branch category: {req.branch_category}")

    continuation = req.branch_content or node["prompt"]
    last_event.selected_branch = req.branch_category

    turns = _get_turns(db, session_id)
    _append_turn(db, sess, "user", continuation, next_index=len(turns))

    turns = _get_turns(db, session_id)
    reply = get_llm_client().generate(_history_as_messages(turns))
    _append_turn(db, sess, "assistant", reply, next_index=len(turns))

    turns = _get_turns(db, session_id)
    embeddings = [t.embedding() for t in turns]
    fixation_result = analyzer.analyze(embeddings)
    metrics = FixationMetrics(
        avg_similarity=fixation_result.avg_similarity,
        dispersion=fixation_result.dispersion,
        trajectory_movement=fixation_result.trajectory_movement,
        fixation_score=fixation_result.fixation_score,
        is_fixated=fixation_result.is_fixated,
    )

    db.commit()
    return SendMessageResponse(
        turn_index=len(turns) - 1,
        assistant_message=reply,
        metrics=metrics,
        intervened=False,
        branches=None,
    )


@router.get("/{session_id}", response_model=SessionHistoryResponse)
def get_session(session_id: str, db: DBSession = Depends(get_db)):
    sess = _get_session_or_404(db, session_id)
    turns = _get_turns(db, session_id)
    turns_out = [TurnOut(turn_index=t.turn_index, role=t.role, content=t.content) for t in turns]

    metrics_timeline = []
    embeddings_running: list[list[float]] = []
    for t in turns:
        embeddings_running.append(t.embedding())
        r = analyzer.analyze(embeddings_running)
        metrics_timeline.append(
            FixationMetrics(
                avg_similarity=r.avg_similarity,
                dispersion=r.dispersion,
                trajectory_movement=r.trajectory_movement,
                fixation_score=r.fixation_score,
                is_fixated=r.is_fixated,
            )
        )

    intervention_count = db.query(InterventionEvent).filter(InterventionEvent.session_id == session_id).count()

    return SessionHistoryResponse(
        session_id=sess.id,
        task_type=sess.task_type,
        strategy=sess.strategy,
        turns=turns_out,
        metrics_timeline=metrics_timeline,
        intervention_count=intervention_count,
    )


@router.get("/{session_id}/evaluation", response_model=EvaluationResponse)
def get_evaluation(session_id: str, db: DBSession = Depends(get_db)):
    _get_session_or_404(db, session_id)
    turns = _get_turns(db, session_id)
    turn_dicts = [{"content": t.content, "embedding": t.embedding()} for t in turns]
    if not turn_dicts:
        raise HTTPException(status_code=400, detail="Session has no turns yet")

    result = evaluate_conversation(turn_dicts)
    intervention_count = db.query(InterventionEvent).filter(InterventionEvent.session_id == session_id).count()
    return EvaluationResponse(**result, num_interventions=intervention_count)
