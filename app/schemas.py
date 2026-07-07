from pydantic import BaseModel, Field
from typing import Optional, Literal


TaskType = Literal["story_generation", "product_brainstorming", "interface_design", "other"]
StrategyName = Literal["baseline", "static", "fixed_interval", "user_triggered", "adaptive"]


class CreateSessionRequest(BaseModel):
    task_type: TaskType = "other"
    strategy: StrategyName = "adaptive"
    strategy_params: dict = Field(default_factory=dict)
    seed_prompt: Optional[str] = Field(
        default=None, description="Optional first user message to seed the conversation"
    )


class CreateSessionResponse(BaseModel):
    session_id: str
    strategy: StrategyName
    initial_branches: Optional[dict] = None  # populated if strategy == 'static'
    assistant_message: Optional[str] = None


class SendMessageRequest(BaseModel):
    content: str
    force_intervene: bool = Field(
        default=False, description="Set true for the user-triggered strategy's 'give me alternatives' button"
    )


class FixationMetrics(BaseModel):
    avg_similarity: float
    dispersion: float
    trajectory_movement: float
    fixation_score: float
    is_fixated: bool


class SendMessageResponse(BaseModel):
    turn_index: int
    assistant_message: Optional[str] = None
    metrics: FixationMetrics
    intervened: bool
    branches: Optional[dict] = None


class SelectBranchRequest(BaseModel):
    branch_category: str
    branch_content: Optional[str] = None  # user can edit the branch text before continuing


class TurnOut(BaseModel):
    turn_index: int
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    task_type: str
    strategy: str
    turns: list[TurnOut]
    metrics_timeline: list[FixationMetrics]
    intervention_count: int


class EvaluationResponse(BaseModel):
    semantic_diversity: float
    embedding_dispersion: float
    lexical_diversity: float
    novelty_score: float
    num_turns: int
    num_interventions: int
