from pydantic import BaseModel, Field
from typing import Literal, List

class UserIntent(BaseModel):
    """Classifies what the user is asking for."""
    intent_type: Literal["qa", "summarize", "calculate"] = Field(
        ..., description="The classified intent of the user's request"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        ..., description="Brief explanation for why this intent was chosen"
    )

class AnswerResponse(BaseModel):
    """The final structured response returned to the user."""
    answer: str = Field(..., description="The main response content")
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in the answer"
    )
    sources: List[str] = Field(
        default_factory=list, description="Document sections or sources used"
    )
    tool_calls_made: List[str] = Field(
        default_factory=list, description="Names of tools invoked to produce this answer"
    )
