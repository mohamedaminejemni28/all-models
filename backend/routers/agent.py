"""API router for the GaitML Research Assistant."""

from pydantic import BaseModel
from fastapi import APIRouter

from Agents.gait_agent import GaitModelAgent

router = APIRouter(prefix="/api/agent", tags=["agent"])
agent = GaitModelAgent()


class AgentChatRequest(BaseModel):
    message: str
    dataset: str | None = None


class AgentChatResponse(BaseModel):
    answer: str
    dataset: str | None = None
    intent: str | None = None


@router.post("/chat", response_model=AgentChatResponse)
def chat(request: AgentChatRequest):
    result = agent.answer_structured(request.message, request.dataset)
    return AgentChatResponse(**result)