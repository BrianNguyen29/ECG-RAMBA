from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from app.services.tutor import tutor_service

router = APIRouter(
    prefix="/tutor",
    tags=["tutor"],
    responses={404: {"description": "Not found"}},
)

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    reply: str
    status: str

@router.post("/chat", response_model=ChatResponse)
async def chat_with_tutor(request: ChatRequest):
    """
    Send a message to the AI Clinical Tutor.
    """
    try:
        reply = await tutor_service.generate_reply(
            message=request.message,
            context=request.context,
            history=request.history
        )
        return ChatResponse(reply=reply, status="success")
    except Exception as e:
        print(f"Error in chat_with_tutor: {e}")
        raise HTTPException(status_code=500, detail=str(e))
