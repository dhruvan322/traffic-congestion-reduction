from functools import lru_cache

from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse
from engine.osm_traffic_router import load_traffic_dataset
from nlp.assistant import process_query


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@lru_cache(maxsize=1)
def _load_chat_dataset():
    """
    Load the historical traffic dataset once per API process.
    """
    return load_traffic_dataset()


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a traffic-related question using the existing NLP assistant.
    """

    query = request.query.strip()
    session_id = request.session_id.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query must not be empty.",
        )

    if not session_id:
        raise HTTPException(
            status_code=422,
            detail="Session ID must not be empty.",
        )

    try:
        df = _load_chat_dataset()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The historical traffic dataset is unavailable.",
        ) from exc

    data_context = {
        "traffic_data": df,
        "total_records": len(df),
    }

    try:
        result = process_query(
            query=query,
            session_id=session_id,
            data_context=data_context,
        )

        return ChatResponse(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The chat assistant could not process this request.",
        ) from exc