from fastapi import APIRouter
from app.services.explanation_service import generate_explanation

router = APIRouter()

@router.post("/explain")
def explain(data: dict):
    return generate_explanation(data["features"], data["scores"])