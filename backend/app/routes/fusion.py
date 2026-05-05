from fastapi import APIRouter
from app.services.fusion_service import fuse_scores

router = APIRouter()

@router.post("/fusion-decision")
def fusion(data: dict):
    """
    Expects: { "model_scores": {...}, "heuristic_scores": {...} }
    """
    return fuse_scores(data["model_scores"], data["heuristic_scores"])