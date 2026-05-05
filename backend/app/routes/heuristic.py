from fastapi import APIRouter
from app.services.heuristic_service import heuristic_analysis

router = APIRouter()

@router.post("/heuristic-analysis")
def heuristic(data: dict):
    return heuristic_analysis(data["features"])