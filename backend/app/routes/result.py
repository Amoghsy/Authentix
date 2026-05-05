from fastapi import APIRouter

router = APIRouter()

@router.get("/result")
def result():
    return {
        "message": "Use POST /detect for the full end-to-end deepfake detection pipeline.",
        "docs": "Swagger available at /docs"
    }